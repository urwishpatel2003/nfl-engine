"""
ml/fantasy.py  —  Underdog best-ball fantasy value from play-by-play
====================================================================
Everything a best-ball drafter wants, derived from data we already own:

  • season_board(season)  — ranked half-PPR board for a completed season (PPG + total + pos rank)
  • project()             — a 2026 DRAFT board: recency-weighted production mapped onto each
                            player's CURRENT team, with a draft-capital prior for rookies
  • breakouts()           — "undervalued" by OPPORTUNITY vs output: players getting the volume
                            (target / touch share) but not yet the fantasy points → positive
                            regression. Needs NO market ADP.
  • with_adp(board)        — optional overlay: if data/raw/adp_underdog.csv exists (columns
                            player, position, adp) we merge it and flag value vs our rank.

Scoring = Underdog Best Ball Mania (half-PPR):
    passing   0.04/yd, 4/TD, -1/INT
    rushing   0.1/yd,  6/TD
    receiving 0.5/rec, 0.1/yd, 6/TD

Why from PBP: the trimmed feed has no fantasy_points column and player_stats.parquet only runs
through 2024, so we compute straight off pbp_{season}.parquet (covers 2025). Fumbles / 2-pt
conversions aren't in the trimmed feed — omitted (a rounding error on season totals).

HONEST SCOPE: these are OUR projections from production + opportunity, not a market consensus.
Fantasy scoring is noisy; treat the board as a lean, not gospel. The value/undervalued read is
strongest where opportunity and production disagree.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"

# Underdog half-PPR weights
_PASS_Y, _PASS_TD, _INT = 0.04, 4.0, 1.0
_RUSH_Y, _RUSH_TD = 0.1, 6.0
_REC, _REC_Y, _REC_TD = 0.5, 0.1, 6.0

FANTASY_POS = ["QB", "RB", "WR", "TE"]      # Underdog rosters are all offense
_PROJ_SEASONS = [2025, 2024, 2023]          # recency window for the draft board
_PROJ_W = {2025: 0.60, 2024: 0.30, 2023: 0.10}

# Value-over-replacement baselines: the positional rank whose production is "freely available"
# in a 12-team Underdog best-ball draft (1QB/2RB/3WR/1TE/1FLEX, ~18-man rosters). Ranking by raw
# points buries RB/WR under QBs (who score most but you only start one); VOR vs these baselines
# is how real draft boards flow — elite RB/WR first, QBs and TEs slide to their true cost.
_REPLACEMENT = {"QB": 16, "RB": 30, "WR": 42, "TE": 14}

_STATS = {}       # season -> per-player stats DataFrame (cached)
_ROSTER = None


def clear():
    global _STATS, _ROSTER
    _STATS = {}
    _ROSTER = None


# ──────────────────────────────────────────────────────────────────
#  Per-season half-PPR stats + opportunity, straight from PBP
# ──────────────────────────────────────────────────────────────────
def season_stats(season: int) -> pd.DataFrame:
    """Per-player half-PPR points + volume/opportunity for one season, from pbp_{season}."""
    if season in _STATS:
        return _STATS[season]
    path = RAW / f"pbp_{season}.parquet"
    if not path.exists():
        _STATS[season] = pd.DataFrame()
        return _STATS[season]
    p = pd.read_parquet(path)
    if "season_type" in p.columns:
        p = p[p["season_type"].eq("REG")]

    comp = p[p["complete_pass"].eq(1)]
    passing = (p[p["pass_attempt"].eq(1) & p["complete_pass"].eq(1) & p["passer_player_id"].notna()]
               .groupby("passer_player_id")
               .agg(pass_yds=("yards_gained", "sum"), pass_td=("touchdown", "sum")))
    ints = (p[p["pass_attempt"].eq(1) & p["passer_player_id"].notna()]
            .groupby("passer_player_id").agg(interceptions=("interception", "sum")))
    rushing = (p[p["rush_attempt"].eq(1) & p["rusher_player_id"].notna()]
               .groupby("rusher_player_id")
               .agg(rush_yds=("yards_gained", "sum"), rush_td=("touchdown", "sum"),
                    carries=("rush_attempt", "sum")))
    receiving = (comp[comp["receiver_player_id"].notna()]
                 .groupby("receiver_player_id")
                 .agg(rec_yds=("yards_gained", "sum"), rec_td=("touchdown", "sum"),
                      receptions=("complete_pass", "sum")))
    targeted = p[p["pass_attempt"].eq(1) & p["receiver_player_id"].notna()]
    tgt = targeted.groupby("receiver_player_id").agg(targets=("pass_attempt", "sum"))
    ay = (targeted.groupby("receiver_player_id").agg(air_yards=("air_yards", "sum"))
          if "air_yards" in targeted.columns else pd.DataFrame())

    # team volume (for share metrics)
    team_pass = targeted.groupby("posteam").size().rename("team_targets")
    team_rush = p[p["rush_attempt"].eq(1)].groupby("posteam").size().rename("team_carries")
    rec_team = (targeted.dropna(subset=["receiver_player_id"])
                .groupby("receiver_player_id")["posteam"]
                .agg(lambda s: s.value_counts().index[0]))       # primary team of a receiver
    rush_team = (p[p["rush_attempt"].eq(1)].dropna(subset=["rusher_player_id"])
                 .groupby("rusher_player_id")["posteam"].agg(lambda s: s.value_counts().index[0]))

    # games played in any role
    wk = []
    for idc in ["passer_player_id", "rusher_player_id", "receiver_player_id"]:
        wk.append(p[p[idc].notna()][[idc, "week"]].rename(columns={idc: "pid"}))
    games = pd.concat(wk).drop_duplicates().groupby("pid").size().rename("games")

    fp = pd.DataFrame(index=games.index)
    fp.index.name = "pid"
    fp = fp.join(games)
    for df in (passing, ints, rushing, receiving, tgt, ay):
        if len(df):
            df.index.name = "pid"
            fp = fp.join(df)
    fp = fp.fillna(0.0)

    fp["points"] = (fp.get("pass_yds", 0) * _PASS_Y + fp.get("pass_td", 0) * _PASS_TD
                    - fp.get("interceptions", 0) * _INT
                    + fp.get("rush_yds", 0) * _RUSH_Y + fp.get("rush_td", 0) * _RUSH_TD
                    + fp.get("receptions", 0) * _REC + fp.get("rec_yds", 0) * _REC_Y
                    + fp.get("rec_td", 0) * _REC_TD)
    fp["ppg"] = fp["points"] / fp["games"].clip(lower=1)

    # opportunity shares (map each player to his primary team's volume)
    prim = rec_team.reindex(fp.index).fillna(rush_team.reindex(fp.index))
    fp["team"] = prim
    fp["target_share"] = (fp.get("targets", 0).values
                          / prim.map(team_pass).replace(0, np.nan).values)
    fp["rush_share"] = (fp.get("carries", 0).values
                        / prim.map(team_rush).replace(0, np.nan).values)
    fp["adot"] = np.where(fp.get("targets", 0) > 0,
                          fp.get("air_yards", 0) / fp.get("targets", 1).replace(0, 1), 0.0)

    # position + name from the seasonal roster
    r = pd.read_parquet(RAW / "rosters_seasonal.parquet")
    r = r[r["season"] == season][["player_id", "position", "player_name"]].drop_duplicates("player_id")
    fp = fp.join(r.set_index("player_id"))
    fp["season"] = season
    _STATS[season] = fp.reset_index()
    return _STATS[season]


def season_board(season: int, pos: str | None = None, min_games: int = 1) -> pd.DataFrame:
    """Ranked half-PPR board for a completed season (offense skill positions only)."""
    fp = season_stats(season)
    if fp.empty:
        return fp
    d = fp[fp["position"].isin(FANTASY_POS) & (fp["games"] >= min_games)].copy()
    if pos:
        d = d[d["position"] == pos.upper()]
    d = d.sort_values("points", ascending=False)
    d["pos_rank"] = d.groupby("position")["points"].rank(ascending=False, method="min").astype(int)
    d["overall_rank"] = range(1, len(d) + 1)
    return d


# ──────────────────────────────────────────────────────────────────
#  2026 draft board: recency-weighted production onto current rosters
# ──────────────────────────────────────────────────────────────────
def _rosters_2026() -> pd.DataFrame:
    global _ROSTER
    if _ROSTER is None:
        r = pd.read_parquet(RAW / "rosters_2026.parquet")
        keep = [c for c in ["player_id", "player_name", "position", "team",
                            "years_exp", "draft_number"] if c in r.columns]
        r = r[keep].dropna(subset=["player_id"]).drop_duplicates("player_id")
        _ROSTER = r[r["position"].isin(FANTASY_POS)].copy()
    return _ROSTER


def _rookie_prior(pos: str, draft_number) -> float:
    """PPG prior for a player with no NFL history, from draft capital. Rough but ordered:
    premium picks project to real roles, day-3 picks to replacement."""
    dn = 260.0 if draft_number is None or pd.isna(draft_number) else float(draft_number)
    cap = max(0.0, 1.0 - dn / 260.0)                 # 1.0 = pick 1, ~0 = UDFA
    base = {"QB": 8.0, "RB": 5.0, "WR": 5.0, "TE": 3.0}.get(pos, 4.0)
    span = {"QB": 8.0, "RB": 8.0, "WR": 8.0, "TE": 5.0}.get(pos, 6.0)
    return round(base + span * cap, 2)


def project() -> pd.DataFrame:
    """2026 fantasy draft board: each rostered skill player projected on recency-weighted
    half-PPR PPG from 2023-25 (renormalized over the seasons he actually played), with a
    draft-capital prior for players who have no history yet (rookies)."""
    hist = {s: season_stats(s).set_index("pid") for s in _PROJ_SEASONS}
    roster = _rosters_2026()
    rows = []
    for r in roster.itertuples():
        pid, pos = r.player_id, r.position
        num, den, gsum, gden = 0.0, 0.0, 0.0, 0.0
        for s in _PROJ_SEASONS:
            h = hist[s]
            if pid in h.index and h.loc[pid, "games"] >= 1:
                w = _PROJ_W[s]
                num += w * float(h.loc[pid, "ppg"]); den += w
                gsum += w * float(h.loc[pid, "games"]); gden += w
        exp = getattr(r, "years_exp", None)
        dn = getattr(r, "draft_number", None)
        if den > 0:
            ppg = num / den
            src = "production"
            games = min(17.0, max(12.0, gsum / gden))     # durability from recent availability
        elif exp is not None and not pd.isna(exp) and exp <= 1:
            ppg = _rookie_prior(pos, dn); src = "rookie"; games = 15.0
        else:
            ppg = {"QB": 6.0, "RB": 3.5, "WR": 3.5, "TE": 2.0}.get(pos, 3.0); src = "depth"; games = 12.0
        rows.append({"player_id": pid, "player": r.player_name, "position": pos, "team": r.team,
                     "proj_ppg": round(ppg, 2), "proj_points": round(ppg * games, 1),
                     "games": round(games, 1), "years_exp": None if exp is None or pd.isna(exp) else int(exp),
                     "source": src})
    d = pd.DataFrame(rows)
    d["pos_rank"] = d.groupby("position")["proj_points"].rank(ascending=False, method="min").astype(int)
    # value over replacement → the honest draft-order metric
    repl = {}
    for pos, base in _REPLACEMENT.items():
        sub = d[d["position"] == pos].sort_values("proj_points", ascending=False)
        repl[pos] = float(sub.iloc[min(base, len(sub)) - 1]["proj_points"]) if len(sub) else 0.0
    d["vor"] = (d["proj_points"] - d["position"].map(repl)).round(1)
    d = d.sort_values("vor", ascending=False).reset_index(drop=True)
    d["overall_rank"] = range(1, len(d) + 1)
    return d


# ──────────────────────────────────────────────────────────────────
#  Undervalued by OPPORTUNITY vs output (positive-regression / breakout)
# ──────────────────────────────────────────────────────────────────
def breakouts(season: int = 2025, top: int = 25) -> pd.DataFrame:
    """Players whose OPPORTUNITY (target share for WR/TE, touch share for RB) outran their
    fantasy output — the classic 'volume is there, points will follow' profile. No ADP needed.
    Youth is a tiebreaker (ascending players regress up harder)."""
    fp = season_stats(season)
    if fp.empty:
        return fp
    d = fp[fp["position"].isin(["RB", "WR", "TE"]) & (fp["games"] >= 6)].copy()
    # opportunity metric per position
    d["touch_share"] = d["rush_share"].fillna(0) + d["target_share"].fillna(0)
    d["opp"] = np.where(d["position"] == "RB", d["touch_share"], d["target_share"].fillna(0))
    # percentiles within position (2025)
    d["opp_pctl"] = d.groupby("position")["opp"].rank(pct=True) * 100
    d["prod_pctl"] = d.groupby("position")["ppg"].rank(pct=True) * 100
    d["gap"] = d["opp_pctl"] - d["prod_pctl"]              # +ve = getting volume, not yet points
    # youth bonus from current roster
    roster = _rosters_2026().set_index("player_id")
    d["years_exp"] = d["pid"].map(roster["years_exp"] if "years_exp" in roster else pd.Series(dtype=float))
    d["score"] = d["gap"] + np.where(d["years_exp"].fillna(9) <= 2, 8, 0)
    d = d[(d["opp_pctl"] >= 55) & (d["gap"] > 8)]          # must have real volume + a real gap
    d = d.sort_values("score", ascending=False).head(top)
    return d[["player_name", "position", "team", "games", "ppg", "opp_pctl", "prod_pctl",
              "gap", "years_exp", "target_share", "rush_share"]]


# ──────────────────────────────────────────────────────────────────
#  Optional market overlay — value vs Underdog ADP (drop-in CSV)
# ──────────────────────────────────────────────────────────────────
def with_adp(board: pd.DataFrame) -> pd.DataFrame:
    """If data/raw/adp_underdog.csv exists (columns: player, position, adp), merge it and
    compute value = our overall_rank − ADP (positive = we like him more than the market =
    'undervalued at ADP'). No file → returns the board unchanged with adp/value = None."""
    board = board.copy()
    path = RAW / "adp_underdog.csv"
    if not path.exists():
        board["adp"] = None
        board["value"] = None
        return board
    adp = pd.read_csv(path)
    adp.columns = [c.strip().lower() for c in adp.columns]
    name_col = "player" if "player" in adp.columns else adp.columns[0]
    adp["_k"] = adp[name_col].astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
    board["_k"] = board.get("player", board.get("player_name")).astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
    board = board.merge(adp[["_k", "adp"]], on="_k", how="left").drop(columns="_k")
    if "overall_rank" in board.columns:
        board["value"] = (board["adp"] - board["overall_rank"]).round(1)   # +ve = falls past our rank
    return board
