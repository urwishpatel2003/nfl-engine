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

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _namekey(s) -> str:
    """Normalize a name for cross-source matching: lowercase, letters only, drop the
    generational suffix (Jr./Sr./II/III/IV) that one feed carries and another drops."""
    k = re.sub(r"[^a-z]", "", str(s).lower())
    return re.sub(r"(iii|iv|ii|jr|sr)$", "", k)

RAW = Path(__file__).parent.parent / "data" / "raw"

# Scoring weights. Passing/rushing/receiving-yard & TD values are identical across the common
# formats; only the per-reception value changes: Underdog Best Ball is HALF-PPR (0.5), Full PPR = 1.0.
_PASS_Y, _PASS_TD, _INT = 0.04, 4.0, 1.0
_RUSH_Y, _RUSH_TD = 0.1, 6.0
_REC_Y, _REC_TD = 0.1, 6.0
_REC_PT = {"half": 0.5, "full": 1.0}

# Three draftable formats. Best Ball (Underdog Best Ball Mania) is season-long half-PPR with NO
# kickers or defenses; the two redraft formats add K/DST. Each grades "value" against the market
# that matches it. (Kicker/defense scoring barely differs by PPR, so K/DST come from the one
# complete source — the ECR export — in both redraft formats.)
FORMATS = {
    "bestball": {"label": "Best Ball", "scoring": "half", "kdst": False, "anchor": "adp", "market": "Underdog ADP"},
    "half":     {"label": "Half PPR",  "scoring": "half", "kdst": True,  "anchor": "adp", "market": "Underdog ADP"},
    "full":     {"label": "Full PPR",  "scoring": "full", "kdst": True,  "anchor": "ecr", "market": "PPR ECR"},
}


def _fmt(fmt):
    return FORMATS.get(fmt, FORMATS["bestball"])

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
_KDST = None


def clear():
    global _STATS, _ROSTER, _KDST
    _STATS = {}
    _ROSTER = None
    _KDST = None


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
    # NOTE: points/ppg are scoring-dependent → computed on demand by _score(), not cached here.

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


def _score(fp: pd.DataFrame, fmt: str = "bestball") -> pd.DataFrame:
    """Attach points + ppg for the chosen format's scoring (only the per-reception value differs).
    Accepts a format key ('bestball'/'half'/'full') or a raw scoring ('half'/'full')."""
    scoring = FORMATS[fmt]["scoring"] if fmt in FORMATS else fmt
    rec_pt = _REC_PT.get(scoring, 0.5)
    d = fp.copy()
    d["points"] = (d.get("pass_yds", 0) * _PASS_Y + d.get("pass_td", 0) * _PASS_TD
                   - d.get("interceptions", 0) * _INT
                   + d.get("rush_yds", 0) * _RUSH_Y + d.get("rush_td", 0) * _RUSH_TD
                   + d.get("receptions", 0) * rec_pt + d.get("rec_yds", 0) * _REC_Y
                   + d.get("rec_td", 0) * _REC_TD)
    d["ppg"] = d["points"] / d["games"].clip(lower=1)
    return d


def season_board(season: int, pos: str | None = None, min_games: int = 1,
                 fmt: str = "bestball") -> pd.DataFrame:
    """Ranked board for a completed season (offense skill positions only), for the given format."""
    fp = season_stats(season)
    if fp.empty:
        return fp
    fp = _score(fp, fmt)
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


def project(fmt: str = "bestball") -> pd.DataFrame:
    """2026 fantasy draft board: each rostered skill player projected on recency-weighted PPG
    from 2023-25 (renormalized over the seasons he actually played) in the format's scoring, with a
    draft-capital prior for players who have no history yet (rookies)."""
    hist = {s: _score(season_stats(s), fmt).set_index("pid") for s in _PROJ_SEASONS}
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
def breakouts(season: int = 2025, top: int = 25, fmt: str = "bestball") -> pd.DataFrame:
    """Players whose OPPORTUNITY (target share for WR/TE, touch share for RB) outran their
    fantasy output — the classic 'volume is there, points will follow' profile. No ADP needed.
    Youth is a tiebreaker (ascending players regress up harder)."""
    fp = season_stats(season)
    if fp.empty:
        return fp
    fp = _score(fp, fmt)
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
def _kdst_pool():
    """K and DST as a draft pool from the ECR export (the only complete K/DST source). Nominal
    projections — these are streamed, low-variance positions — with market = ECR rank so they land
    late, and replacement-level VOR so a path only takes them when the roster forces it."""
    global _KDST
    if _KDST is not None:
        return _KDST
    path = RAW / "ecr_2026.csv"
    if not path.exists():
        _KDST = []
        return _KDST
    e = pd.read_csv(path)
    e = e[e["position"].isin(["K", "DST"])].copy().sort_values("ecr")
    e["pr"] = e.groupby("position")["ecr"].rank(method="min").astype(int)
    pool = []
    for r in e.itertuples():
        base = 8.0 if r.position == "K" else 7.0
        pool.append({"player_id": f"{r.position}_{str(getattr(r, 'team', '')) or r.player}".replace(" ", ""),
                     "player": r.player, "position": r.position, "team": getattr(r, "team", ""),
                     "proj_ppg": base, "proj_points": round(base * 17, 1), "vor": -1.0,
                     "market": float(r.ecr), "value": None, "tier": int(r.tier),
                     "pos_rank": int(r.pr), "source": "market", "overall_rank": 9999,
                     "our_rank": None, "order": float(r.ecr)})
    _KDST = pool
    return _KDST


def draft_path(slot, fmt="bestball", teams=12, roster=None):
    """Round-by-round plan for a custom-size snake with a custom roster. `roster` = per-position
    target counts (QB/RB/WR/TE/K/DST); the draft runs sum(roster) rounds and fills exactly those.
    Best Ball allows no K/DST (they're forced to 0). The room drafts by market; at your picks we
    take the best-value player available (highest VOR) with best-ball structure: wait on QB/TE,
    load RB/WR early, K/DST last, guarantee the core."""
    teams = max(2, int(teams))
    slot = max(1, min(teams, int(slot)))
    default = {"QB": 2, "RB": 5, "WR": 8, "TE": 3, "K": 0, "DST": 0}
    roster = {**default, **{k.upper(): int(v) for k, v in (roster or {}).items()}}
    roster = {k: max(0, int(v)) for k, v in roster.items()}
    if not _fmt(fmt)["kdst"]:                          # Best Ball: no kickers/defenses
        roster["K"] = roster["DST"] = 0
    rounds = sum(roster.values())
    if rounds <= 0:
        roster, rounds = default, sum(default.values())

    b = attach_value(with_adp(project(fmt)), fmt)
    label = b["market_label"].iloc[0] if len(b) else _fmt(fmt)["market"]
    b = b.copy()
    mx = float(b["market"].max()) if b["market"].notna().any() else 0.0
    b["order"] = b["market"].fillna(mx + b["overall_rank"])
    recs = b.sort_values("order").to_dict("records")
    if roster.get("K", 0) or roster.get("DST", 0):        # add K/DST only if the roster uses them
        recs = recs + [p for p in _kdst_pool() if roster.get(p["position"], 0) > 0]
        recs.sort(key=lambda x: x["order"])

    counts = {p: 0 for p in roster}
    taken = set()

    def cap_at(pos, r):
        t = roster.get(pos, 0)
        if t == 0:
            return 0
        if pos in ("K", "DST"):                            # only draftable in the final rounds
            return t if r > rounds - (roster.get("K", 0) + roster.get("DST", 0)) else 0
        if pos in ("QB", "TE"):                             # ramp the 2nd+ later (wait on them)
            return min(t, max(1, math.ceil(t * r / rounds)))
        return t                                            # RB/WR: full target anytime

    def my_overall(r):                                      # snake: even rounds reverse
        return (r - 1) * teams + slot if r % 2 == 1 else r * teams - slot + 1
    mine = {my_overall(r): r for r in range(1, rounds + 1)}

    def _vor(p):
        v = p.get("vor")
        return v if v == v else -1e9

    def _slim(p):
        return {"player": p["player"], "position": p["position"], "team": p.get("team"),
                "proj_ppg": p["proj_ppg"], "vor": p["vor"], "market": p.get("market"),
                "value": p.get("value"), "tier": p.get("tier"), "pos_rank": p.get("pos_rank"),
                "source": p.get("source")}

    picks, total = [], teams * rounds
    for pick in range(1, total + 1):
        avail = [p for p in recs if p["player_id"] not in taken]
        if not avail:
            break
        if pick in mine:
            r = mine[pick]
            my_left = rounds - r + 1
            need = sum(max(0, roster[p] - counts[p]) for p in roster)
            base = [p for p in avail if counts.get(p["position"], 0) < cap_at(p["position"], r)]
            if not base:                                   # gates blocked everything → any unfilled slot
                base = [p for p in avail if counts.get(p["position"], 0) < roster.get(p["position"], 0)] or avail
            if need >= my_left:                            # out of room → force still-unfilled positions
                base = [p for p in base if counts.get(p["position"], 0) < roster.get(p["position"], 0)] or base
            ranked = sorted(base, key=lambda x: -_vor(x))
            # take the best player who likely WON'T last to our next pick; else best-available.
            nxt = my_overall(r + 1) if r < rounds else total + 1
            intervening = max(0, nxt - pick - 1)
            gone = {p["player_id"] for p in avail[:intervening]}   # avail is already market-order sorted
            wontlast = [p for p in ranked if p["player_id"] in gone]
            sel = (wontlast or ranked)[0]
            counts[sel["position"]] = counts.get(sel["position"], 0) + 1
            taken.add(sel["player_id"])
            alts = [a for a in ranked if a["player_id"] != sel["player_id"]][:3]
            picks.append({"round": r, "overall_pick": pick, **_slim(sel),
                          "alts": [_slim(a) for a in alts]})
        else:
            taken.add(avail[0]["player_id"])               # the field takes best-available by market

    return {"slot": slot, "teams": teams, "rounds": rounds, "scoring": fmt,
            "market_label": label, "roster": roster, "counts": counts,
            "roster_ppg": round(sum(p["proj_ppg"] for p in picks), 1), "picks": picks}


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
    adp["_k"] = adp[name_col].map(_namekey)
    adp = adp.drop_duplicates("_k")
    board["_k"] = board.get("player", board.get("player_name")).map(_namekey)
    board = board.merge(adp[["_k", "adp"]], on="_k", how="left").drop(columns="_k")
    return _with_ecr(board)


def attach_value(board: pd.DataFrame, fmt: str = "bestball") -> pd.DataFrame:
    """Compute our_rank + value against the FORMAT-APPROPRIATE market anchor (Best Ball & Half PPR
    → Underdog ADP; Full PPR → PPR ECR). Both are put on the same scale first (re-rank our board
    over just the players the anchor covers) so value = anchor_rank − our_rank; +value = we're
    higher than that market = a target."""
    board = board.copy()
    col, label = _fmt(fmt)["anchor"], _fmt(fmt)["market"]
    board["market"] = board.get(col)
    board["market_label"] = label
    if "overall_rank" in board.columns:
        have = board["market"].notna()
        board["our_rank"] = np.nan
        board.loc[have, "our_rank"] = board.loc[have, "overall_rank"].rank(method="min")
        board["value"] = (board["market"] - board["our_rank"]).round(1)
    return board


def _with_ecr(board: pd.DataFrame) -> pd.DataFrame:
    """Attach FantasyPros expert-consensus rank + draft TIER + strength-of-schedule, if the
    ECR export (data/raw/ecr_2026.csv: player, position, ecr, tier, sos) is present. Tiers are
    the drafter's real tool — they mark the value cliffs where a position falls off."""
    board = board.copy()
    path = RAW / "ecr_2026.csv"
    if not path.exists():
        board["ecr"] = None; board["tier"] = None; board["sos"] = None
        return board
    ecr = pd.read_csv(path)
    ecr["_k"] = ecr["player"].map(_namekey)
    ecr = ecr.drop_duplicates("_k")
    board["_k"] = board.get("player", board.get("player_name")).map(_namekey)
    board = board.merge(ecr[["_k", "ecr", "tier", "sos"]], on="_k", how="left").drop(columns="_k")
    return board


def value_board(max_adp: int = 216, top: int = 30, fmt: str = "bestball"):
    """Market-vs-us within the draftable pool (default ADP ≤ 216 = 18 rounds × 12 teams).
    Restricted to players with real PRODUCTION history — our projection for rookies/depth is a
    crude prior, so calling the market wrong on them is noise, not signal. Sorted by value:
    positive = we're higher than ADP (target), negative = market reaches vs our board (fade).

    The market anchor matches the format (Best Ball / Half PPR → Underdog ADP; Full PPR → PPR
    ECR), so each format is graded against its own market."""
    b = attach_value(with_adp(project(fmt)), fmt)
    label = b["market_label"].iloc[0] if len(b) else _fmt(fmt)["market"]
    d = b[b["market"].notna() & (b["market"] <= max_adp) & (b["source"] == "production")].copy()
    d = d.sort_values("value", ascending=False)
    cols = ["player", "position", "team", "proj_ppg", "our_rank", "market", "value", "pos_rank", "tier"]
    cols = [c for c in cols if c in d.columns]
    targets = d.head(top)[cols]
    fades = d.tail(top).sort_values("value")[cols]
    return targets, fades, label
