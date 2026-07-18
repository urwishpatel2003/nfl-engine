"""
ml/projections.py  —  per-player game stat projections (SportsLine-style)
==========================================================================
Projects each player's expected stat line for a specific matchup, the same core
idea as SportsLine's model (minus their proprietary Monte-Carlo sims):

    player projection = usage share x team volume x efficiency, shaped by game script

Pipeline
  1. player_profiles()  — 2025 per-game usage + efficiency + team shares, from pbp_2025
     (player_stats/seasonal_stats are stale at 2024, so we aggregate box scores from PBP).
  2. team_volume()      — league/team pace: plays, pass & rush attempts per game.
  3. project_matchup()  — combine with the matchup score/total (game script) and
     distribute team volume to the players on the current depth chart.

Honest limits: no snap-count injuries, rookies with no 2025 usage are omitted, TD
projections are noisy, opponent-defense adjustment is coarse. It's a credible
baseline, not a betting tool.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

_PROFILE_CACHE = None
_QBDEPTH_CACHE = None

# Injury statuses that mean "won't play" → excluded from projections. Questionable
# players are assumed active (they suit up ~75% of the time).
OUT_STATUSES = ("Out", "Doubtful")


def unavailable_ids(statuses=OUT_STATUSES) -> set:
    """gsis_ids ruled out in each team's most-recent injury report (same report the
    dashboard's injury panel shows). Empty in the offseason if no report exists."""
    p = RAW / "injuries.parquet"
    if not p.exists():
        return set()
    inj = pd.read_parquet(p)
    if inj.empty or "gsis_id" not in inj.columns or "report_status" not in inj.columns:
        return set()
    inj = inj.dropna(subset=["gsis_id"]).copy()
    inj["sw"] = inj["season"].astype(int) * 100 + inj["week"].astype(int)
    inj = inj[inj["sw"] == inj.groupby("team")["sw"].transform("max")]   # latest week per team
    return set(inj[inj["report_status"].isin(statuses)]["gsis_id"])


def _depth_qbs() -> dict:
    """{team: [(gsis_id, name), …]} ordered by depth-chart rank."""
    global _QBDEPTH_CACHE
    if _QBDEPTH_CACHE is None:
        dc = pd.read_parquet(RAW / "depth_2026_current.parquet").copy()
        dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
        q = dc[dc.pos_abb == "QB"].sort_values(["team", "pos_rank"])
        _QBDEPTH_CACHE = {t: [(r.gsis_id, r.player_name) for r in g.itertuples()]
                          for t, g in q.groupby("team")}
    return _QBDEPTH_CACHE


def _available_qb(team: str, unavail: set):
    """First depth-chart QB who isn't ruled out (falls back to QB1 if all are)."""
    qs = _depth_qbs().get(team, [])
    for gid, name in qs:
        if gid not in unavail:
            return gid, name
    return qs[0] if qs else (None, None)


# ── 1. player box-score aggregation from PBP ────────────────────────
def _player_box_2025() -> pd.DataFrame:
    """Aggregate 2025 regular-season box-score totals per player from play-by-play."""
    p = pd.read_parquet(RAW / "pbp_2025.parquet")
    p = p[(p["week"] <= 18) & p["play_type"].isin(["pass", "run"])].copy()
    p["ret_td"] = p.get("return_touchdown", 0)
    p["is_td"] = (p["touchdown"] == 1) & (p["ret_td"] != 1)

    # passing (by passer)
    pa = p[p["pass_attempt"] == 1].copy()
    pa["cmp_yds"] = pa["yards_gained"] * pa["complete_pass"]
    pa["ptd"] = (pa["is_td"] & (pa["complete_pass"] == 1)).astype(int)
    passing = pa.groupby("passer_player_id").agg(
        g_pass=("game_id", "nunique"), pass_att=("pass_attempt", "sum"),
        cmp=("complete_pass", "sum"), pass_yds=("cmp_yds", "sum"),
        pass_td=("ptd", "sum"), interc=("interception", "sum")).reset_index() \
        .rename(columns={"passer_player_id": "player_id"})

    # rushing (by rusher)
    ru = p[p["rush_attempt"] == 1].copy()
    ru["rtd"] = ru["is_td"].astype(int)
    rushing = ru.groupby("rusher_player_id").agg(
        g_rush=("game_id", "nunique"), carries=("rush_attempt", "sum"),
        rush_yds=("yards_gained", "sum"), rush_td=("rtd", "sum")).reset_index() \
        .rename(columns={"rusher_player_id": "player_id"})

    # receiving (by receiver)
    rc = p[(p["pass_attempt"] == 1) & p["receiver_player_id"].notna()].copy()
    rc["rec_yds"] = rc["yards_gained"] * rc["complete_pass"]
    rc["rectd"] = (rc["is_td"] & (rc["complete_pass"] == 1)).astype(int)
    receiving = rc.groupby("receiver_player_id").agg(
        g_rec=("game_id", "nunique"), targets=("pass_attempt", "sum"),
        rec=("complete_pass", "sum"), rec_yds=("rec_yds", "sum"),
        rec_td=("rectd", "sum"), air=("air_yards", "sum")).reset_index() \
        .rename(columns={"receiver_player_id": "player_id"})

    # team pass/rush attempts per game (for shares)
    team_pa = pa.groupby("posteam").agg(team_pa=("pass_attempt", "sum"),
                                        team_g=("game_id", "nunique")).reset_index()
    team_ra = ru.groupby("posteam")["rush_attempt"].sum().reset_index(name="team_ra")

    box = passing.merge(rushing, on="player_id", how="outer").merge(receiving, on="player_id", how="outer")
    box["games"] = box[["g_pass", "g_rush", "g_rec"]].max(axis=1)
    return box.fillna(0), team_pa, team_ra


# ── 2. per-game profiles + usage shares ─────────────────────────────
def player_profiles() -> pd.DataFrame:
    """Per-player per-game usage + efficiency, with current team & position from the roster."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    box, team_pa, team_ra = _player_box_2025()

    # attach current team + position + name from the 2026 roster / depth chart
    rost = pd.read_parquet(RAW / "rosters_2026.parquet")[["player_id", "player_name", "team", "position"]]
    df = box.merge(rost, on="player_id", how="inner")
    df = df[df["games"] > 0].copy()

    g = df["games"].clip(lower=1)
    # per-game volume
    df["att_pg"] = df["pass_att"] / g
    df["cmp_pct"] = np.where(df["pass_att"] > 0, df["cmp"] / df["pass_att"], 0.63)
    df["ypa"] = np.where(df["pass_att"] > 0, df["pass_yds"] / df["pass_att"], 0)
    df["ptd_pa"] = np.where(df["pass_att"] > 0, df["pass_td"] / df["pass_att"], 0)
    df["int_pa"] = np.where(df["pass_att"] > 0, df["interc"] / df["pass_att"], 0)
    df["carry_pg"] = df["carries"] / g
    df["ypc"] = np.where(df["carries"] > 0, df["rush_yds"] / df["carries"], 0)
    df["rtd_carry"] = np.where(df["carries"] > 0, df["rush_td"] / df["carries"], 0)
    df["tgt_pg"] = df["targets"] / g
    df["catch_pct"] = np.where(df["targets"] > 0, df["rec"] / df["targets"], 0)
    df["ypt"] = np.where(df["targets"] > 0, df["rec_yds"] / df["targets"], 0)
    df["rectd_tgt"] = np.where(df["targets"] > 0, df["rec_td"] / df["targets"], 0)

    # team shares (of season attempts)
    df = df.merge(team_pa[["posteam", "team_pa", "team_g"]], left_on="team", right_on="posteam", how="left")
    df = df.merge(team_ra, left_on="team", right_on="posteam", how="left", suffixes=("", "_r"))
    df["target_share"] = np.where(df["team_pa"] > 0, df["targets"] / df["team_pa"], 0)
    df["carry_share"] = np.where(df["team_ra"] > 0, df["carries"] / df["team_ra"], 0)
    num = df.select_dtypes("number").columns          # float32 -> float64 for clean rounding
    df[num] = df[num].astype("float64")
    _PROFILE_CACHE = df
    return df


# ── 3. team volume (pace) ───────────────────────────────────────────
def team_volume() -> dict:
    """League-average and per-team pass/rush attempts per game (2025)."""
    _, team_pa, team_ra = _player_box_2025()
    t = team_pa.merge(team_ra, on="posteam", how="outer").fillna(0)
    t["pa_pg"] = t["team_pa"] / t["team_g"].clip(lower=1)
    t["ra_pg"] = t["team_ra"] / t["team_g"].clip(lower=1)
    return {
        "by_team": t.set_index("posteam")[["pa_pg", "ra_pg"]].to_dict("index"),
        "lg_pa": float(t["pa_pg"].mean()), "lg_ra": float(t["ra_pg"].mean()),
    }


# QB starters (depth chart) and a rookie/replacement prior
_QB_START = None


def _qb_starters() -> dict:
    """{team: (gsis_id, name)} for the current depth-chart QB1."""
    global _QB_START
    if _QB_START is None:
        dc = pd.read_parquet(RAW / "depth_2026_current.parquet")
        dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
        qb1 = dc[(dc.pos_abb == "QB") & (dc.pos_rank == 1)].drop_duplicates("team")
        _QB_START = {r.team: (r.gsis_id, r.player_name) for r in qb1.itertuples()}
    return _QB_START


# league-average starter line (used for rookies / no-2025-usage starters)
ROOKIE_QB = {"cmp_pct": 0.62, "ypa": 6.4, "ptd_pa": 0.036, "int_pa": 0.028, "carry_pg": 3.0, "ypc": 4.0}


# ── 4. distribute team volume to players for a matchup ──────────────
def _distribute(team: str, team_pa: float, team_ra: float, off_tds: float, prof: pd.DataFrame,
                pass_factor: float = 1.0, rush_factor: float = 1.0, unavail: set = frozenset()) -> dict:
    """Allocate a team's projected pass/rush volume to its AVAILABLE players (injured players
    are dropped so their carries/targets redistribute), with the opponent-defense adjustment
    (pass_factor for the air game, rush_factor for the ground)."""
    roster = prof[(prof.team == team) & (~prof.player_id.isin(unavail))]
    pass_tds = off_tds * 0.62 * pass_factor
    rush_tds = off_tds - off_tds * 0.62          # ground TDs unaffected by pass factor

    # QB: first depth-chart QB who isn't ruled out (rookie/no-2025 -> replacement prior)
    sid, sname = _available_qb(team, unavail)
    qrow = roster[roster.player_id == sid]
    q = qrow.iloc[0] if not qrow.empty else None
    if q is None:
        # rookie / unknown starter — use the depth-chart name with a replacement line
        rk = ROOKIE_QB
        qb_line = {"name": sname or "Starter", "pos": "QB", "rookie": True,
                   "pass_att": round(team_pa), "cmp": round(team_pa * rk["cmp_pct"]),
                   "pass_yds": round(team_pa * rk["ypa"] * pass_factor), "pass_td": round(pass_tds, 1),
                   "int": round(team_pa * rk["int_pa"], 1), "rush_yds": round(rk["carry_pg"] * rk["ypc"])}
    else:
        qb_line = {"name": q.player_name, "pos": "QB", "rookie": False,
                   "pass_att": round(team_pa), "cmp": round(team_pa * q.cmp_pct),
                   "pass_yds": round(team_pa * q.ypa * pass_factor), "pass_td": round(pass_tds, 1),
                   "int": round(team_pa * q.int_pa, 1), "rush_yds": round(q.carry_pg * q.ypc)}

    # Rushers: concentrate carries on the actual backfield (top 4), scaled by run matchup
    rbs = roster[(roster.position == "RB") & (roster.carry_pg > 1)].sort_values(
        "carry_pg", ascending=False).head(4).copy()
    rush_lines = []
    if not rbs.empty:
        denom = rbs["carry_pg"].sum(); tdw = max(1e-6, (rbs.carry_pg * rbs.rtd_carry).sum())
        for _, r in rbs.iterrows():
            car = team_ra * (r.carry_pg / denom)
            rush_lines.append({"name": r.player_name, "pos": "RB",
                               "carries": round(car), "rush_yds": round(car * r.ypc * rush_factor),
                               "rush_td": round(rush_tds * (r.carry_pg * r.rtd_carry) / tdw, 1),
                               "targets": round(r.tgt_pg), "rec": round(r.tgt_pg * r.catch_pct),
                               "rec_yds": round(r.tgt_pg * r.ypt * pass_factor), "rec_td": 0})

    # Receivers: concentrate targets on the actual pass-catchers (top 6), scaled by pass matchup
    recs = roster[roster.position.isin(["WR", "TE", "RB"]) & (roster.tgt_pg > 0.5)].sort_values(
        "tgt_pg", ascending=False).head(6).copy()
    rec_lines = []
    if not recs.empty:
        denom = recs["tgt_pg"].sum(); tdw = max(1e-6, (recs.tgt_pg * recs.rectd_tgt).sum())
        for _, r in recs.iterrows():
            tg = team_pa * 0.95 * (r.tgt_pg / denom)
            rec_lines.append({"name": r.player_name, "pos": r.position,
                              "targets": round(tg), "rec": round(tg * r.catch_pct),
                              "rec_yds": round(tg * r.ypt * pass_factor),
                              "rec_td": round(pass_tds * (r.tgt_pg * r.rectd_tgt) / tdw, 1)})
    return {"qb": qb_line, "rush": rush_lines, "rec": rec_lines}


def injury_impact(team: str, unavail: set = None) -> dict:
    """Points penalty for a team's ruled-out contributors (drives the spread). QB dominates;
    skill players give a smaller, capped hit since a replacement recovers most of the usage."""
    if unavail is None:
        unavail = unavailable_ids()
    prof = player_profiles()
    r = prof[prof.team == team]
    pen, who = 0.0, []
    # QB1 out → penalty scaled by the gap to the best available backup
    from ml.squad import _qb_value_table
    qbp = _qb_value_table().rank(pct=True) * 100
    qs = _depth_qbs().get(team, [])
    if qs and qs[0][0] in unavail:
        s = float(qbp.get(qs[0][0], 60.0))
        b = next((float(qbp.get(g, 40.0)) for g, _ in qs[1:] if g not in unavail), 35.0)
        d = max(0.0, (s - b) / 100.0 * 7.0)                # elite→replacement ≈ up to 7 pts
        if d > 0.1:
            pen += d; who.append(f"{qs[0][1]} (QB)")
    # skill starters out → net loss after a ~65% replacement recovers most of the share
    for _, p in r[r.player_id.isin(unavail)].iterrows():
        if p.position in ("RB", "WR", "TE"):
            share = float(p.get("target_share", 0) or 0) + float(p.get("carry_share", 0) or 0)
            loss = min(2.0, share * 6.0 * 0.35)
            if loss > 0.2:
                pen += loss; who.append(f"{p.player_name} ({p.position})")
    return {"pts": round(min(pen, 10.0), 1), "players": who}


def project_matchup(home: str, away: str, neutral: bool = False) -> dict:
    """Full matchup: unit-vs-unit score + opponent-adjusted player stat lines, using only
    AVAILABLE players (injured players are excluded and their usage redistributed)."""
    from ml.matchup_engine import project_game, team_units
    pred = project_game(home, away, neutral=neutral)
    u = team_units()
    tv = team_volume()
    prof = player_profiles()
    unavail = unavailable_ids()
    points = {home: pred["pred_home_score"], away: pred["pred_away_score"]}
    margin = {home: pred["pred_margin"], away: -pred["pred_margin"]}

    teams = {}
    for team, opp in [(home, away), (away, home)]:
        vol = tv["by_team"].get(team, {"pa_pg": tv["lg_pa"], "ra_pg": tv["lg_ra"]})
        # game script: trailing team throws more (each point of deficit ~0.35 plays pass-ward)
        shift = float(np.clip(-margin[team] * 0.35, -6, 6))
        team_pa = vol["pa_pg"] + shift
        team_ra = max(12.0, vol["ra_pg"] - shift)
        off_tds = max(0.0, (points[team] - 1.2) / 7.0)   # approx offensive TDs
        # opponent-defense adjustment: bad D (positive z_def EPA allowed) -> more player yards
        pass_factor = float(np.clip(1 + 0.14 * u.loc[opp, "z_def_pass"], 0.75, 1.30)) if opp in u.index else 1.0
        rush_factor = float(np.clip(1 + 0.14 * u.loc[opp, "z_def_rush"], 0.75, 1.30)) if opp in u.index else 1.0
        teams[team] = _distribute(team, team_pa, team_ra, off_tds, prof, pass_factor, rush_factor, unavail)

    return {"home": home, "away": away, "pred": pred, "teams": teams}


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        import json
        r = project_matchup(sys.argv[1].upper(), sys.argv[2].upper())
        print(f"{r['away']} @ {r['home']}: {r['pred']['pred_away_score']}-{r['pred']['pred_home_score']}")
        for tm in (r["home"], r["away"]):
            t = r["teams"][tm]
            print(f"\n{tm}:")
            if t["qb"]:
                q = t["qb"]; print(f"  QB {q['name']}: {q['cmp']}/{q['pass_att']}, {q['pass_yds']} yds, {q['pass_td']} TD, {q['int']} INT")
            for x in t["rush"][:3]:
                print(f"  RB {x['name']}: {x['carries']} car, {x['rush_yds']} yds, {x['rush_td']} TD | {x['rec']}-{x['rec_yds']} rec")
            for x in t["rec"][:4]:
                print(f"  {x['pos']} {x['name']}: {x['rec']}/{x['targets']}, {x['rec_yds']} yds, {x['rec_td']} TD")
        sys.exit(0)
    prof = player_profiles()
    print(f"players with 2025 usage: {len(prof)}")
    for pos in ["QB", "RB", "WR"]:
        top = prof[prof.position == pos].nlargest(3, "att_pg" if pos == "QB" else "tgt_pg" if pos != "RB" else "carry_pg")
        print(f"\nTop {pos}:")
        cols = (["player_name", "team", "att_pg", "ypa", "cmp_pct", "ptd_pa"] if pos == "QB"
                else ["player_name", "team", "carry_pg", "ypc", "tgt_pg", "carry_share"] if pos == "RB"
                else ["player_name", "team", "tgt_pg", "ypt", "catch_pct", "target_share"])
        print(top[cols].round(2).to_string(index=False))
