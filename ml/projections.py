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


# ── 4. distribute team volume to players for a matchup ──────────────
def _distribute(team: str, team_pa: float, team_ra: float, off_tds: float,
                prof: pd.DataFrame) -> dict:
    """Allocate a team's projected pass/rush volume to its current players."""
    roster = prof[prof.team == team]
    pass_tds = off_tds * 0.62          # ~62% of offensive TDs are passing
    rush_tds = off_tds - pass_tds

    # QB: the highest-usage passer on the roster gets the team's pass attempts
    qbs = roster[(roster.position == "QB") & (roster.att_pg > 3)].nlargest(1, "att_pg")
    qb_line = None
    if not qbs.empty:
        q = qbs.iloc[0]
        qb_line = {"name": q.player_name, "pos": "QB",
                   "pass_att": round(team_pa), "cmp": round(team_pa * q.cmp_pct),
                   "pass_yds": round(team_pa * q.ypa), "pass_td": round(pass_tds, 1),
                   "int": round(team_pa * q.int_pa, 1),
                   "rush_yds": round(q.carry_pg * q.ypc)}   # QB scramble yards approx

    # Rushers: normalize carry_pg among rostered RBs (+ mobile QB) to team_ra
    rbs = roster[roster.position.isin(["RB"])].copy()
    rbs = rbs[rbs.carry_pg > 1]
    rush_lines = []
    if not rbs.empty:
        w = rbs["carry_pg"] / rbs["carry_pg"].sum()
        for _, r in rbs.sort_values("carry_pg", ascending=False).head(4).iterrows():
            car = team_ra * (r.carry_pg / rbs["carry_pg"].sum())
            rush_lines.append({"name": r.player_name, "pos": "RB",
                               "carries": round(car), "rush_yds": round(car * r.ypc),
                               "rush_td": round(rush_tds * (r.carry_pg * r.rtd_carry) /
                                                max(1e-6, (rbs.carry_pg * rbs.rtd_carry).sum()), 1),
                               "targets": round(r.tgt_pg), "rec": round(r.tgt_pg * r.catch_pct),
                               "rec_yds": round(r.tgt_pg * r.ypt)})

    # Receivers: normalize tgt_pg among WR/TE/RB to the team's pass attempts (~95%)
    recs = roster[roster.position.isin(["WR", "TE", "RB"])].copy()
    recs = recs[recs.tgt_pg > 0.5]
    rec_lines = []
    tgt_pool = team_pa * 0.95
    if not recs.empty:
        share = recs["tgt_pg"] / recs["tgt_pg"].sum()
        td_wt = (recs.tgt_pg * recs.rectd_tgt)
        for _, r in recs.sort_values("tgt_pg", ascending=False).head(6).iterrows():
            tg = tgt_pool * (r.tgt_pg / recs["tgt_pg"].sum())
            rec_lines.append({"name": r.player_name, "pos": r.position,
                              "targets": round(tg), "rec": round(tg * r.catch_pct),
                              "rec_yds": round(tg * r.ypt),
                              "rec_td": round(pass_tds * (r.tgt_pg * r.rectd_tgt) /
                                              max(1e-6, td_wt.sum()), 1)})
    return {"qb": qb_line, "rush": rush_lines, "rec": rec_lines}


def project_matchup(home: str, away: str, neutral: bool = False) -> dict:
    """Full matchup: predicted score + each team's projected player stat lines."""
    from ml.squad import predict_matchup       # same ratings as the power rankings
    pred = predict_matchup(home, away, neutral=neutral)
    tv = team_volume()
    prof = player_profiles()
    points = {home: pred["pred_home_score"], away: pred["pred_away_score"]}
    margin = {home: pred["pred_margin"], away: -pred["pred_margin"]}

    teams = {}
    for team in (home, away):
        vol = tv["by_team"].get(team, {"pa_pg": tv["lg_pa"], "ra_pg": tv["lg_ra"]})
        # game script: trailing team throws more (each point of deficit ~0.35 plays pass-ward)
        shift = float(np.clip(-margin[team] * 0.35, -6, 6))
        team_pa = vol["pa_pg"] + shift
        team_ra = max(12.0, vol["ra_pg"] - shift)
        off_tds = max(0.0, (points[team] - 1.2) / 7.0)   # approx offensive TDs
        teams[team] = _distribute(team, team_pa, team_ra, off_tds, prof)

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
