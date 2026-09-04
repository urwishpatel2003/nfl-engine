"""
ml/history.py  —  opponent-adjusted QB seasons + team unit EPA history
======================================================================
Two research tables for the dashboard, both ROLLING: they enumerate whatever
pbp_{season}.parquet files exist in data/raw, so when the daily refresh starts
downloading pbp_2026 in-season, the current season appears and grows weekly
with no code change.

qb_seasons(n=3)
    Per QB per season (latest n seasons with PBP): dropbacks (pass plays incl.
    sacks, plus scrambles), raw EPA/play and OPPONENT-ADJUSTED EPA/play — each
    play's EPA minus the opposing defense's pass-defense rating from
    ml.adjust.adjusted_unit_epa (an ALS fit over every play, so facing the
    league's best pass defenses no longer hides a good QB) — plus success rate,
    CPOE, TD%, INT% and sack% per season. A season cell needs MIN_DROPBACKS to
    count; a QB appears if any season qualifies.

unit_epa_history()
    Per team per season (ALL available seasons): the four opponent-adjusted
    unit values (off_pass/off_rush/def_pass/def_rush) straight from
    ml.adjust.adjusted_unit_epa. def_* keep the raw "EPA allowed" convention
    (negative = good defense).
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"

MIN_DROPBACKS = 150          # a season cell needs a real sample
_QB_CACHE = None
_UNIT_CACHE = None


def _pbp_seasons() -> list:
    return sorted(int(p.stem.split("_")[1]) for p in RAW.glob("pbp_*.parquet"))


def clear():
    global _QB_CACHE, _UNIT_CACHE
    _QB_CACHE = _UNIT_CACHE = None


def qb_seasons(n: int = 3) -> dict:
    global _QB_CACHE
    if _QB_CACHE is not None:
        return _QB_CACHE
    from ml.adjust import adjusted_unit_epa
    seasons = _pbp_seasons()[-n:]
    per = {}                                           # (gsis, season) -> stats
    names = {}
    for season in seasons:
        p = pd.read_parquet(RAW / f"pbp_{season}.parquet")
        p = p[(p["week"] <= 18) & p["epa"].notna() & p["defteam"].notna()]
        adj = adjusted_unit_epa(season)
        dpass = {t: v.get("def_pass", 0.0) for t, v in adj.items()}

        drop = p[p["passer_player_id"].notna() & (p["pass_attempt"] == 1) | (p["sack"] == 1)]
        drop = drop[drop["passer_player_id"].notna()].copy()
        scr = p[(p["qb_scramble"] == 1) & p["rusher_player_id"].notna()].copy()
        scr["passer_player_id"] = scr["rusher_player_id"]
        qb = pd.concat([drop, scr], ignore_index=True)
        qb["adj_epa"] = qb["epa"] - qb["defteam"].map(dpass).fillna(0.0)

        g = qb.groupby("passer_player_id")
        att = drop[drop["pass_attempt"] == 1].groupby("passer_player_id")
        stats = pd.DataFrame({
            "db": g.size(),
            "epa": g["epa"].mean(),
            "adj_epa": g["adj_epa"].mean(),
            "success": g["success"].mean(),
            "cpoe": att["cpoe"].mean(),
            "td_pct": att["touchdown"].mean(),
            "int_pct": att["interception"].mean(),
            "sacks": g["sack"].sum(),
        })
        stats["sack_pct"] = stats["sacks"] / stats["db"].clip(lower=1)
        nm = drop.dropna(subset=["passer_player_name"]) \
            .groupby("passer_player_id")["passer_player_name"].last()
        names.update(nm.to_dict())
        for gsis, r in stats[stats["db"] >= MIN_DROPBACKS].iterrows():
            per[(gsis, season)] = {k: round(float(r[k]), 4) for k in
                                   ("db", "epa", "adj_epa", "success", "cpoe",
                                    "td_pct", "int_pct", "sack_pct")}

    # current team + full name for each QB (depth charts / roster; blank team for FA)
    team, full = {}, {}
    try:
        dc = pd.read_parquet(RAW / "depth_2026_current.parquet")
        team = dc.drop_duplicates("gsis_id").set_index("gsis_id")["team"].to_dict()
        full = dc.drop_duplicates("gsis_id").set_index("gsis_id")["player_name"].to_dict()
    except Exception:
        pass
    try:
        r = pd.read_parquet(RAW / "rosters_2026.parquet").dropna(subset=["player_id"])
        full = {**r.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict(), **full}
    except Exception:
        pass
    names = {g: full.get(g, n) for g, n in names.items()}

    qbs = {}
    for (gsis, season), st in per.items():
        q = qbs.setdefault(gsis, {"name": names.get(gsis, gsis), "team": team.get(gsis, ""),
                                  "seasons": {}})
        q["seasons"][season] = st
    # headline: dropback-weighted adjusted EPA across the window, recency-tilted
    w = {s: 1.0 + 0.25 * i for i, s in enumerate(seasons)}
    for q in qbs.values():
        num = den = 0.0
        for s, st in q["seasons"].items():
            num += st["adj_epa"] * st["db"] * w[s]
            den += st["db"] * w[s]
        q["adj_epa_w"] = round(num / den, 4) if den else None
    _QB_CACHE = {"seasons": seasons,
                 "qbs": sorted(qbs.values(), key=lambda x: -(x["adj_epa_w"] or -9))}
    return _QB_CACHE


def unit_epa_history() -> dict:
    global _UNIT_CACHE
    if _UNIT_CACHE is not None:
        return _UNIT_CACHE
    from ml.adjust import adjusted_unit_epa
    seasons = _pbp_seasons()
    teams = {}
    for season in seasons:
        for t, u in adjusted_unit_epa(season).items():
            teams.setdefault(t, {})[season] = {k: round(float(v), 4) for k, v in u.items()}
    _UNIT_CACHE = {"seasons": seasons, "teams": teams}
    return _UNIT_CACHE
