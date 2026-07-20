"""
ml/adjust.py  —  opponent-adjusted team unit ratings from play-by-play
======================================================================
Raw EPA/play flatters teams that faced soft schedules. This adjusts each team's
offensive and defensive pass/rush EPA for the quality of opponents faced, using an
alternating-least-squares fit (the same idea as an SRS / ridge rating):

    epa(play) ≈ league_avg + off_rating[posteam] + def_rating[defteam]

Fit off/def ratings by alternating conditional means until convergence, re-centering
each to mean 0 every pass. The result is the team's opponent-adjusted contribution to
EPA/play — offense: higher = better; defense: lower (more negative) = better (suppresses
EPA), matching the "EPA allowed" convention used by the matchup engine.

Feeds matchup_engine.team_units() so the 2025-performance component of every prediction
is schedule-adjusted rather than raw.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"

_ADJ_CACHE = {}


def adjusted_unit_epa(season: int, iters: int = 15) -> dict:
    """{team: {off_pass, off_rush, def_pass, def_rush}} opponent-adjusted EPA/play."""
    if season in _ADJ_CACHE:
        return _ADJ_CACHE[season]
    p = RAW / f"pbp_{season}.parquet"
    if not p.exists():
        return {}
    d = pd.read_parquet(p)
    d = d[(d["week"] <= 18) & d["epa"].notna() & d["posteam"].notna() & d["defteam"].notna()]

    out = {}
    for kind, sub in [("pass", d[d["play_type"] == "pass"]), ("rush", d[d["play_type"] == "run"])]:
        if sub.empty:
            continue
        lg = float(sub["epa"].mean())
        off_g = {t: g for t, g in sub.groupby("posteam")}
        def_g = {t: g for t, g in sub.groupby("defteam")}
        teams = sorted(set(off_g) | set(def_g))
        o = {t: 0.0 for t in teams}
        de = {t: 0.0 for t in teams}
        for _ in range(iters):
            for t, g in off_g.items():                       # offense given current defenses
                o[t] = float((g["epa"] - lg - g["defteam"].map(de)).mean())
            mo = float(np.mean(list(o.values())))
            for t in o:
                o[t] -= mo
            for t, g in def_g.items():                       # defense given current offenses
                de[t] = float((g["epa"] - lg - g["posteam"].map(o)).mean())
            md = float(np.mean(list(de.values())))
            for t in de:
                de[t] -= md
        for t in teams:
            rec = out.setdefault(t, {})
            rec[f"off_{kind}"] = round(o.get(t, 0.0), 4)     # + = good offense
            rec[f"def_{kind}"] = round(de.get(t, 0.0), 4)    # − = good defense (suppresses EPA)
    _ADJ_CACHE[season] = out
    return out


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    adj = adjusted_unit_epa(yr)
    df = pd.DataFrame(adj).T
    df["net_off"] = df["off_pass"] + df["off_rush"]
    df["net_def"] = df["def_pass"] + df["def_rush"]
    print(f"\nOpponent-adjusted units {yr} — best offenses (net off EPA):")
    print(df.sort_values("net_off", ascending=False).head(6)[["off_pass", "off_rush", "net_off"]].round(3).to_string())
    print(f"\nBest defenses (net def EPA allowed, lower=better):")
    print(df.sort_values("net_def").head(6)[["def_pass", "def_rush", "net_def"]].round(3).to_string())
