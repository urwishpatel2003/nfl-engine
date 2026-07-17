"""
ml/ratings.py  —  transparent power ratings (SRS) + matchup predictions
========================================================================
The product engine for the dashboard. Deliberately SIMPLE and interpretable —
we proved (ml/train.py) the fancy ML model doesn't beat the market, so using it as
a *ranking* is opaque AND distorted (it put JAX #1 over SEA by ignoring strength of
schedule). A power ranking should be defensible, so we use:

  SRS  — Simple Rating System: average scoring margin adjusted for opponent quality,
         solved iteratively. The standard, transparent NFL power rating.
  off/def points-per-game — for matchup score/total estimates.

Preseason projection for an unplayed season regresses toward the mean by the
validated carry-over factor (NFL strength persists year-to-year at only r~0.35).

Usage:
  python -m ml.ratings --season 2025               # SRS ranking (last completed season)
  python -m ml.ratings --season 2025 --project     # 2026 preseason projection (regressed)
  python -m ml.ratings --validate                  # SRS year-over-year carry-over
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAWD = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
CARRYOVER_K = 0.35          # validated (see --validate); regress ~65% toward the mean
HFA = 2.0                   # home-field points (~league avg home margin)


def _games(season: int) -> pd.DataFrame:
    s = pd.read_parquet(RAWD / "schedules.parquet")
    return s[(s.season == season) & (s.game_type.str.upper() == "REG") & s.home_score.notna()].copy()


def season_ratings(season: int) -> pd.DataFrame:
    """Per-team off/def points-per-game, point differential, and SRS for a season."""
    g = _games(season)
    rows = []
    for _, r in g.iterrows():
        rows.append((r.home_team, r.away_team, r.home_score, r.away_score))
        rows.append((r.away_team, r.home_team, r.away_score, r.home_score))
    df = pd.DataFrame(rows, columns=["t", "opp", "pf", "pa"])
    agg = df.groupby("t").agg(off_ppg=("pf", "mean"), def_ppg=("pa", "mean")).reset_index()
    agg["margin"] = agg["off_ppg"] - agg["def_ppg"]

    # SRS: rating = avg margin + avg opponent rating (iterate to convergence)
    margin = df.assign(m=df.pf - df.pa).groupby("t")["m"].mean()
    srs = margin.copy()
    for _ in range(1000):
        opp = df.groupby("t").apply(lambda x: srs[x.opp].mean(), include_groups=False)
        new = (margin + opp); new = new - new.mean()
        if (new - srs).abs().max() < 1e-7:
            srs = new; break
        srs = new
    agg["srs"] = agg["t"].map(srs).round(2)
    return agg.rename(columns={"t": "team"})


def _regress(x: pd.Series, k: float) -> pd.Series:
    return (k * (x - x.mean())).round(2)


def power_ratings(season: int, project: bool = False) -> pd.DataFrame:
    """SRS power ratings. project=True regresses toward the mean for the next (unplayed) season."""
    r = season_ratings(season)[["team", "srs", "off_ppg", "def_ppg"]].copy()
    r["srs"] = (r["srs"] - r["srs"].mean()).round(2)        # center at 0
    r["rating_prev"] = r["srs"]
    r["rating"] = _regress(r["srs"], CARRYOVER_K) if project else r["srs"]
    r = r.sort_values("rating", ascending=False).reset_index(drop=True)
    r.insert(0, "rank", r.index + 1)
    return r


def predict_matchup(home: str, away: str, season: int = 2025,
                    neutral: bool = False, project: bool = False) -> dict:
    """Predict a game from team scoring profiles (regressed if project=True)."""
    r = season_ratings(season)
    if home not in r.team.values or away not in r.team.values:
        return {"error": "unknown team(s)"}
    lg = float(pd.concat([r.off_ppg, r.def_ppg]).mean())
    off, deff = r.set_index("team")["off_ppg"], r.set_index("team")["def_ppg"]
    if project:  # regress each team's scoring toward the league mean
        off = lg + CARRYOVER_K * (off - lg)
        deff = lg + CARRYOVER_K * (deff - lg)
    hfa = 0.0 if neutral else HFA
    home_pts = off[home] + (deff[away] - lg) + hfa / 2
    away_pts = off[away] + (deff[home] - lg) - hfa / 2
    margin = home_pts - away_pts
    wp = float(1 / (1 + np.exp(-margin / 13.5 * np.pi / np.sqrt(3))))
    return {
        "home": home, "away": away,
        "pred_home_score": round(float(home_pts), 1),
        "pred_away_score": round(float(away_pts), 1),
        "pred_margin": round(float(margin), 1),
        "pred_total": round(float(home_pts + away_pts), 1),
        "home_win_prob": round(wp, 3), "away_win_prob": round(1 - wp, 3),
    }


def _validate():
    """SRS year-over-year: how much carries over (the regression factor)."""
    seasons = [2021, 2022, 2023, 2024, 2025]
    srs = pd.concat([season_ratings(s).assign(season=s)[["season", "team", "srs"]] for s in seasons])
    prev = srs.rename(columns={"srs": "prev"}); prev["season"] += 1
    m = srs.merge(prev, on=["season", "team"]).dropna()
    k = float((m.prev * m.srs).sum() / (m.prev ** 2).sum())
    print(f"\n  SRS carry-over on {sorted(m.season.unique())} ({len(m)} team-seasons)")
    print(f"    corr(prev SRS, this SRS): {m.prev.corr(m.srs):.3f}")
    print(f"    optimal k: {k:.2f}  (regress ~{(1-k)*100:.0f}% toward the mean)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        _validate(); return
    r = power_ratings(args.season, project=args.project)
    tag = f"{args.season+1} PRESEASON PROJECTION (regressed)" if args.project else f"{args.season} SRS RATINGS"
    print(f"\n  {tag}\n  {'-'*40}")
    for _, x in r.iterrows():
        bar = "#" * max(0, int(round(x["rating"] + 8)))
        prev = f"  (2025 SRS {x['rating_prev']:+.1f})" if args.project else ""
        print(f"  {x['rank']:2d}. {x['team']:<3} {x['rating']:+5.1f}{prev}  {bar}")
    print()


if __name__ == "__main__":
    main()
