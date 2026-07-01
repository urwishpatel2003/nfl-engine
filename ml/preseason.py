"""
ml/preseason.py  —  regressed preseason projection for an unplayed season
==========================================================================
An unplayed season has no game results, so its "power ranking" can only be a
projection. The single most important, empirically-validated fact for that
projection: NFL team strength carries over only weakly year-to-year.

Measured on 2022->2025 (128 team-seasons):
  corr(prior-year strength, this-year strength) = 0.35
  optimal carry-over coefficient k ~= 0.35  ->  regress prior year ~65% to the mean

So the projection for season N is simply the season N-1 model rating (which already
encodes squad quality + EPA + coaching) shrunk toward the league mean by that factor:

    projected = mean + k * (rating_prev - mean)

Roster/QB/coaching tweaks add little on top: there is only ~0.26 pts/game of
predictable year-over-year signal beyond the mean, so we do NOT bolt on unvalidated
adjustments. This is deliberately humble — that is what the data supports.

Usage:
  python -m ml.preseason                # project 2026 from 2025, print + save
  python -m ml.preseason --validate     # show the carry-over validation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "raw"

CARRYOVER_K = 0.35   # validated (see _validate); regress ~65% toward the mean


def _season_strength() -> pd.DataFrame:
    """Model-free team strength per season = avg point differential / game."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["home_score"].notna()].copy()
    h = s.rename(columns={"home_team": "team"}).assign(m=lambda d: d.home_score - d.away_score)[["season", "team", "m"]]
    a = s.rename(columns={"away_team": "team"}).assign(m=lambda d: d.away_score - d.home_score)[["season", "team", "m"]]
    return pd.concat([h, a]).groupby(["season", "team"])["m"].mean().reset_index().rename(columns={"m": "strength"})


def _validate():
    td = _season_strength()
    prev = td.copy(); prev["season"] += 1
    m = td.merge(prev, on=["season", "team"], suffixes=("", "_prev")).dropna()
    k = float((m.strength_prev * m.strength).sum() / (m.strength_prev ** 2).sum())
    mae = lambda kk: float((m.strength - kk * m.strength_prev).abs().mean())
    print(f"\n  Carry-over validation on {sorted(m.season.unique())} ({len(m)} team-seasons)")
    print(f"    corr(prev, this)      : {m.strength_prev.corr(m.strength):.3f}")
    print(f"    optimal k             : {k:.2f}  (regress ~{(1-k)*100:.0f}% toward mean)")
    print(f"    MAE  k=1 / k={k:.2f} / k=0 : {mae(1):.2f} / {mae(k):.2f} / {mae(0):.2f}  pts/game\n")
    return k


def project(prev_season: int = 2025, k: float = CARRYOVER_K) -> pd.DataFrame:
    """Project the season *after* prev_season by regressing prev_season model ratings."""
    from ml.rank import power_ratings
    r = power_ratings(prev_season)[["team", "rating"]].rename(columns={"rating": "rating_prev"})
    mean = r["rating_prev"].mean()
    # center at 0 (points vs an average team) and regress toward the mean by k
    r["rating_prev"] = (r["rating_prev"] - mean).round(1)
    r["projected"] = (k * r["rating_prev"]).round(1)
    r = r.sort_values("projected", ascending=False).reset_index(drop=True)
    r.insert(0, "rank", r.index + 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-season", type=int, default=2025)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    k = _validate() if args.validate else CARRYOVER_K
    proj = project(args.prev_season, k)
    target = args.prev_season + 1
    out = PROC / f"preseason_projection_{target}.csv"
    proj.to_csv(out, index=False)

    print(f"  {target} PRESEASON PROJECTION  (regress {args.prev_season} ~{(1-k)*100:.0f}% to the mean)")
    print(f"  {'-'*46}")
    for _, x in proj.iterrows():
        bar = "#" * max(0, int(round(x["projected"] + 5)))
        print(f"  {x['rank']:2d}. {x['team']:<3} {x['projected']:+5.1f}  (2025: {x['rating_prev']:+5.1f})  {bar}")
    print(f"\n  Saved -> {out.name}")
    print(f"  NOTE: a projection for an unplayed season — wide uncertainty. Prior-year")
    print(f"  strength predicts next year at only r=0.35, hence the heavy regression.\n")


if __name__ == "__main__":
    main()
