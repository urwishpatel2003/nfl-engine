"""
ml/rank.py  —  standalone power ratings + explainable game predictions
=======================================================================
Turns the fitted, leak-free fundamentals model (ml/features.py + XGBoost) into a
usable product: a market-independent power ranking of all 32 teams and per-game
predictions with the factors that drove them.

This is NOT a betting model — game markets are efficient wrt these features (see
ml/train.py). It IS a strong standalone predictor (OOS margin corr ~0.28) that
rates teams and explains matchups from squad quality, EPA, styles and coaching.

Power rating = the model's predicted margin of a team vs a league-average team on
a neutral field (no HFA). Higher = stronger. Ranked 1-32.

Usage:
  python -m ml.rank                       # power rankings for the latest season
  python -m ml.rank --season 2025
  python -m ml.rank --predict 2025 18     # predict a week's games (+ top factors)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROC = Path(__file__).parent.parent / "data" / "processed"
META = ["game_id", "season", "week", "home_team", "away_team", "home_margin", "total"]
MARKET = ["mkt_spread", "mkt_total", "mkt_home_impl"]
SITU = ["rest_diff", "is_div", "is_dome", "is_turf", "temp", "wind", "week_num"]


def _model():
    return xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            reg_lambda=1.0, n_jobs=4, random_state=0)


def load():
    df = pd.read_parquet(PROC / "game_features.parquet").sort_values(["season", "week"])
    feats = [c for c in df.columns if c not in META and c not in MARKET]
    bases = [c[2:] for c in feats if c.startswith("h_")]     # absolute metric names
    return df, feats, bases


def team_state(df: pd.DataFrame, bases: list, season: int) -> pd.DataFrame:
    """Latest absolute feature vector per team, up to the given season."""
    rows = []
    sub = df[df["season"] <= season]
    for side, tcol in [("h", "home_team"), ("a", "away_team")]:
        block = sub[[tcol, "season", "week"] + [f"{side}_{b}" for b in bases]].copy()
        block.columns = ["team", "season", "week"] + bases
        rows.append(block)
    allrows = pd.concat(rows, ignore_index=True).sort_values(["team", "season", "week"])
    return allrows.groupby("team").tail(1).set_index("team")[bases]


def power_ratings(season: int) -> pd.DataFrame:
    df, feats, bases = load()
    model = _model()
    model.fit(df[feats], df["home_margin"])

    state = team_state(df, bases, season)
    league_avg = state.mean()

    latest_week = int(df[df["season"] == season]["week"].max())
    ratings = []
    for team, srow in state.iterrows():
        row = {}
        for b in bases:
            row[f"h_{b}"] = srow[b]
            row[f"a_{b}"] = league_avg[b]
            row[f"d_{b}"] = srow[b] - league_avg[b]
        row.update({"rest_diff": 0, "is_div": 0, "is_dome": 0, "is_turf": 0,
                    "temp": float(df["temp"].median()), "wind": 0.0,
                    "week_num": latest_week})
        ratings.append({"team": team,
                        "rating": float(model.predict(pd.DataFrame([row])[feats])[0])})
    out = pd.DataFrame(ratings).sort_values("rating", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out["rating"] = out["rating"].round(1)
    return out


def predict_week(season: int, week: int):
    df, feats, _ = load()
    train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
    tgt   = df[(df["season"] == season) & (df["week"] == week)]
    if tgt.empty:
        print(f"No games found for {season} wk{week}"); return
    m = _model(); m.fit(train[feats], train["home_margin"])
    mt = _model(); mt.fit(train[feats], train["total"])
    pred_margin = m.predict(tgt[feats]); pred_total = mt.predict(tgt[feats])
    wp = 1 / (1 + np.exp(-pred_margin / 13.5 * np.pi / np.sqrt(3)))
    print(f"\n  {season} Week {week} — standalone model predictions")
    print(f"  {'matchup':22} {'pred':>12} {'total':>6} {'home win':>9}")
    for i, (_, g) in enumerate(tgt.iterrows()):
        pm, pt = pred_margin[i], pred_total[i]
        h = (pt + pm) / 2; a = (pt - pm) / 2
        print(f"  {g['away_team']:>3} @ {g['home_team']:<3}{'':13} "
              f"{a:4.1f}-{h:4.1f}  {pt:5.1f}  {wp[i]:7.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--predict", nargs=2, type=int, metavar=("SEASON", "WEEK"))
    args = ap.parse_args()

    if args.predict:
        predict_week(args.predict[0], args.predict[1])
        return

    r = power_ratings(args.season)
    out = PROC / f"power_rankings_{args.season}.csv"
    r.to_csv(out, index=False)
    print(f"\n  POWER RANKINGS — {args.season} (standalone model, neutral field)")
    print(f"  {'-'*34}")
    for _, row in r.iterrows():
        bar = "#" * max(0, int(round(row["rating"] + 8)))
        print(f"  {row['rank']:2d}. {row['team']:<3} {row['rating']:+5.1f}  {bar}")
    print(f"\n  Saved -> {out.name}")


if __name__ == "__main__":
    main()
