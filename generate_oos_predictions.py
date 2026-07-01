"""
generate_oos_predictions.py  —  honest out-of-sample predictions for ALL games
==============================================================================
backtest_picks.py runs the model walk-forward (styles rebuilt from prior seasons
only) but saves only the *picks*. To evaluate market regression and benchmark
against nfelo/Vegas fairly we need EVERY game's prediction, generated OOS.

This mirrors backtest_picks' style-swap loop but writes one row per game with the
columns the evaluation tools expect:
  season, week, home_team, away_team, pred_margin, actual_margin,
  pred_total, actual_total, vegas_spread (spread_line), vegas_total (total_line),
  home_win_probability

Output: data/processed/oos_predictions_<seasons>.parquet

Usage:
  python generate_oos_predictions.py --seasons 2023 2024 2025
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    args = ap.parse_args()

    from engine.predict import load_engine_data, predict_game
    from engine.styles import build_team_styles

    print("  Loading base engine data (de-leaked composite)...")
    base_data = load_engine_data()

    sched = pd.read_parquet(RAW / "schedules.parquet")
    sched["season"]    = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["game_type"] = sched["game_type"].str.upper().str.strip()

    all_seasons = sorted(set([2021, 2022] + args.seasons))
    rows = []

    for test_season in args.seasons:
        train = [s for s in all_seasons if s < test_season]
        if not train:
            print(f"  {test_season}: no prior seasons for OOS styles — skipping")
            continue
        print(f"  {test_season}: styles from {train} ...")
        styles_oos = build_team_styles(train)
        data = dict(base_data)
        data["styles"] = styles_oos

        games = sched[
            (sched["season"] == test_season) &
            (sched["game_type"] == "REG") &
            sched["home_score"].notna() &
            sched["spread_line"].notna()
        ]

        for _, g in games.iterrows():
            try:
                p = predict_game(str(g["home_team"]), str(g["away_team"]),
                                 int(test_season), int(g["week"]), data=data)
            except Exception:
                continue
            hp, ap_ = p["predicted_home_score"], p["predicted_away_score"]
            rows.append({
                "season": int(test_season), "week": int(g["week"]),
                "home_team": g["home_team"], "away_team": g["away_team"],
                "pred_margin":   hp - ap_,
                "actual_margin": float(g["home_score"]) - float(g["away_score"]),
                "pred_total":    hp + ap_,
                "actual_total":  float(g["home_score"]) + float(g["away_score"]),
                "vegas_spread":  float(g["spread_line"]),
                "vegas_total":   float(g["total_line"]) if pd.notna(g.get("total_line")) else np.nan,
                "home_win_probability": p["home_win_probability"],
            })

    df = pd.DataFrame(rows)
    out = PROC / f"oos_predictions_{'_'.join(str(s) for s in args.seasons)}.parquet"
    df.to_parquet(out, index=False)

    # restore full styles so other tools keep working
    build_team_styles([2021, 2022, 2023, 2024, 2025])

    mae = (df["pred_margin"] - df["actual_margin"]).abs().mean()
    corr = df["pred_margin"].corr(df["actual_margin"])
    print(f"\n  Saved {len(df)} OOS predictions -> {out.name}")
    print(f"  Raw-model OOS margin: MAE={mae:.2f}  corr={corr:.3f}")


if __name__ == "__main__":
    main()
