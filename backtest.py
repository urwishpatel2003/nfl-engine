"""
backtest.py
-----------
Runs predictions for every completed game in a season and scores
against actual results.

Usage:
    python backtest.py --season 2024
    python backtest.py --seasons 2022 2023 2024
    python backtest.py --season 2024 --rebuild   # rebuild engine first
"""

import sys
import argparse
import time
import json
import statistics
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"
PROC.mkdir(exist_ok=True)


def run_backtest(seasons: list, rebuild: bool = False) -> pd.DataFrame:

    if rebuild:
        from engine.composite  import build_composite
        from engine.styles     import build_team_styles
        from engine.conditions import build_all_conditions
        print("Rebuilding engine components...")
        for s in seasons:
            build_composite(season=s)
        build_team_styles(seasons=seasons)
        build_all_conditions(seasons=seasons)

    from engine.predict import load_engine_data, predict_game
    print("Loading engine data...")
    data = load_engine_data()

    schedules = pd.read_parquet(RAW / "schedules.parquet")

    # Normalize season column — nfl_data_py sometimes returns float or string
    schedules["season"] = pd.to_numeric(schedules["season"], errors="coerce").astype("Int64")

    # Normalize game_type — nflverse uses 'REG', some versions use 'reg'
    schedules["game_type"] = schedules["game_type"].str.upper().str.strip()

    # Debug: show what seasons and game_types are actually present
    print(f"  schedules: {len(schedules)} total rows")
    print(f"  seasons present: {sorted(schedules['season'].dropna().unique().tolist())}")
    print(f"  game_types: {schedules['game_type'].value_counts().to_dict()}")

    schedules = schedules[
        (schedules["season"].isin(seasons)) &
        (schedules["game_type"] == "REG") &
        (schedules["home_score"].notna()) &
        (schedules["away_score"].notna())
    ].copy()

    print(f"  After filter: {len(schedules)} completed REG games across seasons {seasons}")
    print(f"\nRunning backtest: {len(schedules)} games across seasons {seasons}\n")

    results = []
    errors  = []

    for _, game in schedules.iterrows():
        home     = game["home_team"]
        away     = game["away_team"]
        season   = int(game["season"])
        week     = int(game["week"])
        home_act = float(game["home_score"])
        away_act = float(game["away_score"])

        try:
            pred = predict_game(home, away, season, week, data=data)

            home_pred = pred["predicted_home_score"]
            away_pred = pred["predicted_away_score"]

            pred_margin   = home_pred - away_pred
            actual_margin = home_act  - away_act
            pred_total    = home_pred + away_pred
            actual_total  = home_act  + away_act

            winner_correct = (pred_margin > 0) == (actual_margin > 0)
            spread_error   = pred_margin - actual_margin
            total_error    = pred_total  - actual_total

            # ATS vs Vegas line (if available)
            vegas_spread = game.get("spread_line")
            ats_correct  = None
            model_vegas_diff = None
            ats_signal   = "PASS"

            if pd.notna(vegas_spread):
                vegas_spread = float(vegas_spread)

                # SIMPLE HONEST ATS:
                # Step 1: Which team does our model think wins?
                model_picks_home = pred_margin > 0

                # Step 2: Did that team cover the spread?
                # nflverse spread_line is home perspective: POSITIVE = home favored
                # Home covers if actual_margin > spread_line
                # Away covers if actual_margin < spread_line
                actual_home_covers = actual_margin > vegas_spread
                ats_correct = (model_picks_home == actual_home_covers)

                # Model-Vegas gap: how much our predicted home margin differs
                # from the Vegas implied home margin (= spread_line)
                vegas_cover_line = vegas_spread   # home must beat this to cover
                model_vegas_diff = round(pred_margin - vegas_cover_line, 1)

                # Signal: only flag when model disagrees with Vegas on winner
                vegas_picks_home = vegas_spread > 0
                if model_picks_home != vegas_picks_home:
                    ats_signal = "BET HOME" if model_picks_home else "BET AWAY"
                else:
                    ats_signal = "PASS"

            # O/U vs Vegas total
            vegas_total = game.get("total_line")
            ou_correct  = None
            if pd.notna(vegas_total):
                model_over  = pred_total > float(vegas_total)
                actual_over = actual_total > float(vegas_total)
                ou_correct  = (model_over == actual_over)

            results.append({
                "season":          season,
                "week":            week,
                "home_team":       home,
                "away_team":       away,
                "home_pred":       round(home_pred, 1),
                "away_pred":       round(away_pred, 1),
                "home_actual":     home_act,
                "away_actual":     away_act,
                "pred_total":      round(pred_total, 1),
                "actual_total":    actual_total,
                "pred_margin":     round(pred_margin, 1),
                "actual_margin":   actual_margin,
                "winner_correct":  winner_correct,
                "spread_error":    round(spread_error, 1),
                "total_error":     round(total_error, 1),
                "abs_spread_err":  abs(spread_error),
                "abs_total_err":   abs(total_error),
                "ats_correct":     ats_correct,
                "ou_correct":      ou_correct,
                "vegas_spread":    vegas_spread,
                "vegas_total":     vegas_total,
                "model_vegas_diff": model_vegas_diff,
                "ats_signal":      ats_signal,
                "game_script":     pred["game_script"],
                "home_off_label":  pred["home_off_label"],
                "away_off_label":  pred["away_off_label"],
            })

        except Exception as e:
            errors.append(f"  {away}@{home} wk{week}: {e}")

    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(e)

    df = pd.DataFrame(results)

    # Save
    out_path = PROC / f"backtest_{'_'.join(str(s) for s in seasons)}.parquet"
    df.to_parquet(out_path, index=False)

    print_backtest_report(df)
    return df


def print_backtest_report(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS")
    print("=" * 65)

    seasons = sorted(df["season"].unique())
    print(f"  Seasons: {seasons}  |  Games: {len(df)}")
    print()

    # Overall metrics
    winner_acc  = df["winner_correct"].mean()
    spread_mae  = df["abs_spread_err"].mean()
    spread_bias = df["spread_error"].mean()
    total_mae   = df["abs_total_err"].mean()
    total_bias  = df["total_error"].mean()

    print(f"  OVERALL ACCURACY")
    print(f"  Winner correct:    {winner_acc:.1%}  ({df['winner_correct'].sum()}/{len(df)})")
    print(f"  Spread MAE:        {spread_mae:.1f} pts")
    print(f"  Spread bias:       {spread_bias:+.1f} pts  (+ = over-predicting home)")
    print(f"  Total MAE:         {total_mae:.1f} pts")
    print(f"  Total bias:        {total_bias:+.1f} pts  (+ = over-predicting scoring)")

    # ATS vs Vegas
    ats_df = df[df["ats_correct"].notna()]
    if not ats_df.empty:
        ats_acc = ats_df["ats_correct"].mean()
        print(f"\n  vs VEGAS LINES  ({len(ats_df)} games with lines)")
        print(f"  ATS accuracy:      {ats_acc:.1%}  (break-even = 52.4%)")

        # High-confidence filter: only direction disagreement games
        if "model_vegas_diff" in df.columns and "ats_signal" in df.columns:
            for min_diff, label in [(3.0, "3+pts gap"), (6.0, "6+pts gap"), (9.0, "9+pts gap")]:
                hc = ats_df[
                    (ats_df["ats_signal"] != "PASS") &
                    (ats_df["model_vegas_diff"].abs() >= min_diff)
                ]
                if len(hc) >= 10:
                    print(f"  ATS (direction disagree, {label}): "
                          f"{hc['ats_correct'].mean():.1%}  ({len(hc)} games)")

        # Flat bet P&L
        wins   = ats_acc * len(ats_df)
        losses = (1 - ats_acc) * len(ats_df)
        pnl    = wins * 100 - losses * 110
        print(f"  Flat bet P&L:     ${pnl:+,.0f}  (@ $100/game, -110 juice)")
        print(f"  ROI:               {pnl/(len(ats_df)*100)*100:.1f}%")

    ou_df = df[df["ou_correct"].notna()]
    if not ou_df.empty:
        ou_acc = ou_df["ou_correct"].mean()
        print(f"  O/U accuracy:      {ou_acc:.1%}")

    # By week
    print(f"\n  BY WEEK  (winner accuracy)")
    by_week = df.groupby("week")["winner_correct"].agg(["mean", "count"])
    for week, row in by_week.iterrows():
        bar = "#" * int(row["mean"] * 20)
        print(f"  Wk {week:2d}: {row['mean']:.0%}  {bar}  ({int(row['count'])} games)")

    # By season
    if len(seasons) > 1:
        print(f"\n  BY SEASON")
        by_season = df.groupby("season").agg(
            winner_acc=("winner_correct", "mean"),
            spread_mae=("abs_spread_err", "mean"),
            total_mae =("abs_total_err",  "mean"),
            games     =("winner_correct",  "count"),
        )
        for season, row in by_season.iterrows():
            print(f"  {season}: winner {row['winner_acc']:.1%} | "
                  f"spread MAE {row['spread_mae']:.1f} | "
                  f"total MAE {row['total_mae']:.1f} | "
                  f"{int(row['games'])} games")

    # Worst misses
    print(f"\n  WORST SPREAD MISSES (top 10)")
    worst = df.nlargest(10, "abs_spread_err")[
        ["season", "week", "away_team", "home_team",
         "away_pred", "home_pred", "away_actual", "home_actual", "spread_error"]
    ]
    for _, r in worst.iterrows():
        print(f"  Wk{r['week']:2d} {r['away_team']}@{r['home_team']}: "
              f"pred {r['away_pred']:.0f}-{r['home_pred']:.0f}  "
              f"actual {int(r['away_actual'])}-{int(r['home_actual'])}  "
              f"err {r['spread_error']:+.0f}")

    # Error distribution
    errs = df["abs_spread_err"].values
    print(f"\n  SPREAD ERROR DISTRIBUTION")
    print(f"  Within  3pts: {(errs <= 3).mean():.1%}")
    print(f"  Within  7pts: {(errs <= 7).mean():.1%}")
    print(f"  Within 10pts: {(errs <= 10).mean():.1%}")
    print(f"  Within 14pts: {(errs <= 14).mean():.1%}")
    print(f"  Over   14pts: {(errs > 14).mean():.1%}")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",  type=int, default=None)
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.seasons:
        seasons = args.seasons
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [2025]

    run_backtest(seasons, rebuild=args.rebuild)
