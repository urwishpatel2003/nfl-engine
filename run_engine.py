"""
run_engine.py
-------------
Master script. Runs all engine components in order.

Usage:
    # Build everything for 2024
    python run_engine.py --season 2024

    # Build + predict a specific week
    python run_engine.py --season 2024 --week 18

    # Predict a single game
    python run_engine.py --season 2024 --week 18 --home KC --away BUF

    # Full rebuild (all seasons)
    python run_engine.py --seasons 2020 2021 2022 2023 2024 --week 18
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run(label: str, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t = time.time()
    result = fn(*args, **kwargs)
    print(f"  Done in {time.time()-t:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="NFL Engine - full pipeline")
    parser.add_argument("--season",  type=int, default=2025)
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--week",    type=int, default=None)
    parser.add_argument("--home",    type=str, default=None)
    parser.add_argument("--away",    type=str, default=None)
    parser.add_argument("--skip-composite", action="store_true")
    parser.add_argument("--skip-styles",    action="store_true")
    parser.add_argument("--skip-conditions",action="store_true")
    args = parser.parse_args()

    seasons = args.seasons or [args.season]
    season  = args.season

    print(f"\nNFL ENGINE -- Season {season} | Seasons: {seasons}")

    # Step 1: Player composite scores — build for ALL seasons, merge into one parquet
    if not args.skip_composite:
        from engine.composite import build_composite
        import pandas as pd
        from pathlib import Path
        PROC = Path(__file__).parent / "data" / "processed"
        frames = []
        for s in seasons:
            print(f"\n{'='*60}")
            print(f"  Step 1: Player Composite Scores — Season {s}")
            print(f"{'='*60}")
            t = time.time()
            df = build_composite(season=s, week=args.week)
            print(f"  Done in {time.time()-t:.1f}s")
            if df is not None and len(df) > 0:
                frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            out = PROC / "composite_scores.parquet"
            combined.to_parquet(out, index=False)
            print(f"\n  Merged composite: {len(combined):,} player-weeks across {seasons}")

    # Step 2: Team style profiles
    if not args.skip_styles:
        from engine.styles import build_team_styles
        run("Step 2: Team Style Profiles", build_team_styles, seasons=seasons)

    # Step 3: Conditions modifiers
    if not args.skip_conditions:
        from engine.conditions import build_all_conditions
        run("Step 3: Conditions Modifiers", build_all_conditions, seasons=seasons)

    # Step 4: Predictions
    from engine.predict import predict_game, predict_week, print_game_report, load_engine_data

    if args.home and args.away:
        week = args.week or 1
        run(f"Step 4: Predict {args.away} @ {args.home} Week {week}",
            lambda: None)
        data = load_engine_data()
        pred = predict_game(args.home, args.away, season, week, data=data)
        print_game_report(pred)

    elif args.week:
        run(f"Step 4: Predict All Week {args.week} Games",
            predict_week, season, args.week)

    else:
        print(f"\nEngine built. Run with --week N to generate predictions.")
        print(f"Example: python run_engine.py --season 2024 --week 18 --home KC --away BUF")


if __name__ == "__main__":
    main()
