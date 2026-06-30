"""
ats.py
------
ATS signal runner. Run from project root.

Usage:
    python ats.py --season 2026 --week 5
    python ats.py --season 2025 --week 15   # backtest a historical week
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.ats_signals import analyze_week, print_ats_report

def main():
    parser = argparse.ArgumentParser(description="ATS betting signal engine")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week",   type=int, required=True)
    parser.add_argument("--min-diff", type=float, default=2.5,
                        help="Min model-Vegas spread difference to flag (default 2.5)")
    args = parser.parse_args()

    print(f"\nRunning ATS signals: Season {args.season} Week {args.week}")
    print(f"Confidence filter: {args.min_diff}+ pt model-Vegas disagreement\n")

    df = analyze_week(args.season, args.week)
    if df.empty:
        print("No predictions found. Run run_engine.py first.")
        return

    print_ats_report(df)

    # Summary stats
    bets = df[df["ats_signal"].str.startswith("BET", na=False)]
    print(f"Games flagged as bets: {len(bets)}/{len(df)}")
    if len(bets) > 0:
        print(f"Total units recommended: {bets['recommended_units'].sum():.1f}")
        print(f"Avg confidence: {bets['ats_confidence'].mean():.0f}%")

if __name__ == "__main__":
    main()
