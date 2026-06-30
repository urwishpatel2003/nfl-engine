"""
ats_season.py
-------------
Run ATS signals for every week of a completed season
and score against actual results.

Usage:
    python ats_season.py --season 2025
    python ats_season.py --season 2025 --min-prob 0.58
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"


def run_season_ats(season: int, min_win_prob: float = 0.55):
    from engine.predict import load_engine_data, predict_week
    from engine.ats_signals import compute_ats_signals

    print(f"\nLoading engine data...")
    data      = load_engine_data()
    schedules = pd.read_parquet(RAW / "schedules.parquet")

    season_sched = schedules[
        (schedules["season"] == season) &
        (schedules["game_type"] == "REG") &
        (schedules["home_score"].notna()) &
        (schedules["away_score"].notna())
    ].copy()

    weeks = sorted(season_sched["week"].unique())
    print(f"Season {season}: {len(weeks)} weeks, {len(season_sched)} games\n")

    all_bets = []

    for week in weeks:
        week_games = season_sched[season_sched["week"] == week]

        # Get predictions for this week
        preds_path = PROC / f"predictions_{season}_wk{int(week):02d}.parquet"
        if preds_path.exists():
            preds_df = pd.read_parquet(preds_path)
        else:
            preds_df = predict_week(season, int(week))

        if preds_df.empty:
            continue

        # Run ATS signals on each game
        week_bets = []
        for _, pred_row in preds_df.iterrows():
            pred = pred_row.to_dict()
            pred["key_matchups"] = []
            signal = compute_ats_signals(pred, schedules)

            if signal["ats_signal"] == "PASS":
                continue

            # Get actual result
            actual = week_games[
                (week_games["home_team"] == signal["home_team"]) &
                (week_games["away_team"] == signal["away_team"])
            ]
            if actual.empty:
                continue

            g = actual.iloc[0]
            home_score = float(g["home_score"])
            away_score = float(g["away_score"])
            actual_margin = home_score - away_score  # home perspective

            vegas_spread = signal.get("vegas_spread")
            if vegas_spread is None or pd.isna(vegas_spread):
                # Try from schedules
                vs = g.get("spread_line")
                if pd.notna(vs):
                    vegas_spread = float(vs)
                else:
                    continue

            # Did the model's pick cover? nflverse spread_line>0 = home favored;
            # home covers if home_margin > spread_line.
            model_picks_home = signal["predicted_home_score"] > signal["predicted_away_score"]
            actual_home_covers = actual_margin > vegas_spread
            ats_correct = (model_picks_home == actual_home_covers)

            bet_side = "HOME" if model_picks_home else "AWAY"
            game_str = f"{signal['away_team']}@{signal['home_team']}"

            week_bets.append({
                "week":          int(week),
                "game":          game_str,
                "home_team":     signal["home_team"],
                "away_team":     signal["away_team"],
                "signal":        signal["ats_signal"],
                "bet_side":      bet_side,
                "confidence":    signal["ats_confidence"],
                "vegas_spread":  vegas_spread,
                "pred_home":     round(signal["predicted_home_score"], 1),
                "pred_away":     round(signal["predicted_away_score"], 1),
                "actual_home":   home_score,
                "actual_away":   away_score,
                "actual_margin": actual_margin,
                "ats_correct":   ats_correct,
                "home_win_prob": signal["home_win_probability"],
                "signal_reasons": " | ".join(signal.get("signal_reasons", [])[:1]),
                "is_divisional": signal.get("is_divisional", False),
                "units":         signal["recommended_units"],
            })

        all_bets.extend(week_bets)

        # Week summary
        if week_bets:
            wdf = pd.DataFrame(week_bets)
            acc = wdf["ats_correct"].mean()
            wu = wdf[wdf["ats_correct"]]["units"].sum()
            lu = wdf[~wdf["ats_correct"]]["units"].sum()
            print(f"  Wk{int(week):2d}: {len(wdf):2d} bets | "
                  f"{wdf['ats_correct'].sum():.0f}W-{(~wdf['ats_correct']).sum():.0f}L "
                  f"({acc:.0%}) | "
                  f"+{wu:.1f}u / -{lu*1.1:.1f}u")

    if not all_bets:
        print("No bets generated.")
        return pd.DataFrame()

    df = pd.DataFrame(all_bets)

    # ── SEASON SUMMARY ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SEASON {season} ATS RESULTS — ALL WEEKS")
    print(f"{'='*65}")

    total   = len(df)
    wins    = df["ats_correct"].sum()
    losses  = total - wins
    acc     = wins / total
    units_w = df[df["ats_correct"]]["units"].sum()
    units_l = df[~df["ats_correct"]]["units"].sum()
    pnl     = units_w * 100 - units_l * 110
    roi     = pnl / (total * 100) * 100

    print(f"\n  Total bets:    {total}")
    print(f"  Record:        {int(wins)}W - {int(losses)}L  ({acc:.1%})")
    print(f"  Units won:     +{units_w:.1f}u")
    print(f"  Units lost:    -{units_l:.1f}u (at -110)")
    print(f"  Net P&L:       ${pnl:+,.0f}  (@ $100/unit)")
    print(f"  ROI:           {roi:.1f}%")
    print(f"  O/U accuracy:  {df.get('ou_correct', pd.Series()).mean():.1%}" if "ou_correct" in df else "")

    # By signal strength
    print(f"\n  BY SIGNAL STRENGTH:")
    for sig_type, label in [("★★", "Double star"), ("★", "Single star"), ("BET", "No star")]:
        mask = df["signal"].str.contains(sig_type.replace("★", "★"), regex=False, na=False)
        if sig_type == "BET":
            mask = df["signal"].str.startswith("BET") & ~df["signal"].str.contains("★")
        sub = df[mask]
        if len(sub) > 0:
            sa = sub["ats_correct"].mean()
            print(f"    {label} ({sig_type}): {len(sub)} bets | "
                  f"{sub['ats_correct'].sum():.0f}W-{(~sub['ats_correct']).sum():.0f}L "
                  f"({sa:.1%})")

    # Divisional vs non-divisional
    if "is_divisional" in df.columns:
        div = df[df["is_divisional"]]
        ndiv = df[~df["is_divisional"]]
        if len(div) > 0:
            print(f"\n  DIVISIONAL games:     {len(div)} bets | {div['ats_correct'].mean():.1%}")
        if len(ndiv) > 0:
            print(f"  NON-DIVISIONAL games: {len(ndiv)} bets | {ndiv['ats_correct'].mean():.1%}")

    # By week range
    print(f"\n  BY PHASE:")
    early = df[df["week"] <= 6]
    mid   = df[(df["week"] >= 7) & (df["week"] <= 13)]
    late  = df[df["week"] >= 14]
    for label, sub in [("Early (Wk1-6)", early), ("Mid (Wk7-13)", mid), ("Late (Wk14-18)", late)]:
        if len(sub) > 0:
            print(f"    {label}: {len(sub)} bets | {sub['ats_correct'].mean():.1%}")

    # Best and worst weeks
    by_week = df.groupby("week").agg(
        bets=("ats_correct", "count"),
        wins=("ats_correct", "sum"),
        acc=("ats_correct", "mean")
    ).reset_index()
    best  = by_week.nlargest(3, "acc")
    worst = by_week.nsmallest(3, "acc")

    print(f"\n  BEST WEEKS:")
    for _, r in best.iterrows():
        print(f"    Wk{int(r['week']):2d}: {int(r['wins'])}/{int(r['bets'])} ({r['acc']:.0%})")

    print(f"\n  WORST WEEKS:")
    for _, r in worst.iterrows():
        print(f"    Wk{int(r['week']):2d}: {int(r['wins'])}/{int(r['bets'])} ({r['acc']:.0%})")

    # Individual game results
    print(f"\n  ALL BETS — DETAIL:")
    print(f"  {'Wk':>3} {'Game':<16} {'Signal':<16} {'Vegas':>6} "
          f"{'Pred':>10} {'Actual':>10} {'ATS':>5}")
    print(f"  {'-'*75}")
    for _, r in df.sort_values(["week"]).iterrows():
        result = "WIN ✓" if r["ats_correct"] else "LOSS ✗"
        pred_str = f"{r['pred_away']:.0f}-{r['pred_home']:.0f}"
        act_str  = f"{r['actual_away']:.0f}-{r['actual_home']:.0f}"
        vline    = f"{r['vegas_spread']:+.1f}" if pd.notna(r.get("vegas_spread")) else "N/A"
        print(f"  Wk{int(r['week']):2d} {r['game']:<16} {r['signal']:<16} "
              f"{vline:>6} {pred_str:>10} {act_str:>10} {result:>5}")

    print(f"\n{'='*65}\n")

    # Save results
    out = PROC / f"ats_results_{season}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved -> {out}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int, default=2025)
    parser.add_argument("--min-prob", type=float, default=0.55,
                        help="Min win probability for signal to fire")
    args = parser.parse_args()

    run_season_ats(args.season, args.min_prob)
