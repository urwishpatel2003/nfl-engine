"""
backtest_picks.py  —  Walk-forward pick strategy validation
============================================================
Proper out-of-sample test. To predict season N:
  - Player composite: uses rolling 6-week EPA (no future leakage)
  - Team styles: rebuilt using ONLY seasons before N
  - Spreads/results: from schedules parquet

Usage:
    python backtest_picks.py --seasons 2023 2024 2025
    python backtest_picks.py --seasons 2025 --min-gap 6
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"

WEEK_CONFIDENCE = {
    1:0.62,2:0.68,3:0.61,4:0.66,5:0.68,6:0.71,
    7:0.63,8:0.67,9:0.61,10:0.56,11:0.71,12:0.70,
    13:0.64,14:0.72,15:0.75,16:0.64,17:0.73,18:0.68,
}


def build_styles_for(train_seasons: list) -> pd.DataFrame:
    """Build team styles from training seasons only — the leaky component."""
    from engine.styles import build_team_styles
    try:
        styles = build_team_styles(train_seasons)
        # Suppress save message clutter — styles get saved to parquet
        return styles
    except Exception as e:
        print(f"      styles build failed: {e}")
        return pd.DataFrame()


def score_pick(pred: dict, game: pd.Series, week: int):
    home = pred.get("home_team")
    away = pred.get("away_team")
    home_pred   = pred.get("predicted_home_score", 22)
    away_pred   = pred.get("predicted_away_score", 22)
    pred_margin = home_pred - away_pred
    win_prob    = pred.get("home_win_probability", 0.5)
    vegas_spread = float(game.get("spread_line", 0) or 0)

    model_picks_home  = pred_margin > 0
    vegas_favors_home = vegas_spread > 0   # nflverse: positive spread_line = home favored

    if model_picks_home == vegas_favors_home:
        return None  # model agrees with Vegas — no signal

    pick_side = "home" if model_picks_home else "away"
    pick_team = home   if model_picks_home else away
    gap = abs(pred_margin - vegas_spread)   # model home margin vs Vegas home margin

    gap_score  = min(40, gap * 2.5)
    wp_score   = min(20, abs(win_prob - 0.5) * 80)
    week_acc   = WEEK_CONFIDENCE.get(week, 0.65)
    week_score = max(0, min(10, (week_acc - 0.56) / (0.75 - 0.56) * 10))
    home_rest  = float(game.get("home_rest") or np.nan)
    away_rest  = float(game.get("away_rest") or np.nan)
    rest_diff  = (home_rest - away_rest
                  if not (np.isnan(home_rest) or np.isnan(away_rest)) else 0)
    rest_score = (min(10, abs(rest_diff) * 2)
                  if ((pick_side=="home" and rest_diff>=3) or
                      (pick_side=="away" and rest_diff<=-3)) else 0)
    roof = str(game.get("roof","")).lower()
    char_score = 2 if ("dome" in roof or "retractable" in roof) else 0
    confidence = gap_score + wp_score + week_score + rest_score + char_score

    h_act = game.get("home_score")
    a_act = game.get("away_score")
    covered = None
    if pd.notna(h_act) and pd.notna(a_act):
        actual_margin = float(h_act) - float(a_act)
        # Home covers if home_margin > spread_line (positive = home favored)
        covered = (actual_margin > vegas_spread if pick_side=="home"
                   else actual_margin < vegas_spread)

    return {
        "pick_team": pick_team, "pick_side": pick_side,
        "gap": round(gap,1), "confidence": round(confidence,1),
        "covered": covered, "home": home, "away": away,
        "week": week, "vegas_spread": vegas_spread,
        "home_pred": round(home_pred,1), "away_pred": round(away_pred,1),
        "home_actual": h_act, "away_actual": a_act,
    }


def run_backtest(seasons: list, top_n: int = 5, min_gap: float = 0):
    from engine.predict import load_engine_data, predict_game

    print(f"\n{'='*68}")
    print(f"  WALK-FORWARD PICK BACKTEST — Top {top_n}/week  min_gap={min_gap}")
    print(f"  Test seasons: {seasons}")
    print(f"  Composite: full 2021-2025 (rolling 6-wk, no future leakage)")
    print(f"  Styles:    rebuilt from prior seasons only per test season")
    print(f"{'='*68}\n")

    # Load full engine data once — composite is already walk-forward safe
    # (rolling 6-week EPA means week N only uses weeks 1 through N)
    print("  Loading base engine data (full composite)...")
    base_data = load_engine_data()

    schedules = pd.read_parquet(RAW / "schedules.parquet")
    schedules["season"]    = pd.to_numeric(schedules["season"], errors="coerce").astype("Int64")
    schedules["game_type"] = schedules["game_type"].str.upper().str.strip()

    # Check spread availability
    for s in seasons:
        sg = schedules[(schedules["season"]==s) & (schedules["game_type"]=="REG")]
        n_with_spread = sg["spread_line"].notna().sum()
        print(f"  Season {s}: {len(sg)} REG games, {n_with_spread} with spread lines")

    all_results   = []
    season_totals = {}
    all_seasons   = sorted(set([2021,2022] + seasons))

    for test_season in seasons:
        train_seasons = [s for s in all_seasons if s < test_season]
        if not train_seasons:
            print(f"\n  {test_season}: skipping — no prior seasons for style training")
            continue

        print(f"\n  Season {test_season}  (styles from {train_seasons}):")
        print(f"  {'─'*50}")

        # Build OOS styles — only prior seasons
        styles_oos = build_styles_for(train_seasons)

        # Swap styles into a copy of base_data
        data = dict(base_data)
        data["styles"] = styles_oos

        s_games = schedules[
            (schedules["season"] == test_season) &
            (schedules["game_type"] == "REG") &
            (schedules["home_score"].notna()) &
            (schedules["spread_line"].notna())
        ].copy()

        if s_games.empty:
            print(f"    No games found with spread lines for {test_season}")
            season_totals[test_season] = {"correct":0,"total":0,"pct":0,"status":"no data","weeks":{}}
            continue

        season_correct = season_picks = 0
        week_results   = {}

        for week in sorted(s_games["week"].dropna().unique()):
            week_games = s_games[s_games["week"] == week]
            week_picks = []

            for _, game in week_games.iterrows():
                try:
                    pred = predict_game(
                        str(game["home_team"]), str(game["away_team"]),
                        int(test_season), int(week), data=data
                    )
                    p = score_pick(pred, game, int(week))
                    if p and p["gap"] >= min_gap:
                        week_picks.append(p)
                except Exception:
                    pass

            week_picks.sort(key=lambda x: x["confidence"], reverse=True)
            top = week_picks[:top_n]

            wk_correct = sum(1 for p in top if p["covered"] is True)
            wk_total   = sum(1 for p in top if p["covered"] is not None)
            season_correct += wk_correct
            season_picks   += wk_total
            week_results[int(week)] = {"correct":wk_correct,"total":wk_total,"picks":top}
            all_results.extend(top)

        pct    = season_correct / season_picks if season_picks > 0 else 0
        status = "WIN" if season_correct>=60 else "close" if season_correct>=56 else "below"
        season_totals[test_season] = {
            "correct":season_correct,"total":season_picks,
            "pct":pct,"status":status,"weeks":week_results
        }
        print(f"    {season_correct}/{season_picks} = {pct:.1%}  [{status}]")

    # ── Results ────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  WALK-FORWARD RESULTS  (styles OOS, composite rolling)")
    print(f"{'='*68}\n")

    total_correct = sum(s["correct"] for s in season_totals.values())
    total_picks   = sum(s["total"]   for s in season_totals.values())
    overall_pct   = total_correct / total_picks if total_picks > 0 else 0

    for s, res in season_totals.items():
        if res["total"] == 0:
            print(f"  {s}:  no data"); continue
        bar = "█"*res["correct"] + "░"*(res["total"]-res["correct"])
        print(f"  {s}: {res['correct']:2d}/{res['total']} = {res['pct']:.1%}"
              f"  [{res['status']}]  {bar}")

    if total_picks > 0:
        print(f"\n  OVERALL: {total_correct}/{total_picks} = {overall_pct:.1%}")
        print(f"  Per-season pace: {total_correct/len(seasons):.0f}/90")

    # Gap threshold breakdown
    print(f"\n  BY GAP THRESHOLD:")
    for thresh in [0, 3, 6, 9]:
        sub = [p for p in all_results
               if p["gap"]>=thresh and p["covered"] is not None]
        if sub:
            corr = sum(1 for p in sub if p["covered"])
            wpw  = len(sub) / max(len(seasons),1) / 18
            print(f"    Gap ≥{thresh:2d}: {corr}/{len(sub)} = {corr/len(sub):.1%}"
                  f"  ({wpw:.1f}/week avg)")

    # Week by week for last test season
    last = seasons[-1]
    if last in season_totals and season_totals[last]["total"] > 0:
        print(f"\n  WEEK-BY-WEEK ({last}):")
        for wk, wr in sorted(season_totals[last]["weeks"].items()):
            if wr["total"] == 0: continue
            bar = "#"*wr["correct"]+"."*(wr["total"]-wr["correct"])
            print(f"    Wk{wk:2d}: {wr['correct']}/{wr['total']}  [{bar}]")

    # Worst weeks
    all_weeks = [(s,wk,wr) for s,res in season_totals.items()
                 for wk,wr in res["weeks"].items() if wr["total"]>0]
    if all_weeks:
        worst = sorted(all_weeks, key=lambda x: x[2]["correct"]/x[2]["total"])[:5]
        print(f"\n  WORST WEEKS:")
        for s,wk,wr in worst:
            detail = ", ".join(f"{p['pick_team']}{'✓' if p['covered'] else '✗'}"
                               for p in wr["picks"])
            print(f"    {s} Wk{wk:2d}: {wr['correct']}/{wr['total']} = "
                  f"{wr['correct']/wr['total']:.0%}  [{detail}]")

    # Save
    if all_results:
        out = PROC / "pick_backtest_oos.csv"
        pd.DataFrame(all_results).to_csv(out, index=False)
        print(f"\n  Saved -> {out}")

    # Restore original styles so main backtest still works
    print("\n  Restoring full 2021-2025 styles...")
    from engine.styles import build_team_styles
    build_team_styles([2021,2022,2023,2024,2025])
    print("  Done.")

    return season_totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons",  nargs="+", type=int, default=[2025])
    parser.add_argument("--top",      type=int,  default=5)
    parser.add_argument("--min-gap",  type=float, default=0)
    args = parser.parse_args()
    run_backtest(args.seasons, args.top, args.min_gap)
