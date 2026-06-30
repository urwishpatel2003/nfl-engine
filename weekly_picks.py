"""
weekly_picks.py  —  Vegas Contest Pick Ranker
==============================================
Generates your top 5 ATS picks for the week, ranked by confidence.
Optimized for a 5-picks/week contest where you need 60-63/90 correct.

Usage:
    python weekly_picks.py --season 2026 --week 1
    python weekly_picks.py --season 2026 --week 1 --top 5
    python weekly_picks.py --season 2026 --week 1 --show-all

Strategy:
    - Only picks games where model direction DISAGREES with Vegas line
    - Ranks by composite confidence score (model gap + style clash + form)
    - Applies filters that historically correlate with model accuracy
    - Outputs exactly 5 picks with full reasoning
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"


# ── Historical ATS accuracy by model-Vegas gap (from backtest) ──────
# These are IN-SAMPLE. Apply 0.75x discount for true out-of-sample.
HISTORICAL_ATS = {
    3:  0.838,   # model disagrees with Vegas by 3+ pts
    6:  0.870,   # 6+ pts gap
    9:  0.895,   # 9+ pts gap
    12: 0.910,   # 12+ pts gap (extrapolated)
}
OOS_DISCOUNT = 0.78  # true out-of-sample ≈ 78% of in-sample

# ── Game quality filters ─────────────────────────────────────────────
# Based on week-by-week accuracy from backtest
WEEK_CONFIDENCE = {
    1: 0.62, 2: 0.68, 3: 0.61, 4: 0.66, 5: 0.68, 6: 0.71,
    7: 0.63, 8: 0.67, 9: 0.61, 10: 0.56, 11: 0.71, 12: 0.70,
    13: 0.64, 14: 0.72, 15: 0.75, 16: 0.64, 17: 0.73, 18: 0.68,
}


def score_game(pred: dict, game_row: pd.Series, week: int) -> dict:
    """
    Compute a composite confidence score (0-100) for an ATS pick.

    Components:
      1. Model-Vegas gap size (40%) — primary signal
      2. Win probability strength (20%) — how convinced is the model
      3. Style clash advantage (15%) — structural matchup edge
      4. Week confidence factor (10%) — model accuracy by week
      5. Rest differential (10%) — short/long week advantage
      6. Game characteristics (5%) — dome, known turf, etc.

    Returns None if the model AGREES with Vegas (no pick signal).
    """
    home  = pred.get("home_team")
    away  = pred.get("away_team")
    home_pred = pred.get("predicted_home_score", 22)
    away_pred = pred.get("predicted_away_score", 22)
    pred_margin  = home_pred - away_pred          # + = home favored by model
    win_prob     = pred.get("home_win_probability", 0.5)

    # nflverse spread_line: home perspective, POSITIVE = home favored (e.g. +7 = home -7)
    vegas_spread = float(game_row.get("spread_line", 0) or 0)
    # Vegas implied home margin == spread_line.
    # pred_margin > 0 means model picks home, < 0 means model picks away
    # ATS pick: compare model direction to Vegas direction

    model_picks_home  = pred_margin > 0
    vegas_favors_home = vegas_spread > 0   # positive spread_line = home favored

    # Only generate a pick when model direction DISAGREES with Vegas
    if model_picks_home == vegas_favors_home:
        return None

    # Who we're picking — cover_need is the line the pick must beat (home perspective)
    if model_picks_home:
        pick_team   = home
        pick_side   = "home"
        cover_need  = vegas_spread        # home must win by more than spread_line
    else:
        pick_team   = away
        pick_side   = "away"
        cover_need  = -vegas_spread       # away gets +spread_line; flip to away perspective

    # ── Component 1: Model-Vegas gap (40 pts max) ──────────────────
    # How far does our predicted margin deviate from the Vegas spread?
    # Larger gap = stronger disagreement = higher confidence
    model_implied_spread = pred_margin           # model's predicted home margin
    gap = abs(model_implied_spread - vegas_spread)  # vs Vegas home margin (= spread_line)
    gap_score = min(40, gap * 2.5)   # 16pt gap = 40 pts score

    # ── Component 2: Win probability conviction (20 pts max) ──────
    # Distance from 50% = model conviction
    conviction = abs(win_prob - 0.5)
    wp_score = min(20, conviction * 80)  # 25% edge = 20 pts score

    # ── Component 3: Style clash advantage (15 pts max) ───────────
    # Sum of positive matchup clashes for the pick side
    matchup_summary = pred.get("matchup_summary", {})
    pick_team_matchup = matchup_summary.get(pick_team, {})
    clash_edge = pick_team_matchup.get("weighted_matchup_edge", 0)
    clash_score = min(15, max(0, clash_edge * 1.5))

    # ── Component 4: Week confidence (10 pts max) ─────────────────
    week_acc = WEEK_CONFIDENCE.get(week, 0.65)
    week_score = (week_acc - 0.56) / (0.75 - 0.56) * 10
    week_score = max(0, min(10, week_score))

    # ── Component 5: Rest differential (10 pts max) ───────────────
    rest_score = 0.0
    home_rest = float(game_row.get("home_rest") or 7)
    away_rest = float(game_row.get("away_rest") or 7)
    if not np.isnan(home_rest) and not np.isnan(away_rest):
        rest_diff = home_rest - away_rest  # + = home has more rest
        if pick_side == "home" and rest_diff >= 3:
            rest_score = min(10, rest_diff * 2)
        elif pick_side == "away" and rest_diff <= -3:
            rest_score = min(10, abs(rest_diff) * 2)

    # ── Component 6: Game characteristics (5 pts max) ──────────────
    char_score = 0.0
    roof = str(game_row.get("roof", "")).lower()
    surface = str(game_row.get("surface", "")).lower()
    if "dome" in roof or "retractable" in roof:
        char_score += 2  # dome games are more predictable (no weather)
    if "turf" in surface or "astro" in surface:
        char_score += 1  # turf games have less variance

    # ── Total score ────────────────────────────────────────────────
    total = gap_score + wp_score + clash_score + week_score + rest_score + char_score
    total = round(min(100, total), 1)

    # ── Estimated true accuracy for this pick ─────────────────────
    # Find applicable historical bracket
    oos_acc = 0.0
    for threshold in sorted(HISTORICAL_ATS.keys(), reverse=True):
        if gap >= threshold:
            oos_acc = HISTORICAL_ATS[threshold] * OOS_DISCOUNT
            break
    if oos_acc == 0:
        oos_acc = 0.55  # below 3pt gap — weak signal

    # ── Flags ──────────────────────────────────────────────────────
    flags = []
    if gap >= 9:  flags.append("STRONG")
    elif gap >= 6: flags.append("good")
    if rest_score >= 6: flags.append("rest edge")
    if clash_score >= 8: flags.append("style clash")
    if week_acc < 0.63:  flags.append("weak week")
    if abs(vegas_spread) <= 1.5: flags.append("pick6 risk")

    return {
        "pick_team":     pick_team,
        "pick_side":     pick_side,
        "vs_team":       away if pick_side == "home" else home,
        "vegas_spread":  vegas_spread,
        "model_margin":  round(pred_margin, 1),
        "gap":           round(gap, 1),
        "cover_needed":  round(cover_need, 1),
        "confidence":    total,
        "win_prob":      round(win_prob, 3),
        "est_accuracy":  round(oos_acc, 3),
        "flags":         flags,
        "components": {
            "gap_score":    round(gap_score, 1),
            "wp_score":     round(wp_score, 1),
            "clash_score":  round(clash_score, 1),
            "week_score":   round(week_score, 1),
            "rest_score":   round(rest_score, 1),
            "char_score":   round(char_score, 1),
        },
        "home_team": home,
        "away_team": away,
        "home_pred": round(home_pred, 1),
        "away_pred": round(away_pred, 1),
        "home_score_actual": game_row.get("home_score"),
        "away_score_actual": game_row.get("away_score"),
    }


def run_week(season: int, week: int, top_n: int = 5, show_all: bool = False):
    from engine.predict import load_engine_data, predict_game

    print(f"\n{'='*70}")
    print(f"  VEGAS CONTEST PICKS  —  Season {season}  Week {week}")
    print(f"  Strategy: top {top_n} highest-confidence ATS signals")
    print(f"{'='*70}\n")

    data = load_engine_data()
    schedules = pd.read_parquet(RAW / "schedules.parquet")

    week_games = schedules[
        (schedules["season"] == season) &
        (schedules["week"] == week) &
        (schedules["game_type"] == "REG") &
        (schedules["spread_line"].notna())
    ].copy()

    if week_games.empty:
        print(f"  No games with spread lines found for {season} Week {week}")
        return []

    print(f"  Analyzing {len(week_games)} games with Vegas spreads...\n")

    picks = []
    no_signal = []

    for _, game in week_games.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        try:
            pred = predict_game(home, away, season, week, data=data)
            result = score_game(pred, game, week)
            if result:
                picks.append(result)
            else:
                no_signal.append(f"{away}@{home} (model agrees with Vegas)")
        except Exception as e:
            no_signal.append(f"{away}@{home} (error: {e})")

    # Sort by confidence
    picks.sort(key=lambda x: x["confidence"], reverse=True)

    if not picks:
        print("  No ATS signals found this week — model agrees with Vegas on all games.")
        return []

    # Show top N picks
    print(f"  {'='*66}")
    print(f"  TOP {min(top_n, len(picks))} PICKS  (confidence ranked)")
    print(f"  {'='*66}")

    top_picks = picks[:top_n]
    for i, p in enumerate(top_picks, 1):
        result_str = ""
        if pd.notna(p.get("home_score_actual")):
            h_act = p["home_score_actual"]
            a_act = p["away_score_actual"]
            actual_margin = h_act - a_act
            # Did the pick cover? Home covers if home_margin > spread_line.
            if p["pick_side"] == "home":
                covered = actual_margin > p["vegas_spread"]
            else:
                covered = actual_margin < p["vegas_spread"]
            result_str = f"  → {'✓ COVERED' if covered else '✗ MISSED'} ({int(a_act)}-{int(h_act)})"

        flag_str = "  " + " ".join(f"[{f}]" for f in p["flags"]) if p["flags"] else ""

        # Betting lines: home favorite of spread_line shows as -spread_line; away as +spread_line
        home_line = -p['vegas_spread']
        away_line =  p['vegas_spread']
        pick_line = home_line if p['pick_side'] == 'home' else away_line

        print(f"\n  #{i}  PICK: {p['pick_team']} ATS  ({p['away_team']} @ {p['home_team']})")
        print(f"       Line: {p['home_team']} {home_line:+.1f} / {p['away_team']} {away_line:+.1f}  |  Take: {p['pick_team']} {pick_line:+.1f}")
        print(f"       Model:  {p['away_pred']}-{p['home_pred']}  |  Vegas gap: {p['gap']:.1f} pts  |  Win prob: {p['win_prob']:.1%}")
        print(f"       Confidence: {p['confidence']:.0f}/100  |  Est. accuracy: {p['est_accuracy']:.0%}{flag_str}")

        # Component breakdown
        c = p["components"]
        print(f"       Breakdown: gap={c['gap_score']:.0f} + conviction={c['wp_score']:.0f} + "
              f"clash={c['clash_score']:.0f} + week={c['week_score']:.0f} + "
              f"rest={c['rest_score']:.0f} + venue={c['char_score']:.0f}")
        if result_str:
            print(f"      {result_str}")

    # Show remaining signals if requested
    if show_all and len(picks) > top_n:
        print(f"\n  {'─'*66}")
        print(f"  ADDITIONAL SIGNALS (not in top {top_n})")
        print(f"  {'─'*66}")
        for p in picks[top_n:]:
            print(f"  {p['away_team']}@{p['home_team']}: {p['pick_team']} ATS  "
                  f"gap={p['gap']:.1f}  conf={p['confidence']:.0f}  {' '.join(p['flags'])}")

    # Weekly summary
    est_weekly_acc = np.mean([p["est_accuracy"] for p in top_picks])
    print(f"\n  {'─'*66}")
    print(f"  WEEK SUMMARY")
    print(f"  {'─'*66}")
    print(f"  Signals found:        {len(picks)} of {len(week_games)} games")
    print(f"  Picks submitted:      {min(top_n, len(picks))}")
    print(f"  Avg confidence:       {np.mean([p['confidence'] for p in top_picks]):.0f}/100")
    print(f"  Est. weekly accuracy: {est_weekly_acc:.0%}  ({est_weekly_acc*5:.1f}/5 expected correct)")
    print(f"  Season pace (if consistent): {est_weekly_acc*90:.0f}/90")
    print()

    # Save picks
    if top_picks:
        out = PROC / f"picks_{season}_wk{week:02d}.csv"
        pd.DataFrame(top_picks).to_csv(out, index=False)
        print(f"  Saved -> {out}")

    return top_picks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int, default=2026)
    parser.add_argument("--week",     type=int, default=1)
    parser.add_argument("--top",      type=int, default=5)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()
    run_week(args.season, args.week, args.top, args.show_all)
