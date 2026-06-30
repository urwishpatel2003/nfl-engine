"""
engine/ats_signals.py
---------------------
ATS-focused betting signal engine.

Layers signals on top of game predictions to identify
the highest-confidence betting opportunities.

Signals:
  1. Model-Vegas disagreement   (free, highest impact)
  2. Situational spot patterns  (free, from schedules)
  3. Home underdog filter       (free)
  4. Divisional game flag       (free)
  5. Week context               (free)

Output per game:
  - ats_signal:    "BET HOME" / "BET AWAY" / "PASS"
  - ats_confidence: 0-100 score
  - signal_reasons: list of contributing factors
  - recommended_units: 0 / 0.5 / 1.0 / 1.5 / 2.0
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"


# ── Confidence thresholds ──────────────────────────────────────
MIN_MODEL_VEGAS_DIFF  = 2.5   # pts -- minimum spread disagreement to consider betting
STRONG_DISAGREEMENT   = 4.5   # pts -- strong signal threshold
VERY_STRONG           = 7.0   # pts -- very strong signal

# ── Divisional matchups ────────────────────────────────────────
DIVISIONS = {
    "NFC East":  ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB",  "MIN"],
    "NFC South": ["ATL", "CAR", "NO",  "TB"],
    "NFC West":  ["ARI", "LAR", "SF",  "SEA"],
    "AFC East":  ["BUF", "MIA", "NE",  "NYJ"],
    "AFC North": ["BAL", "CLE", "CIN", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West":  ["DEN", "KC",  "LV",  "LAC"],
}

def get_division(team: str) -> str | None:
    for div, teams in DIVISIONS.items():
        if team in teams:
            return div
    return None

def is_divisional(home: str, away: str) -> bool:
    return get_division(home) == get_division(away) and get_division(home) is not None


def compute_ats_signals(pred: dict, schedules: pd.DataFrame = None) -> dict:
    """
    Given a game prediction dict, compute ATS betting signals.

    pred: output from predict_game()
    schedules: schedules.parquet DataFrame (for situational context)

    Returns augmented dict with ats_signal, ats_confidence, signal_reasons.
    """
    home      = pred["home_team"]
    away      = pred["away_team"]
    season    = pred["season"]
    week      = pred["week"]
    model_spread = pred["predicted_spread"]   # home perspective: negative = home favored

    vegas_spread = None
    if schedules is not None:
        g = schedules[
            (schedules["home_team"] == home) &
            (schedules["away_team"] == away) &
            (schedules["season"] == season) &
            (schedules["week"] == week)
        ]
        if not g.empty and "spread_line" in g.columns:
            vs = g.iloc[0]["spread_line"]
            if pd.notna(vs):
                vegas_spread = float(vs)

    # nflverse spread_line convention: HOME team perspective (verified empirically)
    #   Positive spread_line = home team favored (home expected to win by spread_line)
    #   Negative spread_line = away team favored (home is underdog)
    # Our predicted_spread: away_score - home_score
    #   Negative = home wins
    #
    # To compare apples-to-apples (both in home perspective):
    #   Vegas home margin = spread_line ; model home margin = -predicted_spread
    #   model_vegas_diff  = (-predicted_spread) - spread_line
    #   Positive = model thinks HOME is STRONGER than Vegas
    #   Negative = model thinks AWAY is STRONGER than Vegas

    signals   = []
    confidence = 50.0
    bet_side  = None

    # ── Signal 1: Winner direction disagreement ───────────────
    # Most meaningful signal: model picks different winner than Vegas implies.
    # Secondary: model thinks game is significantly closer than Vegas (dog covers).
    if vegas_spread is not None:
        vegas_implies_home  = vegas_spread > 0       # positive spread_line = home favored
        model_implies_home  = model_spread < 0       # predicted_spread<0 = home wins
        abs_vegas           = abs(vegas_spread)
        pred_margin_home    = -model_spread           # home perspective

        if vegas_implies_home != model_implies_home:
            # DIRECTION DISAGREEMENT — strongest signal
            if model_implies_home:
                bet_side = "HOME"
                signals.append(
                    f"Model picks {home} to win; Vegas has {away} favored "
                    f"(-{abs_vegas:.1f})"
                )
            else:
                bet_side = "AWAY"
                signals.append(
                    f"Model picks {away} to win; Vegas has {home} favored "
                    f"(-{abs_vegas:.1f})"
                )
            confidence += 20
            if abs_vegas <= 3:
                confidence += 10
                signals.append(f"Tight line ({abs_vegas:.1f}pts) — genuine toss-up")
            elif abs_vegas <= 6:
                confidence += 5

        else:
            # Same direction — only bet if model thinks game is much closer
            # (underdog covers) or model is more bullish on the fav
            margin_gap = abs_vegas - abs(pred_margin_home)

            if margin_gap >= 4.5 and abs_vegas >= 3.5:
                # Model much softer on favorite → underdog likely covers
                if vegas_implies_home:
                    bet_side = "AWAY"
                    signals.append(
                        f"Vegas: {home} -{abs_vegas:.1f} | "
                        f"Model: {home} by only {abs(pred_margin_home):.1f} — "
                        f"{away} likely covers"
                    )
                else:
                    bet_side = "HOME"
                    signals.append(
                        f"Vegas: {away} -{abs_vegas:.1f} | "
                        f"Model: {away} by only {abs(pred_margin_home):.1f} — "
                        f"{home} likely covers"
                    )
                confidence += 12 if margin_gap >= 7 else 7

            elif margin_gap < -3 and abs_vegas >= 3:
                # Model more bullish on favorite than Vegas
                if vegas_implies_home:
                    bet_side = "HOME"
                    signals.append(
                        f"Model {home} by {abs(pred_margin_home):.1f}; "
                        f"Vegas only -{abs_vegas:.1f} — model thinks {home} covers"
                    )
                else:
                    bet_side = "AWAY"
                    signals.append(
                        f"Model {away} by {abs(pred_margin_home):.1f}; "
                        f"Vegas only -{abs_vegas:.1f} — model thinks {away} covers"
                    )
                confidence += 8


    # ── Signal 2: Situational spots ───────────────────────────
    if schedules is not None and week > 1:
        # Get last week's results for both teams
        prev = schedules[
            (schedules["season"] == season) &
            (schedules["week"] == week - 1) &
            (schedules["home_score"].notna())
        ]

        home_prev = prev[
            (prev["home_team"] == home) | (prev["away_team"] == home)
        ]
        away_prev = prev[
            (prev["home_team"] == away) | (prev["away_team"] == away)
        ]

        # Letdown spot: big win last week, lesser opponent this week
        if not home_prev.empty:
            hp = home_prev.iloc[0]
            if hp["home_team"] == home:
                home_margin_last = float(hp["home_score"] or 0) - float(hp["away_score"] or 0)
            else:
                home_margin_last = float(hp["away_score"] or 0) - float(hp["home_score"] or 0)

            if home_margin_last >= 17 and vegas_spread is not None and vegas_spread > 7:
                # Home team won big last week and is a big favorite = letdown risk
                if bet_side != "HOME":
                    signals.append(f"{home} letdown spot: won by {home_margin_last:.0f} last wk, big fav this wk")
                    confidence -= 8
                    if bet_side is None:
                        bet_side = "AWAY"

        # Revenge spot: away team lost to home team by 14+ last meeting
        last_meeting = schedules[
            (schedules["season"] == season - 1) &
            (
                ((schedules["home_team"] == home) & (schedules["away_team"] == away)) |
                ((schedules["home_team"] == away) & (schedules["away_team"] == home))
            ) &
            (schedules["home_score"].notna())
        ]
        if not last_meeting.empty:
            lm = last_meeting.iloc[-1]
            if lm["home_team"] == away:
                # Away team was home last year
                away_margin = float(lm["home_score"] or 0) - float(lm["away_score"] or 0)
            else:
                away_margin = float(lm["away_score"] or 0) - float(lm["home_score"] or 0)

            if away_margin <= -14:
                signals.append(f"{away} revenge spot: lost by {abs(away_margin):.0f} in last meeting")
                confidence += 6
                if bet_side == "AWAY" or bet_side is None:
                    bet_side = "AWAY"

        # Short week: away team on short rest (already in conditions but flag it explicitly)
        away_game_prev = schedules[
            (schedules["season"] == season) &
            (schedules["week"] == week - 1) &
            ((schedules["away_team"] == away) | (schedules["home_team"] == away))
        ]
        if not away_game_prev.empty:
            apg = away_game_prev.iloc[0]
            away_rest = apg.get("away_rest", 7) if apg["away_team"] == away else apg.get("home_rest", 7)
            if pd.notna(away_rest) and float(away_rest) <= 5:
                signals.append(f"{away} on short week ({away_rest:.0f} days rest)")
                confidence += 7
                if bet_side == "HOME" or bet_side is None:
                    bet_side = "HOME"

    # ── Signal 3: Home underdog ───────────────────────────────
    if vegas_spread is not None and vegas_spread < -3:
        # spread_line < -3 means away team favored by 3+ = home is underdog
        home_dog_pts = abs(vegas_spread)
        signals.append(f"{home} is a home underdog (+{home_dog_pts:.1f}) -- historical 53-55% ATS")
        confidence += 5
        if bet_side is None:
            bet_side = "HOME"

    # ── Signal 4: Divisional game ─────────────────────────────
    if is_divisional(home, away):
        signals.append(f"Divisional game ({get_division(home)}) -- underdogs cover at 54%")
        confidence += 3
        if vegas_spread is not None:
            if vegas_spread < -3 and bet_side is None:
                bet_side = "HOME"   # home dog in divisional
            elif vegas_spread > 6 and bet_side is None:
                bet_side = "AWAY"   # big away dog in divisional

    # ── Signal 5: Week context ────────────────────────────────
    if week in [17, 18]:
        signals.append(f"Late season (Wk{week}): model historically 75-66% accurate")
        confidence += 5
    elif week in [1, 2]:
        signals.append(f"Early season (Wk{week}): higher variance, reduce unit size")
        confidence -= 5

    # ── Quality gate ───────────────────────────────────────────
    # Only fire direction disagreement when there's genuine model conviction.
    # Require: win probability of model's pick > 58% (not a coin flip)
    # This filters out games where model barely disagrees with Vegas.
    home_win_prob = pred.get("home_win_probability", 0.5)
    if bet_side == "HOME" and home_win_prob < 0.55:
        bet_side = None   # model not confident enough
        confidence = max(confidence - 15, 40)
    elif bet_side == "AWAY" and home_win_prob > 0.45:
        bet_side = None
        confidence = max(confidence - 15, 40)

    # ── Final decision ─────────────────────────────────────────
    confidence = float(np.clip(confidence, 0, 100))

    # Only bet when we have a clear side AND meaningful confidence
    if bet_side is None or len(signals) == 0:
        ats_signal = "PASS"
        units = 0.0
    elif confidence >= 72:
        ats_signal = f"BET {bet_side} ★★"
        units = 2.0
    elif confidence >= 63:
        ats_signal = f"BET {bet_side} ★"
        units = 1.5
    elif confidence >= 56:
        ats_signal = f"BET {bet_side}"
        units = 1.0
    else:
        ats_signal = "PASS"
        units = 0.0

    return {
        **pred,
        "vegas_spread":      vegas_spread,
        "model_vegas_diff":  round(abs(vegas_spread or 0) - abs(model_spread), 1),
        "ats_signal":        ats_signal,
        "ats_confidence":    round(confidence, 1),
        "ats_bet_side":      bet_side,
        "recommended_units": units,
        "signal_reasons":    signals,
        "is_divisional":     is_divisional(home, away),
    }


def analyze_week(season: int, week: int, data: dict = None) -> pd.DataFrame:
    """
    Run ATS signal analysis for all games in a week.
    Returns DataFrame with signals sorted by confidence.
    """
    from engine.predict import load_engine_data, predict_week

    if data is None:
        data = load_engine_data()

    schedules = pd.read_parquet(RAW / "schedules.parquet")

    # Get predictions
    preds = predict_week(season, week)
    if preds.empty:
        print(f"No predictions for {season} week {week}")
        return pd.DataFrame()

    results = []
    for _, row in preds.iterrows():
        pred_dict = row.to_dict()
        # key_matchups is a list -- safe for to_dict but need to handle
        pred_dict["key_matchups"] = []
        signals = compute_ats_signals(pred_dict, schedules)
        results.append(signals)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Sort: bets first by confidence, then leans, then passes
    priority = {"BET HOME ★★": 0, "BET AWAY ★★": 0,
                "BET HOME ★": 1,  "BET AWAY ★": 1,
                "BET HOME": 2,    "BET AWAY": 2,
                "LEAN HOME": 3,   "LEAN AWAY": 3,
                "PASS": 4}
    df["_sort"] = df["ats_signal"].map(lambda x: priority.get(x, 5))
    df = df.sort_values(["_sort", "ats_confidence"], ascending=[True, False]).drop("_sort", axis=1)

    # Volume cap: max 5 bets per week — only the highest confidence ones
    # Games beyond the top 5 are downgraded to PASS
    bet_mask = df["ats_signal"].str.startswith("BET", na=False)
    if bet_mask.sum() > 5:
        top5_idx = df[bet_mask].head(5).index
        df.loc[bet_mask & ~df.index.isin(top5_idx), "ats_signal"] = "PASS"
        df.loc[bet_mask & ~df.index.isin(top5_idx), "recommended_units"] = 0.0

    return df


def print_ats_report(df: pd.DataFrame):
    """Pretty print the ATS signal report for a week."""
    if df.empty:
        print("No games to report.")
        return

    bets  = df[df["ats_signal"].str.startswith("BET")]
    leans = df[df["ats_signal"].str.startswith("LEAN")]
    passes= df[df["ats_signal"] == "PASS"]

    print(f"\n{'='*65}")
    print(f"  ATS SIGNAL REPORT  |  Season {df.iloc[0]['season']} Week {df.iloc[0]['week']}")
    print(f"{'='*65}")

    if not bets.empty:
        print(f"\n  BETS ({len(bets)} games)")
        print(f"  {'Game':<18} {'Signal':<18} {'Conf':>6} {'M-V Diff':>9} {'Units':>6}")
        print(f"  {'-'*60}")
        for _, r in bets.iterrows():
            game = f"{r['away_team']}@{r['home_team']}"
            diff = r.get('model_vegas_diff', 0)
            print(f"  {game:<18} {r['ats_signal']:<18} {r['ats_confidence']:>5.0f}%"
                  f"  {diff:>+7.1f}pt  {r['recommended_units']:>4.1f}u")
            for reason in r.get("signal_reasons", [])[:2]:
                print(f"    → {reason}")

    if not leans.empty:
        print(f"\n  LEANS ({len(leans)} games)")
        for _, r in leans.iterrows():
            game = f"{r['away_team']}@{r['home_team']}"
            print(f"  {game:<18} {r['ats_signal']:<18} {r['ats_confidence']:>5.0f}%")

    print(f"\n  PASS ({len(passes)} games) — no edge identified")
    print(f"\n  Total recommended units this week: {bets['recommended_units'].sum():.1f}u")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week",   type=int, default=15)
    args = parser.parse_args()

    df = analyze_week(args.season, args.week)
    print_ats_report(df)
