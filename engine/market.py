"""
engine/market.py
----------------
Market-regression layer — blends the raw model prediction toward the betting
market (the nfelo idea). The market (Vegas closing line) is a very strong prior;
our backtests show the raw model trails it and its disagreements are mostly noise.
So instead of trusting the raw model outright, we treat the market as the anchor
and let the model nudge it only in proportion to a tunable weight.

Conventions (see CLAUDE.md):
  * nflverse spread_line: POSITIVE = home favored; market home margin == spread_line
  * model home margin = home_score - away_score
  * we keep the *total* and the *margin* as separate knobs, then re-split into scores

Key knob:
  margin_weight (alpha): weight on the MODEL's margin.
      alpha = 0.0  -> pure market (you become Vegas; can't beat the vig, but well calibrated)
      alpha = 1.0  -> pure raw model (what we have today; trails the market)
      0 < alpha < 1 -> shrink the model's deviation from the market toward 0
  deadband: zero out model deviations smaller than this many points (kills noise;
            only act when the model disagrees with the market by a meaningful amount)

Find the best alpha empirically with optimize_market_weight.py — do NOT hand-pick it.
"""

import numpy as np

WIN_PROB_SIGMA = 13.5   # must match engine/predict.py


def win_prob_from_margin(margin: float, sigma: float = WIN_PROB_SIGMA) -> float:
    """Home win probability from a predicted home margin (logistic approx of normal)."""
    return float(1.0 / (1.0 + np.exp(-margin / sigma * np.pi / np.sqrt(3))))


def regress_margin_to_market(model_margin: float, market_margin: float,
                             margin_weight: float = 0.35,
                             deadband: float = 0.0) -> float:
    """
    Blend a model home-margin toward the market home-margin.

    market_margin is the Vegas spread_line (positive = home favored). Returns the
    regressed home margin. With deadband>0, any residual deviation from the market
    smaller than `deadband` points is snapped back to the market (treated as noise).
    """
    if market_margin is None or (isinstance(market_margin, float) and np.isnan(market_margin)):
        return model_margin  # no market anchor available — fall back to raw model

    blended = margin_weight * model_margin + (1.0 - margin_weight) * market_margin
    deviation = blended - market_margin
    if abs(deviation) < deadband:
        return float(market_margin)
    return float(blended)


def regress_to_market(home_score: float, away_score: float,
                      spread_line: float = None, total_line: float = None,
                      margin_weight: float = 0.35, total_weight: float = 0.5,
                      deadband: float = 0.0) -> dict:
    """
    Apply market regression to a raw model score prediction.

    Splits the prediction into margin + total, regresses each toward the market
    (spread_line / total_line) by the given weights, then recombines into scores.
    Returns a dict with the regressed scores, margin, total, and home win prob.
    """
    model_margin = home_score - away_score
    model_total  = home_score + away_score

    reg_margin = regress_margin_to_market(model_margin, spread_line,
                                          margin_weight, deadband)

    if total_line is not None and not (isinstance(total_line, float) and np.isnan(total_line)):
        reg_total = total_weight * model_total + (1.0 - total_weight) * float(total_line)
    else:
        reg_total = model_total

    reg_home = (reg_total + reg_margin) / 2.0
    reg_away = (reg_total - reg_margin) / 2.0

    return {
        "home_score":  round(float(reg_home), 1),
        "away_score":  round(float(reg_away), 1),
        "margin":      round(float(reg_margin), 1),
        "total":       round(float(reg_total), 1),
        "home_win_probability": round(win_prob_from_margin(reg_margin), 3),
        "raw_margin":  round(float(model_margin), 1),
        "market_margin": (round(float(spread_line), 1) if spread_line is not None
                          and not (isinstance(spread_line, float) and np.isnan(spread_line))
                          else None),
    }
