"""
ml/spreads.py  —  key-number-aware cover / total probabilities
==============================================================
A single predicted margin isn't enough to price a spread: NFL margins cluster hard on
key numbers (3, 7, 10, 6, 14), so the value of a half-point depends entirely on *which*
number it crosses. This turns the model's point estimate into probabilities using the
empirical distribution of real NFL margins (and totals):

    actual_margin ≈ pred_margin + ε,   ε ~ (historical margins − their mean)

We recenter the empirical margin distribution on the model's predicted margin and read off
P(cover). Because the empirical distribution carries the key-number spikes, the cover
probability naturally plateaus around 3 and 7 — exactly how a bookmaker prices it. Same
method for totals (over/under).

All history comes from schedules.parquet (every completed game, all seasons).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"

_MARGINS = None
_TOTALS = None


def _hist():
    """Arrays of historical final margins (home − away) and totals (home + away)."""
    global _MARGINS, _TOTALS
    if _MARGINS is None:
        s = pd.read_parquet(RAW / "schedules.parquet")
        s = s[s["home_score"].notna() & s["away_score"].notna()]
        _MARGINS = (s["home_score"] - s["away_score"]).round().to_numpy(dtype=float)
        _TOTALS = (s["home_score"] + s["away_score"]).round().to_numpy(dtype=float)
    return _MARGINS, _TOTALS


def cover_prob(pred_margin: float, spread_line: float) -> dict:
    """P(home covers), P(away covers), P(push) for the model's margin vs the line.
    spread_line is home-perspective (positive = home favored), matching nflverse."""
    m, _ = _hist()
    sim = m + (pred_margin - float(m.mean()))          # empirical outcomes recentered on the model
    home = float(np.mean(sim > spread_line))
    push = float(np.mean(np.abs(sim - spread_line) < 0.5)) if abs(spread_line - round(spread_line)) < 1e-6 else 0.0
    home = max(0.0, home - push / 2)
    away = max(0.0, 1.0 - home - push)
    return {"home_cover": round(home, 3), "away_cover": round(away, 3), "push": round(push, 3)}


def total_prob(pred_total: float, total_line: float) -> dict:
    """P(over), P(under), P(push) for the model's total vs the line."""
    _, t = _hist()
    sim = t + (pred_total - float(t.mean()))
    over = float(np.mean(sim > total_line))
    push = float(np.mean(np.abs(sim - total_line) < 0.5)) if abs(total_line - round(total_line)) < 1e-6 else 0.0
    over = max(0.0, over - push / 2)
    under = max(0.0, 1.0 - over - push)
    return {"over": round(over, 3), "under": round(under, 3), "push": round(push, 3)}


def ats_pick(pred_margin: float, spread_line: float) -> dict:
    """The model's against-the-spread pick + its cover probability and edge."""
    c = cover_prob(pred_margin, spread_line)
    home_side = c["home_cover"] >= c["away_cover"]
    return {
        "side": "home" if home_side else "away",
        "cover_prob": c["home_cover"] if home_side else c["away_cover"],
        "push": c["push"],
        "edge": round(pred_margin - spread_line, 1),
    }
