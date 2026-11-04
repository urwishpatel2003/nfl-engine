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


_RESID_M = None
_RESID_T = None
RESID_FROM = 2015    # closing lines are populated and stable from here


def _hist():
    """Arrays of historical final margins (home − away) and totals (home + away)."""
    global _MARGINS, _TOTALS
    if _MARGINS is None:
        s = pd.read_parquet(RAW / "schedules.parquet")
        s = s[s["home_score"].notna() & s["away_score"].notna()]
        _MARGINS = (s["home_score"] - s["away_score"]).round().to_numpy(dtype=float)
        _TOTALS = (s["home_score"] + s["away_score"]).round().to_numpy(dtype=float)
    return _MARGINS, _TOTALS


def _resid():
    """Noise around a point prediction: actual minus the CLOSING LINE (margin and total),
    regular season 2015+, each shifted so its median is zero.

    Why residuals and not raw margins: the raw home-margin history is skewed (mean +2.3,
    median +3.0 — the unconditional mix of favourites and dogs), so recentring it on a
    prediction by its mean put the median ~1 pt above the prediction and a line sitting
    exactly ON the prediction read as a 52% home cover. Through 2026 week 8 that picked the
    HOME side of every game the model leaned away by up to 0.6 pts (7 of 91 ledger picks
    went against the model's own edge, all 7 home) and priced the same edge ~3-5 pts of
    cover probability higher for a home pick than an away pick, which the top-5 ranks on.
    Actual-minus-line has no such skew (mean −0.1, median −0.5, home cover 47.3%); shifting
    it to a zero median makes P(cover) exactly 50% at zero edge, so the pick follows the
    edge sign, and it keeps the key-number clustering a line crosses. The slight home
    under-cover of 2015-25 is deliberately NOT carried (it is a past market mispricing, not
    a property of our prediction)."""
    global _RESID_M, _RESID_T
    if _RESID_M is None:
        s = pd.read_parquet(RAW / "schedules.parquet")
        s = s[s["home_score"].notna() & s["away_score"].notna() & (s["season"] >= RESID_FROM) & (s["week"] <= 18)]
        m = s[s["spread_line"].notna()]
        rm = (m["home_score"] - m["away_score"] - m["spread_line"]).to_numpy(dtype=float)
        t = s[s["total_line"].notna()]
        rt = (t["home_score"] + t["away_score"] - t["total_line"]).to_numpy(dtype=float)
        # Mirrored (r and −r): the noise around OUR prediction is assumed symmetric, which
        # makes P(cover) exactly 50% at zero edge for either side, with no median shift and
        # no half-point artefacts (an integer-only pool put its mass at 0 entirely on one
        # side of a .5 line). Key numbers are symmetric anyway (±3, ±7).
        _RESID_M = np.concatenate([rm, -rm])
        _RESID_T = np.concatenate([rt, -rt])
    return _RESID_M, _RESID_T


def _side_probs(diff: np.ndarray, whole_line: bool) -> tuple:
    """(P(over/home), P(under/away), P(push)) from prediction+noise minus the line. An exact
    tie is a push on a whole-number line and a coin flip on a half-point line."""
    tie = np.abs(diff) < 0.25
    hi = float(np.mean(diff > 0.25))
    if whole_line:
        push = float(np.mean(tie))
        return hi, max(0.0, 1.0 - hi - push), push
    hi += 0.5 * float(np.mean(tie))
    return hi, max(0.0, 1.0 - hi), 0.0


def cover_prob(pred_margin: float, spread_line: float, center: str = "median") -> dict:
    """P(home covers), P(away covers), P(push) for the model's margin vs the line.
    spread_line is home-perspective (positive = home favored), matching nflverse.
    actual ≈ pred_margin + ε, ε = the line residual history (see _resid; zero median, so a
    line at the prediction is a 50/50 and the pick follows the edge sign). `center` is kept
    for callers (ml/kalshi passed "median"); both modes now use the same residual noise."""
    r, _ = _resid()
    home, away, push = _side_probs(pred_margin + r - spread_line, abs(spread_line - round(spread_line)) < 1e-6)
    return {"home_cover": round(home, 3), "away_cover": round(away, 3), "push": round(push, 3)}


def total_prob(pred_total: float, total_line: float, center: str = "median") -> dict:
    """P(over), P(under), P(push) for the model's total vs the line; same residual noise
    (actual total minus the closing total line, mirrored) as cover_prob."""
    _, r = _resid()
    over, under, push = _side_probs(pred_total + r - total_line, abs(total_line - round(total_line)) < 1e-6)
    return {"over": round(over, 3), "under": round(under, 3), "push": round(push, 3)}


def simulate(pred_margin: float, pred_total: float) -> dict:
    """Empirical game simulation: the model's prediction plus the mirrored line-residual noise
    (same pool as cover_prob, so the margin distribution on the Matchup page and the cover
    probability on the slate are the same distribution) — key-number spikes built in."""
    rm, rt = _resid()
    ms = pred_margin + rm
    ts = pred_total + rt
    rk = np.abs(np.round(ms))
    key_margins = {int(k): round(float(np.mean(rk == k)), 3) for k in (1, 2, 3, 4, 6, 7, 10, 13, 14)}
    return {
        "home_win_prob": round(float(np.mean(ms > 0)), 3),
        "median_margin": round(float(np.median(ms)), 1),
        "margin_p10": round(float(np.percentile(ms, 10)), 1),
        "margin_p90": round(float(np.percentile(ms, 90)), 1),
        "median_total": round(float(np.median(ts)), 1),
        "total_p10": round(float(np.percentile(ts, 10)), 1),
        "total_p90": round(float(np.percentile(ts, 90)), 1),
        "key_margins": key_margins,     # P(|final margin| == k) — the numbers that decide spreads
    }


def ats_pick(pred_margin: float, spread_line: float) -> dict:
    """The model's against-the-spread pick + its cover probability and edge."""
    c = cover_prob(pred_margin, spread_line)
    # the side follows the edge; probabilities only break a tie the edge can't (edge 0 → home)
    if abs(c["home_cover"] - c["away_cover"]) < 1e-9:
        home_side = pred_margin >= spread_line
    else:
        home_side = c["home_cover"] > c["away_cover"]
    return {
        "side": "home" if home_side else "away",
        "cover_prob": c["home_cover"] if home_side else c["away_cover"],
        "push": c["push"],
        "edge": round(pred_margin - spread_line, 1),
    }
