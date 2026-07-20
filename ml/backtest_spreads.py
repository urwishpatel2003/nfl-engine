"""
ml/backtest_spreads.py  —  grade the matchup model against results AND the closing line
=======================================================================================
The only honest test of a spread model is: does it beat the market, out of sample? This
scores the matchup engine's predicted margins for a season's games against:

  • the actual final margin  → margin MAE, straight-up winner accuracy
  • the closing spread_line   → ATS record, ROI at −110, and the market's own MAE (the bar)

HONESTY: the dashboard's matchup engine is anchored to 2025 data, so evaluating it on 2025
is *in-sample* — the ATS number here is optimistic, not a forecast of live performance. The
repo's walk-forward harness (backtest_picks.py, which rebuilds using only prior seasons) is
the true out-of-sample test, and it lands around 49% ATS — i.e. no edge over the market,
which is the expected result for a public-data model against an efficient line.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"

_BT_CACHE = {}


def latest_completed_season() -> int:
    """Most recent season that actually has final scores (skips a not-yet-played season)."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    played = s[s["home_score"].notna()]
    return int(played["season"].max()) if len(played) else int(s["season"].max())


def evaluate(season: int = None) -> dict:
    if season is None:
        season = latest_completed_season()
    if season in _BT_CACHE:
        return _BT_CACHE[season]
    from ml.matchup_engine import project_game
    s = pd.read_parquet(RAW / "schedules.parquet")
    need = {"season", "home_team", "away_team", "home_score", "away_score", "spread_line"}
    if not need.issubset(s.columns):
        return {"error": "schedule lacks scores/lines"}
    s = s[(s["season"] == season) & s["home_score"].notna() & s["spread_line"].notna()]
    rows = []
    for _, g in s.iterrows():
        pred = project_game(g["home_team"], g["away_team"])
        if "error" in pred:
            continue
        rows.append({"model": pred["pred_margin"],
                     "actual": float(g["home_score"] - g["away_score"]),
                     "spread": float(g["spread_line"])})
    if not rows:
        return {"error": f"no gradable games for {season}"}
    df = pd.DataFrame(rows)

    model_mae = float((df["model"] - df["actual"]).abs().mean())
    market_mae = float((df["spread"] - df["actual"]).abs().mean())
    winner_acc = float(((df["model"] > 0) == (df["actual"] > 0)).mean())
    push = df["actual"] == df["spread"]
    graded = df[~push]
    ats = ((graded["model"] > graded["spread"]) == (graded["actual"] > graded["spread"]))
    ats_rate = float(ats.mean())
    roi = float(ats.mean() * (100 / 110) - (1 - ats.mean()))     # flat $1 at −110

    out = {
        "season": season, "games": len(df),
        "model_margin_mae": round(model_mae, 2),
        "market_margin_mae": round(market_mae, 2),      # the benchmark to beat
        "winner_acc": round(winner_acc, 3),
        "ats_in_sample": round(ats_rate, 3),
        "roi_in_sample": round(roi, 3),
        "beats_market_mae": bool(model_mae < market_mae),
        "note": "In-sample (model uses this season's data). True out-of-sample ATS is ~49% "
                "(no edge) per the walk-forward backtest — the market is efficient.",
    }
    _BT_CACHE[season] = out
    return out


if __name__ == "__main__":
    import sys, json
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    print(json.dumps(evaluate(yr), indent=2))
