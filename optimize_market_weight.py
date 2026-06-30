"""
optimize_market_weight.py  —  Find the best model/market blend
===============================================================
Sweeps the market-regression weight alpha (engine/market.py) over a backtest's
predictions and reports, for each alpha:
  * margin MAE / correlation        (accuracy of the blended margin)
  * ATS cover rate on "edge" games  (where the blend disagrees with the market favorite)
  * win-probability Brier           (calibration)

This answers the central question honestly: is there ANY alpha in (0,1] that beats
the pure-market anchor (alpha=0) on ATS? If the best edge cover rate stays below the
52.4% breakeven, the model adds accuracy/calibration but NO profitable betting edge.

Usage:
  python optimize_market_weight.py
  python optimize_market_weight.py --preds data/processed/backtest_2024_2025.parquet
  python optimize_market_weight.py --deadband 3.0     # only bet on >=3pt disagreements
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from engine.market import regress_margin_to_market, win_prob_from_margin

PROC = Path(__file__).parent / "data" / "processed"
BREAKEVEN = 0.5238


def evaluate(df: pd.DataFrame, alpha: float, deadband: float) -> dict:
    """Apply regression at this alpha and score margin accuracy + ATS + calibration."""
    market = df["vegas_spread"].to_numpy(dtype=float)        # home margin (positive=home fav)
    raw    = df["pred_margin"].to_numpy(dtype=float)         # model home margin
    actual = df["actual_margin"].to_numpy(dtype=float)
    home_win = (actual > 0).astype(int)

    reg = np.array([regress_margin_to_market(r, m, alpha, deadband)
                    for r, m in zip(raw, market)])

    err = reg - actual
    mae = float(np.abs(err).mean())
    corr = float(np.corrcoef(reg, actual)[0, 1]) if np.std(reg) > 1e-9 else float("nan")

    # ATS: model takes the side it likes vs the line; "edge" = disagrees with market favorite
    pick_home   = reg > market
    home_covers = actual > market
    disagree    = reg != market
    edge        = disagree & (pick_home != (market > 0))
    ats_edge    = float((pick_home == home_covers)[edge].mean()) if edge.any() else float("nan")

    # win-prob calibration from regressed margin
    wp = np.array([win_prob_from_margin(m) for m in reg]).clip(1e-6, 1 - 1e-6)
    brier = float(((wp - home_win) ** 2).mean())

    return {"alpha": alpha, "MAE": mae, "corr": corr,
            "n_edge": int(edge.sum()), "ATS_edge": ats_edge, "Brier": brier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(PROC / "backtest_2021_2022_2023_2024_2025.parquet"))
    ap.add_argument("--deadband", type=float, default=0.0,
                    help="snap model deviations smaller than this (pts) back to the market")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    need = {"vegas_spread", "pred_margin", "actual_margin"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{args.preds}: needs columns {need}")
    df = df.dropna(subset=list(need)).copy()

    print(f"\n  Market-weight sweep on {Path(args.preds).name}  "
          f"(n={len(df)}, deadband={args.deadband})")
    is_insample = "backtest_2021" in args.preds or "backtest_2024" in args.preds
    if is_insample:
        print(f"  NOTE: in-sample styles — optimistic; rerun on OOS preds before trusting ATS.")
    print(f"  alpha=0 is pure market; alpha=1 is the raw model. Breakeven ATS = {BREAKEVEN:.1%}")
    print(f"  {'-'*72}")
    print(f"  {'alpha':>6} {'MAE':>7} {'corr':>7} {'ATS_edge':>9} {'n_edge':>7} {'Brier':>7}")

    rows = [evaluate(df, a, args.deadband) for a in np.round(np.arange(0.0, 1.01, 0.1), 2)]
    for r in rows:
        ats = f"{r['ATS_edge']:.1%}" if not np.isnan(r["ATS_edge"]) else "   n/a"
        print(f"  {r['alpha']:>6.1f} {r['MAE']:>7.2f} {r['corr']:>7.3f} "
              f"{ats:>9} {r['n_edge']:>7} {r['Brier']:>7.3f}")

    best_mae = min(rows, key=lambda r: r["MAE"])
    valid_ats = [r for r in rows if not np.isnan(r["ATS_edge"]) and r["n_edge"] >= 20]
    best_ats = max(valid_ats, key=lambda r: r["ATS_edge"]) if valid_ats else None

    print(f"\n  Best MAE : alpha={best_mae['alpha']:.1f}  ({best_mae['MAE']:.2f})")
    if best_ats:
        verdict = "beats breakeven" if best_ats["ATS_edge"] > BREAKEVEN else "still below breakeven"
        print(f"  Best ATS : alpha={best_ats['alpha']:.1f}  ({best_ats['ATS_edge']:.1%}, "
              f"n={best_ats['n_edge']})  -> {verdict}")
    print(f"\n  Read: if MAE keeps dropping as alpha->0, the market is simply better and the")
    print(f"  model's job is calibration, not edge. A best-ATS alpha that clears {BREAKEVEN:.1%}")
    print(f"  on a healthy sample is the only thing that justifies actually betting.\n")


if __name__ == "__main__":
    main()
