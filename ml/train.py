"""
ml/train.py  —  walk-forward margin & edge models
===================================================
Trains on the leak-free feature matrix (ml/features.py) with STRICT walk-forward
validation: to predict season N, train only on seasons < N. Aggregated OOS
predictions are scored against the market.

Two models:
  MARGIN : predict home_margin from ALL features (incl. the market line).
           Goal: OOS correlation above the market's own ~0.456.
  EDGE   : predict the residual (home_margin - spread_line) from NON-market
           features only. If this has OOS skill, the fundamentals know something
           the market doesn't — the only honest source of ATS edge.

Models: XGBoost (handles NaN) + a Ridge linear baseline.

Usage:
  python -m ml.train
  python -m ml.train --save     # also write OOS margin preds for the benchmark tools
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

PROC = Path(__file__).parent.parent / "data" / "processed"
BREAKEVEN = 0.5238

META = ["game_id", "season", "week", "home_team", "away_team", "home_margin", "total"]
MARKET = ["mkt_spread", "mkt_total", "mkt_home_impl"]


def xgb_model():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, reg_alpha=0.0, n_jobs=4, random_state=0)


def ridge_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=10.0))


def walk_forward(df, feat_cols, target, test_seasons, model_fn):
    """Train on seasons < test season, predict it. Returns df with 'pred' for test rows."""
    out = []
    for ts in test_seasons:
        tr = df[df["season"] < ts]
        te = df[df["season"] == ts]
        if len(tr) < 100 or te.empty:
            continue
        m = model_fn()
        m.fit(tr[feat_cols], tr[target])
        p = te.copy()
        p["pred"] = m.predict(te[feat_cols])
        out.append(p)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def ats_edge(pred_margin, market, actual_margin):
    pick_home = pred_margin > market
    home_cov  = actual_margin > market
    disagree  = pred_margin != market
    edge = disagree & (pick_home != (market > 0))
    return (float((pick_home == home_cov)[edge].mean()) if edge.any() else float("nan"),
            int(edge.sum()))


def score(name, pred, actual, market):
    mae = float(np.abs(pred - actual).mean())
    corr = float(np.corrcoef(pred, actual)[0, 1])
    ats, n = ats_edge(pred, market, actual)
    ats_s = f"{ats:.1%}(n={n})" if not np.isnan(ats) else "n/a"
    print(f"  {name:22} MAE={mae:5.2f}  corr={corr:.3f}  ATS_edge={ats_s}")
    return corr, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(PROC / "game_features.parquet").sort_values(["season", "week"])
    feat_all = [c for c in df.columns if c not in META]
    feat_nonmkt = [c for c in feat_all if c not in MARKET]
    test_seasons = [s for s in sorted(df["season"].unique()) if s >= df["season"].min() + 2]

    print(f"\n  Walk-forward ML  (train on seasons < test; test = {test_seasons})")
    print(f"  {len(df)} games | {len(feat_all)} features | market baseline corr ~0.456")
    print(f"  {'='*68}")

    # ── MARGIN model: predict home_margin (all features) ──────────────
    print("\n  [MARGIN model — target home_margin, all features]")
    res = {}
    for label, fn in [("XGB", xgb_model), ("Ridge", ridge_model)]:
        wf = walk_forward(df, feat_all, "home_margin", test_seasons, fn)
        res[label] = wf
        score(f"{label} (all feat)", wf["pred"].values, wf["home_margin"].values, wf["mkt_spread"].values)
    # market-only reference on the same test rows
    ref = res["XGB"]
    score("market (spread) ref", ref["mkt_spread"].values, ref["home_margin"].values, ref["mkt_spread"].values)

    # ── EDGE model: predict residual vs spread (non-market feats) ──────
    print("\n  [EDGE model — target (home_margin - spread), non-market features only]")
    df2 = df.copy()
    df2["resid"] = df2["home_margin"] - df2["mkt_spread"]
    for label, fn in [("XGB", xgb_model), ("Ridge", ridge_model)]:
        wf = walk_forward(df2, feat_nonmkt, "resid", test_seasons, fn)
        # predicted margin = market + predicted residual
        pm = wf["mkt_spread"].values + wf["pred"].values
        rc = float(np.corrcoef(wf["pred"].values, wf["resid"].values)[0, 1])
        ats, n = ats_edge(pm, wf["mkt_spread"].values, wf["home_margin"].values)
        ats_s = f"{ats:.1%}(n={n})" if not np.isnan(ats) else "n/a"
        verd = "EDGE!" if (not np.isnan(ats) and ats > BREAKEVEN and n >= 30) else "no edge"
        print(f"  {label:22} resid_corr={rc:+.3f}  ATS_edge={ats_s}  -> {verd}")

    print(f"\n  Read: MARGIN corr above ~0.456 means the model beats the market on accuracy.")
    print(f"  EDGE resid_corr > 0 with ATS above {BREAKEVEN:.1%} on a healthy n is real betting edge.\n")

    if args.save:
        wf = res["XGB"][["game_id", "season", "week", "home_team", "away_team",
                         "home_margin", "mkt_spread", "mkt_total", "pred"]].copy()
        wf = wf.rename(columns={"home_margin": "actual_margin", "pred": "pred_margin",
                                "mkt_spread": "vegas_spread", "mkt_total": "vegas_total"})
        wf["actual_total"] = np.nan
        wf["home_win_probability"] = 1 / (1 + np.exp(-wf["pred_margin"] / 13.5 * np.pi / np.sqrt(3)))
        out = PROC / "ml_oos_predictions.parquet"
        wf.to_parquet(out, index=False)
        print(f"  Saved OOS margin preds -> {out.name}\n")


if __name__ == "__main__":
    main()
