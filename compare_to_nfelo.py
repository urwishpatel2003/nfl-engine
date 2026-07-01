"""
compare_to_nfelo.py  —  Benchmark our model against nfelo and Vegas
====================================================================
nfelo (https://www.nfeloapp.com, by @greerreNFL) is a strong public Elo-based
NFL model. It ships historical projected spreads for 2021-2025, which gives us a
fixed, independent yardstick on the *same games* our engine predicts.

This script scores three predictors of the home-team margin on a common set of
completed games and reports margin accuracy (MAE / RMSE / correlation), ATS
accuracy vs the closing line, and win-probability calibration (Brier / log-loss):

  1. Vegas   — the nflverse closing spread_line (the market; very hard to beat)
  2. nfelo   — home_closing_line_rounded_nfelo + home_probability_nfelo
  3. Ours    — any predictions file we pass in

Conventions (see CLAUDE.md):
  * nflverse spread_line: POSITIVE = home favored; Vegas home margin == spread_line
  * betting lines (nfelo home_line): NEGATIVE = home favored; home margin == -line
  * our predicted home margin = predicted_home_score - predicted_away_score

Usage:
  # Default: benchmark the in-sample backtest parquet (WARNS: in-sample = optimistic)
  python compare_to_nfelo.py

  # Benchmark a specific predictions file (e.g. an OOS run or a week's predictions)
  python compare_to_nfelo.py --our-preds data/processed/predictions_2024_wk18.parquet
  python compare_to_nfelo.py --our-preds data/processed/backtest_2021_2022_2023_2024_2025.parquet
  python compare_to_nfelo.py --no-ours        # just nfelo vs Vegas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
NFELO_DIR_DEFAULT = ROOT.parent / "nfelo_reference" / "output_data"

BREAKEVEN = 0.5238  # -110 juice


# ── data loading ────────────────────────────────────────────────────
def load_actuals() -> pd.DataFrame:
    """Completed games keyed by game_id, with actual home margin + closing spread."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["home_score"].notna() & s["spread_line"].notna()].copy()
    s["home_margin"] = s["home_score"] - s["away_score"]
    s["home_win"]    = (s["home_margin"] > 0).astype(int)
    return s[["game_id", "season", "week", "home_team", "away_team",
              "home_margin", "home_win", "spread_line"]]


def load_nfelo(nfelo_dir: Path) -> pd.DataFrame:
    """nfelo projected home margin + home win prob, keyed by game_id."""
    f = nfelo_dir / "historic_projected_spreads.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"nfelo benchmark not found at {f}. Point --nfelo-dir at the extracted "
            f"nfelo_reference/output_data folder."
        )
    nf = pd.read_csv(f)
    # nfelo home line is a betting line (negative = home favored) -> margin = -line
    nf["nfelo_margin"] = -nf["home_closing_line_rounded_nfelo"]
    nf = nf.rename(columns={"home_probability_nfelo": "nfelo_wp"})
    return nf[["game_id", "nfelo_margin", "nfelo_wp"]]


def load_our_preds(path: Path, actuals: pd.DataFrame) -> pd.DataFrame:
    """Normalize any of our prediction frames to game_id + our_margin (+ our_wp)."""
    df = pd.read_parquet(path)

    # predicted home margin — accept several column spellings
    if {"predicted_home_score", "predicted_away_score"}.issubset(df.columns):
        df["our_margin"] = df["predicted_home_score"] - df["predicted_away_score"]
    elif {"home_pred", "away_pred"}.issubset(df.columns):
        df["our_margin"] = df["home_pred"] - df["away_pred"]
    elif "predicted_spread" in df.columns:          # away - home (negative = home fav)
        df["our_margin"] = -df["predicted_spread"]
    else:
        raise ValueError(f"{path.name}: can't find predicted scores/margin columns")

    df["our_wp"] = df["home_win_probability"] if "home_win_probability" in df.columns else np.nan

    # attach game_id (these frames key on season/week/home/away)
    if "game_id" not in df.columns:
        key = ["season", "week", "home_team", "away_team"]
        if not set(key).issubset(df.columns):
            raise ValueError(f"{path.name}: need game_id or {key} to join")
        df = df.merge(actuals[["game_id"] + key], on=key, how="left")

    return df[["game_id", "our_margin", "our_wp"]].dropna(subset=["game_id"])


# ── metrics ─────────────────────────────────────────────────────────
def margin_metrics(pred_margin: pd.Series, actual_margin: pd.Series) -> dict:
    err = pred_margin - actual_margin
    return {
        "n":    int(len(err)),
        "MAE":  float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "corr": float(pred_margin.corr(actual_margin)),
    }


def ats_metrics(pred_margin: pd.Series, actual_margin: pd.Series,
                spread_line: pd.Series) -> dict:
    """ATS accuracy vs the closing line: model takes the side it likes vs the line."""
    pick_home   = pred_margin > spread_line          # model thinks home beats the line
    home_covers = actual_margin > spread_line
    disagree    = pred_margin != spread_line         # model has an opinion vs the line
    correct     = (pick_home == home_covers)[disagree]
    # "edge" picks: model disagrees with the market on the favorite
    market_home_fav = spread_line > 0
    edge = disagree & (pick_home != market_home_fav)
    return {
        "ATS_all":  float(correct.mean()) if len(correct) else float("nan"),
        "n_ats":    int(disagree.sum()),
        "ATS_edge": float((pick_home == home_covers)[edge].mean()) if edge.any() else float("nan"),
        "n_edge":   int(edge.sum()),
    }


def calibration(wp: pd.Series, home_win: pd.Series) -> dict:
    mask = wp.notna()
    if mask.sum() == 0:
        return {"Brier": float("nan"), "logloss": float("nan")}
    p = wp[mask].clip(1e-6, 1 - 1e-6)
    y = home_win[mask]
    return {
        "Brier":   float(((p - y) ** 2).mean()),
        "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
    }


def report(name: str, m: pd.DataFrame, has_wp: bool, ats: bool = True):
    mm = margin_metrics(m["pred_margin"], m["home_margin"])
    line = (f"  {name:10} n={mm['n']:4}  MAE={mm['MAE']:5.2f}  RMSE={mm['RMSE']:5.2f}  "
            f"corr={mm['corr']:.3f}")
    if ats:
        am = ats_metrics(m["pred_margin"], m["home_margin"], m["spread_line"])
        edge = f"{am['ATS_edge']:.1%}" if am["n_edge"] else "  n/a"
        line += f"  ATS={am['ATS_all']:.1%}  edge={edge}(n={am['n_edge']})"
    if has_wp and "wp" in m.columns:
        cal = calibration(m["wp"], m["home_win"])
        if not np.isnan(cal["Brier"]):
            line += f"  Brier={cal['Brier']:.3f}  logloss={cal['logloss']:.3f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-preds", default=str(PROC / "backtest_2021_2022_2023_2024_2025.parquet"))
    ap.add_argument("--no-ours", action="store_true", help="benchmark only nfelo vs Vegas")
    ap.add_argument("--nfelo-dir", default=str(NFELO_DIR_DEFAULT))
    ap.add_argument("--by-season", action="store_true", help="also break down per season")
    args = ap.parse_args()

    actuals = load_actuals()
    nfelo   = load_nfelo(Path(args.nfelo_dir))
    base = actuals.merge(nfelo, on="game_id", how="left")

    ours = None
    if not args.no_ours:
        p = Path(args.our_preds)
        ours = load_our_preds(p, actuals)
        base = base.merge(ours, on="game_id", how="left")
        is_insample = "backtest_2021" in p.name
        print(f"\n  Our predictions: {p.name}"
              + ("   [WARNING: in-sample — optimistic vs the OOS numbers]" if is_insample else ""))

    print(f"\n{'='*84}")
    print(f"  BENCHMARK vs nfelo + Vegas   (home-margin prediction, completed games)")
    print(f"  Vegas MAE is the bar to beat; even nfelo barely clears it. Breakeven ATS = {BREAKEVEN:.1%}")
    print(f"{'='*84}")

    def block(df, label):
        print(f"\n  [{label}]")
        # Vegas: predicted home margin == spread_line (no ATS/own-line, no wp here)
        v = df.dropna(subset=["spread_line"]).assign(pred_margin=lambda d: d["spread_line"])
        report("Vegas", v, has_wp=False, ats=False)
        # nfelo
        nf = df.dropna(subset=["nfelo_margin"]).assign(
            pred_margin=lambda d: d["nfelo_margin"], wp=lambda d: d["nfelo_wp"])
        report("nfelo", nf, has_wp=True)
        # ours
        if ours is not None:
            o = df.dropna(subset=["our_margin"]).assign(
                pred_margin=lambda d: d["our_margin"], wp=lambda d: d["our_wp"])
            if len(o):
                report("ours", o, has_wp=True)

    block(base, "ALL SEASONS")
    if args.by_season:
        for s in sorted(base["season"].dropna().unique()):
            block(base[base["season"] == s], f"Season {int(s)}")

    print(f"\n  Notes:")
    print(f"   - MAE ~9.8-9.9 is market-level; that's the realistic ceiling, not a weak score.")
    print(f"   - ATS 'edge' = games where the model disagrees with the market favorite.")
    print(f"   - Lower Brier/log-loss = better-calibrated win probabilities.\n")


if __name__ == "__main__":
    main()
