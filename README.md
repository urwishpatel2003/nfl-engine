# NFL Engine

A data-driven NFL game-prediction engine. It pulls [nflverse](https://github.com/nflverse)
data, models player / team / coaching quality, and predicts each game's **score, spread,
total, win probability, and game script** — then turns those predictions into against-the-spread
(ATS) picks and backtests them.

## How it works

```
fetch_data.py  ─▶  data/raw/        ─▶  engine/  ─▶  data/processed/  ─▶  predictions + picks + dashboard
 (network)         source parquets      models       built outputs
```

1. **Fetch** raw nflverse data (play-by-play, rosters, injuries, Next Gen Stats, PFR, schedules, weather).
2. **Build** four model components:
   - **Composite player scores** — a 0-100 rating per player/week blending positional rank, EPA
     efficiency, usage, NGS tracking, and combine athleticism, scaled by depth-chart role and injury status.
   - **Team styles** — offensive/defensive identity from PBP (pass rate, air yards, pace, EPA, pressure, coverage).
   - **Conditions** — weather, field surface, altitude, rest, travel, and home-field modifiers.
   - **Coaching metrics** — halftime adjustment, discipline (penalties), and ATS coaching edge.
3. **Predict** — combine the components plus a positional matchup matrix into a final game projection.
4. **Apply** — generate weekly ATS picks, backtest the strategy out-of-sample, and explore results in a dashboard.

## Setup

```bash
pip install -r requirements.txt        # pandas, numpy, pyarrow, nfl_data_py, requests, tqdm
pip install flask                      # only for the dashboard
```

## Usage

```bash
# 1. Fetch source data (run locally — needs internet; ~500MB for 6 seasons)
python fetch_data.py --seasons 2020 2021 2022 2023 2024 2025

# 2. Build the engine and predict a week
python run_engine.py --season 2025 --week 1

# Predict a single matchup
python run_engine.py --season 2025 --week 1 --home KC --away BAL

# 3. Weekly ATS picks (top 5 by confidence)
python weekly_picks.py --season 2025 --week 1 --top 5

# 4. Backtest the pick strategy (walk-forward, out-of-sample)
python backtest_picks.py --seasons 2023 2024 2025

# 5. Dashboard
python dashboard/server.py             # http://localhost:5000
```

To rebuild only part of the engine, pass `--skip-composite`, `--skip-styles`, or
`--skip-conditions` to `run_engine.py`.

## Layout

| Path | Purpose |
|------|---------|
| `fetch_data.py` | Pulls all source data into `data/raw/` (the only networked script) |
| `engine/predict.py` | Core prediction model (scores, win prob, game script) |
| `engine/composite.py` | Per-player 0-100 composite scores |
| `engine/styles.py` | Team offensive/defensive style profiles |
| `engine/conditions.py` | Weather / surface / rest / home-field modifiers |
| `engine/matchups.py` | Positional matchup matrix for a game |
| `coaching_metrics.py` | Coach quality signals |
| `run_engine.py` | Orchestrates build steps 1-4 |
| `weekly_picks.py` | ATS pick ranker for a contest (top-5/week) |
| `backtest.py`, `backtest_picks.py` | Prediction & pick validation |
| `*_rankings.py` | Standalone QB / skill / OL / DL / ST rankings → CSVs |
| `roster_update.py` | Roster lifecycle (trades, FA, weekly refresh) |
| `dashboard/` | Flask server + HTML frontend |
| `data/raw/`, `data/processed/` | Source parquets and built outputs |

See [CLAUDE.md](CLAUDE.md) for engine internals, the scoring formula, and gotchas.

## ML pipeline & findings

A second, fitted model lives in `ml/` — built to replace the hand-tuned engine's magic
numbers with weights learned from data, under strict no-leakage discipline.

| File | Purpose |
|------|---------|
| `ml/features.py` | Leak-free per-game feature matrix (market lines, current-season-to-date EPA/success/pace, prior-season baselines, QB/composite quality, coaching, styles) |
| `ml/train.py` | Walk-forward XGBoost + Ridge; margin model and an edge (residual-vs-spread) model |
| `ml/rank.py` | Standalone power ratings (1–32) + per-game predictions |
| `compare_to_nfelo.py` | Benchmark any predictions vs the public [nfelo](https://www.nfeloapp.com) model and Vegas |
| `optimize_market_weight.py` | Sweep the model↔market blend weight (`engine/market.py`) |

```bash
python -m ml.features            # build data/processed/game_features.parquet
python -m ml.train               # walk-forward evaluation
python -m ml.rank --season 2025          # power rankings
python -m ml.rank --predict 2025 18      # week predictions
```

**What the honest evaluation found** (all out-of-sample, walk-forward):

- The original engine's "edge" was a **spread-sign bug × data leakage**, not skill. Fixed both;
  the honest ATS cover rate is ~45–49% (below the 52.4% breakeven), not the 86% once reported.
- **NFL game markets are efficient** with respect to these features: an edge model predicting the
  residual vs the closing spread (or total) has correlation ≈ 0 and ATS ≈ 49%. The market has
  already priced in EPA, points, QB/squad quality, styles, and coaching.
- The fitted **standalone** model (no betting lines) reaches OOS margin correlation ~0.28 — a real
  predictor, but short of the market's ~0.48. Coaching contributes (top-5 feature); current-season
  EPA/points differentials dominate.

**Bottom line:** this is a solid standalone predictor and power-ranking / analysis tool, **not** a
market-beating betting model. Use `ml/rank.py` for rankings and predictions; treat any ATS output
as research, not an edge.

## Notes

- Predictions are for **entertainment / research**. No model beats the market reliably; bet responsibly.
- The model uses hand-tuned coefficients (documented inline in `engine/predict.py`).
- Backtests use walk-forward logic with no future leakage — trust `backtest_picks.py` over in-sample fits.
