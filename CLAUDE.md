# CLAUDE.md

Operational guide for working in this repo. Read this first.

## What this is

An NFL game-prediction engine. It ingests nflverse data, builds player/team/coaching
quality models, and predicts game scores, spreads, totals, win probabilities, and game
script. A picks layer turns predictions into ATS bets; a backtest layer validates them;
a Flask dashboard visualizes them.

- Language: Python 3 (pandas / numpy / pyarrow / nfl_data_py)
- Data format: Parquet (raw + processed), some CSV rankings outputs
- No git, no tests, no CI as of this writing. Not a package — scripts are run directly.

## Pipeline (data flow)

```
fetch_data.py ──▶ data/raw/*.parquet ──▶ engine/ build steps ──▶ data/processed/*.parquet ──▶ predict ──▶ picks / backtest / dashboard
```

1. **Fetch** — `python fetch_data.py --seasons 2020 2021 2022 2023 2024 2025`
   Run LOCALLY (needs internet). Writes ~30 parquet files to `data/raw/`.
2. **Build engine** (in order; orchestrated by `run_engine.py`):
   - `engine/composite.py` → `composite_scores.parquet` — 0-100 player scores per (season, week)
   - `engine/styles.py` → `team_styles.parquet` — team offensive/defensive style profiles from PBP
   - `engine/conditions.py` → `conditions.parquet` — weather/surface/rest/altitude/home-field modifiers
   - `coaching_metrics.py` → `coaching_scores.parquet` + `coaching_flags.csv` (optional, top-level standalone script)
3. **Predict** — `engine/predict.py` consumes the processed parquets + matchups to produce per-game predictions.
4. **Use** — `weekly_picks.py` (ATS picks), `backtest.py` / `backtest_picks.py` (validation), `dashboard/`.

## Common commands

```bash
# Full build for one or more seasons, then predict a week
python run_engine.py --season 2025 --week 1
python run_engine.py --seasons 2021 2022 2023 2024 2025 --week 18

# Single game
python run_engine.py --season 2025 --week 1 --home KC --away BAL

# Just re-predict (engine already built)
python engine/predict.py --season 2025 --week 1 --home KC --away BAL

# Weekly ATS picks (needs Vegas lines in schedules.parquet)
python weekly_picks.py --season 2025 --week 1 --top 5

# Backtest (walk-forward, out-of-sample is the honest one)
python backtest_picks.py --seasons 2023 2024 2025
python backtest.py --season 2024

# Dashboard
python dashboard/server.py        # http://localhost:5000

# Position rankings (standalone, write CSVs to data/processed/)
python qb_rankings.py --season 2025
python skill_rankings.py --season 2025 --pos WR
python ol_dl_st_rankings.py --season 2025 --pos OL
python st_rankings.py --season 2025
```

`run_engine.py` flags: `--skip-composite`, `--skip-styles`, `--skip-conditions` to reuse
already-built processed data.

## Key directories / files

- `engine/` — the model. `predict.py` is the heart (score formula, win prob, game script).
  `composite.py`, `styles.py`, `conditions.py`, `matchups.py` are the imported build modules.
  `coaching_metrics.py` and `update_styles_2025.py` live at top level (run as standalone scripts).
- `data/raw/` — fetched source parquets (pbp_YYYY, rosters, injuries, ngs_*, pfr_*, schedules, weather, …).
- `data/processed/` — built model outputs + predictions_{season}_wk##.parquet + rankings CSVs.
- `dashboard/` — Flask `server.py` + `dashboard.html`.
- `roster_update.py` — manages `depth_charts.parquet` / `rosters_weekly.parquet` lifecycle (trades, FA, weekly refresh).
- `fetch_data.py` — the only thing that touches the network; everything else is offline on cached parquets.

## The prediction formula (engine/predict.py)

Per team, starting from `LEAGUE_AVG_PTS = 24.8`, additively adjusted by:
- offensive quality (composite players, style-weighted by pass rate + OL flags), ×0.09/pt
- team off/def EPA (from styles), defensive quality suppression
- positional matchup edge (`matchups.py`), QB backup penalty, rest differential, ref-crew flag tendency, kicker (close games only)
- home-field points, then weather/surface scoring + pass/rush multipliers
- clipped to [6, 55]. Win prob = sigmoid on margin (σ=13.5), blended 75/25 with moneyline when available.

Most adjustments are hand-tuned magic numbers with inline rationale comments. When changing
weights, keep the comment explaining the *why* next to the number.

## ATS spread convention (do not get this wrong)

nflverse `spread_line` is **home perspective, POSITIVE = home favored** (verified
empirically: home wins ~67% of `spread_line > 0` games). Therefore:
- Home covers ATS iff `(home_score - away_score) > spread_line`.
- Vegas favors home iff `spread_line > 0`.
- Model home margin = `predicted_home_score - predicted_away_score`.
- Note `predict.py` reports `predicted_spread = away - home` (NEGATIVE = home favored) —
  the opposite sign of `spread_line`. Keep them straight.

This was inverted across the entire ATS/picks layer at one point, which faked an ~86%
backtest cover rate (true OOS rate is ~49%). `tests/test_spread_convention.py` guards it —
run `python tests/test_spread_convention.py` after touching any spread/cover logic.

## Gotchas

- **Standalone build scripts**: `coaching_metrics.py` and `update_styles_2025.py` are run directly
  (`python coaching_metrics.py …`), not imported. They derive paths from `Path(__file__).parent`,
  so they MUST stay at the repo root (the `engine/` copies were broken and have been removed).
- **No future leakage**: composite uses `shift(1)` rolling means; `backtest_picks.py` rebuilds
  styles using only prior seasons. Preserve this when touching anything used in backtests.
- **Schema drift**: `predict.load_engine_data()` has defensive normalization for nflverse
  depth-chart schema changes (2026). Expect missing/renamed columns from upstream.
- **Fallbacks everywhere**: lookups fall back to most-recent-season when exact (team, season)
  isn't found. A prediction with sparse data still returns — it doesn't error.
- **Windows + OneDrive**: repo lives under a OneDrive path with spaces. Quote paths. Shell is
  PowerShell; a Bash tool is also available.

## Conventions

- Match the existing style: heavy docstrings at the top of each module describing inputs/outputs,
  section banners (`# ──── …`), and inline comments justifying every tuned constant.
- Scripts use `argparse` with `--season` / `--seasons` / `--week`. Outputs go to `data/processed/`.
- Paths are always derived from `Path(__file__).parent` — keep it relocatable.
