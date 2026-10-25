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
   Needs the fetch-only deps: `pip install -r requirements-fetch.txt` (kept separate
   from `requirements.txt` because `nfl_data_py` pins `pandas<2`, which conflicts with
   the engine's `pandas>=2`; the runtime never imports it).
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

**Pinned contest lines.** The owner's real picks go into the Circa Million contest, whose
spreads are fixed on Wednesday. `ml/line_overrides.py` pins a week's sheet
(`python -m ml.line_overrides --season 2026 --week 7 --source circa CHI:-3 NO:+3 …`, tokens
in the TEAM'S own perspective; the sheet lists the visitor first) into the git-managed
`data/processed/line_overrides.json`; `_finalize_slate` applies them over both the live
Odds-API line and nflverse (`line_source` "circa"), so the slate, top-5, ledger and social post
all grade against the contest numbers. The top five is the SATURDAY board: Thursday/pre-lock
games are not eligible (owner, 2026-10-23, after a one-day trial of including them;
`TOP5_INCLUDE_EARLY=1` lets them in), and `ml.ledger.unrank_early` strips any rank a frozen
early row carries. A pick whose game is already final shows its result on the social post.

## One game, one number

Every page that predicts a SCHEDULED game (Schedule & Picks slate, Matchup page, the props
board) goes through `_predict_scheduled` in `dashboard/server.py`: venue from the schedule row
(a listed Neutral site removes home field), travel/rest/weather deltas, total anchored to the
market line. Never call `_adjusted_prediction`/`project_game` directly for a game that is on
the schedule — in week 7 2026 the Matchup page did, treated Steelers–Saints in Paris as a Saints
home game, and leaned the opposite way to the board being graded. The Matchup page also shows
the board's own row (`_board_row`: its line source and, after the Saturday lock, the frozen
numbers). `python tests/test_matchup_slate_agree.py` (slow, builds the current week) fails if
any page disagrees with the slate; run it after touching prediction routing.

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

## Hosting & data refresh (Railway)

The Flask dashboard is deployed on Railway (`nfl-engine-production.up.railway.app`) via
gunicorn, and is reached publicly at the brand domain **www.halfbloodbettingprince.com**
(canonical; the bare apex redirects to www at Cloudflare's edge): CNAMEs to the Railway
custom-domain targets, Cloudflare SSL mode Full, added 2026-10-24 because the owner's work
network (Zscaler) blocks every `*.up.railway.app` hostname at the TLS handshake. Railway
issued the www certificate; the apex one hit a Let's Encrypt rate limit, which is why www is
canonical. `PUBLIC_BASE_URL` on the service points at www.
Two hard rules for anything that runs ON the server:

- **Never import `nfl_data_py` in the runtime path.** It pins `pandas<2`, which conflicts
  with the engine's pandas 3 and breaks the build. Runtime deps live in `requirements.txt`;
  the fetch-only libs (nfl_data_py/requests-for-fetch/tqdm) live in `requirements-fetch.txt`.
- **The server reads local parquet only.** The one exception is `ml/refresh.py`, which
  downloads nflverse **release** parquets directly over HTTPS (pandas + requests, no
  nfl_data_py) for a *light* current-season refresh: injuries + PBP + schedules, then
  rebuilds `situational_stats` + `team_styles` (`python -m ml.refresh --season 2026`).
  It does **not** rebuild composite/the full engine — that stays an offline `run_engine.py` job.

Refresh is exposed at `POST /api/refresh` (guarded by the `REFRESH_TOKEN` env var) +
`GET /api/refresh/status`, and can run on an in-process daily schedule (`REFRESH_DAILY=1`,
`REFRESH_HOUR`, `REFRESH_SEASON`). After a refresh the server clears its in-process caches
(`clear_caches()` in `dashboard/server.py`). Persistence uses a Railway **Volume** at
`/app/data`; `dashboard/seed.py` (run before gunicorn) seeds it from the build-time
`data_seed/` copy on first boot.

The dashboard (`dashboard/season2026.html`) is a hash-routed SPA: Rankings, Team profile
(`/api/team_profile`), Trends (`/api/team_trends`), Matchup (`/api/matchup_full`), Refresh.
Betting > Player Props (`/api/props_slate`) is schedule-driven like Schedule & Picks: every game
of the week, model props per player (`ml/props.py`), live book lines merged per event
(`ml/odds.event_props`, one Odds-API request per game, cached 30 min), and a "best props" strip
of markets where the model beats the vig-free book price by the floor (`MIN_EDGE` 10 pts,
`MIN_EDGE_BY_MARKET` 15 on the rushing markets — one table in `ml/props_ledger.py` shared by
the board and the record, SET FROM `backtest_props.py`'s floor table; re-run it before
changing). A cold week builds in a
background thread (the route answers 202 + progress; the page polls) because 16 box scores
exceed the gunicorn timeout; the refresh job pre-warms the current week (`warm_props_slate`).
Every book-lined market on the board is written to `ml/props_ledger.py` before kickoff
(provisional until the Saturday lock, frozen after; late-posting markets added once) and
graded on the play-by-play box score; `/api/props_record` feeds Betting > Props Record.

Props pricing (`ml/props.py`, `_price_market` in `dashboard/server.py`): spreads are FITTED
(`_SD_FIT`, game-log sd = a·mean^k × a projection-error scale k from `backtest_props.py`;
receptions are negative binomial, `REC_PHI`), and with a book price the mean is ANCHORED toward
the market-implied mean (`anchor_to_book`: `model_proj` raw, `book_mean` market, `proj` = the
blend that is priced). The weight is `ANCHOR_DEFAULT` (0.65; TD/INT markets higher) until the
calibration log `data/processed/props_calib.parquet` (every lined market + box score, refresh-
managed, git-ignored) has ≥ `CALIB_MIN_N` graded rows for a market — then `anchor_weights()`
measures it by least squares and the server uses that. `python backtest_props.py` is the
walk-forward 2022-25 backtest (mean quality, spread calibration, pick rule vs a naive market;
section D runs against real closing lines once `fetch_prop_history.py` has pulled them from
the Odds API's historical endpoints — paid plan, ~110 credits per game, 2023 season onward).

## Social posts (social/)

The weekly top-5 ATS picks go out as an image (1080×1350) and a vertical video (1080×1920,
~21 s) to YouTube Shorts, Instagram and X. `social/content.py` reads the FROZEN board
(`_slate` + `_overlay_locked`) and the season record; `social/render.py` draws with PIL and
encodes through the ffmpeg binary in the `imageio-ffmpeg` wheel (no system ffmpeg);
`social/publish.py` calls the official APIs with credentials from Railway variables
(`YT_*`, `IG_*`, `X_*`, see `docs/SOCIAL_SETUP.md`); `social/post.py` renders, publishes once
per platform per week (`data/processed/social_posts.jsonl`, refresh-managed on the volume) and
is called from the refresh job (`post_if_due`, needs `SOCIAL_AUTO=1`) once the Saturday lock has
passed. The rendered files are served publicly at `/social/<season>_wk<NN>/…` because Instagram
fetches media by URL. Manual: `POST /api/social/post?week=&dry=1&platforms=` (REFRESH_TOKEN),
`GET /api/social/status`; local: `python -m social.post --season 2026 --week 7 --dry-run`.

## Kalshi (Betting > Kalshi tab)

Ported from the tennis engine. Four modules under `ml/`, one boundary:
- `kalshi.py` reads the account and prices/sizes (RSA-PSS auth over `ts+METHOD+/trade-api/v2/path`,
  fee `ceil(0.07·C·P·(1−P))` per order, Kelly net of fees). Nothing in it can spend.
- `kalshi_match.py` maps a scheduled game to Kalshi's `KXNFLGAME` / `KXNFLSPREAD` / `KXNFLTOTAL`
  events. EXACT series only (dozens of sibling NFL series exist); both teams must sit in the
  same event on the game's date; Kalshi codes differ from nflverse (`LAR` vs `LA`, see `TEAM_CODES`).
- `kalshi_order.py` is the ONLY module that can place an order: one order per call, reachable only
  from `POST /api/kalshi/submit` with `confirm=true`. Re-reads the market first (closed / empty
  book / ask drifted > `KALSHI_MAX_PRICE_DRIFT` / kicked off → refuse). `client_order_id` is minted
  once per ticket and reused on retry. **Dry run unless `KALSHI_ARM=1`; demo API unless `KALSHI_LIVE=1`.**
- `risk.py` enforces `KALSHI_MAX_TICKET_PCT` (2%) and `KALSHI_MAX_DAILY_PCT` (20%) server-side; the
  exposure ledger `data/processed/kalshi_exposure.jsonl` is refresh-managed on the volume and git-ignored.

Tickets are priced from the MARKET-BLENDED probability (`_kalshi_probs` in `dashboard/server.py`),
not the raw model, because the raw model does not beat NFL markets out-of-sample. YES contracts
only (a bid on the team's own market/strike); at most one ticket per market type per game.
Kickoff comes from our schedule (`ml/ledger.kickoff_utc`), never from Kalshi's timestamps.
Credentials: `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY` (PEM text) as Railway service variables —
never in the repo, never handled through chat. Tests: `tests/test_kalshi*.py` (script-style).

## Conventions

- Match the existing style: heavy docstrings at the top of each module describing inputs/outputs,
  section banners (`# ──── …`), and inline comments justifying every tuned constant.
- Scripts use `argparse` with `--season` / `--seasons` / `--week`. Outputs go to `data/processed/`.
- Paths are always derived from `Path(__file__).parent` — keep it relocatable.
