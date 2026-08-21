"""
ml/refresh.py  —  current-season data refresh for the HOSTED dashboard
======================================================================
Pulls the latest nflverse data (injuries, weekly player stats, play-by-play,
schedules, depth charts, rosters) for one season and refreshes the light
processed tables the dashboard reads (situational_stats + team_styles).

Design constraints (why this exists instead of fetch_data.py):
  • Runs ON the server (Railway), so it must NOT import nfl_data_py — that library
    pins pandas<2 and conflicts with the app's pandas 3 (it broke the first deploy).
    Instead we download the nflverse RELEASE parquet files directly over HTTPS with
    pandas + requests.
  • Kept "light": no engine composite rebuild, current season only, so it finishes in
    seconds-to-a-minute and fits a background thread.
  • Multi-season raw files (injuries, player_stats, …) are MERGED by season, not
    replaced, so history is preserved. Per-season PBP is replaced. PBP is filtered to
    the same column subset fetch_data.py uses, to keep files small (~1-2 MB, not 400 MB).

Usage:
    python -m ml.refresh --season 2026
    python -m ml.refresh --season 2025 --skip-download   # just rebuild the light tables
"""

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
STATUS_FILE = PROC / "last_refresh.json"

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLDATA = "https://github.com/nflverse/nfldata/raw/master/data"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {level:5} {msg}")


# ── download helpers ─────────────────────────────────────────────────
def _candidate_urls(season: int) -> dict:
    """local-name -> [candidate URLs]; first that downloads+parses wins.

    Scoped to exactly what the hosted dashboard's LIGHT rebuild + live views consume:
      injuries      → matchup injury panels
      pbp_{season}  → trends/form, team styles, unit ratings (rebuilt into team_styles)
      schedules     → points for/against + upcoming games
      depth_charts  → live team assignments (cuts/trades/signings) → 2026 roster rebuild,
                      so squad ratings track the CURRENT roster, not a stale snapshot
    Player weekly stats are intentionally NOT fetched here: they only feed the engine's
    composite rebuild, which is a heavy/offline step, not part of this light refresh.
    (nflverse has fallback asset names, hence the lists.)"""
    s = season
    return {
        "injuries":  [f"{NFLVERSE}/injuries/injuries_{s}.parquet"],
        f"pbp_{s}":  [f"{NFLVERSE}/pbp/play_by_play_{s}.parquet"],
        "schedules": [f"{NFLDATA}/games.parquet", f"{NFLDATA}/games.csv"],
        "depth_charts": [f"{NFLVERSE}/depth_charts/depth_charts_{s}.parquet"],
        f"rosters_{s}": [f"{NFLVERSE}/rosters/roster_{s}.parquet"],
    }


def _fetch_first(urls: list) -> pd.DataFrame:
    """Download the first URL that succeeds; parse parquet or csv into a DataFrame."""
    last = None
    for u in urls:
        try:
            r = requests.get(u, timeout=90)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            if u.endswith(".csv"):
                return pd.read_csv(buf, low_memory=False)
            return pd.read_parquet(buf)
        except Exception as e:                       # try the next candidate
            last = e
    raise last or RuntimeError("no candidate URL succeeded")


def _safe_to_parquet(df: pd.DataFrame, path: Path):
    """Write parquet; if a mixed/object column (e.g. datetime.date after a concat) breaks
    the arrow conversion, coerce the offending object columns to string and retry once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str)
        df.to_parquet(path, index=False)


def _filter_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns fetch_data.py stores, so the file stays ~1-2 MB not ~400 MB."""
    try:
        sys.path.insert(0, str(ROOT))
        from fetch_data import PBP_COLUMNS         # top-level module, no nfl_data_py import
        want = [c for c in PBP_COLUMNS if c in df.columns]
        if want:
            return df[want]
    except Exception:
        pass
    return df


def _merge_by_season(name: str, df: pd.DataFrame, season: int) -> int:
    """Replace only `season`'s rows in a multi-season raw file, preserving history."""
    path = RAW / f"{name}.parquet"
    if path.exists() and "season" in df.columns:
        try:
            old = pd.read_parquet(path)
            if "season" in old.columns:
                old = old[old["season"] != season]
                df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    _safe_to_parquet(df, path)
    return len(df)


def download_nflverse(season: int, log=_default_log) -> dict:
    """Download the current-season release files. Returns per-file status dict."""
    results = {}
    for name, urls in _candidate_urls(season).items():
        try:
            df = _fetch_first(urls)
            if name.startswith("pbp_"):
                df = _filter_pbp(df)
                _safe_to_parquet(df, RAW / f"{name}.parquet")
                n = len(df)
            elif name == "schedules":                # full multi-season file; authoritative
                _safe_to_parquet(df, RAW / "schedules.parquet")
                n = len(df)
            elif name == "depth_charts":
                # The current-season release is a CUMULATIVE dt-stamped live snapshot with no
                # season column. Keep historical (season-tagged) rows, replace all live
                # (season=NaN) rows with the fresh release — bounded size, no stale dupes.
                path = RAW / "depth_charts.parquet"
                if path.exists():
                    old = pd.read_parquet(path)
                    hist = old[old["season"].notna()] if "season" in old.columns else old
                    combined = pd.concat([hist, df], ignore_index=True)
                else:
                    combined = df
                _safe_to_parquet(combined, path)
                # Also refresh depth_2026_current.parquet — the CANONICAL "current depth
                # chart" every model reads (squad, projections, qb_overlay). It is each
                # team's most recent published snapshot from the cumulative live data.
                cur = df.copy()
                cur["dt"] = cur["dt"].astype(str)
                cur = cur[cur["dt"] == cur.groupby("team")["dt"].transform("max")]
                _safe_to_parquet(cur, RAW / f"depth_{season}_current.parquet")
                log(f"  depth_{season}_current: {len(cur):,} rows (latest per-team snapshot)")
                n = len(combined)
            elif name.startswith("rosters_"):
                # nflverse seasonal roster release → the canonical rosters_{season}.parquet
                # consumed across ml/ (squad, fantasy, matchup_engine, …). Align the release
                # schema to the columns consumers expect.
                df = df.rename(columns={"full_name": "player_name", "gsis_id": "player_id"})
                if "team" in df.columns:              # this release calls Arizona "AZ";
                    df["team"] = df["team"].replace({"AZ": "ARI"})   # everything else here uses ARI
                if "age" not in df.columns and "birth_date" in df.columns:
                    bd = pd.to_datetime(df["birth_date"], errors="coerce")
                    df["age"] = ((pd.Timestamp.now() - bd).dt.days / 365.25).round(1)
                _safe_to_parquet(df, RAW / f"{name}.parquet")
                n = len(df)
            else:
                n = _merge_by_season(name, df, season)
            results[name] = {"rows": int(n)}
            log(f"  {name}: {n:,} rows")
        except Exception as e:
            results[name] = {"error": str(e)[:200]}
            log(f"  {name}: FAILED — {e}", "WARN")
    return results


# ── light rebuild (no network, idempotent) ───────────────────────────
def rebuild_light(log=_default_log) -> dict:
    """Recompute situational_stats (from PBP) + team_styles (all PBP seasons)."""
    out = {}
    sys.path.insert(0, str(ROOT))

    try:
        from fetch_data import compute_situational_stats
        compute_situational_stats(force=True)
        out["situational_stats"] = "ok"
        log("  situational_stats rebuilt")
    except Exception as e:
        out["situational_stats"] = f"error: {str(e)[:200]}"
        log(f"  situational_stats FAILED — {e}", "WARN")

    try:
        seasons = sorted({int(p.stem.split("_")[1]) for p in RAW.glob("pbp_*.parquet")})
        from engine.styles import build_team_styles
        build_team_styles(seasons=seasons)
        out["team_styles"] = f"ok ({seasons})"
        log(f"  team_styles rebuilt for {seasons}")
    except Exception as e:
        out["team_styles"] = f"error: {str(e)[:200]}"
        log(f"  team_styles FAILED — {e}", "WARN")

    # 2026 roster: rebuild week-0 assignments from the just-refreshed live depth charts so
    # squad ratings/depth pages track cuts, trades and signings. refresh=False keeps this
    # network-free and (critically) avoids the nfl_data_py import that breaks the server env.
    try:
        from roster_update import build_2026_roster
        build_2026_roster(refresh=False)
        out["roster_2026"] = "ok"
        log("  2026 roster rebuilt from live depth charts")
    except Exception as e:
        out["roster_2026"] = f"error: {str(e)[:200]}"
        log(f"  2026 roster rebuild FAILED — {e}", "WARN")

    # Season projections: precomputed + stored (win totals fold in completed games from the
    # refreshed schedule; player totals pick up refreshed styles/SOS). This is the weekly update.
    try:
        import importlib
        import ml.season as _season
        importlib.reload(_season)                       # ensure it reads the just-refreshed data
        res = _season.build_season()
        out["season_projections"] = f"ok ({res['games_played']} games played)"
        log(f"  season projections rebuilt ({res['games_played']} games played)")
    except Exception as e:
        out["season_projections"] = f"error: {str(e)[:200]}"
        log(f"  season projections FAILED — {e}", "WARN")

    return out


# ── orchestration ────────────────────────────────────────────────────
def run(season: int, log=_default_log, skip_download: bool = False) -> dict:
    """Full refresh: download current season, then rebuild the light tables."""
    t0 = time.time()
    started = _now()
    log(f"Refresh start — season {season}")

    files = {} if skip_download else download_nflverse(season, log)
    rebuild = rebuild_light(log)

    ok = (skip_download or any("rows" in v for v in files.values())) and \
         all(not str(v).startswith("error") for v in rebuild.values())
    status = {
        "season": season, "started": started, "finished": _now(), "time": _now(),
        "elapsed_sec": round(time.time() - t0, 1),
        "files": files, "rebuild": rebuild, "ok": bool(ok),
    }
    try:
        PROC.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status, indent=2))
    except Exception as e:
        log(f"  could not write status file — {e}", "WARN")
    log(f"Refresh done in {status['elapsed_sec']}s (ok={ok})")
    return status


def last_status() -> dict | None:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Refresh current-season NFL data (no nfl_data_py)")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--skip-download", action="store_true",
                    help="only rebuild situational_stats + team_styles from existing PBP")
    args = ap.parse_args()
    st = run(args.season, skip_download=args.skip_download)
    print("\n" + json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
