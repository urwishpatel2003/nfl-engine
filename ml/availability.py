"""
ml/availability.py  —  owner-pinned availability (a transaction the data has not caught up with)
=============================================================================================
The model learns who cannot play from two nflverse files: the injury report (Wednesday-Friday,
Out/Doubtful) and the roster release (reserve lists: IR, PUP, NFI). Both lag the real world —
the roster release is a daily snapshot, so a Tuesday IR placement (Lamar Jackson, 2026-11-03)
is invisible until the next day's file, and a quarterback who left Sunday's game is "healthy"
until Wednesday's report. The owner usually knows the day it happens.

This pins a player as unavailable from a week on, with an optional last week, exactly like the
Circa sheet pins a week's lines (ml/line_overrides.py). It is merged into
ml.projections.reserve_ids(), so every layer that reads the reserve list sees it: the
QB-out penalty (injury_impact, keyed on the regular starter), the unit injury deltas, the
projections' available-QB lookup, the injury panel and the depth-chart pills. When nflverse
catches up the override is simply redundant; it never has to be removed, but --clear exists.

File: data/processed/availability_overrides.json — git-managed ("curated"), so a committed file
ships with every deploy and dashboard/seed.py syncs it onto the volume.
    {"2026": {"00-0034796": {"name": "Lamar Jackson", "team": "BAL", "label": "IR",
                             "from_week": 9, "to_week": null, "note": "placed on IR 2026-11-03"}}}
from_week..to_week inclusive; to_week null = the rest of the season. A player is pinned for the
week being predicted = the season's first week with an unplayed game.

CLI:
    python -m ml.availability --season 2026 --week 9 --out "Lamar Jackson:IR" ["Name:Out" ...]
    python -m ml.availability --season 2026 --week 9 --until 12 --out "Name:IR"
    python -m ml.availability --season 2026 --show
    python -m ml.availability --season 2026 --clear "Lamar Jackson"
Names resolve against the current depth chart, then the roster release (exact, case-insensitive).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PATH = ROOT / "data" / "processed" / "availability_overrides.json"
_WEEK_CACHE = {}


def load_all() -> dict:
    if PATH.exists():
        try:
            return json.loads(PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(allv: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(allv, indent=1, sort_keys=True), encoding="utf-8")


def predicted_week(season: int) -> int:
    """The week being predicted: the season's first regular-season week with an unplayed game
    (19 once the season is complete)."""
    global _WEEK_CACHE
    if _WEEK_CACHE is None:                      # dashboard.clear_caches() resets module caches to None
        _WEEK_CACHE = {}
    if season in _WEEK_CACHE:
        return _WEEK_CACHE[season]
    wk = 19
    try:
        s = pd.read_parquet(RAW / "schedules.parquet", columns=["season", "week", "home_score", "game_type"])
        s = s[(s["season"] == season) & (s["week"] <= 18)]
        if "game_type" in s.columns:
            s = s[s["game_type"].fillna("REG").str.upper().eq("REG")]
        open_ = s[s["home_score"].isna()]
        if not open_.empty:
            wk = int(open_["week"].min())
    except Exception:
        pass
    _WEEK_CACHE[season] = wk
    return wk


def clear_cache() -> None:
    global _WEEK_CACHE
    _WEEK_CACHE = {}


def active(season: int, week: int | None = None) -> dict:
    """{gsis_id: {"label", "team", "name"}} pinned for `week` (default: the week being predicted)."""
    rows = load_all().get(str(season)) or {}
    if not rows:
        return {}
    wk = predicted_week(season) if week is None else int(week)
    out = {}
    for gid, r in rows.items():
        lo, hi = int(r.get("from_week") or 1), r.get("to_week")
        if lo <= wk and (hi is None or wk <= int(hi)):
            out[str(gid)] = {"label": str(r.get("label") or "OUT"), "team": r.get("team"), "name": r.get("name")}
    return out


def _resolve(name: str):
    """(gsis_id, canonical name, team) for a player name, from the depth chart then the roster."""
    key = name.strip().lower()
    for path, idc, nmc, tc in ((RAW / "depth_2026_current.parquet", "gsis_id", "player_name", "team"),
                               (RAW / "rosters_2026.parquet", "player_id", "player_name", "team")):
        try:
            d = pd.read_parquet(path, columns=[idc, nmc, tc]).dropna(subset=[idc, nmc])
        except Exception:
            continue
        hit = d[d[nmc].str.strip().str.lower() == key].drop_duplicates(idc)
        if len(hit) == 1:
            r = hit.iloc[0]
            return str(r[idc]), str(r[nmc]), str(r[tc])
        if len(hit) > 1:
            raise SystemExit(f"'{name}' matches {len(hit)} players: {hit[[idc, nmc, tc]].to_dict('records')}")
    raise SystemExit(f"'{name}' not found on the depth chart or the roster release")


def pin(season: int, week: int, tokens: list, until: int | None = None, note: str = "") -> dict:
    """NAME:LABEL tokens → overrides from `week` (to `until`, inclusive, or the rest of the season)."""
    allv = load_all()
    rows = allv.setdefault(str(season), {})
    done = {}
    for tok in tokens:
        name, _, label = tok.rpartition(":")
        if not name:
            name, label = tok, "OUT"
        gid, nm, team = _resolve(name)
        rows[gid] = {"name": nm, "team": team, "label": label.upper(), "from_week": int(week),
                     "to_week": int(until) if until is not None else None, "note": note}
        done[gid] = rows[gid]
    _save(allv)
    return done


def unpin(season: int, names: list) -> list:
    allv = load_all()
    rows = allv.get(str(season)) or {}
    gone = []
    for name in names:
        gid, _, _ = _resolve(name)
        if gid in rows:
            gone.append(rows.pop(gid))
    _save(allv)
    return gone


def main():
    ap = argparse.ArgumentParser(description="Pin a player as unavailable ahead of the nflverse files")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None, help="first week the pin applies (default: the week being predicted)")
    ap.add_argument("--until", type=int, default=None, help="last week the pin applies (default: rest of season)")
    ap.add_argument("--out", nargs="*", default=[], help='NAME:LABEL tokens, e.g. "Lamar Jackson:IR"')
    ap.add_argument("--note", default="")
    ap.add_argument("--clear", nargs="*", default=[], help="names to unpin")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.out:
        wk = a.week if a.week is not None else predicted_week(a.season)
        for gid, r in pin(a.season, wk, a.out, a.until, a.note).items():
            print(f"pinned {r['name']} ({r['team']}) {r['label']} from week {r['from_week']}"
                  f"{' to ' + str(r['to_week']) if r['to_week'] else ' (rest of season)'}  [{gid}]")
    if a.clear:
        for r in unpin(a.season, a.clear):
            print(f"unpinned {r['name']} ({r['team']})")
    if a.show or not (a.out or a.clear):
        rows = load_all().get(str(a.season)) or {}
        wk = predicted_week(a.season)
        print(f"{len(rows)} override(s) for {a.season}; predicting week {wk}")
        for gid, r in rows.items():
            on = gid in active(a.season)
            print(f"  {'*' if on else ' '} {r['name']:24} {r['team']:4} {r['label']:5} wk {r['from_week']}-{r['to_week'] or 'end'}  {r.get('note', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
