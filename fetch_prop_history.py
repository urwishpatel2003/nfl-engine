"""
fetch_prop_history.py  —  closing player-prop lines from The Odds API's HISTORICAL endpoints
=============================================================================================
Pulls, for each regular-season game, the player-prop board as it stood HOURS_BEFORE kickoff
(default 1 h → the closing line) and appends it to data/raw/props_history.parquet, one row per
(bookmaker, market, player, side). backtest_props.py section D turns that into the real-line
backtest (hit rate / units by edge, measured anchor weights).

What the API offers (docs, 2026-10-20): historical snapshots at 5-minute intervals; player
props ("additional markets") only from 2023-05-03, i.e. the 2023 season onward; PAID plans
only. Cost: the historical events listing is 1 credit; historical event odds cost
    10 × (unique markets returned) × (regions)
so our 11 markets in one region are ≤ 110 credits per game, ~30k per season, ~90k for 2023-25.
`--dry-run` counts the games and prints the ceiling before anything is spent.

Credentials: ODDS_API_KEY from the environment only (never in the repo, never pasted in chat):
    PowerShell:  $env:ODDS_API_KEY = "<key>"   then   python fetch_prop_history.py --seasons 2025 --dry-run
Resume-safe: games already on file are skipped and the file is saved after every game, so a
run can be stopped and restarted. `--max-credits N` stops when the account's remaining credits
drop below N.

Usage
  python fetch_prop_history.py --seasons 2025 --dry-run
  python fetch_prop_history.py --seasons 2024 2025 --weeks 1 2 3
  python fetch_prop_history.py --seasons 2023 2024 2025 --max-credits 5000
"""

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from ml.odds import _MKT, _REV, _BASE, _SPORT          # noqa: E402  (same market map as the live board)
from ml.ledger import kickoff_utc                       # noqa: E402

RAW = Path(__file__).parent / "data" / "raw"
OUT = RAW / "props_history.parquet"
COLS = ["season", "week", "game_id", "home", "away", "kickoff_utc", "snapshot_ts", "event_id",
        "bookmaker", "market", "player", "side", "line", "price"]


def _get(url):
    r = requests.get(url, timeout=30)
    if r.status_code == 404:                            # no snapshot / event → nothing to store
        return None, r.headers.get("x-requests-remaining")
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def _names():
    ti = pd.read_parquet(RAW / "team_info.parquet", columns=["team_abbr", "team_name"]).drop_duplicates("team_abbr")
    return dict(zip(ti["team_abbr"], ti["team_name"]))


def _match_event(events, home_full, away_full):
    def nick(s):
        return str(s).split()[-1].lower()
    for e in events or []:
        if e.get("home_team") == home_full and e.get("away_team") == away_full:
            return e
    for e in events or []:                              # nickname fallback (Washington, LA naming drift)
        if nick(e.get("home_team")) == nick(home_full) and nick(e.get("away_team")) == nick(away_full):
            return e
    return None


def _rows(data, g, snap):
    out = []
    for bk in (data or {}).get("bookmakers", []):
        for mk in bk.get("markets", []):
            ourm = _REV.get(mk.get("key"))
            if not ourm:
                continue
            for oc in mk.get("outcomes", []):
                player = oc.get("description") or oc.get("name")
                side = (oc.get("name") or "").lower()
                if ourm == "anytime_td":
                    side = "yes"
                out.append({"season": int(g["season"]), "week": int(g["week"]), "game_id": g["game_id"],
                            "home": g["home_team"], "away": g["away_team"], "kickoff_utc": g["ko"].isoformat(),
                            "snapshot_ts": snap, "event_id": data.get("id"), "bookmaker": bk.get("key"),
                            "market": ourm, "player": player, "side": side,
                            "line": oc.get("point"), "price": oc.get("price")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--weeks", nargs="*", type=int, default=None)
    ap.add_argument("--hours-before", type=float, default=1.0, help="snapshot = kickoff − this (closing line)")
    ap.add_argument("--dry-run", action="store_true", help="count games and the credit ceiling; spend nothing")
    ap.add_argument("--max-credits", type=int, default=0, help="stop when the account's remaining credits fall below this")
    a = ap.parse_args()

    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["season"].isin(a.seasons) & (s["game_type"] == "REG")].copy()
    if a.weeks:
        s = s[s["week"].isin(a.weeks)]
    s["ko"] = [kickoff_utc(d, t) for d, t in zip(s["gameday"], s["gametime"])]
    s = s[s["ko"].notna()].sort_values(["season", "week", "ko"])
    done = set()
    if OUT.exists():
        done = set(pd.read_parquet(OUT, columns=["game_id"])["game_id"].unique())
    todo = s[~s["game_id"].isin(done)]
    print(f"{len(s)} games selected, {len(done & set(s['game_id']))} already on file, {len(todo)} to fetch; "
          f"credit ceiling ≈ {len(todo) * (10 * len(_MKT) + 1):,} (10 × {len(_MKT)} markets + 1 per game)")
    if a.dry_run:
        return
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY is not set in this shell")
    names = _names()
    frames = [pd.read_parquet(OUT)] if OUT.exists() else []
    n_rows = 0
    for _, g in todo.iterrows():
        snap_dt = g["ko"] - timedelta(hours=a.hours_before)
        snap = snap_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            ev, remaining = _get(f"{_BASE}/historical/sports/{_SPORT}/events?apiKey={key}&date={snap}")
            events = (ev or {}).get("data", [])
            e = _match_event(events, names.get(g["home_team"], g["home_team"]), names.get(g["away_team"], g["away_team"]))
            if not e:
                print(f"  {g['game_id']}: no event at {snap} ({len(events)} listed)")
                continue
            markets = ",".join(_MKT.values())
            data, remaining = _get(f"{_BASE}/historical/sports/{_SPORT}/events/{e['id']}/odds?apiKey={key}"
                                   f"&regions=us&oddsFormat=american&markets={markets}&date={snap}")
            rows = _rows((data or {}).get("data"), g, (data or {}).get("timestamp", snap))
            if rows:
                frames.append(pd.DataFrame(rows, columns=COLS))
                pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
                n_rows += len(rows)
            print(f"  {g['game_id']}: {len(rows)} rows, credits left {remaining}")
            if a.max_credits and remaining is not None and int(remaining) < a.max_credits:
                print(f"stopping: remaining credits {remaining} < --max-credits {a.max_credits}")
                break
            time.sleep(0.2)
        except requests.HTTPError as ex:
            code = ex.response.status_code if ex.response is not None else "?"
            print(f"  {g['game_id']}: HTTP {code} — {ex.response.text[:160] if ex.response is not None else ''}")
            if code in (401, 402, 422, 429):
                break
    print(f"wrote {n_rows:,} rows → {OUT}")


if __name__ == "__main__":
    main()
