"""
pff_fetch.py  —  pull PFF player grades for all 32 squads via your PFF+ session
================================================================================
KNOWN LIMITATION (verified 2026-08-19): PFF's session JWT rotates every ~60s,
so a copied cookie is usually stale by fetch time — you get rosters but only the
publicly-teased grades (~1 player/team). The reliable path is the live
browser-session fetch driven from Claude Code (see data/raw/pff/README.md);
this script is kept for completeness / in case PFF's auth changes.

pff.com's team-roster endpoint (the one behind /nfl/teams/<slug>/<id>/roster)
returns every player's grade fields as JSON — but zeroed/"locked":"premium"
unless the request carries a logged-in PFF+ session. Grades are subscriber-only,
personal-use data: keep the output out of git (both paths are git-ignored).

ONE-TIME SETUP (your cookie stays local; never share or commit it):
  1. Log into premium.pff.com in your browser.
  2. Open DevTools (F12) → Network tab → reload any pff.com page → click the
     first request → Request Headers → copy the ENTIRE value of the `Cookie:`
     header (one long line).
  3. Paste it into  data/raw/pff/cookie.txt  (create the file; git-ignored).
  4. Run:  python pff_fetch.py
     Re-run whenever you want fresher grades; if the cookie expires you'll get
     a clear "grades locked" message — repeat steps 1-3.

Output: data/processed/pff_grades.parquet — same schema pff_import.py writes
(the CSV-export path still works as a fallback), so the dashboard player-card
badges light up either way.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from ml.squad import _norm, _key
from pff_import import TEAM_MAP, OFFENSE_POS, DEFENSE_POS

ROOT = Path(__file__).parent
COOKIE_PATH = ROOT / "data" / "raw" / "pff" / "cookie.txt"
OUT_PATH = ROOT / "data" / "processed" / "pff_grades.parquet"

API = "https://www.pff.com/api/teams/{tid}/roster?league=nfl"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# API grade field -> our grades_* column (same names the CSV exports use)
GRADE_FIELDS = {
    "offense": "grades_offense", "defense": "grades_defense",
    "pass": "grades_pass", "run": "grades_run",
    "pass_rush": "grades_pass_rush_defense", "run_defense": "grades_run_defense",
    "coverage": "grades_coverage_defense", "receiving": "grades_receiving",
    "pass_block": "grades_pass_block", "run_block": "grades_run_block",
}


def fetch_all(cookie: str, delay: float = 1.0) -> list:
    """One request per franchise id (1-32), polite delay between calls."""
    headers = dict(UA, Cookie=cookie)
    rows = []
    for tid in range(1, 33):
        r = requests.get(API.format(tid=tid), headers=headers, timeout=30)
        r.raise_for_status()
        players = r.json().get("team_players", [])
        rows.extend(players)
        team = players[0].get("team_name", tid) if players else tid
        print(f"  team {tid:2} ({team}): {len(players)} players")
        time.sleep(delay)
    return rows


def build(players: list) -> pd.DataFrame:
    d = pd.DataFrame(players)
    out = pd.DataFrame({
        "pff_id": d["id"],
        "player": d["name"],
        "team": d["team_name"].astype(str).str.upper().replace(TEAM_MAP),
        "position": d["position"],
    })
    out["nm"] = out["player"].map(_norm)
    out["key"] = out["player"].map(_key)
    # PFF's own qualifier (meets_snap_minimum, slimmed to "q" by the browser fetch) — grades
    # on a handful of snaps are real but noisy; comparisons should filter on this.
    qsrc = "q" if "q" in d.columns else "meets_snap_minimum"
    out["qualifies"] = d[qsrc].fillna(False).astype(bool) if qsrc in d.columns else True
    for src, col in GRADE_FIELDS.items():
        if src in d.columns:
            v = pd.to_numeric(d[src], errors="coerce")
            out[col] = v.where(v > 0)          # locked/no-snaps grades come through as 0

    def headline(row):
        pos = str(row.get("position", "")).upper()
        cands = (["grades_offense"] if pos in OFFENSE_POS
                 else ["grades_defense"] if pos in DEFENSE_POS else [])
        cands += ["grades_offense", "grades_defense"]
        for c in cands:
            v = row.get(c)
            if v is not None and pd.notna(v):
                return float(v)
        return None

    out["pff_grade"] = out.apply(headline, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch PFF grades for all squads (needs your PFF+ cookie)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between team requests")
    args = ap.parse_args()

    if not COOKIE_PATH.exists() or not COOKIE_PATH.read_text().strip():
        print(f"No cookie found. Paste your pff.com Cookie header into {COOKIE_PATH}")
        print("(see the setup steps in this file's docstring)")
        return

    cookie = COOKIE_PATH.read_text().strip()
    print("Fetching 32 team rosters from pff.com ...")
    players = fetch_all(cookie, args.delay)
    out = build(players)

    graded = int(out["pff_grade"].notna().sum())
    if graded == 0:
        print("\nEvery grade came back locked/zero — your cookie is missing, expired, or "
              "not a PFF+ session. Re-copy the Cookie header (docstring steps) and rerun.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nSaved {len(out)} players ({graded} with a headline grade) → {args.out}")
    print(out.groupby("position")["pff_grade"].mean().round(1).dropna().to_string())
    print("\nRestart the dashboard (or hit its Refresh) to see PFF badges on player cards.")


if __name__ == "__main__":
    main()
