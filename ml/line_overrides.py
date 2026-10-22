"""
ml/line_overrides.py  —  pinned spreads for a week (contest lines) that the picks are made against
================================================================================================
The owner enters the Circa Million contest, whose point spreads are fixed on Wednesday and
differ from the live market. When a week has an override, the Schedule & Picks slate, the
top-5 ranking, the picks ledger and the social post all price against THESE numbers, and
neither the live Odds-API line nor the nflverse line replaces them (line_source = "circa").

File: data/processed/line_overrides.json — git-managed ("curated"), so a committed file ships
with every deploy and dashboard/seed.py syncs it onto the volume.
    {"2026": {"7": {"2026_07_NE_CHI": {"spread": 3.0, "source": "circa"}, ...}}}
Spreads are stored in nflverse convention: HOME perspective, POSITIVE = home favoured
(see CLAUDE.md, "ATS spread convention").

CLI — tokens are TEAM:line in the TEAM'S OWN perspective, exactly as a contest sheet prints
them (one token per game is enough; the other side is implied):
    python -m ml.line_overrides --season 2026 --week 7 --source circa CHI:-3 NO:+3 CAR:-3 ...
    python -m ml.line_overrides --season 2026 --week 7 --show
    python -m ml.line_overrides --season 2026 --week 7 --clear
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
PATH = ROOT / "data" / "processed" / "line_overrides.json"


def load_all() -> dict:
    if PATH.exists():
        try:
            return json.loads(PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def get(season: int, week: int) -> dict:
    """{game_id: {"spread": float, "source": str}} for the week, else {}."""
    return (load_all().get(str(season)) or {}).get(str(week)) or {}


def save(season: int, week: int, rows: dict) -> None:
    allv = load_all()
    allv.setdefault(str(season), {})[str(week)] = rows
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(allv, indent=1, sort_keys=True), encoding="utf-8")


def _week_games(season: int, week: int) -> pd.DataFrame:
    s = pd.read_parquet(ROOT / "data" / "raw" / "schedules.parquet", columns=["game_id", "season", "week", "home_team", "away_team"])
    return s[(s["season"] == season) & (s["week"] == week)]


def set_lines(season: int, week: int, tokens: list, source: str = "circa") -> dict:
    """TEAM:line tokens (the team's own line) → home-perspective overrides for that week."""
    g = _week_games(season, week)
    by_team = {}
    for _, r in g.iterrows():
        by_team[r["home_team"]] = (r["game_id"], True)
        by_team[r["away_team"]] = (r["game_id"], False)
    rows = get(season, week)
    for tok in tokens:
        team, _, val = tok.upper().partition(":")
        if team not in by_team:
            sys.exit(f"{team} does not play in {season} week {week}")
        try:
            v = float(val.replace("½", ".5").replace("PK", "0"))
        except ValueError:
            sys.exit(f"bad line in {tok!r} (use TEAM:-3 or TEAM:+2.5)")
        gid, is_home = by_team[team]
        home_line = v if is_home else -v
        spread = -home_line                       # nflverse: positive = home favoured
        prev = rows.get(gid)
        if prev is not None and abs(prev["spread"] - spread) > 1e-9:
            print(f"  {gid}: both sides given and they disagree ({prev['spread']:+g} vs {spread:+g}) — last one wins")
        rows[gid] = {"spread": round(spread, 1), "source": source}
    save(season, week, rows)
    return rows


def show(season: int, week: int) -> None:
    g = _week_games(season, week).set_index("game_id")
    rows = get(season, week)
    if not rows:
        print(f"no overrides for {season} week {week}")
        return
    for gid, r in sorted(rows.items()):
        h, a = g.loc[gid, "home_team"], g.loc[gid, "away_team"]
        sp = r["spread"]
        fav = f"{h} {-sp:+g}" if sp > 0 else (f"{a} {sp:+g}" if sp < 0 else "PK")
        print(f"  {gid:16s}  {a}@{h}:  {fav:10s} (home spread_line {sp:+g}, {r['source']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--source", default="circa")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("tokens", nargs="*", help="TEAM:line, the team's own number, e.g. CHI:-3 NO:+3")
    a = ap.parse_args()
    if a.clear:
        save(a.season, a.week, {})
        print(f"cleared {a.season} week {a.week}")
        return
    if a.tokens:
        set_lines(a.season, a.week, a.tokens, a.source)
    show(a.season, a.week)


if __name__ == "__main__":
    main()
