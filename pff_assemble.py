"""
pff_assemble.py  —  turn browser-fetched PFF chunks into the two grade parquets
================================================================================
Part of the weekly in-season PFF refresh (see data/raw/pff/README.md). The flow:

  1. User: "refresh my PFF grades" → Claude opens pff.com in the Browser pane
     (log in if it asks) and pulls, same-origin from the logged-in tab:
       - player rosters:  /api/teams/{1..32}/roster?league=nfl   (8 teams/call,
         slim fields incl. meets_snap_minimum as "q", returns land in
         tool-results/*.txt via the oversized-output capture)
       - team overview:  premium.pff.com /api/v1/teams/overview?league=nfl
         &season=<S>&week=1,...  (navigate the tab to premium.pff.com first —
         cross-origin fetch is CORS-blocked)
  2. Claude runs this script over the captured chunk files:
       python pff_assemble.py --players <chunk1> <chunk2> ... --teams <file> --season 2026
  3. Output: data/processed/pff_grades.parquet + pff_team_grades.parquet
     (git-ignored — licensed). Then push to the live site (railway ssh path, or
     the PFF tab's upload box from any device) and delete the chunk files.

Chunk file format: the tool-result wrapper [{type, text}] where text is the
JSON-quoted JS return, possibly followed by a newline + '#' padding and/or a
"(captured at origin)" trailer — raw_decode handles all of it.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from pff_fetch import build, OUT_PATH
from pff_import import TEAM_MAP

TEAM_OUT = Path(__file__).parent / "data" / "processed" / "pff_team_grades.parquet"


def read_chunk(path: Path):
    """Parse one captured tool-result file down to the JSON payload it carries."""
    raw = path.read_text(encoding="utf-8")
    try:
        wrapper = json.loads(raw)
        text = wrapper[0]["text"] if isinstance(wrapper, list) else wrapper["text"]
    except Exception:
        text = raw                                     # plain file (e.g. saved by hand)
    data, _ = json.JSONDecoder().raw_decode(text)
    if isinstance(data, str):                          # JS string return → unquote once more
        data, _ = json.JSONDecoder().raw_decode(data.split("\n", 1)[0]) \
            if data.lstrip().startswith(("[", "{")) else (json.loads(data.split("\n", 1)[0]), None)
    return data


def assemble_players(paths):
    players, seen = [], set()
    for p in paths:
        chunk = read_chunk(Path(p))
        fresh = [x for x in chunk if x["id"] not in seen]
        seen.update(x["id"] for x in fresh)
        players.extend(fresh)
        print(f"  {Path(p).name}: {len(chunk)} rows, {len(fresh)} new")
    out = build(players)
    graded = int(out["pff_grade"].notna().sum())
    qual = int((out.get("qualifies", True) & out["pff_grade"].notna()).sum()) if "qualifies" in out else graded
    print(f"players: {len(out)} | graded: {graded} | meet snap minimum: {qual}")
    assert graded > 500, "too few grades — PFF session probably dropped mid-fetch; refetch"
    out.to_parquet(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")


def assemble_teams(path, season):
    data = read_chunk(Path(path))
    df = pd.DataFrame(data)
    df["team"] = df["abbreviation"].astype(str).str.upper().replace(TEAM_MAP)
    df["season"] = season
    df["pff_rank"] = df["grades_overall"].rank(ascending=False, method="min").astype(int)
    df.sort_values("pff_rank").to_parquet(TEAM_OUT, index=False)
    print(f"saved {len(df)} teams -> {TEAM_OUT}")


def main():
    ap = argparse.ArgumentParser(description="Assemble browser-fetched PFF chunks into parquets")
    ap.add_argument("--players", nargs="*", default=[], help="captured player-roster chunk files")
    ap.add_argument("--teams", default=None, help="captured team-overview file")
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    if args.players:
        assemble_players(args.players)
    if args.teams:
        assemble_teams(args.teams, args.season)
    if not args.players and not args.teams:
        print("nothing to do — pass --players and/or --teams")


if __name__ == "__main__":
    main()
