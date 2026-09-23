"""
pff_prior.py  —  build LAST season's PFF grade table from the premium facet captures
====================================================================================
Why this exists: pff.com's roster endpoint serves only the CURRENT season's grades. Before
week 1 that meant full 2025 grades; from week 1 on it means 2026 season-to-date — after two
weeks, two games of film. The model weights PFF at 80%, so swapping a 17-game grade for a
2-game grade moved the roster-talent ranking wildly (GB 6th -> 28th, LA 14.9 -> 7.5) on
evidence nobody should trust yet. ml/squad blends the current grade toward this prior by
games played (the same g/(g+4) rule used everywhere in season).

Input: one captured tool-result file holding {facet: [rows]} for the 2025 regular season
from premium.pff.com /api/v1/facet/{defense/summary, passing/summary, receiving/summary,
rushing/summary, offense/blocking}?league=nfl&season=2025&week=1,...,18 (fields slimmed
in the browser: player_id, player, position, team_name, player_game_count, grades_*).

    python pff_prior.py <capture> --season 2025

Output: data/processed/pff_grades_<season>.parquet — same headline logic as pff_fetch.build
(grades_offense for offensive positions, grades_defense for defensive), keyed by pff_id.
Git-ignored (licensed).
"""

import argparse
from pathlib import Path

import pandas as pd

from pff_assemble import read_chunk
from pff_fetch import GRADE_FIELDS, OUT_PATH
from pff_import import TEAM_MAP, OFFENSE_POS, DEFENSE_POS
from ml.squad import _norm, _key

PROC = Path(__file__).parent / "data" / "processed"
_UNIT = {v: v for v in GRADE_FIELDS.values()} | {"grades_pass_route": "grades_pass_route"}


def build_prior(capture: Path, season: int) -> pd.DataFrame:
    data = read_chunk(capture)
    frames = []
    for facet, rows in data.items():
        if rows:
            f = pd.DataFrame(rows); f["facet"] = facet; frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["player_id"])
    # one row per player: keep every grade column's max-game facet value (a QB appears in
    # passing and rushing; a TE in receiving and blocking) — grades are the same PFF grade
    # wherever they appear, so first-non-null is fine
    gcols = [c for c in d.columns if c.startswith("grades_")]
    agg = {c: "first" for c in gcols}
    agg.update({"player": "first", "position": "first", "team_name": "first", "player_game_count": "max"})
    d = d.sort_values("player_game_count", ascending=False)
    g = d.groupby("player_id").agg(agg).reset_index()
    out = pd.DataFrame({
        "pff_id": g["player_id"].astype(int),
        "player": g["player"],
        "team": g["team_name"].astype(str).str.upper().replace(TEAM_MAP),
        "position": g["position"],
        "games": g["player_game_count"],
        "season": season,
    })
    out["nm"] = out["player"].map(_norm)
    out["key"] = out["player"].map(_key)
    for c in gcols:
        out[c] = pd.to_numeric(g[c], errors="coerce")

    def headline(row):
        pos = str(row.get("position", "")).upper()
        cands = (["grades_offense"] if pos in OFFENSE_POS else ["grades_defense"] if pos in DEFENSE_POS else [])
        for c in cands + ["grades_offense", "grades_defense"]:
            v = row.get(c)
            if v is not None and pd.notna(v):
                return float(v)
        return None
    out["pff_grade"] = out.apply(headline, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--season", type=int, default=2025)
    a = ap.parse_args()
    out = build_prior(Path(a.capture), a.season)
    path = PROC / f"pff_grades_{a.season}.parquet"
    out.to_parquet(path, index=False)
    print(f"{len(out)} players, {int(out.pff_grade.notna().sum())} graded -> {path}")
    print(out.groupby("position")["pff_grade"].mean().round(1).dropna().to_dict())


if __name__ == "__main__":
    main()
