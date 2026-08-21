"""
pff_import.py  —  import PFF Premium Stats CSV exports into one grades table
=============================================================================
PFF grades are subscriber-only data with no public API. The workflow is manual
export (allowed for personal subscriber use), then this importer:

  1. You (with a PFF+ subscription) export the Premium Stats tables as CSV:
       premium.pff.com → NFL → Premium Stats → pick a category → "Export CSV"
     Recommended set (one file each, any filename — detection is column-based):
       Passing · Rushing · Receiving · Offense Blocking · Defense · Special Teams
  2. Drop every CSV into  data/raw/pff/   (create it; git-ignored — licensed
     data must never be committed/redistributed).
  3. Run:  python pff_import.py            (optionally --dir/--out for testing)

Output: data/processed/pff_grades.parquet — one row per player:
  pff_id, player, nm (normalized), key (lastname+initial), team (nflverse code),
  position, games, every grades_* column found across the exports, and a single
  headline `pff_grade` (offense grade for offensive positions, defense grade for
  defenders, else best available) used by the dashboard player cards.

Matching downstream is name-based (PFF ids don't map to gsis ids): the dashboard
matches nm+team first, then the lastname+initial key — the same normalization
ml/squad.py uses everywhere else, so player cards stay coherent.

PFF team codes differ from nflverse for four teams (ARZ/BLT/CLV/HST) — mapped here.
"""

import argparse
from pathlib import Path

import pandas as pd

from ml.squad import _norm, _key

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw" / "pff"
OUT_PATH = ROOT / "data" / "processed" / "pff_grades.parquet"

# PFF → nflverse team codes (identity for everyone else)
TEAM_MAP = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}

# PFF position → which grade column is that player's headline number
OFFENSE_POS = {"QB", "HB", "FB", "WR", "TE", "T", "G", "C"}
DEFENSE_POS = {"DI", "ED", "LB", "CB", "S"}


def load_exports(src: Path) -> list:
    """Read every CSV in src that looks like a PFF export (has player + grades_*)."""
    frames = []
    for p in sorted(src.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  SKIP {p.name}: unreadable ({e})")
            continue
        gcols = [c for c in df.columns if c.startswith("grades_")]
        if "player" not in df.columns or not gcols:
            print(f"  SKIP {p.name}: not a PFF export (no player/grades_ columns)")
            continue
        keep = [c for c in ("player", "player_id", "position", "team_name",
                            "player_game_count") if c in df.columns] + gcols
        frames.append(df[keep].copy())
        print(f"  {p.name}: {len(df)} players, grades: {', '.join(gcols)}")
    return frames


def build(frames: list) -> pd.DataFrame:
    """Combine per-category exports into one row per player (first non-null per grade)."""
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"player_id": "pff_id", "team_name": "team",
                          "player_game_count": "games"})
    d["team"] = d["team"].astype(str).str.upper().replace(TEAM_MAP)
    d["nm"] = d["player"].map(_norm)
    d["key"] = d["player"].map(_key)
    # A player without a pff_id (shouldn't happen in real exports) falls back to nm+team.
    d["pid"] = d.get("pff_id", pd.Series(index=d.index)).fillna(d["nm"] + "@" + d["team"])

    gcols = [c for c in d.columns if c.startswith("grades_")]
    agg = {c: "first" for c in ("player", "nm", "key", "team", "position")}
    agg.update({c: "first" for c in gcols})
    if "games" in d.columns:
        agg["games"] = "max"
    # first non-null per column: sort so rows with more grades come first is unnecessary —
    # groupby.first() already skips NaN within each column.
    out = d.groupby("pid", as_index=False).agg(agg)

    def headline(row):
        pos = str(row.get("position", "")).upper()
        cands = []
        if pos in OFFENSE_POS:
            cands = ["grades_offense", "grades_pass", "grades_run"]
        elif pos in DEFENSE_POS:
            cands = ["grades_defense"]
        cands += ["grades_offense", "grades_defense", "grades_special_teams"]
        for c in cands:
            v = row.get(c)
            if v is not None and pd.notna(v):
                return float(v)
        return None

    out["pff_grade"] = out.apply(headline, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Import PFF Premium Stats CSV exports")
    ap.add_argument("--dir", type=Path, default=RAW_DIR, help="folder of PFF CSV exports")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="output parquet path")
    args = ap.parse_args()

    if not args.dir.exists():
        args.dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {args.dir} — drop your PFF Premium Stats CSV exports there and rerun.")
        return

    print(f"Scanning {args.dir} ...")
    frames = load_exports(args.dir)
    if not frames:
        print("No PFF exports found. Export CSVs from premium.pff.com → Premium Stats "
              "and drop them in that folder.")
        return

    out = build(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nSaved {len(out)} players → {args.out}")
    print(f"  with headline grade: {out.pff_grade.notna().sum()}")
    print(f"  teams: {out.team.nunique()}")
    print(out.groupby("position")["pff_grade"].mean().round(1).dropna().to_string())


if __name__ == "__main__":
    main()
