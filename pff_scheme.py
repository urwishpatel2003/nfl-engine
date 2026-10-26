"""
pff_scheme.py  —  PFF man/zone coverage-scheme tables (the coverage-clash inputs) → parquets
============================================================================================
ml.matchup_context._coverage_clash nudges a prediction (±1.5 pts max) by how man-heavy the
defence is (league norm ~27.6%) × how much better the offence's top receivers grade against
man than zone. Both sides come from two premium facets:

    facet/receiving/scheme          -> data/processed/pff_scheme_receiving.parquet
    facet/defense/coverage_scheme   -> data/processed/pff_scheme_defense.parquet

The first build (2026-08-24) pulled the FULL 2025 season straight from the API. Since
2026-10-07 premium.pff.com refuses direct calls (403 invalid_client_token), so in-season the
rows are read from the app's OWN calls, exactly like pff_facets.py: on
/nfl/positions/2026/REGPO install the XHR/fetch recorder, click (through the app's router) the
ALL link of "receiving-scheme" and "defense-coverage-scheme", then return each facet as
JSON.stringify({facet: rows}) padded with '#'. Then:

    python pff_scheme.py <capture1> [<capture2>] --season 2026 --through 7

Output rows keep every column PFF returns plus `team` (nflverse code via pff_import.TEAM_MAP),
`season` and `through_week` — the same shape the 2025 files had, so _coverage_scheme reads
them unchanged. The file being replaced is kept as pff_scheme_<facet>_<its season>.parquet
when it holds a different season (the prior season stays available for a future blend).

In-season the tables are SEASON-TO-DATE (the app requests every week); the receiver side
still needs `_COV_MIN_ROUTES` (50) man+zone routes, which a starter clears by week 3.
Licensed subscriber data — every pff_*.parquet is git-ignored and ships only in the deploy.
"""

import argparse
from pathlib import Path

import pandas as pd

from pff_assemble import read_chunk
from pff_import import TEAM_MAP

PROC = Path(__file__).parent / "data" / "processed"

FACETS = {
    "receiving/scheme":        "receiving",
    "defense/coverage_scheme": "defense",
}


def main():
    ap = argparse.ArgumentParser(description="PFF coverage-scheme facets (browser captures) -> parquets")
    ap.add_argument("capture", nargs="+", help="one capture with both facets, or one file per facet")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--through", type=int, required=True, help="last week included")
    a = ap.parse_args()
    data = {}
    for c in a.capture:
        data.update(read_chunk(Path(c)))
    for facet, stem in FACETS.items():
        rows = data.get(facet) or []
        if not rows:
            print(f"  {facet}: no rows — skipped")
            continue
        d = pd.DataFrame(rows)
        d["team"] = d["team_name"].astype(str).str.upper().replace(TEAM_MAP)
        d["season"] = a.season
        d["through_week"] = a.through
        out = PROC / f"pff_scheme_{stem}.parquet"
        if out.exists():                                   # keep the season being replaced
            old = pd.read_parquet(out)
            old_season = int(old["season"].iloc[0]) if "season" in old.columns and len(old) else None
            if old_season is not None and old_season != a.season:
                keep = PROC / f"pff_scheme_{stem}_{old_season}.parquet"
                if not keep.exists():
                    old.to_parquet(keep, index=False)
                    print(f"  kept the {old_season} table as {keep.name}")
        d.to_parquet(out, index=False)
        # the two numbers the clash actually uses, as a sanity line
        if stem == "defense":
            g = d.groupby("team").agg(man=("man_snap_counts_coverage", "sum"),
                                      zone=("zone_snap_counts_coverage", "sum"))
            rate = g["man"] / (g["man"] + g["zone"]).clip(lower=1)
            print(f"  {facet}: {len(d)} rows, {d['team'].nunique()} teams, league man rate "
                  f"{rate.mean():.1%} (min {rate.min():.0%} {rate.idxmin()}, max {rate.max():.0%} {rate.idxmax()})")
        else:
            routes = d["man_routes"].fillna(0) + d["zone_routes"].fillna(0)
            print(f"  {facet}: {len(d)} rows, {d['team'].nunique()} teams, "
                  f"{int((routes >= 50).sum())} receivers with 50+ man+zone routes")
        print(f"  saved -> {out.name}")


if __name__ == "__main__":
    main()
