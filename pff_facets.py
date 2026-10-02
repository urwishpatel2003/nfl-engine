"""
pff_facets.py  —  in-season PFF unit reports (the premium facet tables) → parquets
=================================================================================
The team-profile "PFF unit reports" card reads nine facet tables. They were first built
from the PRESEASON facets (pff_preseason_*.parquet, weeks -1..-5). This script writes the
REGULAR-SEASON equivalents, pff_season_*.parquet, which the dashboard prefers when present
(dashboard/server.api_pff_units); the preseason files stay as the fallback.

Weekly flow (needs the logged-in Browser pane on premium.pff.com):
  1. In the pane, fetch each facet for the weeks played, e.g.
       /api/v1/facet/<facet>?league=nfl&season=2026&week=1,2,3
     for the facets in FACETS below, collect {facet: rows} and return it padded with
     '#' so it lands in a tool-results file.
  2. python pff_facets.py <capture> --season 2026 --through 3
  3. railway up   (the pff_*.parquet files ship with the deploy; git-ignored)

Rows keep every column PFF returns (same shape as the preseason files) plus `team`
(nflverse code via pff_import.TEAM_MAP), `season` and `through_week`.
"""

import argparse
from pathlib import Path

import pandas as pd

from pff_assemble import read_chunk
from pff_import import TEAM_MAP

PROC = Path(__file__).parent / "data" / "processed"

# facet endpoint -> file stem (matches the preseason stems the unit spec reads)
FACETS = {
    "passing/summary":        "passing",
    "passing/pressure":       "qb_pressure",
    "rushing/summary":        "rushing",
    "receiving/summary":      "receiving",
    "offense/pass_blocking":  "pass_blocking",
    "offense/run_blocking":   "run_blocking",
    "defense/pass_rush":      "pass_rush",
    "defense/coverage":       "coverage",
    "defense/summary":        "defense",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--through", type=int, required=True, help="last week included")
    a = ap.parse_args()
    data = read_chunk(Path(a.capture))
    for facet, stem in FACETS.items():
        rows = data.get(facet) or []
        if not rows:
            print(f"  {facet}: no rows — skipped")
            continue
        d = pd.DataFrame(rows)
        d["team"] = d["team_name"].astype(str).str.upper().replace(TEAM_MAP) if "team_name" in d.columns else None
        d["season"] = a.season
        d["through_week"] = a.through
        path = PROC / f"pff_season_{stem}.parquet"
        d.to_parquet(path, index=False)
        print(f"  {facet}: {len(d)} rows -> {path.name}")


if __name__ == "__main__":
    main()
