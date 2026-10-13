"""
pff_facets.py  —  in-season PFF unit reports (the premium facet tables) → parquets
=================================================================================
The team-profile "PFF unit reports" card reads nine facet tables. They were first built
from the PRESEASON facets (pff_preseason_*.parquet, weeks -1..-5). This script writes the
REGULAR-SEASON equivalents, pff_season_*.parquet, which the dashboard prefers when present
(dashboard/server.api_pff_units); the preseason files stay as the fallback.

Weekly flow (needs the logged-in Browser pane on premium.pff.com):
  Since 2026-10-07 premium's API answers 403 (invalid_client_token) to any fetch that is not
  the app's own, so the data is read from the app's OWN calls instead:
  1. On /nfl/positions/2026/REGPO install a recorder that keeps the response text of every
     XHR/fetch to /api/v1/facet/* (patch XMLHttpRequest.prototype.open + window.fetch; store
     in window.__resp keyed by URL). No headers, no tokens — just what the page received.
  2. Click, through the app's own router (anchor.click(), not a navigation), the ALL link of
     each of the nine reports: passing, passing-pressure, rushing, receiving,
     offense-pass-blocking, offense-run-blocking, defense-pass-rush, defense-coverage,
     defense. Each click makes the app fetch that facet for the season to date.
  3. Return one facet per call as JSON.stringify({facet: rows}) padded with '#' so each
     lands in a tool-results file (the bodies are 50 KB – 1.4 MB).
  4. python pff_facets.py <file1> <file2> ... --season 2026 --through 5
  5. railway up   (the pff_*.parquet files ship with the deploy; git-ignored)
  Rows are the season-to-date view (the app requests every regular + post-season week),
  which equals "through the last week played".

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
    ap.add_argument("capture", nargs="+", help="one capture with all facets, or one file per facet")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--through", type=int, required=True, help="last week included")
    a = ap.parse_args()
    data = {}
    for c in a.capture:                                  # several per-facet captures merge into one dict
        data.update(read_chunk(Path(c)))
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
