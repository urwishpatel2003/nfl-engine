r"""
pff_team_table.py  —  PFF team grades from the premium team table as RENDERED in the browser
===========================================================================================
Since 2026-10-07 premium.pff.com's API refuses direct calls (403 invalid_client_token), so the
weekly team-overview pull reads the table the premium page renders instead. In the Browser
pane, on https://premium.pff.com/nfl/teams/2026/REGPO (logged in), capture the rows:

    const L=document.body.innerText.split('\n').map(s=>s.trim()).filter(Boolean);
    const i0=L.indexOf('Arizona Cardinals'); const names=[]; let i=i0;
    while(i<L.length && !/^\d+ - \d+/.test(L[i])){ names.push(L[i]); i++; }
    const rows=[]; while(i<L.length && rows.length<names.length){
      if(/^\d+ - \d+/.test(L[i])){ rows.push({name:names[rows.length], record:L[i], pf:+L[i+1], pa:+L[i+2],
        g:L.slice(i+3,i+16).map(Number)}); i+=16; } else i++; }
    JSON.stringify({names:names.length, rows})+"\n"+"#".repeat(160000)   // padding → tool-results file

then:  python pff_team_table.py <capture> --season 2026

Column order of the rendered table (verified against the API's own JSON on 2026-10-07):
OVER OFF PASS PBLK RECV RUN RBLK DEF RDEF TACK PRSH COV SPEC. Abbreviation / franchise_id are
carried over from the previous pff_team_grades.parquet by team name, so the schema that
dashboard/server.api_pff_compare reads is unchanged.
"""

import argparse
from pathlib import Path

import pandas as pd

from pff_assemble import read_chunk
from pff_import import TEAM_MAP

OUT = Path(__file__).parent / "data" / "processed" / "pff_team_grades.parquet"
G = ["grades_overall", "grades_offense", "grades_pass", "grades_pass_block", "grades_pass_route", "grades_run",
     "grades_run_block", "grades_defense", "grades_run_defense", "grades_tackle", "grades_pass_rush_defense",
     "grades_coverage_defense", "grades_misc_st"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    rows = read_chunk(Path(a.capture))["rows"]
    old = pd.read_parquet(OUT)
    ids = old.set_index("name")[["abbreviation", "franchise_id"]]
    missing = [r["name"] for r in rows if r["name"] not in ids.index]
    assert len(rows) == 32 and not missing, f"team rows {len(rows)}, unknown names {missing}"
    recs = []
    for r in rows:
        w, l = [int(x) for x in r["record"].split(" - ")[:2]]
        rec = {"abbreviation": ids.loc[r["name"], "abbreviation"], "franchise_id": int(ids.loc[r["name"], "franchise_id"]),
               "name": r["name"], "wins": w, "losses": l, "ties": 0,
               "points_scored": r["pf"], "points_allowed": r["pa"]}
        rec.update(dict(zip(G, r["g"])))
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["team"] = df["abbreviation"].astype(str).str.upper().replace(TEAM_MAP)
    df["season"] = a.season
    df["pff_rank"] = df["grades_overall"].rank(ascending=False, method="min").astype(int)
    df = df[list(old.columns)].sort_values("pff_rank")
    df.to_parquet(OUT, index=False)
    print(df[["team", "wins", "losses", "grades_overall", "pff_rank"]].head(8).to_string(index=False))
    print(f"saved {len(df)} teams -> {OUT}")


if __name__ == "__main__":
    main()
