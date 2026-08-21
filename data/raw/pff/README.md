# PFF grades (subscriber data — stays local)

Everything in this folder except this README is **git-ignored**: PFF grades are
licensed subscriber data and must never be committed or redistributed (the repo
is public). The output `data/processed/pff_grades.parquet` is ignored too.

## Primary path: browser-session fetch (via Claude Code)

Ask Claude to "refresh my PFF grades". The flow:
1. Claude opens pff.com in the in-app Browser pane.
2. **You** log in there yourself (Claude never touches credentials).
3. Claude pulls all 32 team rosters through the page's own authenticated
   session (`/api/teams/{1-32}/roster?league=nfl` — every grade field, ~40s)
   and rebuilds `pff_grades.parquet`.

Why not a saved cookie? PFF's session JWT rotates every ~60 seconds, so any
copied cookie is stale before a fetch finishes — `pff_fetch.py` (the cookie-file
approach) survives only for the handful of publicly-teased grades. The live
browser session refreshes its own token, which is what makes this work.

## Fallback: manual CSV exports (no Claude needed)

premium.pff.com → Premium Stats → Export CSV for Passing / Rushing / Receiving /
Offense Blocking / Defense / Special Teams → drop the files here → run
`python pff_import.py` (filenames don't matter; detection is column-based).

Either way, restart the dashboard and matched players on the Team Profile depth
charts show a gold **PFF** badge (PFF 0–100 scale) next to our rating.
