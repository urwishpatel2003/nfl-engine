# PFF grades (subscriber data — stays local)

Everything in this folder except this README is **git-ignored**: PFF grades and
your session cookie are licensed/private and must never be committed (the repo
is public).

## Easiest path: fetch all squads at once (recommended)

1. Log into premium.pff.com in your browser.
2. DevTools (F12) → Network → reload any pff.com page → click the first request
   → Request Headers → copy the whole `Cookie:` header value.
3. Paste it into `cookie.txt` in this folder (create the file).
4. From the repo root:

   ```
   python pff_fetch.py
   ```

One JSON call per team pulls every player's grade fields (offense, defense,
pass, run, pass rush, coverage, run defense, pass/run block, receiving).
If grades come back locked, the cookie expired — repeat steps 1–3.

## Fallback: manual CSV exports

premium.pff.com → Premium Stats → Export CSV for Passing / Rushing / Receiving /
Offense Blocking / Defense / Special Teams → drop the files here → run
`python pff_import.py` (filenames don't matter; detection is column-based).

Both paths write `data/processed/pff_grades.parquet` (also ignored). Restart
the dashboard and every matched player on the Team Profile depth charts shows a
gold **PFF** badge.
