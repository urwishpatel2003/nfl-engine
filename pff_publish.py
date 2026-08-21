"""
pff_publish.py  —  push the locally built PFF parquets to the private data repo
================================================================================
Final step of the weekly PFF refresh: copies data/processed/pff_grades.parquet
(+ pff_team_grades.parquet) into the sibling clone of the PRIVATE repo
(../nfl-pff-data), commits, and pushes. The hosted dashboard pulls from that
repo on its daily refresh (ml.refresh.pull_pff, env-gated by PFF_DATA_REPO /
PFF_DATA_TOKEN on Railway), so the live site updates itself — no manual upload.

The transfer repo must stay PRIVATE (licensed subscriber data).

Usage:  python pff_publish.py
"""

import shutil
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
DATA_REPO = ROOT.parent / "nfl-pff-data"
FILES = ["pff_grades.parquet", "pff_team_grades.parquet"]


def run(*cmd):
    r = subprocess.run(cmd, cwd=DATA_REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def main():
    if not (DATA_REPO / ".git").exists():
        raise SystemExit(f"private data repo clone not found at {DATA_REPO} — "
                         "git clone https://github.com/urwishpatel2003/nfl-pff-data.git there first")
    copied = []
    for f in FILES:
        src = PROC / f
        if src.exists():
            shutil.copy2(src, DATA_REPO / f)
            copied.append(f)
    if not copied:
        raise SystemExit("no PFF parquets in data/processed — build them first (pff_assemble.py)")
    if not run("git", "status", "--porcelain"):
        print("no changes — data repo already up to date")
        return
    run("git", "add", "-A")
    run("git", "commit", "-m", f"PFF grades {date.today().isoformat()}")
    run("git", "push", "origin", "HEAD")
    print(f"pushed {', '.join(copied)} -> nfl-pff-data (private)")
    print("The live site picks this up on its next daily refresh, or immediately via the ↻ Refresh tab.")


if __name__ == "__main__":
    main()
