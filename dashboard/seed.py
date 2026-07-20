"""
dashboard/seed.py — seed / sync the data/ volume from the image's baked-in copy.

On Railway a persistent Volume is mounted at /app/data, which shadows the committed data
that shipped in the image. We must both:
  1. populate an empty volume on first boot, AND
  2. keep git-managed ("curated") data files in sync on every later deploy — otherwise a new
     or edited committed file (e.g. data/coaching_2026.json) never reaches the server, because
     the volume was seeded once and would otherwise be skipped forever.

So on each boot we copy from data_seed → data:
  • everything, if the volume is empty (first boot);
  • otherwise: curated (git-managed) files are always synced from the image, while
    REFRESH-managed files (updated in place by ml/refresh.py) are preserved on the volume and
    only copied if missing.

Run before gunicorn:  python dashboard/seed.py && gunicorn ...
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SEED = ROOT / "data_seed"

# Files ml/refresh.py rewrites on the server — the volume copy is authoritative for these,
# so we don't clobber a refresh with the (older) baked-in image copy.
#
# NOTE: team_styles.parquet is intentionally NOT here. It's a deterministic derived table
# (pure function of PBP + engine/styles.py), so the committed build must be the baseline that
# carries labeler/code fixes to the volume on every deploy — otherwise a one-time seed freezes
# stale labels forever. A live refresh still rebuilds it at runtime; that output simply persists
# until the next redeploy re-syncs the committed baseline.
_REFRESH_NAMES = {"injuries.parquet", "schedules.parquet", "situational_stats.parquet",
                  "last_refresh.json"}


def _refresh_managed(name: str) -> bool:
    return name in _REFRESH_NAMES or name.startswith("pbp_")


def main():
    if not SEED.exists():
        print("seed: no data_seed/ present — nothing to seed (dev or no volume)")
        return
    volume_empty = not (DATA.exists() and any(DATA.rglob("*.parquet")))
    copied = synced = 0
    for p in SEED.rglob("*"):
        dest = DATA / p.relative_to(SEED)
        if p.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if volume_empty:
            shutil.copy2(p, dest); copied += 1
        elif _refresh_managed(p.name):
            if not dest.exists():
                shutil.copy2(p, dest); copied += 1        # only fill a gap; keep refresh output
        else:
            shutil.copy2(p, dest); synced += 1            # curated git data always matches image
    summary = ("first-boot copied " + str(copied) if volume_empty
               else f"synced {synced} curated files, filled {copied} missing")
    print(f"seed: {summary} -> {DATA}")


if __name__ == "__main__":
    main()
