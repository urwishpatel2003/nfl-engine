"""
dashboard/seed.py — seed the data/ volume from the image's baked-in copy.

On Railway a persistent Volume is mounted at /app/data, which shadows the committed
data that shipped in the image. On first boot that mount is empty, so we copy the
seed (baked to /app/data_seed at build time) into it. No-op when data/ is already
populated — i.e. when there is no volume, or it was seeded on a previous boot.

Run before gunicorn:  python dashboard/seed.py && gunicorn ...
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SEED = ROOT / "data_seed"


def main():
    if not SEED.exists():
        print("seed: no data_seed/ present — nothing to seed (dev or no volume)")
        return
    already = DATA.exists() and any(DATA.rglob("*.parquet"))
    if already:
        print("seed: data/ already populated — skipping")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in SEED.rglob("*"):
        dest = DATA / p.relative_to(SEED)
        if p.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(p, dest)
                n += 1
    print(f"seed: copied {n} files into {DATA}")


if __name__ == "__main__":
    main()
