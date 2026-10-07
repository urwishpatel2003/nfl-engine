"""
ml/props_ledger.py  —  the player-prop record: leans frozen before kickoff, graded on the box score
=================================================================================================
Companion to ml/ledger.py (game picks). Every BOOK-LINED market on the props board — a player,
a market, the book's line and price, the model's side and probability — is written to a ledger
before kickoff and graded afterwards against the play-by-play box score. Only a row written
before kickoff counts: the model moves with injuries and the lines move with the market, so a
re-projection after the game is not the lean that was on the board.

record(season, week, games)
    Upsert one row per (game, player, market) from a priced props slate (dashboard
    /api/props_slate payload with book lines). Same lock rule as the picks ledger:
      • before the week's lock time (Saturday 9am ET, ml.ledger.week_lock_time) rows are
        provisional and re-written on every slate computation;
      • from the lock on, existing rows are FROZEN (never overwritten); a market the book posts
        later is added the first time it is seen and frozen from then on — player props often
        post Saturday/Sunday morning, so a hard "nothing after the lock" rule would record
        almost nothing;
      • nothing is ever written once the game has kicked off.
    tier = "best" when the model beats the vig-free book price by >= MIN_EDGE (the board's
    strip), else "lined" (every other market the book priced — the "all games" analogue).

grade(season)
    Join the ledger with the season's play-by-play (per game, per player box score: passing /
    rushing / receiving lines aggregated exactly the way nflverse's weekly player stats are —
    verified against player_stats 2024: completions, yards, TDs, INTs, receptions exact;
    carries/attempts exact once kneels and spikes are counted and non-scrimmage plays dropped)
    and grade each row W / L / P (push on the number) / V (void: the player did not appear in
    the game, i.e. inactive — books void those). Units at the book's price for the side taken.
    Summaries: all lined vs best, by position (QB / RB / WR / TE), by market, week by week.

Storage: data/processed/props_ledger.parquet — refresh-managed on the server volume (never
clobbered by a deploy, see dashboard/seed.py) and git-ignored (the laptop's ledger is a sandbox).
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.ledger import kickoff_utc, week_lock_time, units

PROC = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "raw"
LEDGER = PROC / "props_ledger.parquet"

MIN_EDGE = 0.05          # model prob − vig-free book prob for a row to be a "best" prop (board strip)
POSITIONS = ("QB", "RB", "WR", "TE")

COLS = ["key", "game_id", "season", "week", "home", "away", "kickoff_utc", "locked_at", "frozen",
        "team", "opp", "player", "player_id", "pos", "market", "label", "dist",
        "side", "line", "proj", "prob", "fair", "book_prob", "book_odds", "edge", "tier", "rank", "books"]
NUM = ["line", "proj", "prob", "fair", "book_prob", "book_odds", "edge", "rank", "books"]


def load() -> pd.DataFrame:
    if LEDGER.exists():
        try:
            df = pd.read_parquet(LEDGER)
            for c in COLS:
                if c not in df.columns:
                    df[c] = False if c == "frozen" else None
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


# ── write ─────────────────────────────────────────────────────────────
def record(season: int, week: int, games: list, now: datetime | None = None, strip_n: int = 12) -> int:
    """Upsert the priced slate's book-lined markets. Returns rows written."""
    now = now or datetime.now(timezone.utc)
    lock_at = week_lock_time(games)
    frozen = bool(lock_at is not None and now >= lock_at)
    old = load()
    keep = set()
    if frozen and len(old):
        wk = (old["season"] == int(season)) & (old["week"] == int(week))
        if wk.any() and not old.loc[wk, "frozen"].fillna(False).astype(bool).all():
            old.loc[wk, "frozen"] = True                       # first post-lock call: stamp the board
            old.to_parquet(LEDGER, index=False)
        keep = set(old.loc[wk, "key"])
    rows = []
    for g in games:
        gid = g.get("game_id")
        if not gid or g.get("final"):
            continue
        ko = kickoff_utc(g.get("gameday"), g.get("gametime"))
        if ko is None or now >= ko:
            continue                                            # kicked off → the book is closed
        for team, pls in (g.get("teams") or {}).items():
            opp = g["away"] if team == g["home"] else g["home"]
            for pl in pls:
                for m in pl.get("markets", []):
                    book = m.get("book")
                    if not book or m.get("edge") is None or m.get("thin"):
                        continue
                    key = f"{gid}|{m['market']}|{pl.get('player_id') or pl['name']}"
                    if key in keep:
                        continue                                # frozen board: never overwrite
                    rows.append({
                        "key": key, "game_id": gid, "season": int(season), "week": int(week),
                        "home": g["home"], "away": g["away"],
                        "kickoff_utc": ko.isoformat(), "locked_at": now.isoformat(), "frozen": frozen,
                        "team": team, "opp": opp, "player": pl["name"], "player_id": pl.get("player_id"),
                        "pos": pl["pos"], "market": m["market"], "label": m["label"], "dist": m["dist"],
                        "side": m["side"], "line": m.get("line"), "proj": m["proj"], "prob": m["prob"],
                        "fair": m["fair"], "book_prob": m.get("book_prob"),
                        "book_odds": m.get("book_odds", book.get("yes_odds")),
                        "edge": m["edge"], "tier": "best" if m["edge"] >= MIN_EDGE else "lined",
                        "rank": None, "books": book.get("books"),
                    })
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=COLS)
    # strip rank: the board's top-N best props by edge (over the rows written in this call)
    best = new.index[new["tier"] == "best"]
    order = new.loc[best].sort_values("edge", ascending=False).index[:strip_n]
    new.loc[order, "rank"] = range(1, len(order) + 1)
    if len(old):
        old = old[~old["key"].isin(new["key"])]
    out = pd.concat([old, new], ignore_index=True)
    for c in NUM:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LEDGER, index=False)
    return len(rows)


# ── box scores from play-by-play ──────────────────────────────────────
_BOX_CACHE = {}        # season -> (mtime, frame)


def game_boxes(season: int) -> tuple:
    """(box, games_present): per (game_id, player_id) box score for every game in the season's
    play-by-play, and the set of game_ids the play-by-play covers:
    pass_att, cmp, pass_yds, pass_td, int, carries, rush_yds, rush_td, targets, rec, rec_yds,
    rec_td. Aggregation rules match nflverse's weekly player stats (see module docstring):
      passing  : play_type pass or qb_spike on a scrimmage down; attempt = complete, incomplete
                 or intercepted (sacks are not attempts); yards on completions only
      rushing  : play_type run or qb_kneel on a scrimmage down (two-point tries have no down)
      receiving: targets = pass plays with a receiver; yards/TDs on completions
    Return TDs are excluded from scrimmage TDs."""
    path = RAW / f"pbp_{season}.parquet"
    if not path.exists():
        return pd.DataFrame(), set()
    mt = path.stat().st_mtime
    c = _BOX_CACHE.get(season)
    if c and c[0] == mt:
        return c[1]
    p = pd.read_parquet(path)
    p = p[p["down"].notna()].copy() if "down" in p.columns else p.copy()
    ret = p["return_touchdown"].fillna(0) if "return_touchdown" in p.columns else 0
    p["is_td"] = (p["touchdown"].fillna(0) == 1) & (ret != 1)

    pa = p[p["play_type"].isin(["pass", "qb_spike"])].copy()
    pa["att"] = ((pa["complete_pass"] == 1) | (pa["incomplete_pass"] == 1) | (pa["interception"] == 1)).astype(int)
    pa["cmp_yds"] = pa["yards_gained"].fillna(0) * pa["complete_pass"].fillna(0)
    pa["ptd"] = (pa["is_td"] & (pa["complete_pass"] == 1)).astype(int)
    passing = pa.dropna(subset=["passer_player_id"]).groupby(["game_id", "passer_player_id"]).agg(
        pass_att=("att", "sum"), cmp=("complete_pass", "sum"), pass_yds=("cmp_yds", "sum"),
        pass_td=("ptd", "sum"), int=("interception", "sum")).reset_index() \
        .rename(columns={"passer_player_id": "player_id"})

    ru = p[p["play_type"].isin(["run", "qb_kneel"])].copy()
    ru["rtd"] = ru["is_td"].astype(int)
    rushing = ru.dropna(subset=["rusher_player_id"]).groupby(["game_id", "rusher_player_id"]).agg(
        carries=("rush_attempt", "sum"), rush_yds=("yards_gained", "sum"), rush_td=("rtd", "sum")).reset_index() \
        .rename(columns={"rusher_player_id": "player_id"})

    rc = pa[pa["receiver_player_id"].notna()].copy()
    rc["rectd"] = (rc["is_td"] & (rc["complete_pass"] == 1)).astype(int)
    receiving = rc.groupby(["game_id", "receiver_player_id"]).agg(
        targets=("att", "sum"), rec=("complete_pass", "sum"), rec_yds=("cmp_yds", "sum"),
        rec_td=("rectd", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"})

    box = passing.merge(rushing, on=["game_id", "player_id"], how="outer") \
                 .merge(receiving, on=["game_id", "player_id"], how="outer").fillna(0)
    box["rushrec_yds"] = box["rush_yds"] + box["rec_yds"]
    box["anytime_td"] = ((box["rush_td"] + box["rec_td"]) >= 1).astype(int)
    box["games_in_pbp"] = 1
    games_present = set(p["game_id"].unique())
    _BOX_CACHE[season] = (mt, (box, games_present))
    return _BOX_CACHE[season][1]


# ── grade ─────────────────────────────────────────────────────────────
def _tally(df: pd.DataFrame) -> dict:
    g = df[df["res"].isin(["W", "L", "P"])]
    w, l, p = int((g["res"] == "W").sum()), int((g["res"] == "L").sum()), int((g["res"] == "P").sum())
    dec = w + l
    u = float(g["units"].sum())
    return {"w": w, "l": l, "p": p, "n": len(g), "void": int((df["res"] == "V").sum()),
            "pct": round(w / dec, 3) if dec else None,
            "units": round(u, 2), "roi": round(u / len(g), 3) if len(g) else None}


def grade(season: int) -> dict:
    led = load()
    led = led[led["season"] == season].copy() if len(led) else led
    sched = pd.read_parquet(RAW / "schedules.parquet", columns=["game_id", "home_score", "away_score"]) \
        if (RAW / "schedules.parquet").exists() else pd.DataFrame(columns=["game_id", "home_score", "away_score"])
    final = set(sched.loc[sched["home_score"].notna(), "game_id"])
    box, present = game_boxes(season)
    bx = box.set_index(["game_id", "player_id"]) if len(box) else None
    now = datetime.now(timezone.utc)

    rows = []
    for r in led.to_dict("records"):
        row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
        gid = row["game_id"]
        row.update({"res": None, "units": 0.0, "actual": None})
        if gid in final and gid in present and bx is not None:
            row["status"] = "final"
            pid = row.get("player_id")
            hit = (gid, pid) in bx.index if pid else False
            if not hit:
                row["res"] = "V"                               # no appearance → inactive, books void
            else:
                act = float(bx.loc[(gid, pid), row["market"]])
                row["actual"] = act
                if row["market"] == "anytime_td":
                    won = act >= 1
                    row["res"] = "W" if won else "L"
                else:
                    line = float(row["line"])
                    if abs(act - line) < 1e-9:
                        row["res"] = "P"
                    else:
                        over = act > line
                        row["res"] = "W" if (over == (row["side"] == "OVER")) else "L"
                odds = row.get("book_odds")
                if row["res"] in ("W", "L"):
                    row["units"] = units(odds, row["res"] == "W") if odds is not None else (1.0 if row["res"] == "W" else -1.0)
        elif gid in final:
            row["status"] = "final"                            # scored, play-by-play not pulled yet
        else:
            try:
                row["status"] = "live" if now >= datetime.fromisoformat(row["kickoff_utc"]) else "pending"
            except Exception:
                row["status"] = "pending"
        rows.append(row)
    gdf = pd.DataFrame(rows) if rows else pd.DataFrame(columns=COLS + ["res", "units", "actual", "status"])
    fin = gdf[gdf["res"].notna()] if len(gdf) else gdf

    def _sub(mask):
        return fin[mask] if len(fin) else fin

    summary, by_pos, by_market, weeks = {}, {}, [], []
    if len(fin):
        best = fin["tier"] == "best"
        summary = {"all": _tally(fin), "best": _tally(_sub(best)),
                   "strip": _tally(_sub(fin["rank"].notna()))}
        for pos in POSITIONS:
            by_pos[pos] = {"all": _tally(_sub(fin["pos"] == pos)), "best": _tally(_sub(best & (fin["pos"] == pos)))}
        for mk, g in fin.groupby("market"):
            by_market.append({"market": mk, "label": g["label"].iloc[0], "all": _tally(g),
                              "best": _tally(g[g["tier"] == "best"]),
                              "over": _tally(g[g["side"].isin(["OVER", "YES"])]), "under": _tally(g[g["side"] == "UNDER"])})
        by_market.sort(key=lambda x: -x["all"]["n"])
    if len(gdf):
        for wk, g in gdf.groupby("week"):
            f = g[g["res"].notna()]
            weeks.append({"week": int(wk), "rows": len(g), "graded": len(f),
                          "all": _tally(f) if len(f) else None,
                          "best": _tally(f[f["tier"] == "best"]) if len(f) else None,
                          **{pos: _tally(f[(f["pos"] == pos) & (f["tier"] == "best")]) if len(f) else None for pos in POSITIONS}})
    # the log: best-tier rows (every lined market would be thousands); newest first
    log = gdf[gdf["tier"] == "best"].sort_values(["week", "kickoff_utc", "edge"], ascending=[False, False, False]) \
        if len(gdf) else gdf
    return {"season": season, "n_rows": int(len(gdf)), "n_best": int((gdf["tier"] == "best").sum()) if len(gdf) else 0,
            "n_graded": int(len(fin)), "n_pending": int((gdf["res"].isna()).sum()) if len(gdf) else 0,
            "summary": summary, "by_pos": by_pos, "by_market": by_market, "weeks": weeks,
            "log": log.head(400).to_dict("records") if len(log) else [],
            "rules": {"min_edge": MIN_EDGE, "positions": list(POSITIONS)}}
