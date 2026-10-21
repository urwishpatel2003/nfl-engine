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
    Only the BEST tier is kept: the model beats the vig-free book price by >= MIN_EDGE. The
    other lined markets are priced for the matchup page but not recorded (owner decision
    2026-10-14: "only show and keep the best props"). Rows from before that date may carry
    tier "lined"; grade() reports the best tier.

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

# model prob − vig-free book prob for a row to be a "best" prop (the board and this record).
# SET FROM THE BACKTEST (backtest_props.py section C, anchored model, walk-forward 2022-25,
# 48k markets against a naive trailing-average market, breakeven 52.4 % at −110):
#     claimed edge   5-7.5   7.5-10   10-12.5   12.5-15   15-20   20+
#     hit rate       55 %    60 %     63 %      62 %      68 %    85 %
# The claimed edge is calibrated against that naive market (realised ≈ claimed), but a real
# book is sharper, so plan on roughly half of it surviving: a 10-pt claimed edge still clears
# the vig with margin, a 5-pt one is breakeven. 15 pts is safe but leaves ~10 picks a week.
# Rushing markets are weaker (carries 59 %, rush yds 54 %, rush+rec 54 % in the 10-15 bucket)
# and need 15. Owner history: 0.05 at launch → 0.15 on 2026-10-20 → this table the same day.
MIN_EDGE = 0.10                                       # default floor
MIN_EDGE_BY_MARKET = {"rush_yds": 0.15, "carries": 0.15, "rushrec_yds": 0.15}


def min_edge(market: str) -> float:
    return MIN_EDGE_BY_MARKET.get(market, MIN_EDGE)
POSITIONS = ("QB", "RB", "WR", "TE")

COLS = ["key", "game_id", "season", "week", "home", "away", "kickoff_utc", "locked_at", "frozen",
        "team", "opp", "player", "player_id", "pos", "market", "label", "dist",
        "side", "line", "proj", "model_proj", "book_mean", "anchor_w",
        "prob", "fair", "book_prob", "book_odds", "edge", "tier", "rank", "books"]
NUM = ["line", "proj", "model_proj", "book_mean", "anchor_w", "prob", "fair", "book_prob", "book_odds", "edge", "rank", "books"]

# ── calibration log ───────────────────────────────────────────────────
# EVERY book-lined, non-thin market (not just the best tier) with the raw model mean, the
# market-implied mean and the book's vig-free over price, so the anchor weight can be
# MEASURED against the box score instead of assumed (ml.props.ANCHOR_DEFAULT is the prior).
# Same lifecycle as the record: provisional until the Saturday lock, frozen after. Never shown.
CALIB = PROC / "props_calib.parquet"
CALIB_COLS = ["key", "game_id", "season", "week", "kickoff_utc", "locked_at", "frozen",
              "team", "player", "player_id", "pos", "market", "dist", "line", "sd",
              "model_proj", "book_mean", "book_over_prob", "book_odds"]
CALIB_MIN_N = 100          # graded rows a market needs before its measured weight replaces the default


# The record starts here. Rows before this week were priced by the pre-2026-10-20 model (fixed
# CV spreads, no market anchor) and the owner asked for them removed so the record measures
# the current model only: week 7 is the first week the anchored model priced before kickoff.
# Earlier rows are dropped from the file the first time it is loaded after the deploy, on both
# the record and the calibration log. Change the week here to move the start; never edit the
# parquet by hand.
RECORD_START = {2026: 7}            # season -> first week kept


def _since(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not len(df) or "season" not in df.columns:
        return df
    drop = pd.Series(False, index=df.index)
    for season, wk in RECORD_START.items():
        drop |= (df["season"] == season) & (df["week"] < wk)
    if drop.any():
        df = df[~drop].copy()
        try:
            df.to_parquet(path, index=False)              # physically gone, not just filtered
        except Exception:
            pass
    return df


def load() -> pd.DataFrame:
    if LEDGER.exists():
        try:
            df = pd.read_parquet(LEDGER)
            for c in COLS:
                if c not in df.columns:
                    df[c] = False if c == "frozen" else None
            return _since(df, LEDGER)
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
    rows, calib = [], []
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
                    if key not in keep and m.get("book_mean") is not None:
                        calib.append({
                            "key": key, "game_id": gid, "season": int(season), "week": int(week),
                            "kickoff_utc": ko.isoformat(), "locked_at": now.isoformat(), "frozen": frozen,
                            "team": team, "player": pl["name"], "player_id": pl.get("player_id"), "pos": pl["pos"],
                            "market": m["market"], "dist": m["dist"], "line": m.get("line"), "sd": m.get("sd"),
                            "model_proj": m.get("model_proj", m["proj"]), "book_mean": m.get("book_mean"),
                            "book_over_prob": m.get("book_prob") if m["side"] in ("OVER", "YES") else (
                                None if m.get("book_prob") is None else round(1 - m["book_prob"], 3)),
                            "book_odds": m.get("book_odds", book.get("yes_odds")),
                        })
                    if m["edge"] < min_edge(m["market"]):  # only the best tier is kept (owner decision 2026-10-14)
                        continue
                    if key in keep:
                        continue                                # frozen board: never overwrite
                    rows.append({
                        "key": key, "game_id": gid, "season": int(season), "week": int(week),
                        "home": g["home"], "away": g["away"],
                        "kickoff_utc": ko.isoformat(), "locked_at": now.isoformat(), "frozen": frozen,
                        "team": team, "opp": opp, "player": pl["name"], "player_id": pl.get("player_id"),
                        "pos": pl["pos"], "market": m["market"], "label": m["label"], "dist": m["dist"],
                        "side": m["side"], "line": m.get("line"), "proj": m["proj"],
                        "model_proj": m.get("model_proj"), "book_mean": m.get("book_mean"), "anchor_w": m.get("anchor_w"),
                        "prob": m["prob"], "fair": m["fair"], "book_prob": m.get("book_prob"),
                        "book_odds": m.get("book_odds", book.get("yes_odds")),
                        "edge": m["edge"], "tier": "best" if m["edge"] >= min_edge(m["market"]) else "lined",
                        "rank": None, "books": book.get("books"),
                    })
    _calib_write(calib, season, week, frozen)
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=COLS)
    # board rank: every best prop by edge (over the rows written in this call)
    best = new.index[new["tier"] == "best"]
    order = new.loc[best].sort_values("edge", ascending=False).index
    new.loc[order, "rank"] = range(1, len(order) + 1)
    if len(old):
        old = old[~old["key"].isin(new["key"])]
    out = pd.concat([old, new], ignore_index=True)
    for c in NUM:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LEDGER, index=False)
    return len(rows)


def calib_load() -> pd.DataFrame:
    if CALIB.exists():
        try:
            df = pd.read_parquet(CALIB)
            for c in CALIB_COLS:
                if c not in df.columns:
                    df[c] = False if c == "frozen" else None
            return _since(df, CALIB)
        except Exception:
            pass
    return pd.DataFrame(columns=CALIB_COLS)


def _calib_write(rows: list, season: int, week: int, frozen: bool) -> int:
    """Upsert this call's calibration rows. Before the lock every row is overwritten by the next
    call (the line moves); after it the week's rows are stamped frozen and kept, and only
    markets not yet on file are added (late-posting lines)."""
    old = calib_load()
    if len(old):
        wk = (old["season"] == int(season)) & (old["week"] == int(week))
        if frozen and wk.any():
            if not old.loc[wk, "frozen"].fillna(False).astype(bool).all():
                old.loc[wk, "frozen"] = True
                old.to_parquet(CALIB, index=False)
            keep = set(old.loc[wk, "key"])
            rows = [r for r in rows if r["key"] not in keep]
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=CALIB_COLS)
    if len(old):
        old = old[~old["key"].isin(new["key"])]
    out = pd.concat([old, new], ignore_index=True)
    for c in ("line", "sd", "model_proj", "book_mean", "book_over_prob", "book_odds"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CALIB, index=False)
    return len(rows)


_ANCHOR_CACHE = {}     # season -> (calib mtime, pbp mtime, weights)


def anchor_weights(season: int) -> dict:
    """Per-market anchor weight MEASURED from the calibration log: over graded rows the
    least-squares w for actual ≈ (1−w)·model + w·book_mean, i.e. w = Σ(d·r)/Σ(d²) with
    d = book − model and r = actual − model. A market needs CALIB_MIN_N graded rows; until
    then its entry carries the default and n, so the record page can show the state.
    Returns {market: {"w": float|None, "n": int, "default": float, "mae_model", "mae_book",
    "mae_blend"}}; ml.props.anchor_weight() falls back to the default when "w" is None."""
    from ml.props import ANCHOR_DEFAULT, ANCHOR_DEFAULT_BY_MARKET, ANCHOR_MIN, ANCHOR_MAX
    cm = CALIB.stat().st_mtime if CALIB.exists() else 0
    pb = RAW / f"pbp_{season}.parquet"
    pm = pb.stat().st_mtime if pb.exists() else 0
    c = _ANCHOR_CACHE.get(season)
    if c and c[0] == cm and c[1] == pm:
        return c[2]
    out = {}
    df = calib_load()
    df = df[df["season"] == int(season)] if len(df) else df
    box, present = game_boxes(season) if len(df) else (pd.DataFrame(), set())
    if len(df) and len(box):
        bx = box.set_index(["game_id", "player_id"])
        df = df[df["game_id"].isin(present) & df["player_id"].notna()].copy()
        idx = list(zip(df["game_id"], df["player_id"]))
        hit = np.array([i in bx.index for i in idx], dtype=bool)      # a plain list would select COLUMNS when empty
        df = df[hit].copy() if len(df) else df
        df["actual"] = [float(bx.loc[i, mk]) for i, mk in zip(list(zip(df["game_id"], df["player_id"])), df["market"])]
        for mk, g in df.groupby("market"):
            g = g.dropna(subset=["model_proj", "book_mean", "actual"])
            n = len(g)
            d = (g["book_mean"] - g["model_proj"]).to_numpy(dtype=float)
            r = (g["actual"] - g["model_proj"]).to_numpy(dtype=float)
            default = ANCHOR_DEFAULT_BY_MARKET.get(mk, ANCHOR_DEFAULT)
            w = None
            if n >= CALIB_MIN_N and float((d * d).sum()) > 0:
                w = float(min(ANCHOR_MAX, max(ANCHOR_MIN, (d * r).sum() / (d * d).sum())))
            we = w if w is not None else default
            blend = (1 - we) * g["model_proj"] + we * g["book_mean"]
            out[mk] = {"w": None if w is None else round(w, 3), "n": int(n), "default": default,
                       "mae_model": round(float((g["actual"] - g["model_proj"]).abs().mean()), 2) if n else None,
                       "mae_book": round(float((g["actual"] - g["book_mean"]).abs().mean()), 2) if n else None,
                       "mae_blend": round(float((g["actual"] - blend).abs().mean()), 2) if n else None}
    _ANCHOR_CACHE[season] = (cm, pm, out)
    return out


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

    # Rows priced off a malformed book price (|odds| < 100 — the old raw-median consensus
    # produced −14 / −1 when books straddled ±100, and the edge was computed from it) are
    # not a real market: left out of every tally and the log.
    if "book_odds" in led.columns and len(led):
        bad = pd.to_numeric(led["book_odds"], errors="coerce").abs() < 100
        n_bad = int(bad.sum())
        led = led[~bad]
    else:
        n_bad = 0
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
                # American odds are <= -100 or >= +100; anything else (0, a missing price
                # stored as 0, a decimal slipped through) is malformed → price it at -110
                odds = row.get("book_odds")
                try:
                    odds = float(odds)
                    if not (abs(odds) >= 100):
                        odds = -110.0
                except (TypeError, ValueError):
                    odds = -110.0
                if row["res"] in ("W", "L"):
                    row["units"] = units(odds, row["res"] == "W")
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
            gb = g[g["tier"] == "best"]
            by_market.append({"market": mk, "label": g["label"].iloc[0], "all": _tally(g),
                              "best": _tally(gb),
                              "over": _tally(g[g["side"].isin(["OVER", "YES"])]), "under": _tally(g[g["side"] == "UNDER"]),
                              "over_best": _tally(gb[gb["side"].isin(["OVER", "YES"])]), "under_best": _tally(gb[gb["side"] == "UNDER"])})
        by_market.sort(key=lambda x: -x["best"]["n"])
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
            "n_malformed": n_bad,
            "n_graded": int(len(fin)), "n_pending": int((gdf["res"].isna()).sum()) if len(gdf) else 0,
            "summary": summary, "by_pos": by_pos, "by_market": by_market, "weeks": weeks,
            "log": log.head(400).to_dict("records") if len(log) else [],
            "anchor": anchor_weights(season), "calib_min_n": CALIB_MIN_N,
            "since_week": RECORD_START.get(season),
            "rules": {"min_edge": MIN_EDGE, "min_edge_by_market": MIN_EDGE_BY_MARKET, "positions": list(POSITIONS)}}
