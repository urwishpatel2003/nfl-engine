"""
ml/ledger.py  —  the model's betting record: picks frozen BEFORE kickoff, graded after
=====================================================================================
Why a ledger instead of re-predicting finished games: the model moves every day (rosters,
injuries, live lines), so a prediction regenerated after the final score is not the pick
that was actually on the board. Only a pick written before kickoff is a fair test.

record(season, week, games)
    Upsert one row per game from a freshly computed slate (dashboard /api/schedule payload):
    the model's numbers, the line it was priced against, and the three picks —
      ATS   : ats_pick / edge / cover_prob / pick_rank (top-5 flag)
      Total : total_pick (Over/Under) / total_prob
      ML    : ml_pick (model's straight-up favorite) / ml_prob / posted odds / no-vig implied
    A game is re-locked on every slate computation until its kickoff, so the stored row is the
    LAST pick before the game started. Nothing is ever written once the game has kicked off.

grade(season)
    Join the ledger with schedules (final scores) and grade every pick: W / L / P (push),
    units at -110 for ATS & totals, units at the posted price for ML. Returns the per-game
    log, week-by-week rows and market summaries (all picks + the model's higher-conviction
    subsets: top-5 ATS, totals >= 57% and ML value >= +5 pts over the market's no-vig price).

Storage: data/processed/picks_ledger.parquet — refresh-managed on the server volume (never
overwritten by a deploy) and git-ignored (the laptop's ledger is not the site's).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "raw"
LEDGER = PROC / "picks_ledger.parquet"

JUICE = -110                     # standard price assumed for ATS / totals
TOTAL_CONF = 0.57                # "confident total" threshold on the model's over/under prob
ML_VALUE = 0.05                  # model win prob must beat the no-vig market price by 5 pts

COLS = ["game_id", "season", "week", "home", "away", "kickoff_utc", "locked_at",
        "pred_home", "pred_away", "pred_margin", "pred_total", "home_win_prob",
        "vegas_spread", "vegas_total", "home_ml", "away_ml", "line_source",
        "ats_pick", "edge", "cover_prob", "pick_rank",
        "total_pick", "total_prob",
        "ml_pick", "ml_prob", "ml_odds", "ml_implied"]


# ── time / odds helpers ───────────────────────────────────────────────
def kickoff_utc(gameday, gametime) -> datetime | None:
    """nflverse gameday (YYYY-MM-DD) + gametime (HH:MM, US Eastern) → aware UTC datetime."""
    if not isinstance(gameday, str) or not gameday:
        return None
    hm = gametime if isinstance(gametime, str) and gametime else "13:00"
    try:
        naive = datetime.strptime(f"{gameday} {hm}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    try:
        from zoneinfo import ZoneInfo
        return naive.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
    except Exception:                                  # no tz database: EDT Mar-Nov, else EST
        off = -4 if 3 <= naive.month <= 11 else -5
        return naive.replace(tzinfo=timezone(timedelta(hours=off))).astimezone(timezone.utc)


def implied(odds) -> float | None:
    """American odds → implied probability (with vig)."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    o = float(odds)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def novig(home_odds, away_odds) -> tuple:
    """(home, away) fair probabilities with the book's margin removed."""
    h, a = implied(home_odds), implied(away_odds)
    if h is None or a is None or h + a <= 0:
        return None, None
    return h / (h + a), a / (h + a)


def units(odds, won: bool) -> float:
    """Profit on a 1-unit stake at American odds."""
    if not won:
        return -1.0
    o = float(odds)
    return o / 100 if o > 0 else 100 / -o


# ── write ─────────────────────────────────────────────────────────────
def load() -> pd.DataFrame:
    if LEDGER.exists():
        try:
            return pd.read_parquet(LEDGER)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def record(season: int, week: int, games: list, now: datetime | None = None) -> int:
    """Upsert the not-yet-kicked-off games of a computed slate. Returns rows written."""
    now = now or datetime.now(timezone.utc)
    rows = []
    for g in games:
        if g.get("pred_margin") is None or not g.get("game_id"):
            continue
        ko = kickoff_utc(g.get("gameday"), g.get("gametime"))
        if ko is None or now >= ko or g.get("final"):
            continue                                   # kicked off → the book is closed
        hp = g.get("home_win_prob")
        ml_pick = ml_prob = ml_odds = ml_imp = None
        if hp is not None:
            home_side = hp >= 0.5
            ml_pick = g["home"] if home_side else g["away"]
            ml_prob = hp if home_side else 1 - hp
            ml_odds = g.get("home_ml") if home_side else g.get("away_ml")
            fh, fa = novig(g.get("home_ml"), g.get("away_ml"))
            ml_imp = fh if home_side else fa
        rows.append({
            "game_id": g["game_id"], "season": int(season), "week": int(week),
            "home": g["home"], "away": g["away"],
            "kickoff_utc": ko.isoformat(), "locked_at": now.isoformat(),
            "pred_home": g.get("pred_home"), "pred_away": g.get("pred_away"),
            "pred_margin": g.get("pred_margin"), "pred_total": g.get("pred_total"),
            "home_win_prob": hp,
            "vegas_spread": g.get("vegas_spread"), "vegas_total": g.get("vegas_total"),
            "home_ml": g.get("home_ml"), "away_ml": g.get("away_ml"),
            "line_source": g.get("line_source"),
            "ats_pick": g.get("ats_pick"), "edge": g.get("edge"), "cover_prob": g.get("cover_prob"),
            "pick_rank": g.get("pick_rank"),
            "total_pick": g.get("total_pick"), "total_prob": g.get("total_prob"),
            "ml_pick": ml_pick, "ml_prob": ml_prob, "ml_odds": ml_odds, "ml_implied": ml_imp,
        })
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=COLS)
    old = load()
    if len(old):
        old = old[~old["game_id"].isin(new["game_id"])]
    out = pd.concat([old, new], ignore_index=True)
    for c in ("pred_home", "pred_away", "pred_margin", "pred_total", "home_win_prob", "vegas_spread",
              "vegas_total", "home_ml", "away_ml", "edge", "cover_prob", "pick_rank", "total_prob",
              "ml_prob", "ml_odds", "ml_implied"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LEDGER, index=False)
    return len(rows)


# ── grade ─────────────────────────────────────────────────────────────
def _grade_row(r, hs, as_) -> dict:
    """W/L/P + units for the three markets of one finished game."""
    margin, total = hs - as_, hs + as_
    out = {"ats": None, "ats_units": 0.0, "ou": None, "ou_units": 0.0, "ml": None, "ml_units": 0.0}
    # ATS: nflverse spread is home-perspective, positive = home favored → home covers iff margin > line
    if r.get("ats_pick") and pd.notna(r.get("vegas_spread")):
        cover = margin - float(r["vegas_spread"])            # >0 home covers
        side = 1 if r["ats_pick"] == r["home"] else -1
        res = "P" if abs(cover) < 1e-9 else ("W" if cover * side > 0 else "L")
        out["ats"] = res
        out["ats_units"] = 0.0 if res == "P" else units(JUICE, res == "W")
    if r.get("total_pick") and pd.notna(r.get("vegas_total")):
        diff = total - float(r["vegas_total"])
        want = 1 if r["total_pick"] == "Over" else -1
        res = "P" if abs(diff) < 1e-9 else ("W" if diff * want > 0 else "L")
        out["ou"] = res
        out["ou_units"] = 0.0 if res == "P" else units(JUICE, res == "W")
    if r.get("ml_pick"):
        if margin == 0:
            res = "P"
        else:
            winner = r["home"] if margin > 0 else r["away"]
            res = "W" if winner == r["ml_pick"] else "L"
        out["ml"] = res
        odds = r.get("ml_odds")
        out["ml_units"] = 0.0 if res == "P" else (units(odds, res == "W") if pd.notna(odds) else 0.0)
    return out


def _tally(df: pd.DataFrame, col: str, ucol: str) -> dict:
    g = df[df[col].notna()]
    w, l, p = int((g[col] == "W").sum()), int((g[col] == "L").sum()), int((g[col] == "P").sum())
    dec = w + l
    u = float(g[ucol].sum())
    return {"w": w, "l": l, "p": p, "n": len(g), "pct": round(w / dec, 3) if dec else None,
            "units": round(u, 2), "roi": round(u / len(g), 3) if len(g) else None}


def grade(season: int) -> dict:
    led = load()
    led = led[led["season"] == season] if len(led) else led
    sched = pd.read_parquet(RAW / "schedules.parquet",
                            columns=["game_id", "home_score", "away_score"]) if (RAW / "schedules.parquet").exists() \
        else pd.DataFrame(columns=["game_id", "home_score", "away_score"])
    sc = sched.set_index("game_id")
    games, now = [], datetime.now(timezone.utc)
    for _, r in led.sort_values(["week", "kickoff_utc"]).iterrows():
        r = r.to_dict()
        row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
        hs = sc["home_score"].get(r["game_id"]) if r["game_id"] in sc.index else None
        as_ = sc["away_score"].get(r["game_id"]) if r["game_id"] in sc.index else None
        if hs is not None and pd.notna(hs) and as_ is not None and pd.notna(as_):
            row.update(_grade_row(r, float(hs), float(as_)))
            row.update({"home_score": int(hs), "away_score": int(as_), "status": "final"})
        else:
            row.update({"ats": None, "ats_units": 0.0, "ou": None, "ou_units": 0.0,
                        "ml": None, "ml_units": 0.0, "home_score": None, "away_score": None})
            try:
                ko = datetime.fromisoformat(r["kickoff_utc"])
                row["status"] = "live" if now >= ko else "pending"
            except Exception:
                row["status"] = "pending"
        games.append(row)
    gdf = pd.DataFrame(games) if games else pd.DataFrame(columns=COLS + ["ats", "ou", "ml", "status"])
    fin = gdf[gdf["status"] == "final"] if len(gdf) else gdf

    def _sub(mask):
        return fin[mask] if len(fin) else fin

    summary = {}
    if len(fin):
        summary = {
            "ats_all": _tally(fin, "ats", "ats_units"),
            "ats_top": _tally(_sub(fin["pick_rank"].notna()), "ats", "ats_units"),
            "ou_all": _tally(fin, "ou", "ou_units"),
            "ou_conf": _tally(_sub(fin["total_prob"] >= TOTAL_CONF), "ou", "ou_units"),
            "ml_all": _tally(fin, "ml", "ml_units"),
            "ml_value": _tally(_sub((fin["ml_prob"] - fin["ml_implied"]) >= ML_VALUE), "ml", "ml_units"),
        }
    weeks = []
    if len(gdf):
        for wk, g in gdf.groupby("week"):
            f = g[g["status"] == "final"]
            weeks.append({"week": int(wk), "games": len(g), "final": len(f),
                          "ats": _tally(f, "ats", "ats_units") if len(f) else None,
                          "ats_top": _tally(f[f["pick_rank"].notna()], "ats", "ats_units") if len(f) else None,
                          "ou": _tally(f, "ou", "ou_units") if len(f) else None,
                          "ml": _tally(f, "ml", "ml_units") if len(f) else None})
    # Was the model wrong, or was the market wrong too? Mean absolute error of the locked
    # model number vs the closing line it was priced against, on the same games. A bad
    # week where the market missed by as much is variance; a bad week where the market
    # was close and the model was not is signal.
    accuracy = None
    if len(fin):
        f = fin[fin["pred_margin"].notna() & fin["vegas_spread"].notna()]
        ft = fin[fin["pred_total"].notna() & fin["vegas_total"].notna()]
        act_m = f["home_score"] - f["away_score"]
        act_t = ft["home_score"] + ft["away_score"]
        accuracy = {
            "n": int(len(f)),
            "margin_mae_model": round(float((act_m - f["pred_margin"]).abs().mean()), 2) if len(f) else None,
            "margin_mae_market": round(float((act_m - f["vegas_spread"]).abs().mean()), 2) if len(f) else None,
            "total_mae_model": round(float((act_t - ft["pred_total"]).abs().mean()), 2) if len(ft) else None,
            "total_mae_market": round(float((act_t - ft["vegas_total"]).abs().mean()), 2) if len(ft) else None,
            "avg_total_actual": round(float(act_t.mean()), 1) if len(ft) else None,
            "avg_total_model": round(float(ft["pred_total"].mean()), 1) if len(ft) else None,
            "avg_total_market": round(float(ft["vegas_total"].mean()), 1) if len(ft) else None,
            "winner_acc": round(float(((f["pred_margin"] > 0) == (act_m > 0)).mean()), 3) if len(f) else None,
            "market_winner_acc": round(float(((f["vegas_spread"] > 0) == (act_m > 0)).mean()), 3) if len(f) else None,
        }
    return {"season": season, "n_locked": len(gdf), "n_final": len(fin),
            "n_pending": int((gdf["status"] != "final").sum()) if len(gdf) else 0,
            "summary": summary, "weeks": weeks, "games": games, "accuracy": accuracy,
            "rules": {"juice": JUICE, "total_conf": TOTAL_CONF, "ml_value": ML_VALUE,
                      "breakeven_110": round(110 / 210, 3)}}
