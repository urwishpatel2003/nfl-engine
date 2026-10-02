"""
dashboard/server.py
-------------------
Lightweight Flask server that serves prediction data as JSON.
The frontend (dashboard.html) fetches from these endpoints.

Usage:
    pip install flask
    python dashboard/server.py

Then open: http://localhost:5000
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder=str(Path(__file__).parent))


@app.after_request
def _no_cache(resp):
    """Never let the browser/edge serve a stale page or API response — the model and data
    change on every deploy/refresh, so always revalidate."""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


PROC = Path(__file__).parent.parent / "data" / "processed"
RAW  = Path(__file__).parent.parent / "data" / "raw"


def safe_json(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj): return None
    return obj


def df_to_json(df: pd.DataFrame) -> list:
    records = df.replace({float('nan'): None}).to_dict('records')
    return [{k: safe_json(v) for k, v in r.items()} for r in records]


@app.route('/')
def index():
    return send_from_directory(str(Path(__file__).parent), 'season2026.html')


@app.route('/legacy')
def legacy():
    return send_from_directory(str(Path(__file__).parent), 'dashboard.html')


# ── Team metadata (colors + logos) ─────────────────────────────────
_TEAM_META = None


def team_meta() -> dict:
    global _TEAM_META
    if _TEAM_META is None:
        p = RAW / "team_info.parquet"
        if not p.exists():
            _TEAM_META = {}
            return _TEAM_META
        ti = pd.read_parquet(p)
        cols = [c for c in ["team_abbr", "team_name", "team_color", "team_color2",
                            "team_logo_espn"] if c in ti.columns]
        _TEAM_META = (ti[cols].drop_duplicates("team_abbr")
                      .set_index("team_abbr").to_dict("index"))
    return _TEAM_META


@app.route('/api/team_meta')
def api_team_meta():
    return jsonify(team_meta())


# ── 2026 projected starting QB (informational; does NOT affect ratings) ─────
_QB1 = None


def qb1_2026() -> dict:
    global _QB1
    if _QB1 is None:
        p = RAW / "depth_2026_current.parquet"
        if not p.exists():
            _QB1 = {}
            return _QB1
        d = pd.read_parquet(p)
        d = d[d["pos_abb"] == "QB"].copy()
        d["pos_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
        starters = d[d["pos_rank"] == 1].drop_duplicates("team")
        _QB1 = dict(zip(starters["team"], starters["player_name"]))
    return _QB1


_SQUAD = None


@app.route('/api/power_rankings')
def api_power_rankings():
    """2026 team ratings. mode=preseason (default) = current roster-talent + coaching
    (the season hasn't been played); mode=final = prior-season results-based ratings."""
    global _SQUAD
    season = int(request.args.get('season', 2025))
    mode = request.args.get('mode', 'preseason')
    meta = team_meta()
    if mode == 'preseason':
        if _SQUAD is None:
            from ml.squad import squad_ratings, WEIGHTS
            bd = squad_ratings(breakdown=True)[0].copy()
            # split the rating into an offense (qb+skill+ol) and defense (def_team+rush+cover)
            # composite, then rank teams on each — far more intuitive than the raw z-score.
            bd["off"] = WEIGHTS["qb"] * bd["qb"] + WEIGHTS["skill"] * bd["skill"] + WEIGHTS["ol"] * bd["ol"]
            bd["def"] = (WEIGHTS["def_team"] * bd["def_team"] + WEIGHTS["rush"] * bd["rush"]
                         + WEIGHTS["cover"] * bd["cover"])
            bd["off_rank"] = bd["off"].rank(ascending=False, method="min").astype(int)
            bd["def_rank"] = bd["def"].rank(ascending=False, method="min").astype(int)
            # IN-SEASON: fold this season's results in. rating = (1−w)·roster talent + w·performance,
            # performance = opponent-adjusted net EPA/play on the same points scale, w = g/(g+6)
            # per team (ml/current.py). Week 1 moves a team ~14% of the way toward its result;
            # by midseason the results carry more than the roster projection — the roster is the
            # prior, the games are the evidence.
            bd["rating_talent"] = bd["rating"]
            bd["perf_w"] = 0.0
            try:
                from ml.current import performance_rating, state as _cstate
                perf, pw_ = performance_rating()
                if len(perf):
                    w = bd["team"].map(pw_).fillna(0.0)
                    pr = bd["team"].map(perf)
                    bd["rating_perf"] = pr
                    bd["rating"] = ((1 - w) * bd["rating"] + w * pr.fillna(bd["rating"])).round(1)
                    bd["perf_w"] = w.round(3)
                    bd["weeks_played"] = _cstate()["weeks_played"]
                    bd = bd.sort_values("rating", ascending=False).reset_index(drop=True)
                    bd["rank"] = bd.index + 1
            except Exception as e:
                print(f"[rankings] performance blend skipped: {e}", flush=True)
            try:                                          # anchor the abstract rating to projected wins
                from ml.season import team_win_totals
                pw = team_win_totals().set_index("team")["proj_wins"]
                bd["proj_wins"] = bd["team"].map(pw)
            except Exception:
                bd["proj_wins"] = None
            _SQUAD = bd
        r = _SQUAD
    else:
        from ml.rank import power_ratings
        r = power_ratings(season)
    qbs = qb1_2026() if mode == 'preseason' else {}
    has = lambda c: c in r.columns
    recs = []
    for _, row in r.iterrows():
        m = meta.get(row["team"], {})
        recs.append({
            "rank": int(row["rank"]), "team": row["team"], "rating": float(row["rating"]),
            "prev": float(row["rating_prev"]) if has("rating_prev") else None,
            "off_rank": int(row["off_rank"]) if has("off_rank") else None,
            "def_rank": int(row["def_rank"]) if has("def_rank") else None,
            "proj_wins": (float(row["proj_wins"]) if has("proj_wins") and pd.notna(row["proj_wins"]) else None),
            "rating_talent": float(row["rating_talent"]) if has("rating_talent") else None,
            "rating_perf": (float(row["rating_perf"]) if has("rating_perf") and pd.notna(row["rating_perf"]) else None),
            "perf_w": float(row["perf_w"]) if has("perf_w") else 0.0,
            "weeks_played": int(row["weeks_played"]) if has("weeks_played") else 0,
            "name": m.get("team_name", row["team"]),
            "color": m.get("team_color") or "#334155",
            "logo": m.get("team_logo_espn", ""),
            "qb": qbs.get(row["team"], ""),
        })
    return jsonify(recs)


_UNIT_EPA_CACHE = {}


@app.route('/api/unit_epa')
def api_unit_epa():
    """Per-team opponent-adjusted EPA/play split into pass vs rush, for both offense and defense —
    powers the quadrant scatter on the Rankings page. Latest completed season (2025 by default), since
    EPA needs games played. Convention: off_* higher = better offense; def_* is EPA ALLOWED so lower =
    better defense (the frontend negates it so 'up-right = elite in both phases' reads the same way)."""
    arg = request.args.get('season', 'current')
    if arg in _UNIT_EPA_CACHE:
        return jsonify(_UNIT_EPA_CACHE[arg])
    from ml.adjust import adjusted_unit_epa
    st = season_state()
    if arg == 'current':          # in-season: this season shrunk toward last season by games played
        from ml.current import adjusted_units
        adj = adjusted_units()
        season = st["season"]
    else:
        season = int(arg)
        adj = adjusted_unit_epa(season)
    meta = team_meta()
    recs = []
    for team, u in adj.items():
        m = meta.get(team, {})
        recs.append({
            "team": team, "name": m.get("team_name", team),
            "color": m.get("team_color") or "#334155", "logo": m.get("team_logo_espn", ""),
            "off_pass": u.get("off_pass"), "off_rush": u.get("off_rush"),
            "def_pass": u.get("def_pass"), "def_rush": u.get("def_rush"),
        })
    payload = {"season": season, "mode": arg, "teams": recs,
               "state": st if (arg == 'current' and st["in_progress"]) else None}
    _UNIT_EPA_CACHE[arg] = payload
    return jsonify(payload)


_LEAGUE_STATS_CACHE = {}

# Per-side stat leaderboards for the Rankings page. Each column pulls a raw team_styles
# metric (or a schedule-derived scoring average) and declares its direction so the frontend
# can rank + color it. `pct` columns are stored 0-1 and rendered ×100; `better` fixes which
# end of the distribution is #1 (defense EPA/points are "lower = better").
_OFF_COLS = [
    {"key": "epa",     "label": "EPA/play",   "src": "off_epa_per_play", "better": "hi", "dec": 2, "pct": False,
     "tip": "Expected Points Added per offensive play — how many points the average snap gains vs a league-average play from the same spot. The best single number for offense quality."},
    {"key": "pts",     "label": "Pts/G",      "src": "_pf",              "better": "hi", "dec": 1, "pct": False,
     "tip": "Points scored per game (regular season, from final scores)."},
    {"key": "success", "label": "Success%",   "src": "off_success_rate", "better": "hi", "dec": 1, "pct": True,
     "tip": "Share of plays that gain positive expected points (stay 'on schedule'). Measures consistency, where EPA can be skewed by a few big plays."},
    {"key": "pass",    "label": "Pass EPA",   "src": "off_epa_per_pass", "better": "hi", "dec": 2, "pct": False,
     "tip": "EPA per dropback — passing-game efficiency including sacks and scrambles."},
    {"key": "rush",    "label": "Rush EPA",   "src": "off_epa_per_rush", "better": "hi", "dec": 2, "pct": False,
     "tip": "EPA per designed rush — running-game efficiency. League average is slightly negative (passing is more efficient)."},
    {"key": "rztd",    "label": "RZ TD%",     "src": "rz_td_rate",       "better": "hi", "dec": 1, "pct": True,
     "tip": "Share of red-zone plays that end in a touchdown — finishing drives with 7 instead of 3."},
    {"key": "sk_all",  "label": "Sack% all'd","src": "sack_rate_allowed","better": "lo", "dec": 1, "pct": True,
     "tip": "Share of dropbacks where the QB is sacked — pass protection (lower is better)."},
    {"key": "pen",     "label": "Pen/G",      "src": "off_penalties_pg", "better": "lo", "dec": 1, "pct": False,
     "tip": "Offensive penalties per game (false starts, holding — negated plays included). Lower = more disciplined."},
]
# NOTE on defense EPA sign: team_styles stores def_epa_per_* already NEGATED
# (def_epa_per_play = -mean(EPA), so HIGHER = better defense) and def_success_rate as a
# STOP rate (1 - success allowed, higher = better). We `neg` the EPA columns back to raw
# "EPA allowed" for display (negative = elite, matching the Unit EPA map + how fans read it)
# and rank them lo=best; points-allowed is the only raw "lower is better" metric.
_DEF_COLS = [
    {"key": "epa",     "label": "EPA/play",   "src": "def_epa_per_play", "better": "lo", "dec": 2, "pct": False, "neg": True,
     "tip": "Expected Points Added allowed per play — how many points the average opposing snap gains against this defense. Negative = the defense takes points off the board; the best single number for defense quality."},
    {"key": "pts",     "label": "Pts/G",      "src": "_pf",              "better": "lo", "dec": 1, "pct": False,
     "tip": "Points allowed per game (regular season, from final scores). Includes points given up by turnovers/special teams, so it can diverge from per-play EPA."},
    {"key": "success", "label": "Stop%",      "src": "def_success_rate", "better": "hi", "dec": 1, "pct": True,
     "tip": "Share of opponent plays stopped for negative expected points — down-to-down consistency of the defense."},
    {"key": "pass",    "label": "Pass EPA",   "src": "def_epa_per_pass", "better": "lo", "dec": 2, "pct": False, "neg": True,
     "tip": "EPA allowed per opponent dropback — pass defense (coverage + pass rush). Negative = elite."},
    {"key": "rush",    "label": "Rush EPA",   "src": "def_epa_per_rush", "better": "lo", "dec": 2, "pct": False, "neg": True,
     "tip": "EPA allowed per opponent rush — run defense. Negative = elite."},
    {"key": "sack",    "label": "Sack%",      "src": "sack_rate_gen",    "better": "hi", "dec": 1, "pct": True,
     "tip": "Share of opponent dropbacks ending in a sack — pass-rush production."},
    {"key": "stop3",   "label": "3rd stop%",  "src": "third_down_stop_rate", "better": "hi", "dec": 1, "pct": True,
     "tip": "Share of opponent third downs that fail to convert — getting off the field."},
    {"key": "pen",     "label": "Pen/G",      "src": "def_penalties_pg", "better": "lo", "dec": 1, "pct": False,
     "tip": "Defensive penalties per game (offsides, DPI, holding — negated plays included). Lower = more disciplined."},
]


def _scoring_avgs(season: int, blend: bool = True) -> dict:
    """{team: (points_for_avg, points_against_avg)} from final scores in schedules.

    blend=True applies the same in-season shrinkage as every other column (ml/current.py):
    an in-progress season's per-game points are blended toward last season's by games
    played. Without this the Pts/G column read a 36-point week-1 game as a full-season
    number beside EPA columns that had been shrunk — two rules in one table."""
    s = schedules_df()
    if not len(s):
        return {}

    def _avg(season_):
        sc = s[(s["season"] == season_) & s["home_score"].notna() & (s["week"] <= 18)]
        pf, pa = {}, {}
        for _, g in sc.iterrows():
            for team, scored, allowed in ((g["home_team"], g["home_score"], g["away_score"]),
                                          (g["away_team"], g["away_score"], g["home_score"])):
                pf.setdefault(team, []).append(scored)
                pa.setdefault(team, []).append(allowed)
        return {t: (float(np.mean(pf[t])), float(np.mean(pa.get(t, [0])))) for t in pf}

    cur = _avg(season)
    if not blend:
        return cur
    try:
        from ml.current import state, weights
        st, w = state(), weights()
        if season != st["season"] or not st["in_progress"] or st["prior"] is None or not len(w):
            return cur
        pri = _avg(st["prior"])
        out = {}
        for t in set(cur) | set(pri):
            c, p, wt = cur.get(t), pri.get(t), float(w.get(t, 0.0))
            if c is None:
                out[t] = p
            elif p is None:
                out[t] = c
            else:
                out[t] = (wt * c[0] + (1 - wt) * p[0], wt * c[1] + (1 - wt) * p[1])
        return out
    except Exception:
        return cur


def _stat_side(df: pd.DataFrame, cols: list, meta: dict, scoring: dict, is_def: bool) -> dict:
    """Build one side (offense/defense) leaderboard: value + league rank per cell."""
    rows = []
    for _, r in df.iterrows():
        team = r["team"]
        vals = {}
        for c in cols:
            src = c["src"]
            if src == "_pf":
                v = scoring.get(team, (None, None))[1 if is_def else 0]
            else:
                v = safe_json(r[src]) if src in r.index else None
                if v is not None and c.get("neg"):     # flip stored "-EPA allowed" back to raw EPA allowed
                    v = -v
            vals[c["key"]] = v
        m = meta.get(team, {})
        rows.append({"team": team, "name": m.get("team_name", team),
                     "color": m.get("team_color") or "#334155",
                     "logo": m.get("team_logo_espn", ""), "vals": vals})
    # rank each column (1 = best), respecting direction; ties share the min rank
    for c in cols:
        present = [x for x in rows if x["vals"][c["key"]] is not None]
        present.sort(key=lambda x: x["vals"][c["key"]], reverse=(c["better"] == "hi"))
        prev, rk = None, 0
        for i, x in enumerate(present):
            val = x["vals"][c["key"]]
            if val != prev:
                rk = i + 1
                prev = val
            x.setdefault("ranks", {})[c["key"]] = rk
    out = []
    for x in rows:
        out.append({"team": x["team"], "name": x["name"], "color": x["color"], "logo": x["logo"],
                    "cells": {k: {"v": x["vals"][k], "r": x.get("ranks", {}).get(k)} for k in x["vals"]}})
    out.sort(key=lambda x: (x["cells"]["epa"]["r"] or 99))
    return {"columns": cols, "rows": out}


@app.route('/api/league_stats')
def api_league_stats():
    """Per-team offense & defense stat leaderboards (value + league rank per metric) for the
    Rankings page. Latest completed season by default — these are actual on-field results, so
    they need games played (unlike the roster-talent power ranking)."""
    season = int(request.args.get('season', latest_style_season()))
    raw = request.args.get('raw') == '1'             # season-to-date only, no shrink toward last season
    key = (season, raw)
    if key in _LEAGUE_STATS_CACHE:
        return jsonify(_LEAGUE_STATS_CACHE[key])
    s = styles_df(raw=raw)
    sub = s[s["season"] == season]
    if sub.empty:
        return jsonify({"error": f"no stats for {season}"}), 404
    meta = team_meta()
    scoring = _scoring_avgs(season, blend=not raw)
    st = season_state()
    payload = {
        "season": season, "raw": raw,
        "state": st if season == st["season"] else {"season": season, "in_progress": False, "weeks_played": 0, "blend_w": 0.0, "prior": None},
        "offense": _stat_side(sub, _OFF_COLS, meta, scoring, is_def=False),
        "defense": _stat_side(sub, _DEF_COLS, meta, scoring, is_def=True),
    }
    _LEAGUE_STATS_CACHE[key] = payload
    return jsonify(payload)


_DEPTH_CACHE = {}
_PFF_COMPARE = None


@app.route('/api/qb_history')
def api_qb_history():
    """Per-QB, per-season opponent-adjusted passing table (rolling — grows as the
    current season's PBP lands via the daily refresh)."""
    from ml.history import qb_seasons
    return jsonify(_native(qb_seasons()))


@app.route('/api/unit_history')
def api_unit_history():
    """Team offense/defense opponent-adjusted unit EPA for every season with PBP."""
    from ml.history import unit_epa_history
    return jsonify(_native(unit_epa_history()))


# ── PFF unit reports (per-team drilldowns from the preseason facet tables) ──────
# (parquet file, unit label, sort/grade column, volume column that must be > 0,
#  then the curated analyst columns as (col, short label, decimals)).
_PFF_UNIT_SPEC = [
    ("passing", "pff_preseason_passing", "Passing", "grades_pass", "dropbacks", [
        ("grades_pass", "Grade", 1), ("dropbacks", "DB", 0), ("completions", "Cmp", 0),
        ("attempts", "Att", 0), ("yards", "Yds", 0), ("touchdowns", "TD", 0),
        ("interceptions", "INT", 0), ("ypa", "YPA", 1), ("accuracy_percent", "Acc%", 1),
        ("btt_rate", "BTT%", 1), ("twp_rate", "TWP%", 1), ("avg_time_to_throw", "TTT", 2),
        ("qb_rating", "Rate", 1)]),
    ("pressure", "pff_preseason_qb_pressure", "QB vs pressure", "grades_pass", "pressure_dropbacks", [
        ("grades_pass", "Grade", 1), ("pressure_dropbacks", "PrsDB", 0),
        ("pressure_completion_percent", "PrsCmp%", 1), ("pressure_qb_rating", "PrsRate", 1),
        ("pressure_twp_rate", "PrsTWP%", 1), ("pressure_to_sack_rate", "Prs→Sk%", 1),
        ("no_pressure_qb_rating", "CleanRate", 1), ("blitz_qb_rating", "BlitzRate", 1)]),
    ("rushing", "pff_preseason_rushing", "Rushing", "grades_run", "attempts", [
        ("grades_run", "Grade", 1), ("attempts", "Att", 0), ("yards", "Yds", 0),
        ("ypa", "YPA", 1), ("touchdowns", "TD", 0), ("yco_attempt", "YAC/A", 1),
        ("avoided_tackles", "MTF", 0), ("elusive_rating", "Elu", 1),
        ("explosive", "Expl", 0), ("fumbles", "Fum", 0)]),
    ("receiving", "pff_preseason_receiving", "Receiving", "grades_pass_route", "targets", [
        ("grades_pass_route", "Grade", 1), ("targets", "Tgt", 0), ("receptions", "Rec", 0),
        ("yards", "Yds", 0), ("touchdowns", "TD", 0), ("yprr", "YPRR", 2),
        ("avg_depth_of_target", "ADOT", 1), ("drops", "Drop", 0),
        ("contested_catch_rate", "Cont%", 1), ("targeted_qb_rating", "TgtRate", 1)]),
    ("pass_block", "pff_preseason_pass_blocking", "Pass blocking", "grades_pass_block", "snap_counts_pass_block", [
        ("grades_pass_block", "Grade", 1), ("snap_counts_pass_block", "Snaps", 0),
        ("pbe", "PBE", 1), ("pressures_allowed", "Prs", 0), ("sacks_allowed", "Sk", 0),
        ("hits_allowed", "Hit", 0), ("hurries_allowed", "Hur", 0),
        ("true_pass_set_pbe", "TPS PBE", 1), ("penalties", "Pen", 0)]),
    ("run_block", "pff_preseason_run_blocking", "Run blocking", "grades_run_block", "snap_counts_run_block", [
        ("grades_run_block", "Grade", 1), ("snap_counts_run_block", "Snaps", 0),
        ("gap_grades_run_block", "Gap", 1), ("zone_grades_run_block", "Zone", 1),
        ("penalties", "Pen", 0)]),
    ("pass_rush", "pff_preseason_pass_rush", "Pass rush", "grades_pass_rush_defense", "snap_counts_pass_rush", [
        ("grades_pass_rush_defense", "Grade", 1), ("snap_counts_pass_rush", "Snaps", 0),
        ("total_pressures", "Prs", 0), ("sacks", "Sk", 0), ("hits", "Hit", 0),
        ("hurries", "Hur", 0), ("pass_rush_win_rate", "Win%", 1), ("prp", "PRP", 1),
        ("batted_passes", "Bat", 0)]),
    ("coverage", "pff_preseason_coverage", "Coverage", "grades_coverage_defense", "snap_counts_coverage", [
        ("grades_coverage_defense", "Grade", 1), ("snap_counts_coverage", "Snaps", 0),
        ("targets", "Tgt", 0), ("receptions", "Rec", 0), ("catch_rate", "Cat%", 1),
        ("yards", "Yds", 0), ("yards_per_coverage_snap", "Y/CS", 2),
        ("qb_rating_against", "RateAg", 1), ("pass_break_ups", "PBU", 0),
        ("interceptions", "INT", 0), ("missed_tackles", "MT", 0), ("stops", "Stop", 0)]),
    ("run_def", "pff_preseason_defense", "Run defense", "grades_run_defense", "player_game_count", [
        ("grades_run_defense", "Grade", 1), ("grades_tackle", "Tackling", 1),
        ("tackles", "Tkl", 0), ("assists", "Ast", 0), ("stops", "Stop", 0),
        ("missed_tackles", "MT", 0), ("forced_fumbles", "FF", 0),
        ("grades_defense", "Def grade", 1)]),
]
_PFF_UNITS_CACHE = {}


@app.route('/api/pff_units')
def api_pff_units():
    """Per-team PFF unit reports (the premium 'Team Reports' drilldowns): every unit's
    player table with the stats an analyst actually reads. Built from the league-wide
    preseason facet parquets; absent files → available:false."""
    team = (request.args.get('team') or '').upper()
    if not team:
        return jsonify({"error": "team required"}), 400
    if team in _PFF_UNITS_CACHE:
        return jsonify(_PFF_UNITS_CACHE[team])
    # In-season facet files (pff_season_*, built by pff_facets.py from the regular-season
    # weeks) take precedence over the preseason ones; the card says which it is showing.
    units, season_tag, source = [], "2026 preseason", "preseason"
    for key, fname, label, grade_col, vol_col, cols in _PFF_UNIT_SPEC:
        p_season = PROC / f"{fname.replace('pff_preseason_', 'pff_season_')}.parquet"
        p = p_season if p_season.exists() else PROC / f"{fname}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if p is p_season:
            source = "season"
            tw = int(d["through_week"].max()) if "through_week" in d.columns and d["through_week"].notna().any() else None
            season_tag = f"2026 regular season{f' · through week {tw}' if tw else ''}"
        if "team" not in d.columns or grade_col not in d.columns:
            continue
        d = d[(d["team"] == team) & d[grade_col].notna()]
        if vol_col in d.columns:
            d = d[pd.to_numeric(d[vol_col], errors="coerce").fillna(0) > 0]
        if d.empty:
            continue
        cols_avail = [(c, lbl, dec) for c, lbl, dec in cols if c in d.columns]
        d = d.sort_values(grade_col, ascending=False)
        rows = []
        for r in d.itertuples():
            row = {"player": r.player, "position": getattr(r, "position", ""),
                   "games": safe_json(getattr(r, "player_game_count", None))}
            for c, _, dec in cols_avail:
                v = getattr(r, c, None)
                row[c] = round(float(v), dec) if v is not None and pd.notna(v) else None
            rows.append(row)
        units.append({"key": key, "label": label,
                      "columns": [{"k": c, "label": lbl, "dec": dec} for c, lbl, dec in cols_avail],
                      "grade_col": grade_col, "rows": rows})
    payload = {"available": bool(units), "team": team, "season": season_tag, "source": source, "units": units}
    _PFF_UNITS_CACHE[team] = payload
    return jsonify(_native(payload))


@app.route('/api/pff_upload', methods=['POST'])
def api_pff_upload():
    """Owner-only upload of the locally built PFF parquets to the server volume.
    Guarded by REFRESH_TOKEN (must be configured — refuses when absent so an unconfigured
    deployment can't accept anonymous uploads). Validates schema before writing."""
    global _PFF_COMPARE
    tok = os.environ.get("REFRESH_TOKEN")
    if not tok or (request.headers.get("X-Refresh-Token") or "") != tok:
        return jsonify({"error": "unauthorized"}), 403
    import io as _io
    saved = {}
    spec = {"grades": ("pff_grades.parquet", {"pff_grade", "nm", "team"}),
            "team_grades": ("pff_team_grades.parquet", {"grades_overall", "team", "pff_rank"})}
    for field, (fname, need) in spec.items():
        f = request.files.get(field)
        if f is None:
            continue
        try:
            df = pd.read_parquet(_io.BytesIO(f.read()))
        except Exception as e:
            return jsonify({"error": f"{field}: not a readable parquet ({e})"}), 400
        if not need.issubset(df.columns):
            return jsonify({"error": f"{field}: missing columns {sorted(need - set(df.columns))}"}), 400
        df.to_parquet(PROC / fname, index=False)
        saved[field] = len(df)
    if not saved:
        return jsonify({"error": "no files provided (fields: grades, team_grades)"}), 400
    _PFF_COMPARE = None                                # rebuilt on next request
    _DEPTH_CACHE.clear()
    _PFF_UNITS_CACHE.clear()
    try:
        import ml.squad as _sq
        _sq._PFF_CACHE = _sq._PFF_PRE_CACHE = _sq._PFF_TABLE = None
    except Exception:
        pass
    return jsonify({"ok": True, "saved": saved})


@app.route('/api/pff_compare')
def api_pff_compare():
    """Model-vs-PFF disagreement view. Needs locally imported PFF data (subscriber-only,
    git-ignored, never on the hosted volume) — returns available:false without it.
    Player comparison is percentile-vs-percentile: our rating IS a position percentile,
    so PFF grades are converted to percentiles within their PFF position group; comparing
    raw grade to percentile would manufacture fake disagreements."""
    global _PFF_COMPARE
    if _PFF_COMPARE is not None:
        return jsonify(_PFF_COMPARE)
    pg_path = PROC / "pff_grades.parquet"
    if not pg_path.exists():
        return jsonify({"available": False})
    from ml.squad import squad_ratings, team_depth_chart, _norm, _key
    d = pd.read_parquet(pg_path).dropna(subset=["pff_grade"])
    # Only compare players PFF itself considers graded (meets_snap_minimum): a grade off a
    # handful of snaps is noise, and it flooded the disagreement lists with depth players.
    # Percentiles are computed within the qualifying population for the same reason.
    if "qualifies" in d.columns:
        d = d[d["qualifies"]]
    d["pff_pctl"] = d.groupby(d["position"].replace({"FB": "HB"}))["pff_grade"] \
        .rank(pct=True) * 100    # FB folded into HB — tiny cohorts mint fake percentiles
    by_nt = {(r.nm, r.team): (float(r.pff_grade), float(r.pff_pctl)) for r in d.itertuples()}
    uniq = d[~d.duplicated(["key", "team"], keep=False)]
    by_key = {(r.key, r.team): (float(r.pff_grade), float(r.pff_pctl)) for r in uniq.itertuples()}

    meta = team_meta()
    ranks, _ = squad_ratings()
    rows = []
    for team in ranks["team"]:
        if team not in _DEPTH_CACHE:
            _DEPTH_CACHE[team] = team_depth_chart(team)
        for g in _DEPTH_CACHE[team]:
            for p in g["players"]:
                # measured ratings only (projections aren't an opinion worth comparing),
                # top-2 depth (starters/primary backups) to keep the list meaningful.
                if p.get("source") != "measured" or p.get("pff") is None:
                    continue
                if p.get("rank") and p["rank"] > 2:
                    continue
                hit = by_nt.get((_norm(p["name"]), team)) or by_key.get((_key(p["name"]), team))
                if not hit:
                    continue
                m = meta.get(team, {})
                rows.append({"name": p["name"], "team": team, "pos": g["pos"],
                             "ours": p["rating"], "pff": round(hit[0], 1), "pff_pctl": int(round(hit[1])),
                             "gap": int(round(p["rating"] - hit[1])),
                             "logo": m.get("team_logo_espn", "")})
    rows.sort(key=lambda x: -x["gap"])
    payload = {"available": True, "n_players": len(rows),
               "model_high": rows[:15], "pff_high": rows[::-1][:15]}

    # 2026 preseason block: PFF's preseason team ranking + standout performers. Preseason
    # grades skew toward depth players getting snaps — clearly labeled, never blended into
    # the team power ranking.
    pre_path = PROC / "pff_preseason_players.parquet"
    pre_teams_path = PROC / "pff_preseason_teams.parquet"
    if pre_teams_path.exists():
        pt = pd.read_parquet(pre_teams_path)
        pre_rows = []
        for r in pt.itertuples():
            m = meta.get(r.team, {})
            pre_rows.append({"team": r.team, "name": m.get("team_name", r.team),
                             "logo": m.get("team_logo_espn", ""),
                             "pre_rank": int(r.pre_rank), "overall": float(r.grades_overall),
                             "off": float(r.grades_offense), "def": float(r.grades_defense),
                             "record": f"{int(r.wins)}-{int(r.losses)}" + (f"-{int(r.ties)}" if r.ties else "")})
        payload["preseason_teams"] = sorted(pre_rows, key=lambda x: x["pre_rank"])
    if pre_path.exists():
        pp = pd.read_parquet(pre_path).dropna(subset=["pre_grade"])
        pp = pp[pp.get("games", pd.Series(1, index=pp.index)).fillna(1) >= 2]  # damp one-series wonders
        top = pp.nlargest(12, "pre_grade")
        payload["preseason_top"] = [{
            "name": r.player, "team": r.team, "pos": r.position,
            "grade": round(float(r.pre_grade), 1), "games": int(r.games) if pd.notna(r.games) else None,
            "logo": meta.get(r.team, {}).get("team_logo_espn", ""),
        } for r in top.itertuples()]

    tg_path = PROC / "pff_team_grades.parquet"
    if tg_path.exists():
        tg = pd.read_parquet(tg_path)
        our_rank = dict(zip(ranks["team"], ranks["rank"]))
        teams = []
        for r in tg.itertuples():
            m = meta.get(r.team, {})
            orank = int(our_rank.get(r.team, 0))
            teams.append({"team": r.team, "name": m.get("team_name", r.team),
                          "logo": m.get("team_logo_espn", ""), "color": m.get("team_color") or "#334155",
                          "our_rank": orank, "pff_rank": int(r.pff_rank),
                          "delta": int(r.pff_rank) - orank,      # + = we're higher on them than PFF
                          "pff_overall": float(r.grades_overall),
                          "record": f"{int(r.wins)}-{int(r.losses)}" + (f"-{int(r.ties)}" if r.ties else "")})
        teams.sort(key=lambda x: x["our_rank"])
        payload["teams"] = {"season": 2025, "rows": teams}
    _PFF_COMPARE = payload
    return jsonify(_native(payload))


@app.route('/api/team')
def api_team():
    """Full 2026 depth chart for a team with per-player 2025 position-percentile ratings."""
    team = request.args.get('team', '').upper()
    if not team:
        return jsonify({"error": "team required"}), 400
    if team not in _DEPTH_CACHE:
        from ml.squad import team_depth_chart
        _DEPTH_CACHE[team] = team_depth_chart(team)
    m = team_meta().get(team, {})
    qb = qb1_2026().get(team, "")
    return jsonify(_native({    # _native: camp players can carry NaN ids → invalid JSON otherwise
        "team": team, "name": m.get("team_name", team),
        "color": m.get("team_color") or "#334155", "logo": m.get("team_logo_espn", ""),
        "qb": qb, "groups": _DEPTH_CACHE[team],
    }))


_PROJ_CACHE = {}


@app.route('/api/matchup_players')
def api_matchup_players():
    """Projected per-player stat lines for a matchup (SportsLine-style box score)."""
    home = (request.args.get('home') or '').upper()
    away = (request.args.get('away') or '').upper()
    if not home or not away or home == away:
        return jsonify({"error": "two different teams required"}), 400
    # Keyed on the AVAILABILITY inputs too: a box score projected before Thursday's injury
    # report must not keep serving a ruled-out starter until the next full cache clear.
    key = (home, away, _avail_sig())
    if key not in _PROJ_CACHE:
        from ml.projections import project_matchup
        _PROJ_CACHE.clear()                            # inputs moved → every cached box is stale
        _PROJ_CACHE[key] = project_matchup(home, away)
    return jsonify(_native(_PROJ_CACHE[key]))


def _avail_sig() -> tuple:
    """mtimes of the files that decide who is available (injury report + roster release)."""
    sig = []
    for name in ("injuries.parquet", "rosters_2026.parquet", "depth_2026_current.parquet"):
        p = RAW / name
        sig.append(int(p.stat().st_mtime) if p.exists() else 0)
    return tuple(sig)


def _native(obj):
    """Recursively convert numpy types / NaN to JSON-native values."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    return safe_json(obj)


@app.route('/api/matchup')
def api_matchup():
    """Predict a matchup with the unit-vs-unit engine (differentiated total + unit edges)."""
    from ml.matchup_engine import project_game
    home = request.args.get('home')
    away = request.args.get('away')
    neutral = request.args.get('neutral', '0') == '1'
    if not home or not away:
        return jsonify({"error": "home and away required"}), 400
    res = project_game(home.upper(), away.upper(), neutral)
    if "error" in res:
        return jsonify(res), 404
    meta = team_meta()
    for side in ("home", "away"):
        m = meta.get(res[side], {})
        res[f"{side}_name"] = m.get("team_name", res[side])
        res[f"{side}_color"] = m.get("team_color") or "#334155"
        res[f"{side}_logo"] = m.get("team_logo_espn", "")
    return jsonify(_native(res))


@app.route('/api/weeks')
def get_weeks():
    """List all available prediction files."""
    files = sorted(PROC.glob("predictions_*.parquet"))
    weeks = []
    for f in files:
        parts = f.stem.split('_')  # predictions_2024_wk15
        if len(parts) >= 3:
            weeks.append({
                "file": f.name,
                "season": parts[1],
                "week": parts[2].replace('wk', ''),
                "label": f"Season {parts[1]} Week {parts[2].replace('wk','')}",
            })
    return jsonify(weeks)


@app.route('/api/predictions')
def get_predictions():
    season = request.args.get('season', '2024')
    week   = request.args.get('week', '15')
    path   = PROC / f"predictions_{season}_wk{int(week):02d}.parquet"
    if not path.exists():
        return jsonify({"error": f"No predictions for season {season} week {week}. Run: python run_engine.py --season {season} --week {week}"}), 404
    df = pd.read_parquet(path)
    return jsonify(df_to_json(df))


@app.route('/api/predict')
def predict_single():
    home   = request.args.get('home')
    away   = request.args.get('away')
    season = int(request.args.get('season', 2024))
    week   = int(request.args.get('week', 18))
    if not home or not away:
        return jsonify({"error": "home and away required"}), 400
    from engine.predict import load_engine_data, predict_game
    data = load_engine_data()
    pred = predict_game(home, away, season, week, data=data)
    pred["key_matchups"] = pred.get("key_matchups", [])
    return jsonify({k: safe_json(v) for k, v in pred.items()})


@app.route('/api/styles')
def get_styles():
    season = int(request.args.get('season', 2024))
    path = PROC / "team_styles.parquet"
    if not path.exists():
        return jsonify({"error": "Run run_engine.py first"}), 404
    df = pd.read_parquet(path)
    df = df[df["season"] == season]
    cols = ["team", "season", "offense_label", "defense_label",
            "pass_rate_overall", "avg_air_yards", "off_epa_per_play",
            "def_epa_per_play", "def_quality_score", "sack_rate",
            "turnover_rate", "third_down_stop_rate", "pace",
            "run_heavy_off", "pass_heavy_off", "blitz_heavy_def",
            "strong_run_def", "strong_pass_def"]
    cols = [c for c in cols if c in df.columns]
    return jsonify(df_to_json(df[cols]))


@app.route('/api/composite')
def get_composite():
    season   = int(request.args.get('season', 2024))
    week     = int(request.args.get('week', 18))
    position = request.args.get('position', None)
    team     = request.args.get('team', None)
    path = PROC / "composite_scores.parquet"
    if not path.exists():
        return jsonify({"error": "Run run_engine.py first"}), 404
    df = pd.read_parquet(path)
    df = df[(df["season"] == season) & (df["week"] <= week)]
    df = df.sort_values("week", ascending=False).drop_duplicates("player_id")
    if position:
        df = df[df["position"] == position.upper()]
    if team:
        df = df[df["recent_team"] == team.upper()]
    df = df.sort_values("adjusted_score", ascending=False).head(100)
    cols = ["player_display_name", "position", "recent_team", "season", "week",
            "composite_score", "adjusted_score", "tier", "pos_rank",
            "rank_score", "efficiency_score", "usage_score",
            "tracking_score", "athleticism_score"]
    cols = [c for c in cols if c in df.columns]
    return jsonify(df_to_json(df[cols]))


@app.route('/api/backtest')
def get_backtest():
    season = request.args.get('season', '2024')
    files  = list(PROC.glob(f"backtest_*{season}*.parquet"))
    if not files:
        return jsonify({"error": f"No backtest data. Run: python backtest.py --season {season}"}), 404
    df = pd.read_parquet(files[0])
    summary = {
        "games":        int(len(df)),
        "winner_acc":   round(float(df["winner_correct"].mean()), 3),
        "spread_mae":   round(float(df["abs_spread_err"].mean()), 1),
        "total_mae":    round(float(df["abs_total_err"].mean()), 1),
        "spread_bias":  round(float(df["spread_error"].mean()), 2),
        "total_bias":   round(float(df["total_error"].mean()), 2),
        "ats_acc":      round(float(df[df["ats_correct"].notna()]["ats_correct"].mean()), 3) if "ats_correct" in df.columns else None,
        "ou_acc":       round(float(df[df["ou_correct"].notna()]["ou_correct"].mean()), 3) if "ou_correct" in df.columns else None,
        "by_week":      df.groupby("week")["winner_correct"].agg(["mean","count"]).reset_index().rename(columns={"mean":"acc","count":"games"}).to_dict("records"),
    }
    return jsonify(summary)


@app.route('/api/teams')
def get_teams():
    path = RAW / "team_info.parquet"
    if not path.exists():
        return jsonify([])
    df = pd.read_parquet(path)
    cols = ["team_abbr","team_name","team_nick","team_conf","team_division","team_color","team_color2"]
    cols = [c for c in cols if c in df.columns]
    return jsonify(df_to_json(df[cols]))


# ═══════════════════════════════════════════════════════════════════
#  RESEARCH FEATURES — team profile, trends, full matchup
#  All read the parquet already in the repo; no network at request time.
# ═══════════════════════════════════════════════════════════════════

# ── lazy dataframe caches ───────────────────────────────────────────
_STYLES = None
_INJ = None
_SCHED = None
_PBP_CACHE = {}


_STYLES_RAW = None


def styles_df(raw: bool = False) -> pd.DataFrame:
    """team_styles with the IN-PROGRESS season's rows blended toward last season by games
    played (ml/current.py) — the one rule every in-season view shares, so League Stats,
    profiles, labels and the matchup grid never rank 32 teams off a single game.
    raw=True returns the built table untouched (season-to-date only)."""
    global _STYLES, _STYLES_RAW
    if _STYLES_RAW is None:
        _STYLES_RAW = pd.read_parquet(PROC / "team_styles.parquet")
    if raw:
        return _STYLES_RAW
    if _STYLES is None:
        try:
            from ml.current import blended_styles
            _STYLES = blended_styles(_STYLES_RAW)
        except Exception as e:
            print(f"[styles] blend failed, serving raw: {e}", flush=True)
            _STYLES = _STYLES_RAW
    return _STYLES


def season_state() -> dict:
    """Current-season progress for view captions: {season, prior, weeks_played, blend_w, in_progress}."""
    try:
        from ml.current import state, league_weight
        st = state()
        return {"season": st["season"], "prior": st["prior"], "weeks_played": st["weeks_played"],
                "in_progress": st["in_progress"], "blend_w": league_weight() if st["in_progress"] else 0.0}
    except Exception:
        return {"season": latest_style_season(), "prior": None, "weeks_played": 0, "in_progress": False, "blend_w": 0.0}


def injuries_df() -> pd.DataFrame:
    global _INJ
    if _INJ is None:
        p = RAW / "injuries.parquet"
        _INJ = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _INJ


def schedules_df() -> pd.DataFrame:
    global _SCHED
    if _SCHED is None:
        p = RAW / "schedules.parquet"
        _SCHED = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _SCHED


def pbp_season(season: int) -> pd.DataFrame:
    if season not in _PBP_CACHE:
        p = RAW / f"pbp_{season}.parquet"
        _PBP_CACHE[season] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _PBP_CACHE[season]


def latest_style_season() -> int:
    s = styles_df()
    return int(s["season"].max()) if len(s) else 2025


# ── team strengths / weaknesses via league percentiles ──────────────
# (column, human label, higher_is_better) — direction-normalised so a high
# percentile always means "good".
_PROFILE_METRICS = [
    ("off_epa_per_play",     "Offense EPA/play",     True),
    ("off_epa_per_pass",     "Passing offense",      True),
    ("off_epa_per_rush",     "Rushing offense",      True),
    ("off_success_rate",     "Offensive efficiency", True),
    ("rz_td_rate",           "Red-zone TD rate",     True),
    ("two_min_epa",          "Two-minute offense",   True),
    # team_styles stores def_epa_per_* NEGATED (higher = better) and def_success_rate as a
    # STOP rate — so higher IS better for all four, despite the "allowed" mental model.
    ("def_epa_per_play",     "Defense EPA/play",     True),
    ("def_epa_per_pass",     "Pass defense",         True),
    ("def_epa_per_rush",     "Run defense",          True),
    ("def_success_rate",     "Defensive efficiency", True),
    ("pressure_rate_gen",    "Pass-rush pressure",   True),
    ("sack_rate_gen",        "Sack rate",            True),
    ("third_down_stop_rate", "Third-down defense",   True),
    ("def_quality_score",    "Overall defense grade", True),
]

# Tendency chips from REAL league percentiles. The boolean archetype flags in team_styles
# are miscalibrated (poor_qb_contain is True for all 32 teams, elite_mobile_qb for 27, and
# most others never fire), so we derive tendencies from the continuous columns instead.
# (column, tag when top of league, tag when bottom, top/bottom fraction that qualifies)
_TENDENCY_SPEC = [
    ("pass_rate_overall",    "Pass-heavy offense",        "Run-heavy offense",   0.18),
    ("avg_air_yards",        "Deep passing attack",       "Short passing game",  0.16),
    ("pace",                 "Fast tempo",                "Slow tempo",          0.16),
    ("play_action_rate",     "High play-action",          None,                  0.16),
    ("motion_rate",          "Heavy pre-snap motion",     None,                  0.16),
    ("qb_rush_rate",         "Mobile QB",                 None,                  0.16),
    ("avg_blitzers",         "Blitz-heavy defense",       None,                  0.16),
    ("off_epa_per_play",     "Explosive offense",         None,                  0.12),
    ("rz_td_rate",           "Elite red-zone offense",    None,                  0.12),
    ("two_min_epa",          "Strong two-minute offense", None,                  0.15),
    ("third_down_stop_rate", "Strong third-down defense", None,                  0.16),
    ("sack_rate_gen",        "Heavy pass rush",           None,                  0.16),
    ("def_epa_per_play",     "Stingy defense",            None,                  0.15),  # stored higher=better
    ("off_penalties_pg",     "Penalty-prone offense",     "Disciplined offense", 0.15),
    ("def_penalties_pg",     "Penalty-prone defense",     None,                  0.12),
]


def _tendencies(team: str, season: int) -> list:
    """Real per-team tendencies from where the team sits in the league distribution."""
    s = styles_df()
    s = s[s["season"] == season]
    tags = []
    for col, hi, lo, frac in _TENDENCY_SPEC:
        if col not in s.columns:
            continue
        cv = s[["team", col]].dropna()
        if team not in set(cv["team"]):
            continue
        p = float(cv[col].rank(pct=True)[cv["team"] == team].iloc[0])
        if hi and p >= 1 - frac:
            tags.append(hi)
        elif lo and p <= frac:
            tags.append(lo)
    return tags


def _profile_percentiles(team: str, season: int) -> list:
    """League-relative percentile (0-100, higher=better) for each curated metric."""
    s = styles_df()
    s = s[s["season"] == season]
    out = []
    for col, label, higher in _PROFILE_METRICS:
        if col not in s.columns:
            continue
        cv = s[["team", col]].dropna()
        if team not in set(cv["team"]):
            continue
        ranks = cv[col].rank(pct=True)
        pr = float(ranks[cv["team"] == team].iloc[0])
        if not higher:
            pr = 1.0 - pr
        val = float(cv[cv["team"] == team][col].iloc[0])
        out.append({"metric": col, "label": label,
                    "value": round(val, 3), "pctl": int(round(pr * 100))})
    return out


def _units_display(team: str) -> dict:
    """Per-team unit z-scores in 'good = high' convention (matches the matchup UI)."""
    from ml.matchup_engine import team_units
    u = team_units()
    if team not in u.index:
        return {}
    r = u.loc[team]
    return {
        "pass_off": round(float(r["z_off_pass"]), 2), "rush_off": round(float(r["z_off_rush"]), 2),
        "pass_def": round(float(-r["z_def_pass"]), 2), "rush_def": round(float(-r["z_def_rush"]), 2),
        "st": round(float(r["z_st"]), 2), "coach": round(float(r["z_coaching"]), 2),
        "cont_off": round(float(r["cont_off"]), 2), "cont_def": round(float(r["cont_def"]), 2),
    }


@app.route('/api/team_profile')
def api_team_profile():
    """Full team research profile: style, strengths/weaknesses, situational, units, depth."""
    team = (request.args.get('team') or '').upper()
    if not team:
        return jsonify({"error": "team required"}), 400
    s = styles_df()
    season = int(request.args.get('season', latest_style_season()))
    ss = s[(s["season"] == season) & (s["team"] == team)]
    if ss.empty:                                   # fall back to team's most recent season
        alt = s[s["team"] == team]
        if alt.empty:
            return jsonify({"error": f"no style data for {team}"}), 404
        season = int(alt["season"].max())
        ss = alt[alt["season"] == season]
    row = ss.iloc[0]

    meta = team_meta().get(team, {})
    from ml.squad import squad_ratings, team_depth_chart
    ranks, _ = squad_ratings()
    rr = ranks[ranks["team"] == team]

    style_keys = ["offense_label", "defense_label", "pass_rate_overall", "pass_rate_early_down",
                  "rz_pass_rate", "third_down_pass_rate", "avg_air_yards", "avg_yac", "pace",
                  "play_action_rate", "motion_rate", "screen_pass_rate", "no_huddle_rate",
                  "blitz_rate", "avg_blitzers", "scramble_rate", "qb_rush_rate"]
    sit_keys = ["pressure_rate_gen", "sack_rate_gen", "pressure_rate_allowed", "sack_rate_allowed",
                "rz_td_rate", "rz_td_rate_allowed_y", "two_min_epa", "fourth_go_rate",  # _x is a broken all-1.0 merge artifact
                "def_points_allowed_avg", "turnover_rate", "third_down_stop_rate",
                "off_penalties_pg", "def_penalties_pg", "off_penalty_yds_pg", "def_penalty_yds_pg"]
    style = {k: safe_json(row[k]) for k in style_keys if k in row.index}
    situational = {k: safe_json(row[k]) for k in sit_keys if k in row.index}

    pcts = _profile_percentiles(team, season)
    strengths = sorted(pcts, key=lambda x: -x["pctl"])[:5]
    weaknesses = sorted(pcts, key=lambda x: x["pctl"])[:5]

    if team not in _DEPTH_CACHE:
        _DEPTH_CACHE[team] = team_depth_chart(team)
    from ml.coaching import team_coaching

    return jsonify(_native({
        "team": team, "season": season,
        "name": meta.get("team_name", team),
        "color": meta.get("team_color") or "#334155",
        "logo": meta.get("team_logo_espn", ""),
        "qb": qb1_2026().get(team, ""),
        "rank": int(rr["rank"].iloc[0]) if len(rr) else None,
        "rating": float(rr["rating"].iloc[0]) if len(rr) else None,
        "style": style, "situational": situational,
        "strengths": strengths, "weaknesses": weaknesses,
        "tendencies": _tendencies(team, season),
        "units": _units_display(team),
        "coaching": team_coaching(team),
        "groups": _DEPTH_CACHE[team],
        "injuries": team_injury_map(team),
    }))


# ── weekly form / trends ────────────────────────────────────────────
def team_weekly_form(team: str, season: int) -> list:
    """Per-week offensive/defensive EPP + points for/against for a team's season."""
    p = pbp_season(season)
    if p.empty:
        return []
    p = p[p["week"] <= 18]
    plays = p[p["play_type"].isin(["pass", "run"]) & p["epa"].notna()]
    off = plays[plays["posteam"] == team]
    deff = plays[plays["defteam"] == team]

    # points for / against by week from final scores
    pf, pa = {}, {}
    s = schedules_df()
    if len(s):
        sc = s[(s["season"] == season) & s["home_score"].notna()]
        for _, g in sc.iterrows():
            w = int(g["week"])
            if g["home_team"] == team:
                pf[w], pa[w] = g["home_score"], g["away_score"]
            elif g["away_team"] == team:
                pf[w], pa[w] = g["away_score"], g["home_score"]

    weeks = sorted(set(off["week"].dropna().astype(int)) | set(deff["week"].dropna().astype(int)) | set(pf))
    rows = []
    for w in weeks:
        o, d = off[off["week"] == w], deff[deff["week"] == w]
        rows.append({
            "week": int(w),
            "off_epa": round(float(o["epa"].mean()), 3) if len(o) else None,
            "def_epa": round(float(d["epa"].mean()), 3) if len(d) else None,
            "success": round(float(o["success"].mean()), 3) if len(o) and "success" in o else None,
            "pass_rate": round(float((o["play_type"] == "pass").mean()), 3) if len(o) else None,
            "pf": safe_json(pf.get(w)), "pa": safe_json(pa.get(w)),
        })
    return rows


def _form_summary(weeks: list, last_n: int = 3) -> dict:
    def avg(rows, k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    last = weeks[-last_n:]
    return {
        "season": {k: avg(weeks, k) for k in ("off_epa", "def_epa", "pf", "pa")},
        "last3": {k: avg(last, k) for k in ("off_epa", "def_epa", "pf", "pa")},
        "games": len(weeks),
    }


@app.route('/api/team_trends')
def api_team_trends():
    team = (request.args.get('team') or '').upper()
    if not team:
        return jsonify({"error": "team required"}), 400
    # default to the latest season that actually has play-by-play
    season = int(request.args.get('season', 0)) or None
    if season is None:
        for cand in range(latest_style_season(), 2018, -1):
            if not pbp_season(cand).empty:
                season = cand
                break
        season = season or latest_style_season()
    weeks = team_weekly_form(team, season)
    return jsonify(_native({
        "team": team, "season": season, "weeks": weeks,
        "summary": _form_summary(weeks),
        "empty": len(weeks) == 0,
    }))


# ── latest injuries + full matchup ──────────────────────────────────
# Panel ordering: this week's game designations → practice-only notes (no designation yet,
# but DNP / limited on the mid-week report is the earliest signal a status is coming) →
# season-long reserve lists last (IR/PUP/… aren't week-to-week news).
_STATUS_RANK = {"Out": 0, "Doubtful": 1, "Questionable": 2, "DNP": 3, "LP": 4,
                "PUP": 5, "NFI": 5, "EXE": 5, "IR": 6, "RES": 6, "RET": 7, "CUT": 7}
_NOT_ON_TEAM = ("CUT", "RET")      # released/retired: unavailable to the model, but not "injuries"
_PRACTICE_SHORT = {"Did Not Participate In Practice": "DNP",
                   "Limited Participation in Practice": "LP"}


def _isna(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v)


def _team_report(team: str):
    """(season, week, rows) of this team's CURRENT-SEASON report via ml.projections —
    the one rule every layer shares, so the panel never shows a list the model isn't using."""
    from ml.projections import current_reports
    cur = current_reports(injuries_df())
    if cur.empty:
        return None, None, cur
    t = cur[cur["team"] == team]
    if t.empty:                    # no report yet this season → nothing (NOT last year's)
        return int(cur["season"].max()), None, t
    return int(t["season"].iloc[0]), int(t["week"].iloc[0]), t


def _reserve_rows(team: str) -> list:
    """Reserve-list players (IR/PUP/NFI/exempt) from the roster release, with name+position."""
    from ml.projections import reserve_ids
    res = reserve_ids(team)
    if not res:
        return []
    r = pd.read_parquet(RAW / "rosters_2026.parquet",
                        columns=["team", "player_id", "player_name", "position"])
    r = r[(r["team"] == team) & r["player_id"].isin(res)].drop_duplicates("player_id")
    return [{"gsis_id": str(x.player_id), "name": x.player_name, "position": x.position,
             "status": res[str(x.player_id)], "injury": ""} for x in r.itertuples()]


def latest_injuries(team: str) -> dict:
    """This team's current-season injury report (empty before it publishes) + reserve lists.
    Game designations (Out/Doubtful/Questionable) carry the reported injury; players with
    no designation yet but DNP/limited practice are listed as DNP/LP so a Wednesday report
    is still informative. Reserve-list players (IR/PUP/…) are appended from the roster."""
    season, week, t = _team_report(team)
    reserve = _reserve_rows(team)
    res_by_id = {p["gsis_id"]: p["status"] for p in reserve}
    players, seen = [], set()
    for _, r in t.iterrows():
        gid, st = str(r.get("gsis_id")), r.get("report_status")
        if _isna(st):
            # no game designation: the roster's reserve list (IR/PUP/exempt) outranks a
            # practice note — an exempt-list player "limited in practice" still can't play
            st = res_by_id.get(gid) or _PRACTICE_SHORT.get(str(r.get("practice_status") or ""))
            if not st:
                continue                                   # full practice, no designation
            inj = r.get("practice_primary_injury")
        else:
            inj = r.get("report_primary_injury")
        seen.add(gid)
        players.append({"name": r.get("full_name"), "position": r.get("position"),
                        "status": str(st), "injury": "" if _isna(inj) else str(inj)})
    players += [p for p in reserve if p["gsis_id"] not in seen and p["status"] not in _NOT_ON_TEAM]
    players.sort(key=lambda x: (_STATUS_RANK.get(x["status"], 9), str(x["name"])))
    return {"season": season, "week": week, "players": players}


def team_injury_map(team: str) -> dict:
    """Current-season report for a team keyed by gsis_id — for the profile depth chart.
    Id-keyed (not name) to stay coherent with the id-first depth-chart join. Game
    designations + practice-only DNP/LP + reserve-list labels; healthy players absent."""
    season, week, t = _team_report(team)
    reserve = _reserve_rows(team)
    res_by_id = {p["gsis_id"]: p["status"] for p in reserve}
    by_id = {}
    for _, r in t.iterrows():
        gid, st = r.get("gsis_id"), r.get("report_status")
        if _isna(gid):
            continue
        if _isna(st):                                      # reserve list outranks a practice note
            st = res_by_id.get(str(gid)) or _PRACTICE_SHORT.get(str(r.get("practice_status") or ""))
            if not st:
                continue                                   # no designation → healthy, skip
            inj = r.get("practice_primary_injury")
        else:
            inj = r.get("report_primary_injury")
        by_id[str(gid)] = {"status": str(st), "injury": "" if _isna(inj) else str(inj)}
    for p in reserve:
        by_id.setdefault(p["gsis_id"], {"status": p["status"], "injury": ""})
    return {"season": season, "week": week, "by_id": by_id}


def _adjusted_prediction(home: str, away: str, neutral: bool = False, unavail=None) -> dict:
    """project_game score with three second-order layers: (1) injury→unit routing so a hurt
    unit loses harder to a strong opposing unit (interaction), (2) scheme/play-caller mismatch
    nudges, (3) the flat QB/skill availability points penalty — then recompute margin/total/wp."""
    from ml.matchup_engine import project_game
    from ml.projections import injury_impact, unavailable_ids
    from ml.matchup_context import unit_injury_deltas, scheme_matchup
    if unavail is None:
        unavail = unavailable_ids()
    unit_adj = {home: unit_injury_deltas(home), away: unit_injury_deltas(away)}
    res = project_game(home, away, neutral, unit_adj=unit_adj)
    if "error" in res:
        return res
    imp = {home: injury_impact(home, unavail), away: injury_impact(away, unavail)}
    sch = scheme_matchup(home, away)
    res["pred_home_score"] = round(res["pred_home_score"] - imp[home]["pts"] + sch["home_delta"], 1)
    res["pred_away_score"] = round(res["pred_away_score"] - imp[away]["pts"] + sch["away_delta"], 1)
    res["pred_margin"] = round(res["pred_home_score"] - res["pred_away_score"], 1)
    res["pred_total"] = round(res["pred_home_score"] + res["pred_away_score"], 1)
    _wp = float(1 / (1 + np.exp(-res["pred_margin"] / 13.5 * np.pi / np.sqrt(3))))
    res["home_win_prob"], res["away_win_prob"] = round(_wp, 3), round(1 - _wp, 3)
    res["injury_impact"] = imp
    res["scheme_matchup"] = sch
    res["unit_injuries"] = {t: {k: v for k, v in d.items() if abs(v) > 1e-9}
                            for t, d in unit_adj.items()}
    return res


# ── Detailed matchup analysis helpers ───────────────────────────────
# Each dimension pits an offensive metric against the defensive metric it attacks.
# `obet`/`dbet` = which end of the STORED value ranks #1 (team_styles def_* are pre-flipped
# higher=better, see the leaderboard note). `dneg` displays defensive EPA as raw "EPA allowed".
_MDIMS = [  # key, label, off_col, obet, def_col, dbet, pct, dec, dneg
    ("epa",     "EPA / play",      "off_epa_per_play", "hi", "def_epa_per_play", "hi", False, 2, True),
    ("pass",    "Passing",         "off_epa_per_pass", "hi", "def_epa_per_pass", "hi", False, 2, True),
    ("rush",    "Rushing",         "off_epa_per_rush", "hi", "def_epa_per_rush", "hi", False, 2, True),
    ("succ",    "Success rate",    "off_success_rate", "hi", "def_success_rate", "hi", True,  1, False),
    ("rz",      "Red-zone TD%",    "rz_td_rate",       "hi", "rz_td_rate_allowed_y", "lo", True, 1, False),  # _x is a broken all-1.0 merge artifact
    ("protect", "Pass pro vs rush","sack_rate_allowed","lo", "sack_rate_gen",    "hi", True,  1, False),
    ("disc",    "Penalties / G",   "off_penalties_pg", "lo", "def_penalties_pg", "lo", False, 1, False),
]


def _col_rank(sub: pd.DataFrame, col: str, better: str) -> dict:
    """{team: league rank} for one styles column; 1 = best, ties share the min rank."""
    if col not in sub.columns:
        return {}
    s = sub[["team", col]].dropna(subset=[col]).sort_values(col, ascending=(better == "lo"))
    ranks, prev, rk = {}, None, 0
    for i, (_, row) in enumerate(s.iterrows()):
        v = row[col]
        if v != prev:
            rk, prev = i + 1, v
        ranks[row["team"]] = rk
    return ranks


def _matchup_situational(home: str, away: str, season: int) -> dict:
    """For each phase, pit each team's offense against the opponent's defense with league ranks.
    Returns two directions (away offense vs home defense, and vice-versa)."""
    sub = styles_df()
    sub = sub[sub["season"] == season]
    if sub.empty:
        return None
    ranks = {}
    for _, lbl, ocol, obet, dcol, dbet, *_r in _MDIMS:
        ranks.setdefault(ocol, _col_rank(sub, ocol, obet))
        ranks.setdefault(dcol, _col_rank(sub, dcol, dbet))

    def val(team, col):
        if col not in sub.columns:                    # a styles build missing a metric → blank cell, not a 500
            return None
        v = sub.loc[sub["team"] == team, col]
        return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else None

    def direction(off_t, def_t):
        rows = []
        for key, lbl, ocol, obet, dcol, dbet, pct, dec, dneg in _MDIMS:
            ov, dv = val(off_t, ocol), val(def_t, dcol)
            if dv is not None and dneg:
                dv = -dv                              # show defensive EPA as allowed (neg = elite)
            orank, drank = ranks[ocol].get(off_t), ranks[dcol].get(def_t)
            edge = (drank - orank) if (orank and drank) else None   # >0 = offense is the stronger unit
            rows.append({"key": key, "label": lbl, "pct": pct, "dec": dec,
                         "off_val": ov, "off_rank": orank, "def_val": dv, "def_rank": drank, "edge": edge})
        return rows

    return {"season": season,
            "away_off": {"off": away, "def": home, "rows": direction(away, home)},
            "home_off": {"off": home, "def": away, "rows": direction(home, away)}}


def _head_to_head(home: str, away: str, limit: int = 6) -> dict:
    """Past meetings between the two teams (either venue) + series record & ATS/O-U trends.

    DISPLAY-ONLY, by evidence: walk-forward study (2023-25 targets, 302 games with 2+ prior
    meetings, strength-controlled) found H2H margin history predicts nothing incremental
    (coef +0.012, t=+0.11; sign-unstable by season; frequent rivals 4+ meetings: -0.057,
    t=-0.41 — if anything mild mean reversion). Do not wire H2H into predictions."""
    s = schedules_df()
    if not len(s):
        return None
    d = s[(((s["home_team"] == home) & (s["away_team"] == away)) |
           ((s["home_team"] == away) & (s["away_team"] == home))) & s["home_score"].notna()]
    if d.empty:
        return {"games": [], "n": 0}
    d = d.sort_values(["season", "week"])
    games, hw, aw, ov, un, h_cov = [], 0, 0, 0, 0, 0
    for _, g in d.iterrows():
        hs, as_ = float(g["home_score"]), float(g["away_score"])
        winner = g["home_team"] if hs > as_ else (g["away_team"] if as_ > hs else "TIE")
        if winner == home:
            hw += 1
        elif winner == away:
            aw += 1
        sp, tot = g.get("spread_line"), g.get("total_line")
        total_pts = hs + as_
        ou = None
        if pd.notna(tot):
            ou = "O" if total_pts > tot else ("U" if total_pts < tot else "P")
            if ou == "O": ov += 1
            elif ou == "U": un += 1
        covered = None                                # did the (query) home team cover vs the line?
        if pd.notna(sp):                              # spread_line home-perspective, >0 = home favored
            margin = hs - as_
            home_covered = margin > sp
            covered = g["home_team"] if home_covered else g["away_team"]
            if covered == home: h_cov += 1
        games.append({
            "season": int(g["season"]), "week": int(g["week"]) if pd.notna(g["week"]) else None,
            "home": g["home_team"], "away": g["away_team"],
            "home_score": int(hs), "away_score": int(as_), "winner": winner,
            "spread_line": safe_json(sp), "total_line": safe_json(tot), "total_pts": int(total_pts), "ou": ou,
        })
    recent = games[-limit:][::-1]
    n_ats = sum(1 for x in [g for g in games if g["spread_line"] is not None])
    return {"n": len(games), "games": recent,
            "record": {home: hw, away: aw, "ties": len(games) - hw - aw},
            "avg_total": round(sum(g["total_pts"] for g in games) / len(games), 1),
            "over": ov, "under": un, "ou_n": ov + un,
            f"{home}_covers": h_cov, "ats_n": n_ats}


def _scheduled_game(home: str, away: str) -> dict:
    """The upcoming (or most recent) scheduled game for this exact home/away pairing, with
    its Vegas line and venue conditions. None if these teams aren't paired in the schedule."""
    s = schedules_df()
    if not len(s):
        return None
    d = s[(s["home_team"] == home) & (s["away_team"] == away)]
    if d.empty:
        return None
    fut = d[d["home_score"].isna()]
    row = fut.sort_values(["season", "week"]).iloc[0] if len(fut) else d.sort_values(["season", "week"]).iloc[-1]
    return {
        "season": int(row["season"]), "week": int(row["week"]) if pd.notna(row["week"]) else None,
        "played": bool(pd.notna(row.get("home_score"))),
        "spread_line": safe_json(row.get("spread_line")), "total_line": safe_json(row.get("total_line")),
        "roof": safe_json(row.get("roof")), "surface": safe_json(row.get("surface")),
        "temp": safe_json(row.get("temp")), "wind": safe_json(row.get("wind")),
        "div_game": bool(row.get("div_game")) if pd.notna(row.get("div_game")) else None,
        "stadium": safe_json(row.get("stadium")),
        "gameday": str(row.get("gameday")) if pd.notna(row.get("gameday")) else None,
    }


def _matchup_betting(res: dict, sched: dict) -> dict:
    """Model line vs Vegas line, with the ATS lean + cover prob and total lean + O/U prob.
    Uses the scheduled nflverse line; falls back to model-only when no line exists."""
    from ml.spreads import ats_pick, total_prob
    out = {"model_margin": res["pred_margin"], "model_total": res["pred_total"], "has_line": False}
    sp = sched.get("spread_line") if sched else None
    if sp is None:
        return out
    out["has_line"] = True
    out["vegas_spread"] = sp                          # home-perspective, >0 = home favored
    a = ats_pick(res["pred_margin"], sp)
    out["edge"] = a["edge"]
    out["ats_side"] = res["home"] if a["side"] == "home" else res["away"]
    out["cover_prob"] = a["cover_prob"]
    out["push_prob"] = a["push"]
    tot = sched.get("total_line")
    if tot is not None and res.get("pred_total") is not None:
        out["vegas_total"] = tot
        tp = total_prob(res["pred_total"], tot)
        over = tp["over"] >= tp["under"]
        out["total_side"] = "Over" if over else "Under"
        out["total_prob"] = tp["over"] if over else tp["under"]
    return out


def _game_script(res: dict, pace: float = None) -> dict:
    """A plain-language read of the expected game flow from the prediction + combined pace."""
    margin = res.get("pred_margin") or 0
    total = res.get("pred_total") or 0
    fav = res["home"] if margin >= 0 else res["away"]
    dog = res["away"] if margin >= 0 else res["home"]
    am = abs(margin)
    if am < 3:      tightness = "coin-flip"
    elif am < 7:    tightness = "one-score game"
    elif am < 10.5: tightness = "clear edge"
    else:           tightness = "potential blowout"
    tempo = None
    if pace is not None:
        pace = round(pace, 1)
        tempo = "up-tempo" if pace >= 64 else ("methodical" if pace <= 61 else "average-paced")
    return {"favorite": fav, "underdog": dog, "margin": round(margin, 1), "total": round(total, 1),
            "tightness": tightness, "pace": pace, "tempo": tempo,
            "shootout": total >= 48, "grind": total <= 41}


def _combined_pace(home: str, away: str, season: int) -> float:
    """Average offensive pace (plays/game) of the two teams, for the game-script read."""
    sub = styles_df()
    sub = sub[sub["season"] == season]
    vals = [float(sub.loc[sub["team"] == t, "pace"].iloc[0])
            for t in (home, away) if len(sub.loc[sub["team"] == t, "pace"].dropna())]
    return sum(vals) / len(vals) if vals else None


@app.route('/api/matchup_full')
def api_matchup_full():
    """Matchup prediction + schemes + recent form + latest injuries for both teams."""
    home = (request.args.get('home') or '').upper()
    away = (request.args.get('away') or '').upper()
    neutral = request.args.get('neutral', '0') == '1'
    if not home or not away or home == away:
        return jsonify({"error": "two different teams required"}), 400
    res = _adjusted_prediction(home, away, neutral)
    if "error" in res:
        return jsonify(res), 404

    meta = team_meta()
    styles = styles_df()
    season = latest_style_season()
    form_season = None
    for cand in range(season, 2018, -1):
        if not pbp_season(cand).empty:
            form_season = cand
            break

    def scheme(t):
        r = styles[(styles["season"] == season) & (styles["team"] == t)]
        if r.empty:
            return {}
        r = r.iloc[0]
        return {
            "offense_label": safe_json(r.get("offense_label")),
            "defense_label": safe_json(r.get("defense_label")),
            "pass_rate": safe_json(r.get("pass_rate_overall")),
            "pace": safe_json(r.get("pace")),
            "blitz_rate": safe_json(r.get("blitz_rate")),
            "play_action_rate": safe_json(r.get("play_action_rate")),
            "tendencies": _tendencies(t, season),
        }

    def form(t):
        weeks = team_weekly_form(t, form_season) if form_season else []
        return {"season": form_season, "weeks": weeks, "summary": _form_summary(weeks)}

    for side, t in [("home", home), ("away", away)]:
        m = meta.get(t, {})
        res[f"{side}_name"] = m.get("team_name", t)
        res[f"{side}_color"] = m.get("team_color") or "#334155"
        res[f"{side}_logo"] = m.get("team_logo_espn", "")
    res["schemes"] = {home: scheme(home), away: scheme(away)}
    res["form"] = {home: form(home), away: form(away)}
    res["injuries"] = {home: latest_injuries(home), away: latest_injuries(away)}
    from ml.spreads import simulate
    res["simulation"] = simulate(res["pred_margin"], res["pred_total"])
    # detailed analysis blocks
    sched = _scheduled_game(home, away)
    res["situational"] = _matchup_situational(home, away, season)
    res["h2h"] = _head_to_head(home, away)
    res["betting"] = _matchup_betting(res, sched)
    res["conditions"] = sched
    res["game_script"] = _game_script(res, _combined_pace(home, away, season))
    return jsonify(_native(res))


_SCHED_PRED = {}   # (season, week) -> games list; cleared on refresh


def _finalize_slate(base_games):
    """Overlay live Vegas lines (The Odds API, cached) onto the cached model predictions, then
    compute the key-number ATS picks + top-5. Runs every request (cheap) so lines/picks stay
    current while the expensive predictions stay cached. Falls back to the nflverse line when the
    book has nothing (offseason / later weeks)."""
    from ml.spreads import ats_pick as _ats_pick, total_prob as _total_prob
    from ml.backtest_spreads import blend_weight
    games = [dict(g) for g in base_games]            # copy so the cached base stays clean
    odds_status = None
    try:
        from ml.odds import game_lines, have_key, _namekey
        if have_key():
            lines, odds_status = game_lines()
            for g in games:
                lv = None if g.get("final") else lines.get(
                    (_namekey(g.get("home_name", g["home"])), _namekey(g.get("away_name", g["away"]))))
                if lv and lv.get("spread") is not None:
                    g["nfl_spread"] = g.get("vegas_spread")
                    g["vegas_spread"] = lv["spread"]
                    g["line_source"] = "live"
                    if lv.get("total") is not None:
                        g["vegas_total"] = lv["total"]
                elif g.get("vegas_spread") is not None:
                    g["line_source"] = "nflverse"
    except Exception:
        odds_status = None
    w = blend_weight()                               # optimal market-anchored ensemble weight
    scored = [g for g in games if g.get("pred_margin") is not None and g.get("vegas_spread") is not None]
    for g in scored:
        a = _ats_pick(g["pred_margin"], g["vegas_spread"])
        g["edge"] = a["edge"]
        g["ats_pick"] = g["home"] if a["side"] == "home" else g["away"]
        g["cover_prob"] = a["cover_prob"]
        g["push_prob"] = a["push"]
        g["blend_margin"] = round((1 - w) * g["pred_margin"] + w * g["vegas_spread"], 1)
        g["blend_weight"] = w
        if g.get("pred_total") is not None and g.get("vegas_total") is not None:
            tp = _total_prob(g["pred_total"], g["vegas_total"])
            over = tp["over"] >= tp["under"]
            g["total_pick"] = "Over" if over else "Under"
            g["total_prob"] = tp["over"] if over else tp["under"]
    _mark_early(games)
    _rank_top5(games, [])
    return games, odds_status


MIN_EDGE = 2.5          # points vs the line for a pick to count as a real EDGE play


def _rank_top5(games: list, taken: list) -> None:
    """Assign pick_rank 1-5 (skipping ranks in `taken`) and pick_tier to the live games.

    The pick'em needs FIVE entries every week, so the board always carries five: first the
    games that disagree with the line by >= MIN_EDGE, ranked by cover probability
    (tier 'edge'); if fewer than five clear the floor, the remaining slots are filled by
    the next-largest disagreements (tier 'fill'). The tier is stored and graded separately
    — the three-week review showed 4+ pt edges 8-4 and 1-2.5 pt picks 3-10, so the record
    must be able to tell a forced fifth pick from a real one. Early (pre-lock) games never
    qualify for either tier."""
    free = [r for r in range(1, 6) if r not in taken]
    for g in games:
        if not g.get("locked"):
            g["pick_rank"] = None; g["pick_tier"] = None
    pool = [g for g in games if not g.get("locked") and not g.get("early")
            and g.get("cover_prob") is not None and g.get("edge") is not None]
    edge = sorted([g for g in pool if abs(g["edge"]) >= MIN_EDGE], key=lambda x: -x["cover_prob"])
    fill = sorted([g for g in pool if abs(g["edge"]) < MIN_EDGE], key=lambda x: (-abs(x["edge"]), -x["cover_prob"]))
    for g, r in zip(edge + fill, free):
        g["pick_rank"] = r
        g["pick_tier"] = "edge" if abs(g["edge"]) >= MIN_EDGE else "fill"


def _mark_early(games: list) -> None:
    """Flag games that kick off BEFORE the week's Saturday-morning lock (Thursday night,
    Friday/Saturday internationals) as early=True. They stay on the slate and in the
    all-games record, but are NOT eligible for the top-5: the pick'em deadline for those
    games comes before the board is set, so the contest picks never include them."""
    try:
        from ml.ledger import week_lock_time, kickoff_utc
        lock = week_lock_time(games)
        for g in games:
            ko = kickoff_utc(g.get("gameday"), g.get("gametime"))
            g["early"] = bool(lock is not None and ko is not None and ko < lock)
    except Exception:
        for g in games:
            g["early"] = False


def _reg_weeks(season: int):
    """(regular-season frame, sorted week list) for a season."""
    s = schedules_df()
    d = s[s["season"] == season].copy()
    if "game_type" in d.columns:                     # regular season for the weekly view
        d = d[d["game_type"].fillna("REG").str.upper().eq("REG")]
    return d, sorted(int(x) for x in d["week"].dropna().unique())


def _current_week(season: int):
    """The first regular-season week with an unplayed game (None once the season is done)."""
    d, weeks = _reg_weeks(season)
    for w in weeks:
        if d[(d["week"] == w) & d["home_score"].isna()].shape[0]:
            return w
    return None


def _lock_picks(season: int, week: int, games: list) -> None:
    """Freeze this slate's picks for games that haven't kicked off (ml/ledger.py). Only the
    current season is a live record; browsing old seasons must not write history."""
    try:
        from ml.ledger import record
        record(season, week, games)
    except Exception as e:
        print(f"[ledger] lock failed {season} wk{week}: {e}")


def lock_current_week() -> str:
    """Compute + freeze the current week's slate. Called after every data refresh so a pick is
    on the record before kickoff even if nobody opened the Schedule page that day."""
    try:
        s = schedules_df()
        season = int(s["season"].max())
        week = _current_week(season)
        if week is None:
            return "season complete"
        _SCHED_PRED.pop((season, week), None)         # refresh changed rosters/injuries → re-predict
        slate = _slate(season, week)
        _overlay_locked(season, slate["games"])       # rank the live games around the frozen ones
        _lock_picks(season, week, slate["games"])
        return f"locked {season} wk{week}"
    except Exception as e:
        return f"error: {str(e)[:120]}"


@app.route('/api/schedule')
def api_schedule():
    """A week's slate: every game with the model's roster+injury-adjusted prediction
    (and the Vegas line / final score when available). Auto-pairs home/away from the schedule."""
    s = schedules_df()
    if s.empty:
        return jsonify({"error": "no schedule data"}), 404
    seasons = sorted(int(x) for x in s["season"].dropna().unique())
    season = int(request.args.get('season', seasons[-1]))
    d, weeks = _reg_weeks(season)
    if not weeks:
        return jsonify(_native({"season": season, "week": None, "seasons": seasons, "weeks": [], "games": []}))
    # default week: the CURRENT week (first with an unplayed game) for the live season, so the
    # page opens on this weekend's slate instead of week 1; a finished season opens on week 1
    default_week = (_current_week(season) if season == seasons[-1] else None) or weeks[0]
    week = int(request.args.get('week', default_week))
    out = _slate(season, week)
    if season == seasons[-1]:
        # overlay FIRST so the live games' top-5 ranks are slotted around the games already
        # frozen (a Thursday game keeps its rank; Sunday games fill the free slots), and it is
        # those consistent ranks that get written to the record
        _overlay_locked(season, out["games"])         # started/frozen games show the pick that was on the board
        _lock_picks(season, week, out["games"])       # the live season writes the record
    return jsonify(_native({**out, "seasons": seasons, "weeks": weeks}))


_LOCK_FIELDS = ("pred_home", "pred_away", "pred_margin", "pred_total", "home_win_prob",
                "vegas_spread", "vegas_total", "line_source", "ats_pick", "edge", "cover_prob",
                "pick_rank", "pick_tier", "total_pick", "total_prob")


def _overlay_locked(season: int, games: list) -> None:
    """Replace the live re-prediction with the ledger row (ml/ledger.py) for every game that
    is FROZEN (the week passed its lock time) or has KICKED OFF. The model moves every day
    — results feed the blend, injuries change, lines close — so the board a reader sees
    after Saturday 9am ET must be the board that gets graded, not a re-prediction. Before
    the lock time the live model prices the slate and the ledger is provisional."""
    try:
        from ml.ledger import load, kickoff_utc, week_lock_time
        from datetime import datetime, timezone
        led = load()
        led = led[led["season"] == season] if len(led) else led
        if not len(led):
            return
        rows = led.set_index("game_id")
        now = datetime.now(timezone.utc)
        lock_at = week_lock_time(games)
        week_frozen = bool(lock_at is not None and now >= lock_at)
        for g in games:
            g["week_lock_at"] = lock_at.isoformat() if lock_at else None
            g["week_frozen"] = week_frozen
            gid = g.get("game_id")
            ko = kickoff_utc(g.get("gameday"), g.get("gametime"))
            started = g.get("final") or (ko is not None and now >= ko)
            if not (started or week_frozen) or gid not in rows.index:
                continue
            r = rows.loc[gid]
            for k in _LOCK_FIELDS:
                v = r.get(k)
                g[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            g["blend_margin"] = None                     # not part of the locked record
            g["locked"] = True
            g["locked_at"] = r.get("locked_at")
        # ONE top-5 per week. Locked games keep the rank they were locked with; the live games
        # are then ranked into whatever slots remain. Ranking the live games over the whole
        # slate (the old way) let a locked #2 and a live #2 coexist, or a locked #3 with no #1
        # — mid-week the strip showed duplicates or gaps and read as "not five picks".
        if any(x.get("locked") for x in games):
            taken = [int(g["pick_rank"]) for g in games if g.get("locked") and g.get("pick_rank")]
            _rank_top5(games, taken)
    except Exception as e:
        print(f"[ledger] overlay failed: {e}", flush=True)


def _slate(season: int, week: int) -> dict:
    """Predictions (cached) + fresh lines/picks for one week."""
    from ml.projections import unavailable_ids
    d, _ = _reg_weeks(season)
    if (season, week) not in _SCHED_PRED:            # cache only the EXPENSIVE predictions (no lines/picks)
        dw = d[d["week"] == week]
        sort_cols = [c for c in ["gameday", "gametime"] if c in dw.columns]
        if sort_cols:
            dw = dw.sort_values(sort_cols)
        from ml.context import game_context
        meta = team_meta()
        unavail = unavailable_ids()
        base = []
        for _, g in dw.iterrows():
            home, away = g.get("home_team"), g.get("away_team")
            if not isinstance(home, str) or not isinstance(away, str):
                continue
            hm, am = meta.get(home, {}), meta.get(away, {})
            played = pd.notna(g.get("home_score"))
            ctx = game_context(home, away, g)
            rec = {
                "game_id": g.get("game_id"), "gameday": g.get("gameday"), "gametime": g.get("gametime"),
                "home": home, "away": away,
                "home_name": hm.get("team_name", home), "away_name": am.get("team_name", away),
                "home_logo": hm.get("team_logo_espn", ""), "away_logo": am.get("team_logo_espn", ""),
                "home_color": hm.get("team_color") or "#334155", "away_color": am.get("team_color") or "#334155",
                "vegas_spread": safe_json(g.get("spread_line")), "vegas_total": safe_json(g.get("total_line")),
                "home_ml": safe_json(g.get("home_moneyline")), "away_ml": safe_json(g.get("away_moneyline")),
                "home_score": safe_json(g.get("home_score")), "away_score": safe_json(g.get("away_score")),
                "final": bool(played),
                "neutral": ctx["neutral"], "stadium": ctx["stadium"], "context_notes": ctx["notes"],
            }
            # neutral site removes home field (via project_game); travel/weather nudge each score
            pred = _adjusted_prediction(home, away, neutral=ctx["neutral"], unavail=unavail)
            if "error" not in pred:
                hs = round(pred["pred_home_score"] + ctx["home_delta"], 1)
                as_ = round(pred["pred_away_score"] + ctx["away_delta"], 1)
                margin = round(hs - as_, 1)
                wp = float(1 / (1 + np.exp(-margin / 13.5 * np.pi / np.sqrt(3))))
                rec.update({
                    "pred_home": hs, "pred_away": as_,
                    "pred_margin": margin, "pred_total": round(hs + as_, 1),
                    "home_win_prob": round(wp, 3),
                    "inj_home": pred["injury_impact"][home], "inj_away": pred["injury_impact"][away],
                    "context_delta": {"home": ctx["home_delta"], "away": ctx["away_delta"]},
                })
            base.append(rec)
        _SCHED_PRED[(season, week)] = base

    # live Vegas lines + picks are applied fresh each request (cheap; predictions stay cached)
    games, odds_status = _finalize_slate(_SCHED_PRED[(season, week)])
    return {"season": season, "week": week, "games": games, "odds_status": odds_status}


# ── Kalshi (Betting > Kalshi) ────────────────────────────────────────
# Ported from the tennis engine. Tickets are built from the current week's slate; the only
# code that can spend money is ml/kalshi_order.py, reachable solely via /api/kalshi/submit
# with confirm=true. Dry run unless KALSHI_ARM=1; demo API unless KALSHI_LIVE=1.

# ticker -> our schedule's kickoff (UTC ISO), filled by the last ticket scan. Kalshi cannot
# be asked this reliably, and the order guard refuses a ticket with no known kickoff.
# In-process state: the Procfile pins gunicorn to one worker, which this relies on.
TICKET_STARTS: dict[str, str] = {}
KALSHI_MIN_EDGE = 0.03     # probability points over the ask before a ticket is even sized;
                           # below this the "edge" is inside the model's own error bar


@app.after_request
def _never_cache_money(resp):
    """Nothing under /api/kalshi may be cached: these endpoints price real orders against a
    live balance, and a cached ticket list sizes a bet against a bankroll that no longer exists."""
    if request.path.startswith("/api/kalshi"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


def _kalshi_probs(g: dict) -> dict:
    """
    The probabilities a ticket is priced from, for one slate game.

    Uses the MARKET-BLENDED margin (ml/backtest_spreads.blend_weight — the weight that
    minimised margin error out-of-sample), not the raw model, because the raw model does
    not reliably beat NFL markets (CLAUDE.md). Pricing Kalshi against the pure model would
    manufacture an "edge" on nearly every game. Totals blend the same way against the
    Vegas total. Both raw and blended are returned so the page can show the gap.
    """
    from ml.spreads import cover_prob, total_prob
    w = float(g.get("blend_weight") or 0.0)
    m_raw = float(g["pred_margin"])
    m = float(g["blend_margin"]) if g.get("blend_margin") is not None else m_raw
    t_raw = float(g["pred_total"]) if g.get("pred_total") is not None else None
    t = ((1 - w) * t_raw + w * float(g["vegas_total"])) if (t_raw is not None and g.get("vegas_total") is not None) else t_raw
    wp = lambda mm: float(1 / (1 + np.exp(-mm / 13.5 * np.pi / np.sqrt(3))))
    # MEDIAN-centred: a strike sitting exactly on the blended number must read 50%, not the
    # ~52% the mean-centred ATS helper gives (NFL margins are skewed). That 2 pts is most of
    # a Kalshi "edge" on a near-line strike, so it is removed here rather than bet on.
    cp = lambda mm, line: cover_prob(mm, line, center="median")
    tp = lambda tt, line: total_prob(tt, line, center="median")["over"]
    return {"margin": m, "margin_raw": m_raw, "total": t, "total_raw": t_raw,
            "home_win": wp(m), "home_win_raw": wp(m_raw),
            "cover": lambda line: cp(m, line),                  # P(home margin > line)
            "cover_raw": lambda line: cp(m_raw, line),
            "over": (lambda line: tp(t, line)) if t is not None else None,
            "over_raw": (lambda line: tp(t_raw, line)) if t_raw is not None else None}


# Spread / total strikes are priced only within this many points of the VEGAS number (the
# blended model number when no line is posted). The model's validated information is about
# the CENTRE of the outcome distribution; a 20.5-point strike at 11c is a bet on the tail
# shape of a recentred historical distribution, which the model has never been tested on —
# and it is exactly where a naive Kelly sees the fattest "edge". Winner markets are unaffected.
KALSHI_STRIKE_WINDOW = 3.5


def _kalshi_tickets_for_game(g: dict, events: dict, bankroll: float, budget: dict,
                             held: dict) -> tuple:
    """(tickets, skipped) for one game across winner / spread / total markets.

    At most ONE ticket per market type per game (the best net EV%), because the strikes of
    one event are the same bet at different prices, not independent opportunities."""
    from ml import kalshi, kalshi_match, kalshi_order, risk
    from ml.ledger import kickoff_utc
    tickets, skipped = [], []
    ko = kickoff_utc(g.get("gameday"), g.get("gametime"))
    starts = ko.isoformat() if ko else None
    label_game = f"{g['away']} @ {g['home']}"
    P = _kalshi_probs(g)

    def note(kind, backing, reason, price=None, prob=None):
        skipped.append({"game": label_game, "kind": kind, "backing": backing, "starts": starts,
                        "model_prob": None if prob is None else round(prob, 4),
                        "price": price, "reason": reason})

    # candidate (kind, backing label, market, prob) — YES contracts only (a bid on the ticker)
    cands = []
    for kind in ("winner", "spread", "total"):
        found = kalshi_match.find_event(g["home"], g["away"], g.get("gameday"), events.get(kind, {}))
        if not found.get("ok"):
            note(kind, None, found.get("reason"))
            continue
        if kind == "winner":
            wm = kalshi_match.winner_markets(found)
            if not wm:
                note(kind, None, "event does not have exactly one contract per team")
                continue
            cands += [(kind, f"{g['home']} to win", wm["home"], P["home_win"], P["home_win_raw"]),
                      (kind, f"{g['away']} to win", wm["away"], 1 - P["home_win"], 1 - P["home_win_raw"])]
        elif kind == "spread":
            centre = float(g["vegas_spread"]) if g.get("vegas_spread") is not None else P["margin"]
            n_in = 0
            for side, strike, m in kalshi_match.spread_markets(found):
                team = g["home"] if side == "home" else g["away"]
                # the strike expressed as a home-perspective line: home > strike, or home < -strike
                line = strike if side == "home" else -strike
                if abs(line - centre) > KALSHI_STRIKE_WINDOW:
                    continue
                n_in += 1
                if side == "home":
                    p, pr = P["cover"](strike)["home_cover"], P["cover_raw"](strike)["home_cover"]
                else:                              # away wins by > strike ⇔ home margin < -strike
                    p, pr = P["cover"](-strike)["away_cover"], P["cover_raw"](-strike)["away_cover"]
                cands.append((kind, f"{team} wins by over {strike:g}", m, p, pr))
            if not n_in:
                note(kind, None, f"no strike within {KALSHI_STRIKE_WINDOW:g} pts of the line")
        else:
            if P["over"] is None:
                note(kind, None, "no model total for this game")
                continue
            centre = float(g["vegas_total"]) if g.get("vegas_total") is not None else P["total"]
            n_in = 0
            for strike, m in kalshi_match.total_markets(found):
                if abs(strike - centre) > KALSHI_STRIKE_WINDOW:
                    continue
                n_in += 1
                cands.append((kind, f"Over {strike:g} points", m, P["over"](strike),
                              P["over_raw"](strike) if P["over_raw"] else None))
            if not n_in:
                note(kind, None, f"no strike within {KALSHI_STRIKE_WINDOW:g} pts of the total")

    if kalshi_order.started({}, starts=starts):
        for kind in ("winner", "spread", "total"):
            note(kind, None, "the game has already kicked off")
        return [], skipped

    best: dict[str, dict] = {}
    best_reason: dict[str, tuple] = {}
    for kind, backing, m, prob, prob_raw in cands:
        price = kalshi_match.ask_price(m)
        if price is None:
            best_reason.setdefault(kind, (backing, "no offer resting on that contract", None, prob))
            continue
        if prob - price < KALSHI_MIN_EDGE:
            best_reason.setdefault(kind, (backing, f"edge {100*(prob-price):+.1f} pts is under the {100*KALSHI_MIN_EDGE:.0f}-pt floor", price, prob))
            continue
        sized = kalshi.size_position(prob, price, bankroll, maker=risk.MAKER,
                                     max_stake_pct=budget["ticket_pct"])
        if sized["contracts"] < 1:
            best_reason[kind] = (backing, sized.get("reason") or "sized to nothing", price, prob)
            continue
        offered = kalshi_match.ask_size(m)
        capped = False
        if offered is not None and sized["contracts"] > int(offered):
            if int(offered) < 1:
                best_reason[kind] = (backing, "nothing offered at the ask", price, prob)
                continue
            capped = True
            sized = kalshi.size_position(prob, price, bankroll, maker=risk.MAKER,
                                         max_stake_pct=min(budget["ticket_pct"],
                                                           100.0 * int(offered) * price / max(bankroll, 1e-9)))
            if sized["contracts"] < 1:
                best_reason[kind] = (backing, "offered size too small to trade", price, prob)
                continue
        t = {"game": label_game, "kind": kind, "backing": backing, "ticker": m.get("ticker"),
             "event": m.get("event_ticker"), "model_prob": round(prob, 4),
             "model_prob_raw": None if prob_raw is None else round(prob_raw, 4),
             "price": price, "contracts": sized["contracts"], "stake": sized["stake"],
             "fee": sized["fee"], "ev": sized["ev"], "ev_pct": sized["ev_pct"],
             "offered": offered, "size_capped": capped, "starts": starts,
             "held": held.get(str(m.get("ticker"))),
             "already_placed": str(m.get("ticker")) in held,
             "client_order_id": kalshi_order.new_client_order_id()}
        if kind not in best or t["ev_pct"] > best[kind]["ev_pct"]:
            best[kind] = t
    for kind, t in best.items():
        tickets.append(t)
    for kind, (backing, reason, price, prob) in best_reason.items():
        if kind not in best:
            note(kind, backing, reason, price, prob)
    return tickets, skipped


@app.route('/api/kalshi/tickets')
def api_kalshi_tickets():
    """Costed order tickets for the current week's games. Builds only — nothing here can
    place an order; submission is a separate endpoint requiring explicit confirmation."""
    from ml import kalshi, kalshi_match, kalshi_order, risk
    if not kalshi.configured():
        return jsonify({"available": False, "reason": "Kalshi credentials are not set", "tickets": []})
    try:
        bankroll = float(kalshi.balance().get("dollars") or 0.0)
    except Exception as e:
        return jsonify({"available": False, "reason": f"could not read balance: {str(e)[:120]}", "tickets": []})
    budget = risk.budget(bankroll, kalshi_order.open_tickers())
    try:
        s = schedules_df()
        season = int(s["season"].max())
        week = _current_week(season)
        if week is None:
            return jsonify({"available": True, "tickets": [], "skipped": [], "budget": budget,
                            "armed": kalshi_order.armed(), "live": kalshi.live_mode(),
                            "note": "season complete"})
        slate = _slate(season, week)
    except Exception as e:
        return jsonify({"available": False, "reason": f"slate failed: {str(e)[:160]}", "tickets": []})
    try:
        events = {k: kalshi_match.open_events(k) for k in ("winner", "spread", "total")}
    except Exception as e:
        return jsonify({"available": False, "reason": f"could not read Kalshi markets: {str(e)[:160]}", "tickets": []})

    # What is ALREADY held. Positions reflect fills (authoritative); the ledger covers an
    # order that was sent but has not filled yet.
    held: dict[str, float] = {}
    try:
        for pos in kalshi.positions(limit=200):
            c = kalshi.num(pos.get("position_fp")) or 0.0
            if c:
                held[str(pos.get("ticker"))] = c
    except Exception:
        pass
    try:
        for t in risk.placed_here()["tickers"]:
            held.setdefault(str(t), 0.0)
    except Exception:
        pass

    tickets, skipped = [], []
    for g in slate["games"]:
        if g.get("pred_margin") is None or g.get("final"):
            continue
        t, sk = _kalshi_tickets_for_game(g, events, bankroll, budget, held)
        tickets += t
        skipped += sk
    for t in tickets:
        if t.get("starts"):
            TICKET_STARTS[str(t["ticker"])] = str(t["starts"])
    # Sorted by the blended probability, highest first; EV alone sorts longshots to the top.
    tickets.sort(key=lambda t: (-(t["model_prob"] or 0), -(t["ev"] or 0)))
    return jsonify(_native({"available": True, "season": season, "week": week,
                            "tickets": tickets, "skipped": skipped, "budget": budget,
                            "armed": kalshi_order.armed(), "live": kalshi.live_mode(),
                            "min_edge": KALSHI_MIN_EDGE, "n_events": {k: len(v) for k, v in events.items()}}))


@app.route('/api/kalshi/report')
def api_kalshi_report():
    """Everything about the Kalshi side of the account, read-only: orders (sent), fills
    (traded), positions (open, marked to the bid), and a realised ledger built from FILLS
    plus settlements (a cash-out never settles). Scope defaults to what this page sent."""
    from ml import kalshi, kalshi_match, kalshi_order, risk
    if not kalshi.configured():
        return jsonify({"available": False, "reason": "Kalshi credentials are not set"})
    out = {"available": True, "live": kalshi.live_mode(), "armed": kalshi_order.armed(), "errors": {}}

    def pull(name, fn, default):
        try:
            return fn()
        except Exception as e:
            out["errors"][name] = f"{type(e).__name__}: {str(e)[:120]}"
            return default

    bal = pull("balance", kalshi.balance, {})
    bankroll = float(bal.get("dollars") or 0.0)
    out["balance"] = bal.get("dollars")
    out["budget"] = risk.budget(bankroll, kalshi_order.open_tickers())
    out["caps"] = {"ticket_pct": risk.MAX_TICKET_PCT, "daily_pct": risk.MAX_DAILY_PCT,
                   "maker": risk.MAKER, "kelly_fraction": kalshi.KELLY_FRACTION,
                   "max_price_drift": kalshi_order.MAX_PRICE_DRIFT}
    raw_orders = pull("orders", lambda: kalshi.orders(limit=200), [])
    raw_fills = pull("fills", lambda: kalshi.fills(limit=200), [])
    raw_pos = pull("positions", lambda: kalshi.positions(limit=200), [])
    raw_settle = pull("settlements", lambda: kalshi.settlements(limit=200), [])

    scope = str(request.args.get("scope", "page")).lower()
    if scope not in ("page", "nfl", "account"):
        scope = "page"
    mine = pull("ledger", risk.placed_here, {"tickers": set(), "order_ids": set(), "labels": {}})
    out["scope"] = scope
    nfl_series = tuple(v + "-" for v in kalshi_match.SERIES.values())

    def ours(rec: dict) -> bool:
        if scope == "account":
            return True
        t = str(rec.get("ticker") or rec.get("market_ticker") or "")
        if scope == "nfl":
            return t.startswith(nfl_series)
        coid = str(rec.get("client_order_id") or "")
        return (coid in mine["order_ids"]) or (t in mine["tickers"])

    out["scope_counts"] = {}
    for label, rows in (("orders", raw_orders), ("fills", raw_fills), ("positions", raw_pos), ("settlements", raw_settle)):
        out["scope_counts"][label] = {"account": len(rows), "shown": sum(1 for r in rows if ours(r))}
    raw_orders = [r for r in raw_orders if ours(r)]
    raw_fills = [r for r in raw_fills if ours(r)]
    raw_pos = [r for r in raw_pos if ours(r)]
    raw_settle = [r for r in raw_settle if ours(r)]

    mkts: dict[str, dict] = {}

    def market_of(ticker):
        t = str(ticker or "")
        if t and t not in mkts:
            try:
                mkts[t] = kalshi.market(t) or {}
            except Exception:
                mkts[t] = {}
        return mkts.get(t, {})

    def name_of(ticker):
        t = str(ticker or "")
        return mine["labels"].get(t) or str(market_of(t).get("yes_sub_title") or market_of(t).get("title") or t)

    sport_of = kalshi_match.sport_of
    seen = {str(r.get("ticker") or r.get("market_ticker") or "") for r in (raw_orders + raw_fills + raw_pos + raw_settle)}
    for t in list(seen)[:60]:
        name_of(t)
    n = kalshi.num
    out["orders"] = [{
        "order_id": o.get("order_id"), "ticker": o.get("ticker"), "backing": name_of(o.get("ticker")),
        "sport": sport_of(o.get("ticker")), "status": o.get("status"),
        "side": o.get("book_side") or o.get("side"), "price": n(o.get("yes_price_dollars")),
        "placed": n(o.get("initial_count_fp")), "filled": n(o.get("fill_count_fp")),
        "remaining": n(o.get("remaining_count_fp")),
        "fees": (n(o.get("taker_fees_dollars")) or 0.0) + (n(o.get("maker_fees_dollars")) or 0.0),
        "created": o.get("created_time")} for o in raw_orders]
    out["positions"] = [{
        "ticker": p_.get("ticker"), "backing": name_of(p_.get("ticker")), "sport": sport_of(p_.get("ticker")),
        "contracts": n(p_.get("position_fp")), "exposure": n(p_.get("market_exposure_dollars")),
        "traded": n(p_.get("total_traded_dollars")), "realized": n(p_.get("realized_pnl_dollars")),
        "fees": n(p_.get("fees_paid_dollars")),
        "bid": n(market_of(p_.get("ticker")).get("yes_bid_dollars")),        # marked at the BID
        "mark": round((n(p_.get("position_fp")) or 0.0) * (n(market_of(p_.get("ticker")).get("yes_bid_dollars")) or 0.0), 2),
        "updated": p_.get("last_updated_ts")} for p_ in raw_pos if (n(p_.get("position_fp")) or 0) != 0]
    for p_ in out["positions"]:
        p_["unrealised"] = round((p_["mark"] or 0.0) - (p_["traded"] or 0.0), 2)

    # realised ledger from FILLS (+ settlements for what was still held); cost never doubled
    led: dict[str, dict] = {}

    def row(t):
        return led.setdefault(str(t), {"ticker": str(t), "bought": 0.0, "cost": 0.0, "sold": 0.0, "proceeds": 0.0,
                                       "settled": 0.0, "revenue": 0.0, "fees": 0.0, "settle_cost": 0.0,
                                       "settle_fee": 0.0, "at": None, "kind": set()})
    for f in raw_fills:
        t = f.get("ticker") or f.get("market_ticker")
        if not t:
            continue
        r = row(t)
        cnt, px = n(f.get("count_fp")) or 0.0, n(f.get("yes_price_dollars")) or 0.0
        if str(f.get("book_side")) == "bid":
            r["bought"] += cnt; r["cost"] += cnt * px
        else:
            r["sold"] += cnt; r["proceeds"] += cnt * px; r["kind"].add("cashed out")
        r["fees"] += n(f.get("fee_cost")) or 0.0
        at = f.get("created_time")
        if at and (r["at"] is None or str(at) > str(r["at"])):
            r["at"] = at
    for s_ in raw_settle:
        t = s_.get("ticker")
        if not t:
            continue
        r = row(t)
        r["revenue"] += (n(s_.get("revenue")) or 0.0) / 100.0          # CENTS
        r["settled"] += (n(s_.get("yes_count_fp")) or 0.0) + (n(s_.get("no_count_fp")) or 0.0)
        r["settle_cost"] += (n(s_.get("yes_total_cost_dollars")) or 0.0) + (n(s_.get("no_total_cost_dollars")) or 0.0)
        r["settle_fee"] += n(s_.get("fee_cost")) or 0.0
        r["kind"].add("settled")
        at = s_.get("settled_time")
        if at and (r["at"] is None or str(at) > str(r["at"])):
            r["at"] = at
    closed = []
    for t, r in led.items():
        if r["bought"] > 0:
            avg = r["cost"] / r["bought"]
        elif r["settled"] > 0 and r["settle_cost"] > 0 and r["sold"] <= 0:
            avg = r["settle_cost"] / r["settled"]; r["bought"] = r["settled"]; r["cost"] = r["settle_cost"]
        else:
            avg = 0.0
        remaining = max(0.0, r["bought"] - r["sold"])
        settled_n = min(r["settled"], remaining)               # you can only settle what you still held
        revenue = (r["revenue"] * (settled_n / r["settled"])) if r["settled"] > 0 else 0.0
        closed_n = r["sold"] + settled_n
        if closed_n <= 0:
            continue
        basis = avg * closed_n
        returned = r["proceeds"] + revenue
        fees = r["fees"] if r["fees"] > 0 else r["settle_fee"]
        closed.append({"ticker": t, "backing": name_of(t), "sport": sport_of(t),
                       "kind": (" + ".join(sorted(k for k in r["kind"] if k != "settled" or settled_n > 0)) or "cashed out"),
                       "contracts": round(closed_n, 2), "cost": round(basis, 2), "revenue": round(returned, 2),
                       "fee": round(fees, 2), "pl": round(returned - basis - fees, 2), "at": r["at"]})
    closed.sort(key=lambda x: str(x.get("at") or ""), reverse=True)
    out["settlements"] = closed
    staked = sum(r["cost"] for r in closed)
    realised = sum(r["pl"] for r in closed)
    wins = sum(1 for r in closed if r["pl"] > 0)
    out["record"] = {
        "settled": len(closed), "won": wins, "lost": len(closed) - wins,
        "cashed_out": sum(1 for r in closed if "cashed out" in r["kind"]),
        "win_pct": round(100.0 * wins / len(closed), 1) if closed else None,
        "staked": round(staked, 2), "realised_pl": round(realised, 2),
        "fees": round(sum(r["fee"] for r in closed), 2),
        "roi_pct": round(100.0 * realised / staked, 1) if staked else None,
        "open_positions": len(out["positions"]),
        "open_exposure": round(sum(p_["exposure"] or 0.0 for p_ in out["positions"]), 2),
        "orders_sent": len(out["orders"]),
        "orders_filled": sum(1 for o in out["orders"] if (o["filled"] or 0) > 0),
        "orders_unfilled": sum(1 for o in out["orders"] if not (o["filled"] or 0)),
        "open_mark": round(sum(p_["mark"] or 0.0 for p_ in out["positions"]), 2),
        "open_unrealised": round(sum(p_["unrealised"] or 0.0 for p_ in out["positions"]), 2),
        "avg_stake": round(staked / len(closed), 2) if closed else None,
        "best": max((r["pl"] for r in closed), default=None),
        "worst": min((r["pl"] for r in closed), default=None)}
    out["record"]["total_return"] = round(out["record"]["realised_pl"] + out["record"]["open_unrealised"], 2)

    sports: dict[str, dict] = {}

    def bucket(name):
        return sports.setdefault(name, {"sport": name, "orders": 0, "filled": 0, "open": 0, "open_exposure": 0.0,
                                        "open_mark": 0.0, "settled": 0, "won": 0, "staked": 0.0,
                                        "realised_pl": 0.0, "fees": 0.0})
    for o in out["orders"]:
        b = bucket(o["sport"]); b["orders"] += 1; b["filled"] += 1 if (o["filled"] or 0) > 0 else 0; b["fees"] += o["fees"] or 0.0
    for p_ in out["positions"]:
        b = bucket(p_["sport"]); b["open"] += 1; b["open_exposure"] += p_["exposure"] or 0.0; b["open_mark"] += p_["mark"] or 0.0
    for r in closed:
        b = bucket(r["sport"]); b["settled"] += 1; b["won"] += 1 if r["pl"] > 0 else 0; b["staked"] += r["cost"]; b["realised_pl"] += r["pl"]
    for b in sports.values():
        for k in ("open_exposure", "open_mark", "staked", "realised_pl", "fees"):
            b[k] = round(b[k], 2)
        b["roi_pct"] = round(100.0 * b["realised_pl"] / b["staked"], 1) if b["staked"] else None
        b["win_pct"] = round(100.0 * b["won"] / b["settled"], 1) if b["settled"] else None
    out["by_sport"] = sorted(sports.values(), key=lambda b: (-(b["settled"] + b["open"]), b["sport"]))
    return jsonify(_native(out))


@app.route('/api/kalshi/submit', methods=['POST'])
def api_kalshi_submit():
    """Place ONE order a person has just confirmed. Every limit is re-checked server-side;
    the request must carry confirm=true. Dry run unless KALSHI_ARM=1."""
    from ml import kalshi_order
    body = request.get_json(silent=True) or {}
    if body.get("confirm") is not True:
        return jsonify({"ok": False, "error": "confirmation required"}), 400
    try:
        ticker = str(body["ticker"]); count = int(body["contracts"])
        price = float(body["price"]); coid = str(body["client_order_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "ticker, contracts, price and client_order_id are all required"}), 400
    label = f"{body.get('backing') or ''} — {body.get('game') or ''}".strip(" —")
    result = kalshi_order.create_order(ticker, count, price, coid,
                                       starts=TICKET_STARTS.get(ticker), label=label or None)
    if result.get("error"):
        mkt = result.get("market") or {}
        print(f"[kalshi] REFUSED {ticker} {count}@{price:.2f}: {result['error']}"
              + (f" (ask now {mkt.get('ask')}, offered {mkt.get('offered')})"
                 if mkt.get("ask") is not None or mkt.get("offered") is not None else ""), flush=True)
        return jsonify({"ok": False, **result}), 400
    if result.get("sent"):
        print(f"[kalshi] SENT {ticker} {count}@{price:.2f} -> {json.dumps(result.get('response'))[:300]}", flush=True)
    return jsonify({"ok": True, **result})


def _kalshi_selftest() -> None:
    """Boot report: do the credentials work, demo or LIVE, armed or not, balance and row
    counts. Reads only; never prints the key; no code path here places an order."""
    try:
        from ml import kalshi, kalshi_order, risk
    except Exception as e:
        print(f"[kalshi] import failed: {e}", flush=True)
        return
    if not kalshi.configured():
        print("[kalshi] credentials not set - Kalshi tab disabled", flush=True)
        return
    mode = "LIVE (real money)" if kalshi.live_mode() else "demo"
    print(f"[kalshi] ARMED - confirmed tickets WILL be sent, cap {risk.MAX_TICKET_PCT}% per ticket"
          if kalshi_order.armed() else "[kalshi] not armed - confirmed tickets are logged, not sent", flush=True)
    try:
        bal = kalshi.balance()
        print(f"[kalshi] OK - {mode}, balance ${bal.get('dollars')}, ledger {risk._ledger_path()}", flush=True)
    except kalshi.KalshiError as e:
        hint = " - key rejected; a demo key cannot sign production requests or vice versa (KALSHI_LIVE)" if e.status == 401 else ""
        print(f"[kalshi] FAILED ({mode}) - {e}{hint}", flush=True)
    except Exception as e:
        print(f"[kalshi] FAILED ({mode}) - {type(e).__name__}: {str(e)[:140]}", flush=True)


@app.route('/api/record')
def api_record():
    """The model's betting record: every pick frozen before kickoff (ml/ledger.py), graded
    against final scores — ATS, totals and moneyline, all picks and the conviction subsets."""
    from ml.ledger import grade
    s = schedules_df()
    season = int(request.args.get('season', s["season"].max() if len(s) else 2026))
    return jsonify(_native(grade(season)))


@app.route('/api/backtest')
def api_backtest():
    """Honest accuracy: model margin MAE vs the market's, straight-up accuracy, and
    (in-sample) ATS/ROI, with the out-of-sample caveat baked into the payload."""
    from ml.backtest_spreads import evaluate, latest_completed_season
    arg = request.args.get('season')
    season = int(arg) if arg else None
    res = evaluate(season)
    if "error" in res:                               # requested season not gradable → latest completed
        res = evaluate(latest_completed_season())
    return jsonify(_native(res))


@app.route('/api/fantasy')
def api_fantasy():
    """Best-ball fantasy: our 2026 VOR-ranked draft board (value-over-replacement so it flows like
    a real draft), plus opportunity-vs-production 'undervalued/breakout' candidates. If an ADP CSV
    is dropped at data/raw/adp_underdog.csv, the board also shows ADP + value vs our rank."""
    from ml.fantasy import (project, breakouts, with_adp, attach_value, value_board,
                            draft_path, FORMATS)
    view = request.args.get('view', 'board')
    pos = request.args.get('pos')
    fmt = request.args.get('scoring', 'bestball')        # 'scoring' param carries the format key
    if fmt not in FORMATS:
        fmt = 'bestball'
    if view == 'path':
        from ml.fantasy import DEFAULT_ROSTER
        dr = DEFAULT_ROSTER.get(fmt, DEFAULT_ROSTER['bestball'])
        slot = int(request.args.get('slot', 6))
        teams = int(request.args.get('teams', 12))
        roster = {p: int(request.args.get(p.lower(), dr[p])) for p in ('QB', 'RB', 'WR', 'TE', 'K', 'DST')}
        return jsonify(_native(draft_path(slot, fmt=fmt, teams=teams, roster=roster)))
    if view == 'breakouts':
        d = breakouts(2025, top=int(request.args.get('top', 25)), fmt=fmt)
        return jsonify(_native({"view": "breakouts", "season": 2025, "scoring": fmt,
                                "players": d.to_dict('records')}))
    if view == 'values':
        tg, fd, market = value_board(top=int(request.args.get('top', 30)), fmt=fmt)
        return jsonify(_native({"view": "values", "scoring": fmt, "market_label": market,
                                "targets": tg.to_dict('records'), "fades": fd.to_dict('records')}))
    b = attach_value(with_adp(project(fmt)), fmt)
    label = b['market_label'].iloc[0] if len(b) else ''
    if pos and pos.upper() in ('QB', 'RB', 'WR', 'TE'):
        b = b[b['position'] == pos.upper()]
    has_adp = bool(b['adp'].notna().any())
    b = b.head(int(request.args.get('limit', 240)))
    return jsonify(_native({"view": "board", "has_adp": has_adp, "scoring": fmt,
                            "market_label": label, "count": len(b), "players": b.to_dict('records')}))


@app.route('/api/props')
def api_props():
    """Per-player prop markets for a matchup: projected number + distribution params so the
    frontend can price any book line as over/under + fair odds. Built on the same opponent-
    adjusted, game-script-shaped projections as the matchup box score."""
    from ml.props import player_props
    home = request.args.get('home')
    away = request.args.get('away')
    if not home or not away or home == away:
        return jsonify({"error": "pick two different teams"}), 400
    neutral = request.args.get('neutral') == '1'
    data = player_props(home, away, neutral=neutral)
    if request.args.get('lines') == '1':                 # merge real book lines (spends odds credits)
        from ml.odds import event_props, have_key, _namekey
        if not have_key():
            data["odds_status"] = {"error": "ODDS_API_KEY not set on the server"}
        else:
            meta = team_meta()
            hn = meta.get(home, {}).get('team_name', home)
            an = meta.get(away, {}).get('team_name', away)
            props_map, status = event_props(hn, an)
            data["odds_status"] = status
            for team in (home, away):
                for pl in data["teams"].get(team, []):
                    pk = _namekey(pl["name"])
                    for m in pl["markets"]:
                        book = props_map.get((pk, m["market"]))
                        if book:
                            m["book"] = book
    return jsonify(_native(data))


@app.route('/api/season')
def api_season():
    """Season-long projections: team win totals (expected wins + fair O/U line + P(over) from the
    Poisson-binomial over each team's 2026 schedule) or player season stat totals + fantasy points."""
    from ml.season import team_win_totals, player_season_totals, status
    view = request.args.get('view', 'wins')
    st = status()
    if view == 'implied':
        from ml.season import implied_totals
        return jsonify(_native({"view": "implied", "status": st,
                                "teams": implied_totals().to_dict('records')}))
    if view == 'players':
        from ml.fantasy import FORMATS
        fmt = request.args.get('scoring', 'half')
        if fmt not in FORMATS:
            fmt = 'half'
        pos = request.args.get('pos')
        d = player_season_totals(fmt)
        if pos and pos.upper() in ('QB', 'RB', 'WR', 'TE'):
            d = d[d['position'] == pos.upper()]
        d = d.head(int(request.args.get('limit', 200)))
        return jsonify(_native({"view": "players", "scoring": fmt, "status": st,
                                "players": d.to_dict('records')}))
    from ml.season import win_total_lines, p_over_line
    w = team_win_totals()
    teams = w.to_dict('records')
    lines = win_total_lines()
    if lines:
        def _amprob(o):
            return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)
        for t in teams:
            bk = lines.get(t['team'])
            if not bk:
                continue
            po = p_over_line(t['dist'], bk['line'])
            oo, uo = bk.get('over_odds') or -110, bk.get('under_odds') or -110
            novig = _amprob(oo) / (_amprob(oo) + _amprob(uo))
            t['book'] = {"line": bk['line'], "over_odds": oo, "under_odds": uo,
                         "priced": bk.get('over_odds') is not None,
                         "our_over": round(po, 3), "novig_over": round(novig, 3),
                         "edge": round(po - novig, 3)}
    return jsonify(_native({"view": "wins", "status": st,
                            "has_book": bool(lines), "teams": teams}))


# ═══════════════════════════════════════════════════════════════════
#  DATA REFRESH — download latest nflverse data + rebuild light tables
#  Runs in a background thread (POST /api/refresh) or on an in-process
#  daily schedule. Uses ml.refresh, which downloads release parquets
#  directly (no nfl_data_py — that conflicts with pandas 3).
# ═══════════════════════════════════════════════════════════════════
import os
import threading

_REFRESH_STATE = {"running": False, "log": []}
_REFRESH_LOCK = threading.Lock()


def clear_caches(scope: str = "full"):
    """Drop in-process caches so freshly refreshed data is served immediately.

    scope='light' (the 4-hourly availability pull: injuries, rosters, depth charts,
    schedules) keeps the heavy, unchanged tables — play-by-play frames, team_styles, the
    adjusted-EPA fits, history, backtests — and clears only what those feeds touch. Clearing
    and rebuilding everything six times a day left the process at ~1 GB of half-released
    frames between pulls, which is what Railway was billing."""
    global _TEAM_META, _QB1, _SQUAD, _INJ, _SCHED, _PFF_COMPARE
    _QB1 = _SQUAD = _INJ = _SCHED = _PFF_COMPARE = None
    _DEPTH_CACHE.clear()
    _PROJ_CACHE.clear()
    _SCHED_PRED.clear()
    for mod, attr in [("ml.matchup_engine", "_UNITS"), ("ml.matchup_engine", "_CAL"), ("ml.squad", "_PCT_CACHE"),
                      ("ml.squad", "_META_CACHE"), ("ml.squad", "_PBP_AGG"), ("ml.squad", "_SKILL_CACHE"),
                      ("ml.projections", "_PROFILE_CACHE"), ("ml.projections", "_QBDEPTH_CACHE"),
                      ("ml.projections", "_RANK_CACHE"), ("ml.projections", "_TV_CACHE")]:
        try:
            import importlib
            setattr(importlib.import_module(mod), attr, None)
        except Exception:
            pass
    for modname in ("ml.matchup_context", "ml.current", "ml.season"):
        try:
            import importlib
            importlib.import_module(modname).clear()
        except Exception:
            pass
    # blended views depend on games played (ml.current weights) — rebuild them from the kept raw tables
    global _STYLES
    _STYLES = None
    _UNIT_EPA_CACHE.clear()
    _LEAGUE_STATS_CACHE.clear()
    try:
        import ml.squad as _sq
        _sq._PFF_TABLE = None                         # PFF prior blend weight moves with games played
    except Exception:
        pass
    if scope == "light":
        return
    _TEAM_META = None
    _PBP_CACHE.clear()
    _PFF_UNITS_CACHE.clear()
    try:                                              # PFF grade lookups (squad player cards)
        import ml.squad as _sq
        _sq._PFF_CACHE = _sq._PFF_PRE_CACHE = _sq._PFF_TABLE = None
    except Exception:
        pass
    try:                                              # QB/unit history tables
        import ml.history as _hist
        _hist.clear()
    except Exception:
        pass
    for modname, cachename in [("ml.adjust", "_ADJ_CACHE"), ("ml.backtest_spreads", "_BT_CACHE"),
                               ("ml.projections", "_BOX_CACHE")]:
        try:
            import importlib
            getattr(importlib.import_module(modname), cachename).clear()
        except Exception:
            pass
    try:
        import ml.backtest_spreads
        ml.backtest_spreads._BLEND_W = None           # recompute optimal blend after refresh
    except Exception:
        pass
    global _STYLES_RAW
    _STYLES_RAW = None
    _UNIT_EPA_CACHE.clear()
    _LEAGUE_STATS_CACHE.clear()
    for modname in ("ml.coaching", "ml.fantasy", "ml.odds"):
        try:
            import importlib
            importlib.import_module(modname).clear()
        except Exception:
            pass


def _run_refresh(season: int, light: bool = False):
    from ml import refresh as R

    def log(msg, level="INFO"):
        _REFRESH_STATE["log"].append(str(msg))
        del _REFRESH_STATE["log"][:-40]
        print(f"[refresh] {msg}", flush=True)         # also to stdout so `railway logs` shows it

    try:
        R.run(season, log=log, light=light)
    except Exception as e:
        log(f"FATAL {e}", "WARN")
    finally:
        clear_caches("light" if light else "full")
        _release_memory()
        log(f"picks ledger: {lock_current_week()}")   # freeze this week's picks on fresh data
        _release_memory()
        _REFRESH_STATE["running"] = False


def _release_memory() -> None:
    """Hand freed frames back to the OS. Dropping a cache frees Python objects, but glibc keeps
    the arenas, so the container's RSS — what Railway bills — stayed near its peak until the
    next restart. gc + malloc_trim returns it; a no-op where malloc_trim is unavailable."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _start_refresh(season: int, light: bool = False) -> bool:
    """Start a refresh thread if none is running. Returns False if already running."""
    with _REFRESH_LOCK:
        if _REFRESH_STATE["running"]:
            return False
        _REFRESH_STATE["running"] = True
        _REFRESH_STATE["log"] = []
    threading.Thread(target=_run_refresh, args=(season, light), daemon=True).start()
    return True


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Trigger a background data refresh. Guarded by the REFRESH_TOKEN env var."""
    token = os.environ.get("REFRESH_TOKEN")
    if not token:
        return jsonify({"error": "refresh disabled (no REFRESH_TOKEN configured)"}), 403
    supplied = request.headers.get("X-Refresh-Token") or request.args.get("token")
    if supplied != token:
        return jsonify({"error": "invalid token"}), 401
    season = int(request.args.get("season", os.environ.get("REFRESH_SEASON", 2026)))
    light = request.args.get("light") == "1"          # availability-only pull (seconds)
    if not _start_refresh(season, light):
        return jsonify({"error": "refresh already running"}), 409
    return jsonify({"started": True, "season": season, "light": light})


@app.route('/api/refresh/status')
def api_refresh_status():
    from ml import refresh as R
    return jsonify({
        "running": _REFRESH_STATE["running"],
        "last_refresh": R.last_status(),
        "log_tail": "\n".join(_REFRESH_STATE["log"][-12:]),
    })


def _daily_scheduler():
    """Optional in-process daily refresh. Enable with REFRESH_DAILY=1 (hour = REFRESH_HOUR
    UTC, default 8). One worker only (gunicorn --workers 1), so a single thread suffices."""
    import time as _t
    from datetime import datetime, timezone
    hour = int(os.environ.get("REFRESH_HOUR", 8))
    season = int(os.environ.get("REFRESH_SEASON", 2026))
    # Boot catch-up: a deploy lands whenever it lands; if the volume's last refresh is older
    # than REFRESH_STALE_HOURS (default 6) pull now instead of waiting for tomorrow's slot.
    # In-season that's what makes a game-day deploy carry the same-day injury report.
    try:
        from ml import refresh as R
        last = (R.last_status() or {}).get("finished")
        stale_h = float(os.environ.get("REFRESH_STALE_HOURS", 6))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 3600 \
            if last else 1e9
        if age >= stale_h:
            print(f"[boot] last refresh {age:.1f}h old (>= {stale_h}h) — refreshing now")
            _t.sleep(15)                     # let gunicorn finish binding first
            _start_refresh(season)
    except Exception as e:
        print(f"[boot] catch-up refresh check failed: {e}")
    # Between the daily full refresh, LIGHT availability pulls (injuries + depth charts +
    # roster release, seconds of work) every REFRESH_LIGHT_HOURS (default 4). Injury reports
    # publish Wed–Fri afternoons ET and IR moves happen any day; a once-a-day pull left the
    # matchup box scores projecting players who had been ruled out hours earlier.
    light_h = float(os.environ.get("REFRESH_LIGHT_HOURS", 4))
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target = target.replace(day=now.day)
            secs = (target - now).total_seconds() + 86400
        else:
            secs = (target - now).total_seconds()
        nap = min(secs, light_h * 3600) if light_h > 0 else secs
        _t.sleep(max(60, nap))
        if nap >= secs - 1:                          # reached the daily slot → full refresh
            _start_refresh(season)
            _t.sleep(3600)                           # avoid double-firing within the same hour
        else:
            _start_refresh(season, light=True)


_kalshi_selftest()

if os.environ.get("REFRESH_DAILY") == "1":
    threading.Thread(target=_daily_scheduler, daemon=True).start()
    print(f"[boot] daily refresh scheduled: {int(os.environ.get('REFRESH_HOUR', 8)):02d}:00 UTC, "
          f"season {os.environ.get('REFRESH_SEASON', 2026)}")


def _pff_boot_sync():
    """Pull the PFF parquets from the private data repo on boot, so a redeploy alone
    brings the site current — no manual refresh needed after a weekly grade push."""
    import time as _t
    _t.sleep(5)                                       # let gunicorn finish booting
    try:
        from ml.refresh import pull_pff
        res = pull_pff()
        if "parquet" in (res or ""):
            global _PFF_COMPARE
            _PFF_COMPARE = None
            import ml.squad as _sq
            _sq._PFF_CACHE = _sq._PFF_PRE_CACHE = _sq._PFF_TABLE = None
            _DEPTH_CACHE.clear()
            _PFF_UNITS_CACHE.clear()
        print(f"[boot] pff sync: {res}")
    except Exception as e:
        print(f"[boot] pff sync failed: {e}")


if os.environ.get("PFF_DATA_REPO") and os.environ.get("PFF_DATA_TOKEN"):
    threading.Thread(target=_pff_boot_sync, daemon=True).start()


if __name__ == '__main__':
    # Local dev entrypoint. In production (Railway) gunicorn imports `app` directly
    # and this block never runs — but honor $PORT / $HOST if someone runs it directly.
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print("NFL 2026 Dashboard — power rankings + matchup predictions")
    print(f"Open: http://localhost:{port}   (legacy engine dashboard at /legacy)")
    print()
    app.run(debug=debug, host=host, port=port, use_reloader=False)
