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
    season = int(request.args.get('season', 2025))
    if season in _UNIT_EPA_CACHE:
        return jsonify(_UNIT_EPA_CACHE[season])
    from ml.adjust import adjusted_unit_epa
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
    payload = {"season": season, "teams": recs}
    _UNIT_EPA_CACHE[season] = payload
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
]


def _scoring_avgs(season: int) -> dict:
    """{team: (points_for_avg, points_against_avg)} from final scores in schedules."""
    s = schedules_df()
    if not len(s):
        return {}
    sc = s[(s["season"] == season) & s["home_score"].notna() & (s["week"] <= 18)]
    pf, pa = {}, {}
    for _, g in sc.iterrows():
        for team, scored, allowed in ((g["home_team"], g["home_score"], g["away_score"]),
                                      (g["away_team"], g["away_score"], g["home_score"])):
            pf.setdefault(team, []).append(scored)
            pa.setdefault(team, []).append(allowed)
    return {t: (float(np.mean(pf[t])), float(np.mean(pa.get(t, [0])))) for t in pf}


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
    if season in _LEAGUE_STATS_CACHE:
        return jsonify(_LEAGUE_STATS_CACHE[season])
    s = styles_df()
    sub = s[s["season"] == season]
    if sub.empty:
        return jsonify({"error": f"no stats for {season}"}), 404
    meta = team_meta()
    scoring = _scoring_avgs(season)
    payload = {
        "season": season,
        "offense": _stat_side(sub, _OFF_COLS, meta, scoring, is_def=False),
        "defense": _stat_side(sub, _DEF_COLS, meta, scoring, is_def=True),
    }
    _LEAGUE_STATS_CACHE[season] = payload
    return jsonify(payload)


_DEPTH_CACHE = {}
_PFF_COMPARE = None


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
    d["pff_pctl"] = d.groupby("position")["pff_grade"].rank(pct=True) * 100
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
    key = (home, away)
    if key not in _PROJ_CACHE:
        from ml.projections import project_matchup
        _PROJ_CACHE[key] = project_matchup(home, away)
    return jsonify(_native(_PROJ_CACHE[key]))


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


def styles_df() -> pd.DataFrame:
    global _STYLES
    if _STYLES is None:
        _STYLES = pd.read_parquet(PROC / "team_styles.parquet")
    return _STYLES


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
                "def_points_allowed_avg", "turnover_rate", "third_down_stop_rate"]
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
_STATUS_RANK = {"Out": 0, "Doubtful": 1, "Questionable": 2}


def latest_injuries(team: str) -> dict:
    """Most-recent available injury report for a team (empty in the offseason)."""
    inj = injuries_df()
    if inj.empty or "team" not in inj.columns:
        return {"season": None, "week": None, "players": []}
    t = inj[inj["team"] == team]
    if t.empty:
        return {"season": None, "week": None, "players": []}
    season = int(t["season"].max())
    t = t[t["season"] == season]
    week = int(t["week"].max())
    t = t[t["week"] == week]
    players = []
    for _, r in t.iterrows():
        st = r.get("report_status")
        if not st or (isinstance(st, float) and pd.isna(st)):
            continue
        players.append({
            "name": r.get("full_name"), "position": r.get("position"),
            "status": st, "injury": r.get("report_primary_injury") or "",
        })
    players.sort(key=lambda x: _STATUS_RANK.get(x["status"], 3))
    return {"season": season, "week": week, "players": players}


def team_injury_map(team: str) -> dict:
    """Latest injury report for a team keyed by gsis_id — for the profile depth chart.
    Id-keyed (not name) to stay coherent with the id-first depth-chart join. Empty in
    the offseason before that season's game reports publish (nflverse has no file yet)."""
    inj = injuries_df()
    if inj.empty or "team" not in inj.columns or "gsis_id" not in inj.columns:
        return {"season": None, "week": None, "by_id": {}}
    t = inj[inj["team"] == team]
    if t.empty:
        return {"season": None, "week": None, "by_id": {}}
    season = int(t["season"].max())
    t = t[t["season"] == season]
    week = int(t["week"].max())
    t = t[t["week"] == week]
    by_id = {}
    for _, r in t.iterrows():
        gid, st = r.get("gsis_id"), r.get("report_status")
        if not gid or (isinstance(gid, float) and pd.isna(gid)):
            continue
        if not st or (isinstance(st, float) and pd.isna(st)):
            continue                                   # no designation → healthy, skip
        by_id[str(gid)] = {"status": str(st),
                           "injury": r.get("report_primary_injury") or ""}
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
    """Past meetings between the two teams (either venue) + series record & ATS/O-U trends."""
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
    MIN_EDGE = 2.5                                    # min points vs the line to count as a real play
    qualified = [g for g in scored if abs(g.get("edge") or 0) >= MIN_EDGE]
    for i, g in enumerate(sorted(qualified, key=lambda x: -x["cover_prob"])[:5], 1):
        g["pick_rank"] = i
    return games, odds_status


@app.route('/api/schedule')
def api_schedule():
    """A week's slate: every game with the model's roster+injury-adjusted prediction
    (and the Vegas line / final score when available). Auto-pairs home/away from the schedule."""
    from ml.projections import unavailable_ids
    s = schedules_df()
    if s.empty:
        return jsonify({"error": "no schedule data"}), 404
    seasons = sorted(int(x) for x in s["season"].dropna().unique())
    season = int(request.args.get('season', seasons[-1]))
    d = s[s["season"] == season].copy()
    if "game_type" in d.columns:                     # regular season for the weekly view
        d = d[d["game_type"].fillna("REG").str.upper().eq("REG")]
    weeks = sorted(int(x) for x in d["week"].dropna().unique())
    if not weeks:
        return jsonify(_native({"season": season, "week": None, "seasons": seasons, "weeks": [], "games": []}))
    week = int(request.args.get('week', weeks[0]))
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
    return jsonify(_native({"season": season, "week": week, "seasons": seasons, "weeks": weeks,
                            "games": games, "odds_status": odds_status}))


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


def clear_caches():
    """Drop every in-process cache so freshly refreshed data is served immediately."""
    global _TEAM_META, _QB1, _SQUAD, _STYLES, _INJ, _SCHED, _PFF_COMPARE
    _TEAM_META = _QB1 = _SQUAD = _STYLES = _INJ = _SCHED = _PFF_COMPARE = None
    _DEPTH_CACHE.clear()
    _PROJ_CACHE.clear()
    _PBP_CACHE.clear()
    _SCHED_PRED.clear()
    for modname, cachename in [("ml.adjust", "_ADJ_CACHE"), ("ml.backtest_spreads", "_BT_CACHE")]:
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
    for modname in ("ml.matchup_context", "ml.coaching", "ml.fantasy", "ml.odds", "ml.season"):
        try:
            import importlib
            importlib.import_module(modname).clear()
        except Exception:
            pass
    for mod, attr in [("ml.matchup_engine", "_UNITS"), ("ml.squad", "_PCT_CACHE"),
                      ("ml.squad", "_META_CACHE"), ("ml.squad", "_SKILL_CACHE"),
                      ("ml.squad", "_PBP_AGG"), ("ml.projections", "_PROFILE_CACHE"),
                      ("ml.projections", "_QBDEPTH_CACHE")]:
        try:
            import importlib
            setattr(importlib.import_module(mod), attr, None)
        except Exception:
            pass


def _run_refresh(season: int):
    from ml import refresh as R

    def log(msg, level="INFO"):
        _REFRESH_STATE["log"].append(str(msg))
        del _REFRESH_STATE["log"][:-40]

    try:
        R.run(season, log=log)
    except Exception as e:
        _REFRESH_STATE["log"].append(f"FATAL {e}")
    finally:
        clear_caches()
        _REFRESH_STATE["running"] = False


def _start_refresh(season: int) -> bool:
    """Start a refresh thread if none is running. Returns False if already running."""
    with _REFRESH_LOCK:
        if _REFRESH_STATE["running"]:
            return False
        _REFRESH_STATE["running"] = True
        _REFRESH_STATE["log"] = []
    threading.Thread(target=_run_refresh, args=(season,), daemon=True).start()
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
    if not _start_refresh(season):
        return jsonify({"error": "refresh already running"}), 409
    return jsonify({"started": True, "season": season})


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
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target = target.replace(day=now.day)
            secs = (target - now).total_seconds() + 86400
        else:
            secs = (target - now).total_seconds()
        _t.sleep(max(60, secs))
        _start_refresh(season)
        _t.sleep(3600)   # avoid double-firing within the same hour


if os.environ.get("REFRESH_DAILY") == "1":
    threading.Thread(target=_daily_scheduler, daemon=True).start()


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
