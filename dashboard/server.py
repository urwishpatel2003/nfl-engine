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
            from ml.squad import squad_ratings
            _SQUAD = squad_ratings()[0]
        r = _SQUAD
    else:
        from ml.rank import power_ratings
        r = power_ratings(season)
    qbs = qb1_2026() if mode == 'preseason' else {}
    recs = []
    for _, row in r.iterrows():
        m = meta.get(row["team"], {})
        recs.append({
            "rank": int(row["rank"]), "team": row["team"], "rating": float(row["rating"]),
            "prev": float(row["rating_prev"]) if "rating_prev" in r.columns else None,
            "name": m.get("team_name", row["team"]),
            "color": m.get("team_color") or "#334155",
            "logo": m.get("team_logo_espn", ""),
            "qb": qbs.get(row["team"], ""),
        })
    return jsonify(recs)


_DEPTH_CACHE = {}


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
    return jsonify({
        "team": team, "name": m.get("team_name", team),
        "color": m.get("team_color") or "#334155", "logo": m.get("team_logo_espn", ""),
        "qb": qb, "groups": _DEPTH_CACHE[team],
    })


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
    ("def_epa_per_play",     "Defense EPA/play",     False),
    ("def_epa_per_pass",     "Pass defense",         False),
    ("def_epa_per_rush",     "Run defense",          False),
    ("def_success_rate",     "Defensive efficiency", False),
    ("pressure_rate_gen",    "Pass-rush pressure",   True),
    ("sack_rate_gen",        "Sack rate",            True),
    ("third_down_stop_rate", "Third-down defense",   True),
    ("def_quality_score",    "Overall defense grade", True),
]

# boolean archetype flags in team_styles → human tendency labels
_FLAG_LABELS = {
    "run_heavy_off": "Run-heavy offense", "pass_heavy_off": "Pass-heavy offense",
    "deep_pass_off": "Deep passing attack", "short_pass_off": "Short passing game",
    "fast_pace": "Fast tempo", "slow_pace": "Slow tempo",
    "blitz_heavy_def": "Blitz-heavy defense", "high_play_action": "High play-action",
    "motion_heavy_off": "Heavy pre-snap motion", "mobile_qb_offense": "Mobile QB",
    "elite_mobile_qb": "Elite mobile QB", "fourth_down_aggressive": "Aggressive on 4th down",
    "elite_rz_offense": "Elite red-zone offense", "elite_rz_defense": "Elite red-zone defense",
    "elite_2min": "Elite two-minute offense", "leaky_under_pressure": "Struggles under pressure",
    "poor_qb_contain": "Poor QB contain", "elite_qb_contain": "Elite QB contain",
}


def _tendencies(row) -> list:
    return [lbl for flag, lbl in _FLAG_LABELS.items()
            if flag in row and bool(row[flag]) and not pd.isna(row[flag])]


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
                "rz_td_rate", "rz_td_rate_allowed_x", "two_min_epa", "fourth_go_rate",
                "def_points_allowed_avg", "turnover_rate", "third_down_stop_rate"]
    style = {k: safe_json(row[k]) for k in style_keys if k in row.index}
    situational = {k: safe_json(row[k]) for k in sit_keys if k in row.index}

    pcts = _profile_percentiles(team, season)
    strengths = sorted(pcts, key=lambda x: -x["pctl"])[:5]
    weaknesses = sorted(pcts, key=lambda x: x["pctl"])[:5]

    if team not in _DEPTH_CACHE:
        _DEPTH_CACHE[team] = team_depth_chart(team)

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
        "tendencies": _tendencies(row),
        "units": _units_display(team),
        "groups": _DEPTH_CACHE[team],
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
            "tendencies": _tendencies(r),
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
    return jsonify(_native(res))


_SCHED_PRED = {}   # (season, week) -> games list; cleared on refresh


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
    if (season, week) in _SCHED_PRED:
        return jsonify(_native({"season": season, "week": week, "seasons": seasons,
                                "weeks": weeks, "games": _SCHED_PRED[(season, week)]}))
    dw = d[d["week"] == week]
    sort_cols = [c for c in ["gameday", "gametime"] if c in dw.columns]
    if sort_cols:
        dw = dw.sort_values(sort_cols)

    from ml.context import game_context
    meta = team_meta()
    unavail = unavailable_ids()
    games = []
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
        games.append(rec)

    # Turn each point estimate into a key-number-aware ATS pick + cover probability, and an
    # over/under read. Rank the top-5 plays by cover probability (accounts for key numbers,
    # not just raw edge).
    from ml.spreads import ats_pick as _ats_pick, total_prob as _total_prob
    from ml.backtest_spreads import blend_weight
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
    for i, g in enumerate(sorted(scored, key=lambda x: -x["cover_prob"])[:5], 1):
        g["pick_rank"] = i

    _SCHED_PRED[(season, week)] = games
    return jsonify(_native({"season": season, "week": week, "seasons": seasons,
                            "weeks": weeks, "games": games}))


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
    global _TEAM_META, _QB1, _SQUAD, _STYLES, _INJ, _SCHED
    _TEAM_META = _QB1 = _SQUAD = _STYLES = _INJ = _SCHED = None
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
    try:
        import ml.matchup_context
        ml.matchup_context.clear()
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
