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


@app.route('/api/power_rankings')
def api_power_rankings():
    """Standalone-model power ratings for the upcoming season (entering `season`+1)."""
    season = int(request.args.get('season', 2025))
    from ml.rank import power_ratings
    r = power_ratings(season)
    meta = team_meta()
    recs = []
    for _, row in r.iterrows():
        m = meta.get(row["team"], {})
        recs.append({
            "rank": int(row["rank"]), "team": row["team"], "rating": float(row["rating"]),
            "name": m.get("team_name", row["team"]),
            "color": m.get("team_color") or "#334155",
            "logo": m.get("team_logo_espn", ""),
        })
    return jsonify(recs)


@app.route('/api/matchup')
def api_matchup():
    """Predict any matchup from current team strength."""
    from ml.rank import predict_matchup
    home = request.args.get('home')
    away = request.args.get('away')
    season = int(request.args.get('season', 2025))
    neutral = request.args.get('neutral', '0') == '1'
    if not home or not away:
        return jsonify({"error": "home and away required"}), 400
    res = predict_matchup(home.upper(), away.upper(), season, neutral)
    if "error" in res:
        return jsonify(res), 404
    meta = team_meta()
    for side in ("home", "away"):
        m = meta.get(res[side], {})
        res[f"{side}_name"] = m.get("team_name", res[side])
        res[f"{side}_color"] = m.get("team_color") or "#334155"
        res[f"{side}_logo"] = m.get("team_logo_espn", "")
    return jsonify({k: safe_json(v) for k, v in res.items()})


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


if __name__ == '__main__':
    print("NFL 2026 Dashboard — power rankings + matchup predictions")
    print("Open: http://localhost:5000   (legacy engine dashboard at /legacy)")
    print()
    app.run(debug=True, port=5000, use_reloader=False)
