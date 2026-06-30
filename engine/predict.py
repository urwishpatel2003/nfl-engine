"""
engine/predict.py
-----------------
Master prediction engine. Combines all components into a final
gameday output prediction.

For each game produces:
  TEAM LEVEL:
    predicted_home_score, predicted_away_score
    predicted_total, predicted_spread (home perspective)
    home_win_probability
    offensive_output_score (0-100)
    defensive_output_score (0-100)

  PLAY STYLE PREDICTION:
    home_predicted_pass_rate  -- will they be forced to pass more/less?
    home_predicted_rush_rate
    away_predicted_pass_rate
    away_predicted_rush_rate
    game_script               -- "shootout" / "defensive battle" / "run heavy" / "normal"

  KEY FACTORS:
    top_matchup_advantage     -- most impactful positional edge
    conditions_summary
    style_clash_summary

Formula (simplified):
  base_score = league_avg_points_per_game (22 pts/team historically)
  + team_off_quality    (from composite + style scores, max ±8 pts)
  + opponent_def_quality (from composite + style, max ±8 pts)
  + matchup_weighted_edge (from matchup matrix, max ±6 pts)
  + conditions_modifier (from conditions engine, max ±5 pts)
  * scoring_mult        (weather/surface multiplier)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

# League average points per game per team (2021-2025 avg)
LEAGUE_AVG_PTS = 24.8   # Recalibrated: NFL avg ~44.5 pts/game = 22.25/team
                        # +2.8 offset corrects for composite under-prediction bias
LEAGUE_AVG_PASS_RATE = 0.565


def load_engine_data():
    """Load all pre-built engine components."""
    data = {}

    p = PROC / "composite_scores.parquet"
    if p.exists():
        data["composite"] = pd.read_parquet(p)
    else:
        raise FileNotFoundError("composite_scores.parquet not found. Run composite.py first.")

    p = PROC / "team_styles.parquet"
    if p.exists():
        data["styles"] = pd.read_parquet(p)
    else:
        print("WARNING: team_styles.parquet not found. Style context will be skipped.")
        data["styles"] = None

    p = PROC / "conditions.parquet"
    if p.exists():
        data["conditions"] = pd.read_parquet(p)
    else:
        print("WARNING: conditions.parquet not found. Conditions will use defaults.")
        data["conditions"] = None

    data["schedules"]       = pd.read_parquet(RAW / "schedules.parquet")
    data["depth"]           = pd.read_parquet(RAW / "depth_charts.parquet")

    # Load officials and compute per-crew penalty tendency (NEW)
    off_path = RAW / "officials.parquet"
    if off_path.exists():
        officials_raw = pd.read_parquet(off_path)
        # Compute game-level penalty counts from PBP if available, else use game count proxy
        # For now: compute ref crew's historical avg using game_id frequency as proxy
        # A full implementation would join to PBP penalty column
        ref_counts = officials_raw.groupby(["game_id","official_id"]).size().reset_index()
        # Use referee (off_pos == "R" = head ref) as crew identifier
        head_refs = officials_raw[officials_raw.get("off_pos", pd.Series("R")) == "R"][["game_id","official_id"]].copy() \
            if "off_pos" in officials_raw.columns else officials_raw[["game_id","official_id"]].drop_duplicates("game_id")
        data["officials"] = officials_raw
        data["head_refs"] = head_refs
    else:
        data["officials"] = None
        data["head_refs"] = None

    # Normalize depth chart schema (nflverse changed in 2026)
    _d = data["depth"]
    if "season" not in _d.columns:
        _d["season"] = 2025
        _d["week"]   = 1
        _d["depth_team"] = _d.get("pos_slot", pd.Series(2, index=_d.index)).fillna(2).clip(lower=1).astype(int)
    if "club_code" not in _d.columns and "team" in _d.columns:
        _d["club_code"] = _d["team"]
    data["depth"] = _d
    data["rosters_weekly"]  = pd.read_parquet(RAW / "rosters_weekly.parquet")

    # Load coaching scores if available
    coaching_path = PROC / "coaching_scores.parquet"
    data["coaching"] = pd.read_parquet(coaching_path) if coaching_path.exists() else None

    return data


def get_starting_qb(rosters: pd.DataFrame, composite: pd.DataFrame,
                    team: str, season: int, week: int) -> dict:
    """
    Fix 3: Determine who actually started at QB for a team in a given week.

    Checks rosters_weekly for active QBs on that team that week.
    Compares against composite to get starter vs backup scores.
    Returns dict with starter info and any backup penalty to apply.
    """
    # Get QBs on this team this week from rosters
    team_qbs = rosters[
        (rosters["team"] == team) &
        (rosters["season"] == season) &
        (rosters["week"] == week) &
        (rosters["position"] == "QB") &
        (rosters["status"].isin(["ACT", "Active", "A"]) | rosters["status"].isna())
    ].copy()

    if team_qbs.empty:
        # Widen search — any week near this one
        team_qbs = rosters[
            (rosters["team"] == team) &
            (rosters["season"] == season) &
            (rosters["position"] == "QB")
        ].sort_values("week", ascending=False)
        team_qbs = team_qbs[team_qbs["week"] <= week].head(5)

    if team_qbs.empty:
        return {"backup_penalty": 0.0, "starter_name": None, "is_backup": False}

    # Get composite scores for these QBs
    qb_ids = team_qbs["player_id"].dropna().unique()
    qb_scores = composite[
        (composite["player_id"].isin(qb_ids)) &
        (composite["season"] == season) &
        (composite["week"] <= week)
    ].sort_values("week", ascending=False).drop_duplicates("player_id")

    if qb_scores.empty:
        return {"backup_penalty": 0.0, "starter_name": None, "is_backup": False}

    # The top-scoring QB is our presumed starter
    qb_scores = qb_scores.sort_values("adjusted_score", ascending=False)
    starter   = qb_scores.iloc[0]
    starter_score = float(starter["adjusted_score"])
    starter_name  = str(starter.get("player_display_name", "Unknown"))

    # Check schedules for known starting QB (if available)
    # schedules has home_qb_id / away_qb_id columns in some seasons
    backup_penalty = 0.0
    is_backup      = False

    # If there's a clear backup (2nd QB has much lower score), flag it
    if len(qb_scores) >= 2:
        backup_score = float(qb_scores.iloc[1]["adjusted_score"])
        score_gap    = starter_score - backup_score

        # If the top QB score is unusually LOW for a starter (< 35),
        # it might mean the real starter is injured and this IS the backup
        if starter_score < 35 and score_gap < 15:
            # Both QBs look like backups — apply a team-level penalty
            backup_penalty = -6.0
            is_backup = True
        elif starter_score < 40:
            # Starting QB is below average — smaller penalty
            backup_penalty = -3.0

    return {
        "backup_penalty":  round(backup_penalty, 1),
        "starter_name":    starter_name,
        "starter_score":   round(starter_score, 1),
        "is_backup":       is_backup,
    }


def get_team_quality(composite: pd.DataFrame, styles: pd.DataFrame,
                     team: str, season: int, week: int,
                     coaching: pd.DataFrame = None) -> dict:
    """
    Compute team offensive and defensive quality scores
    from composite player scores + style metrics.
    """
    # Get this team's players this week
    players = composite[
        (composite["recent_team"] == team) &
        (composite["season"] == season) &
        (composite["week"] <= week)
    ].sort_values("week", ascending=False).drop_duplicates("player_id")

    # Offensive quality: top QB + top skill players
    qb_score  = players[players["position"] == "QB"]["adjusted_score"].max()
    wr_scores = players[players["position"] == "WR"]["adjusted_score"].nlargest(3).mean()
    rb_score  = players[players["position"] == "RB"]["adjusted_score"].max()
    te_score  = players[players["position"] == "TE"]["adjusted_score"].max()

    # ── Dynamic position weights based on play style ──────────────
    # The value of each position group depends on HOW the team plays.
    # A run-heavy team's RB matters far more than their QB for scoring.
    # A pass-heavy team's QB + WRs dominate. OL quality modifies these weights.
    #
    # Base weights (league average team, ~56% pass rate):
    #   QB: 0.40  WR: 0.28  RB: 0.19  TE: 0.13
    #
    # Style adjustments (interpolated from pass_rate before styles are loaded):
    #   Run heavy (≤50% pass):  QB 0.28  WR 0.20  RB 0.35  TE 0.17
    #   Balanced  (56% pass):   QB 0.40  WR 0.28  RB 0.19  TE 0.13
    #   Pass heavy (≥62% pass): QB 0.50  WR 0.32  RB 0.09  TE 0.09
    #
    # OL interaction (applied after style weights):
    #   Elite OL    → shifts 3pts weight from QB toward RB (clean pocket = QB less critical; run game amplified)
    #   Leaky OL    → shifts 3pts weight from RB toward QB (run game disrupted; QB must carry more)
    #   This reflects the empirical finding that BAL's OL is built for Henry, not Lamar

    # Start with style-aware pass rate (will be overridden if styles available below)
    _pass_rate = LEAGUE_AVG_PASS_RATE
    _is_elite_ol = False
    _is_leaky_ol = False
    _off_label = "Balanced"

    # Peek at styles to get pass rate and OL flags before computing weights
    # Fall back to most recent available season if exact season not present
    if styles is not None:
        s_peek = styles[(styles["team"] == team) & (styles["season"] == season)]
        if s_peek.empty:
            # Use most recent season available for this team
            team_styles = styles[styles["team"] == team]
            if not team_styles.empty:
                s_peek = team_styles.sort_values("season", ascending=False).head(1)
        if not s_peek.empty:
            r_peek = s_peek.iloc[0]
            _pass_rate   = float(r_peek.get("pass_rate_overall", LEAGUE_AVG_PASS_RATE) or LEAGUE_AVG_PASS_RATE)
            _is_elite_ol = bool(r_peek.get("elite_ol", False))
            _is_leaky_ol = bool(r_peek.get("leaky_under_pressure", False))
            _off_label   = str(r_peek.get("offense_label", "Balanced"))
            _off_label   = str(r_peek.get("offense_label", "Balanced"))

    # Interpolate weights between run-heavy and pass-heavy anchors
    # pass_rate: 0.50 = run heavy, 0.565 = balanced, 0.62+ = pass heavy
    t = np.clip((_pass_rate - 0.50) / (0.62 - 0.50), 0.0, 1.0)  # 0=run heavy, 1=pass heavy
    w_qb = 0.28 + t * (0.50 - 0.28)   # 0.28 → 0.50
    w_wr = 0.20 + t * (0.32 - 0.20)   # 0.20 → 0.32
    w_rb = 0.35 - t * (0.35 - 0.09)   # 0.35 → 0.09
    w_te = 0.17 - t * (0.17 - 0.09)   # 0.17 → 0.09

    # OL interaction: elite OL amplifies RB (+3pts weight), leaky OL amplifies QB dependency
    if _is_elite_ol:
        # Clean pockets mean the OL is doing work — shift weight toward RB
        # Elite run AND pass blocking reduces QB's marginal contribution
        shift = 0.04
        w_rb += shift
        w_qb -= shift
    elif _is_leaky_ol:
        # Leaky OL disrupts run game and puts pressure on QB to escape
        # QB decision-making under pressure becomes the critical factor
        shift = 0.04
        w_qb += shift
        w_rb -= shift

    # Normalize to sum to 1.0 (rounding may drift)
    total_w = w_qb + w_wr + w_rb + w_te
    w_qb /= total_w; w_wr /= total_w; w_rb /= total_w; w_te /= total_w

    # Weighted offensive composite with style-dependent weights
    off_quality = (
        (qb_score  if pd.notna(qb_score)  else 50) * w_qb +
        (wr_scores if pd.notna(wr_scores) else 50) * w_wr +
        (rb_score  if pd.notna(rb_score)  else 50) * w_rb +
        (te_score  if pd.notna(te_score)  else 50) * w_te
    )

    # Style quality (from PBP-derived efficiency)
    off_epa_pts    = 0.0
    def_epa_pts    = 0.0
    def_quality    = 50.0   # 0-100 composite defensive quality
    pass_rate      = LEAGUE_AVG_PASS_RATE
    off_label      = "Balanced"
    def_label      = "Bend Dont Break"

    if styles is not None:
        s = styles[(styles["team"] == team) & (styles["season"] == season)]
        if s.empty:
            # Fall back to most recent available season for this team
            team_hist = styles[styles["team"] == team]
            if not team_hist.empty:
                s = team_hist.sort_values("season", ascending=False).head(1)
        if not s.empty:
            row = s.iloc[0]
            off_epa_raw  = float(row.get("off_epa_per_play", 0) or 0)
            def_epa_raw  = float(row.get("def_epa_per_play", 0) or 0)
            off_epa_pts  = np.clip(off_epa_raw * 18, -5, 5)
            def_epa_pts  = np.clip(def_epa_raw * 18, -5, 5)
            def_quality  = float(row.get("def_quality_score", 50) or 50)
            pass_rate    = float(row.get("pass_rate_overall", LEAGUE_AVG_PASS_RATE) or LEAGUE_AVG_PASS_RATE)
            off_label    = str(row.get("offense_label", "Balanced"))
            def_label    = str(row.get("defense_label", "Bend Dont Break"))

            # Play-action rate boost (NEW): high PA% teams score more efficiently
            # League avg PA ~22%, elite ~30%+. Each 1% above avg = +0.08 pts
            # Only apply when FTN data is available (2022+); guard against NaN/0
            pa_rate_raw = row.get("play_action_rate")
            if pa_rate_raw is not None:
                try:
                    pa_rate = float(pa_rate_raw)
                    if not np.isnan(pa_rate) and 0.05 < pa_rate < 0.70:
                        pa_boost = np.clip((pa_rate - 0.22) * 8.0, -1.0, 2.0)
                        off_epa_pts = np.clip(off_epa_pts + pa_boost, -6, 6)
                except (TypeError, ValueError):
                    pass

            # Boost offensive quality for elite red zone and 2-min drill teams
            rz_boost  = 2.0 if row.get("elite_rz_offense") else 0.0
            min2_boost = 1.5 if row.get("elite_2min") else 0.0
            off_quality += rz_boost + min2_boost

            # QB mobility boost (NEW): mobile QBs add a scoring dimension beyond
            # their passing composite. The scramble EPA from PBP feeds this.
            # elite_mobile_qb (Lamar tier): +3 pts off_quality
            # mobile_qb (Allen/Hurts/Maye tier): +1.5 pts off_quality
            # This stacks on top of their composite score which already captures
            # passing quality — this is the ADDITIONAL value from legs
            if row.get("elite_mobile_qb"):
                off_quality = min(100, off_quality + 3.0)
            elif row.get("mobile_qb_offense"):
                off_quality = min(100, off_quality + 1.5)

    # ── Coaching quality adjustments ──────────────────────────────
    # Lookup coaching scores — use most recent available season as fallback
    coaching_score   = 50.0
    is_elite_coach   = False
    is_poor_coach    = False
    is_elite_adjuster= False
    is_undisciplined = False

    if coaching is not None and not coaching.empty:
        c = coaching[(coaching["team"] == team) & (coaching["season"] == season)]
        if c.empty:
            # Fall back to most recent season
            team_c = coaching[coaching["team"] == team]
            if not team_c.empty:
                c = team_c.sort_values("season", ascending=False).head(1)
        if not c.empty:
            r = c.iloc[0]
            coaching_score    = float(r.get("coaching_score", 50) or 50)
            is_elite_coach    = bool(r.get("elite_coaching", False))
            is_poor_coach     = bool(r.get("poor_coaching", False))
            is_elite_adjuster = bool(r.get("elite_adjuster", False))
            is_undisciplined  = bool(r.get("undisciplined", False))

    # Apply coaching modifiers to off/def quality
    # Elite coach: +3 off_quality (scheme execution, adjustment, clock mgmt)
    # Poor coach:  −3 off_quality (wasted possessions, bad decisions)
    # Elite adjuster: +2 def_quality (second-half scheme changes confuse opponents)
    # Undisciplined: −2 off_quality (penalties kill drives, momentum)
    if is_elite_coach:
        off_quality = min(100, off_quality + 3.0)
    elif is_poor_coach:
        off_quality = max(0,   off_quality - 3.0)

    if is_elite_adjuster:
        # Good halftime adjusters also improve defensively in second half
        def_quality = min(100, def_quality + 2.0)

    if is_undisciplined:
        off_quality = max(0, off_quality - 2.0)

    if styles is not None:
        s2 = styles[(styles["team"] == team) & (styles["season"] == season)]
        if s2.empty:
            team_hist2 = styles[styles["team"] == team]
            if not team_hist2.empty:
                s2 = team_hist2.sort_values("season", ascending=False).head(1)
        if not s2.empty:
            row2 = s2.iloc[0]

            # Penalise offensive quality for leaky OL — scaled by pass rate
            # A pass-heavy team suffers far more from a leaky OL than a run team
            # Pass heavy (62%+): full −4.5 pts  |  Run heavy (50%): only −1.5 pts
            if row2.get("leaky_under_pressure"):
                ol_penalty = np.interp(_pass_rate, [0.50, 0.62], [1.5, 4.5])
                off_quality = max(0, off_quality - ol_penalty)

            # Elite OL boost — also scaled by play style
            # Run-heavy teams benefit MORE from elite OL (amplifies ground game)
            # Pass-heavy teams benefit less (QB still does the work)
            # Run heavy (50%): +3.5 pts  |  Pass heavy (62%+): +1.5 pts
            if row2.get("elite_ol"):
                ol_boost = np.interp(_pass_rate, [0.50, 0.62], [3.5, 1.5])
                off_quality = min(100, off_quality + ol_boost)

            # Defensive quality boost for elite pass rush
            if row2.get("elite_pass_rush"):
                def_quality = min(100, def_quality + 4.0)

            # Kicker quality: stored for use in score clipping / close game adjust
            kicker_score  = float(row2.get("kicker_score",  50) or 50)
            poor_kicker   = bool(row2.get("poor_kicker",    False))
            elite_kicker  = bool(row2.get("elite_kicker",   False))

            # Punter quality: elite punter gives slight def_quality boost (field position)
            if row2.get("elite_punter"):
                def_quality = min(100, def_quality + 1.5)

    return {
        "off_quality":  round(off_quality, 1),
        "def_quality":  round(def_quality, 1),
        "off_epa_pts":  round(np.clip(off_epa_pts, -8, 8), 2),
        "def_epa_pts":  round(np.clip(def_epa_pts, -8, 8), 2),
        "pass_rate":    round(pass_rate, 3),
        "off_label":    off_label,
        "def_label":    def_label,
        "qb_score":     round(float(qb_score) if pd.notna(qb_score) else 50, 1),
        "rb_score":     round(float(rb_score) if pd.notna(rb_score) else 50, 1),
        "wr_score":     round(float(wr_scores) if pd.notna(wr_scores) else 50, 1),
        "kicker_score": round(kicker_score, 1),
        "poor_kicker":  poor_kicker,
        "elite_kicker": elite_kicker,
    }


def predict_game(home_team: str, away_team: str,
                 season: int, week: int,
                 data: dict = None) -> dict:
    """
    Full game prediction.
    Returns comprehensive dict with scores, probabilities, and game script.
    """
    if data is None:
        data = load_engine_data()

    composite  = data["composite"]
    styles     = data.get("styles")
    conditions = data.get("conditions")
    rosters    = data.get("rosters_weekly", pd.DataFrame())
    schedules  = data.get("schedules")
    officials  = data.get("officials")
    coaching   = data.get("coaching")

    # ── Get team quality metrics ───────────────────────────────────
    home_q = get_team_quality(composite, styles, home_team, season, week, coaching)
    away_q = get_team_quality(composite, styles, away_team, season, week, coaching)

    # ── Fix 3: QB starting status check ───────────────────────────
    home_qb_info = get_starting_qb(rosters, composite, home_team, season, week)
    away_qb_info = get_starting_qb(rosters, composite, away_team, season, week)

    home_qb_penalty = home_qb_info["backup_penalty"]
    away_qb_penalty = away_qb_info["backup_penalty"]

    # ── Get matchup matrix ─────────────────────────────────────────
    from engine.matchups import build_game_matchups, summarize_game_matchups
    matchups = build_game_matchups(home_team, away_team, season, week,
                                    composite=composite,
                                    depth=data["depth"],
                                    styles=styles)
    matchup_summary = summarize_game_matchups(matchups)

    home_matchup_edge = matchup_summary.get(home_team, {}).get("weighted_matchup_edge", 0)
    away_matchup_edge = matchup_summary.get(away_team, {}).get("weighted_matchup_edge", 0)

    # Scale matchup edge to points (edge of 10 composite points ~ 1.5 score points)
    home_matchup_pts = home_matchup_edge * 0.15
    away_matchup_pts = away_matchup_edge * 0.15

    # ── Get conditions ─────────────────────────────────────────────
    home_adv_pts  = 1.5  # default home field
    pass_mult     = 1.0
    rush_mult     = 1.0
    scoring_mult  = 1.0
    cond_notes    = "Default conditions"

    if conditions is not None:
        cond = conditions[
            (conditions["home_team"] == home_team) &
            (conditions["away_team"] == away_team) &
            (conditions["season"] == season) &
            (conditions["week"] == week)
        ]
        if cond.empty:
            # Broader fallback: just home team + season + week
            cond = conditions[
                (conditions["home_team"] == home_team) &
                (conditions["season"] == season) &
                (conditions["week"] == week)
            ]
        if cond.empty:
            # Last resort: any game with these two teams this season
            cond = conditions[
                (conditions["season"] == season) &
                (
                    ((conditions["home_team"] == home_team) & (conditions["away_team"] == away_team)) |
                    ((conditions["home_team"] == away_team) & (conditions["away_team"] == home_team))
                )
            ]
            if not cond.empty:
                cond = cond.sort_values("week", ascending=False).head(1)
        if not cond.empty:
            c = cond.iloc[0]
            home_adv_pts = float(c.get("total_home_advantage_pts", 1.5))
            pass_mult    = float(c.get("passing_mult", 1.0))
            rush_mult    = float(c.get("rushing_mult", 1.0))
            scoring_mult = float(c.get("scoring_mult", 1.0))
            cond_notes   = str(c.get("conditions_notes", ""))
        else:
            # Apply home field default even without weather data
            from engine.conditions import HOME_FIELD_PTS
            home_adv_pts = HOME_FIELD_PTS.get(home_team, HOME_FIELD_PTS["DEFAULT"])
            cond_notes   = f"Home field ({home_team}): +{home_adv_pts:.1f}pt (no weather data)"

    # ── Rest differential (NEW) ────────────────────────────────────
    # Short week (≤4 days rest) = −1.5 pts. Extra rest (≥10 days) = +1.0 pts
    home_rest_pts = 0.0
    away_rest_pts = 0.0
    if schedules is not None:
        game_row = schedules[
            (schedules["home_team"] == home_team) &
            (schedules["away_team"] == away_team) &
            (schedules["season"] == season) &
            (schedules["week"] == week)
        ]
        if not game_row.empty:
            g = game_row.iloc[0]
            try:
                home_rest = float(g.get("home_rest") or np.nan)
                if not np.isnan(home_rest) and 1 <= home_rest <= 21:
                    if home_rest <= 4:   home_rest_pts = -1.5
                    elif home_rest >= 10: home_rest_pts = 1.0
            except (TypeError, ValueError):
                pass
            try:
                away_rest = float(g.get("away_rest") or np.nan)
                if not np.isnan(away_rest) and 1 <= away_rest <= 21:
                    if away_rest <= 4:   away_rest_pts = -1.5
                    elif away_rest >= 10: away_rest_pts = 1.0
            except (TypeError, ValueError):
                pass

    # ── Ref crew penalty tendency (NEW) ────────────────────────────
    # High-flag crews inflate totals by ~2-3 pts; low-flag crews suppress
    ref_total_adj = 0.0
    if officials is not None:
        crew = officials[
            (officials["season"] == season) &
            (officials["game_id"].str.contains(f"_{week}_", na=False))
        ]
        if crew.empty:
            # Try matching by home team + week
            all_games = schedules[(schedules["season"] == season) & (schedules["week"] == week)] \
                if schedules is not None else pd.DataFrame()
            if not all_games.empty:
                game_ids = all_games[
                    (all_games["home_team"] == home_team)
                ]["game_id"].values
                if len(game_ids):
                    crew = officials[officials["game_id"].isin(game_ids)]

        if not crew.empty and "ref_penalty_rate" in officials.columns:
            ref_rate = float(crew["ref_penalty_rate"].iloc[0] or 0.12)
            # League avg ~0.12 penalties/play; above avg inflates total
            ref_total_adj = np.clip((ref_rate - 0.12) * 40, -1.5, 2.0)

    # ── SCORE PREDICTION ───────────────────────────────────────────
    # Base: league average
    home_base = LEAGUE_AVG_PTS
    away_base = LEAGUE_AVG_PTS

    # Offensive quality adjustment (relative to league average of 50)
    # Each 10 composite points above average = ~0.9 pts
    # Calibrated from CPOE/RYOE signal strength vs noise in composite scores
    home_base += (home_q["off_quality"] - 50) * 0.09
    away_base += (away_q["off_quality"] - 50) * 0.09

    # Team EPA-based efficiency adjustment
    home_base += home_q["off_epa_pts"]
    away_base += away_q["off_epa_pts"]

    # Defensive adjustment — two signals now:
    # 1. EPA-based (continuous efficiency)
    # 2. def_quality_score (composite 0-100: turnovers + 3rd down stops + sack rate)
    # Elite defenses (def_quality > 65) should suppress scoring by 2-4 pts
    home_def_adj = ((home_q["def_quality"] - 50) / 50) * 3.5   # max ±3.5 pts
    away_def_adj = ((away_q["def_quality"] - 50) / 50) * 3.5

    away_base -= home_q["def_epa_pts"] + home_def_adj   # home D suppresses away score
    home_base -= away_q["def_epa_pts"] + away_def_adj   # away D suppresses home score

    # Matchup edge adjustments
    home_base += home_matchup_pts
    away_base += away_matchup_pts

    # Fix 3: QB backup penalty (applied if starter QB appears to be out)
    home_base += home_qb_penalty
    away_base += away_qb_penalty

    # Rest differential adjustment (NEW)
    home_base += home_rest_pts
    away_base += away_rest_pts

    # Ref crew total adjustment — applies equally to both teams (affects O/U not spread)
    home_base += ref_total_adj * 0.5
    away_base += ref_total_adj * 0.5

    # Kicker quality adjustment — only matters in close predicted games
    # Elite kicker: +0.8 pts expected value (more made FGs near end of range)
    # Poor kicker:  -1.2 pts expected value (missed FGs in tight games)
    pred_margin_pre = abs(home_base - away_base)
    if pred_margin_pre <= 7:   # close game — kicker matters
        if home_q.get("elite_kicker"): home_base += 0.8
        if away_q.get("elite_kicker"): away_base += 0.8
        if home_q.get("poor_kicker"):  home_base -= 1.2
        if away_q.get("poor_kicker"):  away_base -= 1.2

    # Apply home field advantage
    home_base += home_adv_pts

    # Apply weather/surface scoring multiplier
    home_base *= scoring_mult
    away_base *= scoring_mult

    # Apply pass/rush multipliers to score components
    # (teams that pass more are more affected by weather)
    home_pass_rate = home_q["pass_rate"]
    away_pass_rate = away_q["pass_rate"]

    home_base *= (home_pass_rate * pass_mult + (1 - home_pass_rate) * rush_mult)
    away_base *= (away_pass_rate * pass_mult + (1 - away_pass_rate) * rush_mult)

    # Final clip to realistic range
    home_score_pred = round(np.clip(home_base, 6, 55), 1)
    away_score_pred = round(np.clip(away_base, 6, 55), 1)

    total_pred  = round(home_score_pred + away_score_pred, 1)
    spread_pred = round(away_score_pred - home_score_pred, 1)  # negative = home favored

    # ── WIN PROBABILITY ────────────────────────────────────────────
    # Sigmoid on predicted margin
    margin = home_score_pred - away_score_pred
    sigma  = 13.5   # empirical std of NFL game margins
    model_win_prob = float(1 / (1 + np.exp(-margin / sigma * np.pi / np.sqrt(3))))

    # Blend with moneyline implied probability (NEW) — 75% model, 25% market
    # Only activates when a valid moneyline is available for this specific game
    if schedules is not None:
        game_row = schedules[
            (schedules["home_team"] == home_team) &
            (schedules["away_team"] == away_team) &
            (schedules["season"] == season) &
            (schedules["week"] == week)
        ]
        if not game_row.empty:
            home_ml = game_row.iloc[0].get("home_moneyline")
            try:
                ml = float(home_ml)
                # Only blend if moneyline is a meaningful value (not 0, not extreme)
                if not np.isnan(ml) and abs(ml) >= 100 and abs(ml) <= 2500:
                    if ml > 0:
                        implied = 100 / (ml + 100)
                    else:
                        implied = (-ml) / (-ml + 100)
                    implied = float(np.clip(implied, 0.05, 0.95))
                    # Blend: 75% model, 25% market
                    model_win_prob = 0.75 * model_win_prob + 0.25 * implied
            except (TypeError, ValueError):
                pass  # No valid moneyline — use model only

    home_win_prob = round(model_win_prob, 3)

    # ── PREDICTED PLAY STYLE ───────────────────────────────────────
    home_pred_pass = round(np.clip(home_pass_rate * pass_mult, 0.35, 0.80), 3)
    away_pred_pass = round(np.clip(away_pass_rate * pass_mult, 0.35, 0.80), 3)

    # Game script prediction
    if total_pred >= 52:
        game_script = "Shootout"
    elif total_pred <= 38:
        game_script = "Defensive Battle"
    elif home_q["off_label"] in ["Run Heavy"] and away_q["off_label"] in ["Run Heavy"]:
        game_script = "Ground and Pound"
    elif abs(margin) >= 10:
        game_script = "Expected Blowout"
    else:
        game_script = "Normal / Competitive"

    # ── FLOOR / CEILING (variance range) ──────────────────────────
    base_std = 9.5
    if game_script == "Shootout":
        score_std = base_std * 1.25
    elif game_script == "Defensive Battle":
        score_std = base_std * 0.80
    else:
        score_std = base_std

    home_floor    = round(max(0, home_score_pred - score_std), 1)
    home_ceiling  = round(home_score_pred + score_std, 1)
    away_floor    = round(max(0, away_score_pred - score_std), 1)
    away_ceiling  = round(away_score_pred + score_std, 1)
    total_floor   = round(home_floor  + away_floor,    1)
    total_ceiling = round(home_ceiling + away_ceiling,  1)

    # ── KEY MATCHUPS ───────────────────────────────────────────────
    key_matchups = []
    if not matchups.empty:
        km = matchups[matchups["key_matchup"]].sort_values(
            "matchup_edge", key=abs, ascending=False
        ).head(3)
        for _, m in km.iterrows():
            key_matchups.append({
                "slot":      m["matchup_slot"],
                "off_team":  m["off_team"],
                "off_player": m["off_player"],
                "def_player": m["def_player"],
                "edge":      m["matchup_edge"],
                "grade":     m["matchup_grade"],
            })

    # ── STYLE CLASH SUMMARY ────────────────────────────────────────
    style_clash_summary = ""
    if styles is not None:
        home_s = styles[(styles["team"] == home_team) & (styles["season"] == season)]
        away_s = styles[(styles["team"] == away_team) & (styles["season"] == season)]
        if not home_s.empty and not away_s.empty:
            from engine.styles import get_style_clash_score
            # Away offense vs home defense
            away_vs_home_def = get_style_clash_score(away_s.iloc[0], home_s.iloc[0])
            home_vs_away_def = get_style_clash_score(home_s.iloc[0], away_s.iloc[0])
            all_clashes = {**away_vs_home_def, **home_vs_away_def}
            if all_clashes:
                style_clash_summary = " | ".join(f"{k}: {v:+d}" for k, v in all_clashes.items())

    return {
        # Scores
        "home_team":            home_team,
        "away_team":            away_team,
        "season":               season,
        "week":                 week,
        "predicted_home_score": home_score_pred,
        "predicted_away_score": away_score_pred,
        "predicted_total":      total_pred,
        "predicted_spread":     spread_pred,
        "home_win_probability": home_win_prob,
        "away_win_probability": round(1 - home_win_prob, 3),

        # Floor / ceiling
        "home_floor":           home_floor,
        "home_ceiling":         home_ceiling,
        "away_floor":           away_floor,
        "away_ceiling":         away_ceiling,
        "total_floor":          total_floor,
        "total_ceiling":        total_ceiling,

        # Game script
        "game_script":          game_script,
        "home_pred_pass_rate":  home_pred_pass,
        "away_pred_pass_rate":  away_pred_pass,
        "home_pred_rush_rate":  round(1 - home_pred_pass, 3),
        "away_pred_rush_rate":  round(1 - away_pred_pass, 3),

        # Team quality
        "home_off_quality":     home_q["off_quality"],
        "away_off_quality":     away_q["off_quality"],
        "home_def_quality":     home_q["def_quality"],
        "away_def_quality":     away_q["def_quality"],
        "home_qb_score":        home_q["qb_score"],
        "away_qb_score":        away_q["qb_score"],
        "home_off_label":       home_q["off_label"],
        "away_off_label":       away_q["off_label"],
        "home_def_label":       home_q["def_label"],
        "away_def_label":       away_q["def_label"],

        # Matchups
        "home_matchup_edge":    round(home_matchup_edge, 1),
        "away_matchup_edge":    round(away_matchup_edge, 1),
        "key_matchups":         key_matchups,

        # Conditions
        "home_advantage_pts":   round(home_adv_pts, 1),
        "passing_mult":         pass_mult,
        "scoring_mult":         scoring_mult,
        "conditions_notes":     cond_notes,
        "style_clash_summary":  style_clash_summary,

        # QB status (Fix 3)
        "home_qb_name":         home_qb_info.get("starter_name"),
        "away_qb_name":         away_qb_info.get("starter_name"),
        "home_qb_penalty":      home_qb_penalty,
        "away_qb_penalty":      away_qb_penalty,
        "home_qb_is_backup":    home_qb_info.get("is_backup", False),
        "away_qb_is_backup":    away_qb_info.get("is_backup", False),
    }


def predict_week(season: int, week: int) -> pd.DataFrame:
    """Predict all games for a given week."""
    data = load_engine_data()
    schedules = data["schedules"]

    games = schedules[
        (schedules["season"] == season) &
        (schedules["week"] == week)
    ]

    results = []
    for _, game in games.iterrows():
        try:
            pred = predict_game(
                game["home_team"], game["away_team"],
                season, week, data=data
            )
            results.append(pred)
            print(f"  {game['away_team']} @ {game['home_team']} -> "
                  f"{pred['predicted_away_score']:.1f}-{pred['predicted_home_score']:.1f} "
                  f"(home win: {pred['home_win_probability']:.1%})")
        except Exception as e:
            print(f"  ERROR: {game['away_team']} @ {game['home_team']}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        out_path = PROC / f"predictions_{season}_wk{week:02d}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\nSaved -> {out_path}")
    return df


def print_game_report(pred: dict):
    """Pretty-print a full game prediction report."""
    h, a = pred["home_team"], pred["away_team"]
    print(f"\n{'='*60}")
    print(f"  {a} @ {h}  |  Season {pred['season']} Week {pred['week']}")
    print(f"{'='*60}")
    print(f"\n  PREDICTED SCORE:  {a} {pred['predicted_away_score']} - {pred['predicted_home_score']} {h}")
    print(f"  Range:            {a} {pred['away_floor']}-{pred['away_ceiling']}  |  {h} {pred['home_floor']}-{pred['home_ceiling']}")
    print(f"  Predicted total:  {pred['predicted_total']}  (range: {pred['total_floor']}-{pred['total_ceiling']})")
    print(f"  Predicted spread: {pred['predicted_spread']:+.1f} (home perspective)")
    print(f"  Win probability:  {h} {pred['home_win_probability']:.1%}  |  {a} {pred['away_win_probability']:.1%}")

    print(f"\n  GAME SCRIPT:      {pred['game_script']}")
    hqb = pred.get('home_qb_name','—') or '—'
    aqb = pred.get('away_qb_name','—') or '—'
    hbk = ' ⚠ BACKUP' if pred.get('home_qb_is_backup') else ''
    abk = ' ⚠ BACKUP' if pred.get('away_qb_is_backup') else ''
    print(f"  {h} offense:       {pred['home_off_label']}  |  QB: {hqb}{hbk}  |  {pred['home_pred_pass_rate']:.1%} pass")
    print(f"  {a} offense:       {pred['away_off_label']}  |  QB: {aqb}{abk}  |  {pred['away_pred_pass_rate']:.1%} pass")
    print(f"  {h} defense:       {pred['home_def_label']}")
    print(f"  {a} defense:       {pred['away_def_label']}")

    print(f"\n  TEAM QUALITY:")
    print(f"  {h}: OFF {pred['home_off_quality']:.0f}/100  DEF {pred['home_def_quality']:.0f}/100  QB {pred['home_qb_score']:.0f}")
    print(f"  {a}: OFF {pred['away_off_quality']:.0f}/100  DEF {pred['away_def_quality']:.0f}/100  QB {pred['away_qb_score']:.0f}")

    if pred["key_matchups"]:
        print(f"\n  KEY MATCHUPS:")
        for km in pred["key_matchups"]:
            print(f"    [{km['slot']}] {km['off_team']} {km['off_player']} vs {km['def_player']}  "
                  f"edge: {km['edge']:+.1f}  ({km['grade']})")

    print(f"\n  CONDITIONS:  {pred['home_advantage_pts']:+.1f}pt home advantage")
    if pred["conditions_notes"]:
        for note in pred["conditions_notes"].split(" | ")[:3]:
            print(f"    {note}")

    if pred["style_clash_summary"]:
        print(f"\n  STYLE CLASHES:  {pred['style_clash_summary']}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--home",   default=None)
    parser.add_argument("--away",   default=None)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--week",   type=int, default=18)
    args = parser.parse_args()

    if args.home and args.away:
        data = load_engine_data()
        pred = predict_game(args.home, args.away, args.season, args.week, data=data)
        print_game_report(pred)
    else:
        print(f"Predicting all Week {args.week} {args.season} games...")
        df = predict_week(args.season, args.week)
        if not df.empty:
            print(f"\nWeek {args.week} Summary:")
            print(df[["away_team", "home_team", "predicted_away_score",
                       "predicted_home_score", "predicted_total",
                       "predicted_spread", "home_win_probability",
                       "game_script"]].to_string(index=False))
