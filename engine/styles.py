"""
engine/styles.py
----------------
Profiles each team's offensive and defensive playing style
from play-by-play data.

Offensive tags derived from PBP:
  - pass_rate_overall       : % of plays that are passes
  - pass_rate_early_down    : pass% on 1st/2nd down
  - pass_rate_run_situations: pass% when run expected (short yardage, lead)
  - air_yards_per_att       : avg depth of throw (deep vs short passing team)
  - rz_pass_rate            : red zone pass rate
  - third_down_pass_rate    : 3rd down pass%
  - scramble_rate           : QB scramble frequency
  - pace                    : plays per game (fast vs slow)
  - epa_per_pass            : passing efficiency
  - epa_per_rush            : rushing efficiency

Defensive tags:
  - pass_epa_allowed        : EPA allowed per dropback
  - rush_epa_allowed        : EPA allowed per rush
  - sack_rate               : sacks per dropback
  - stuff_rate              : TFL per rush attempt
  - coverage_grade          : pass_success_rate_allowed (lower = better)
  - third_down_stop_rate    : % of opp 3rd downs stopped

Style cluster labels (assigned after scoring):
  Offense: "Air Raid", "West Coast", "Run Heavy", "Balanced", "RPO"
  Defense: "Aggressive Blitz", "Cover 2 Zone", "Man Coverage", "Run Stopper", "Bend Dont Break"
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load_pbp(seasons: list) -> pd.DataFrame:
    """Load and concatenate PBP parquets for given seasons."""
    frames = []
    for s in seasons:
        p = RAW / f"pbp_{s}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if len(df) > 0:
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No PBP parquets found in {RAW}")
    return pd.concat(frames, ignore_index=True)


def build_team_styles(seasons: list = None, n_games: int = None) -> pd.DataFrame:
    """
    Compute per-team offensive and defensive style metrics.

    seasons: list of seasons to include (default: all available)
    n_games: if set, only use last N games per team (rolling window)

    Returns one row per (team, season) with style metrics + labels.
    """
    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024]

    print(f"Loading PBP for seasons {seasons}...")
    pbp = load_pbp(seasons)

    # Keep only scrimmage plays
    pbp = pbp[pbp["play_type"].isin(["pass", "run", "qb_spike", "qb_kneel"])].copy()
    pbp = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()]

    # Identify run-heavy situations (short yardage or team winning by 7+)
    pbp["run_situation"] = (
        ((pbp["down"].isin([1, 2])) & (pbp["ydstogo"] <= 3)) |
        (pbp["score_differential"] >= 7)
    ).astype(int)

    pbp["red_zone"] = (pbp["yardline_100"] <= 20).astype(int)
    pbp["is_pass"]  = pbp["pass_attempt"].fillna(0).astype(int)
    pbp["is_run"]   = pbp["rush_attempt"].fillna(0).astype(int)
    pbp["is_scramble"] = pbp["qb_scramble"].fillna(0).astype(int)
    pbp["early_down"]  = pbp["down"].isin([1, 2]).astype(int)
    pbp["third_down"]  = (pbp["down"] == 3).astype(int)

    # ── OFFENSIVE METRICS ──────────────────────────────────────────
    print("Computing offensive style metrics...")

    def safe_mean(x): return x.mean() if len(x) > 0 else np.nan

    off = pbp.groupby(["season", "posteam"]).apply(lambda g: pd.Series({
        # Volume / pass rate
        "plays_total":          len(g),
        "pass_rate_overall":    safe_mean(g["is_pass"]),
        "pass_rate_early_down": safe_mean(g.loc[g["early_down"] == 1, "is_pass"]),
        "pass_rate_run_sit":    safe_mean(g.loc[g["run_situation"] == 1, "is_pass"]),
        "rz_pass_rate":         safe_mean(g.loc[g["red_zone"] == 1, "is_pass"]),
        "third_down_pass_rate": safe_mean(g.loc[g["third_down"] == 1, "is_pass"]),
        "scramble_rate":        safe_mean(g["is_scramble"]),

        # Depth / style
        "avg_air_yards":        g.loc[g["is_pass"] == 1, "air_yards"].mean() if "air_yards" in g.columns else np.nan,
        "avg_yac":              g.loc[g["is_pass"] == 1, "yards_after_catch"].mean() if "yards_after_catch" in g.columns else np.nan,

        # Efficiency
        "off_epa_per_play":     safe_mean(g["epa"]) if "epa" in g.columns else np.nan,
        "off_epa_per_pass":     safe_mean(g.loc[g["is_pass"] == 1, "epa"]) if "epa" in g.columns else np.nan,
        "off_epa_per_rush":     safe_mean(g.loc[g["is_run"] == 1, "epa"]) if "epa" in g.columns else np.nan,
        "off_success_rate":     safe_mean(g["success"]) if "success" in g.columns else np.nan,

        # ── QB MOBILITY METRICS (NEW) ──────────────────────────────
        # Scramble EPA: how much value the QB generates on scrambles
        "qb_scramble_epa":      safe_mean(g.loc[g["is_scramble"] == 1, "epa"]) if "epa" in g.columns else np.nan,
        # QB rush volume: scrambles + designed runs (carries by passer_player_id)
        "qb_rush_rate":         safe_mean(g["is_scramble"]),  # scrambles per dropback
        # QB rush yards per scramble
        "qb_scramble_yds":      g.loc[g["is_scramble"] == 1, "yards_gained"].mean() if "yards_gained" in g.columns else np.nan,

        # Pace (plays per game estimate -- total / weeks played)
        "games_played":         g["game_id"].nunique(),
    }), include_groups=False).reset_index()

    off["pace"] = off["plays_total"] / off["games_played"].replace(0, np.nan)

    # ── DEFENSIVE METRICS ──────────────────────────────────────────
    print("Computing defensive style metrics...")

    defp = pbp.copy()
    defp["sack"]         = defp["sack"].fillna(0)
    defp["tfl"]          = defp.get("tackled_for_loss", pd.Series(0, index=defp.index)).fillna(0)
    defp["incomplete"]   = defp["incomplete_pass"].fillna(0)
    defp["interception"] = defp["interception"].fillna(0)
    defp["fumble_lost"]  = defp.get("fumble_lost", pd.Series(0, index=defp.index)).fillna(0)
    defp["touchdown"]    = defp["touchdown"].fillna(0)

    def_metrics = defp.groupby(["season", "defteam"]).apply(lambda g: pd.Series({
        # Core EPA quality
        "def_epa_per_play":      -safe_mean(g["epa"]) if "epa" in g.columns else np.nan,
        "def_epa_per_pass":      -safe_mean(g.loc[g["is_pass"] == 1, "epa"]) if "epa" in g.columns else np.nan,
        "def_epa_per_rush":      -safe_mean(g.loc[g["is_run"] == 1, "epa"]) if "epa" in g.columns else np.nan,

        # Pass rush
        "sack_rate":             safe_mean(g.loc[g["is_pass"] == 1, "sack"]),

        # Run defense
        "stuff_rate":            safe_mean(g.loc[g["is_run"] == 1, "tfl"]),
        "rush_yards_allowed_avg": g.loc[g["is_run"] == 1, "yards_gained"].mean() if "yards_gained" in g.columns else np.nan,

        # Turnover generation
        "int_rate":              safe_mean(g.loc[g["is_pass"] == 1, "interception"]),
        "fumble_rate":           safe_mean(g["fumble_lost"]),
        "turnover_rate":         safe_mean(g["interception"]) + safe_mean(g["fumble_lost"]),

        # Coverage / passing defense
        "def_success_rate":      1 - (safe_mean(g["success"]) if "success" in g.columns else 0.5),
        "def_completion_rate":   safe_mean(g.loc[g["is_pass"] == 1, "complete_pass"]) if "complete_pass" in g.columns else np.nan,
        "def_air_yards_allowed": g.loc[g["is_pass"] == 1, "air_yards"].mean() if "air_yards" in g.columns else np.nan,

        # Situational
        "third_down_stop_rate":  1 - safe_mean(g.loc[g["third_down"] == 1, "success"]) if "success" in g.columns else np.nan,
        "rz_td_rate_allowed":    safe_mean(g.loc[(g["red_zone"] == 1) & (g["touchdown"] == 1), "touchdown"]) if "red_zone" in g.columns else np.nan,

        # Points allowed proxy
        "def_points_allowed_avg": g.groupby("game_id")["defteam_score_post"].max().mean() if "defteam_score_post" in g.columns else np.nan,

        # How often offenses pass against this defense
        "def_pass_rate_faced":   safe_mean(g["is_pass"]),

        # ── QB MOBILITY CONTAINMENT (NEW) ──────────────────────────
        # How much EPA this defense allows on QB scrambles/runs
        # Lower (more negative from offense perspective) = better at containing mobile QBs
        "def_qb_scramble_epa_allowed": safe_mean(g.loc[g["is_scramble"] == 1, "epa"]) if "epa" in g.columns else np.nan,
        # Scrambles allowed per dropback (lower = better at keeping QB in pocket)
        "def_scramble_rate_allowed":   safe_mean(g["is_scramble"]),
        # Yards allowed per QB scramble (lower = better)
        "def_scramble_yds_allowed":    g.loc[g["is_scramble"] == 1, "yards_gained"].mean() if "yards_gained" in g.columns else np.nan,
    }), include_groups=False).reset_index()

    # ── COMPOSITE DEFENSIVE QUALITY SCORE (0-100) ──────────────────
    # Combine the best defensive signals into one number per team
    # Higher = better defense (stops opponents)
    def compute_def_quality(df_row):
        score = 50.0
        w_total = 0.0

        signals = {
            "def_epa_per_play":    (df_row.get("def_epa_per_play",  0), 30),
            "def_epa_per_pass":    (df_row.get("def_epa_per_pass",  0), 20),
            "def_epa_per_rush":    (df_row.get("def_epa_per_rush",  0), 15),
            "sack_rate":           (df_row.get("sack_rate",       0.06), 10),
            "turnover_rate":       (df_row.get("turnover_rate",   0.03), 15),
            "third_down_stop_rate":(df_row.get("third_down_stop_rate", 0.6), 10),
        }
        # Each signal: convert to a 0-100 contribution
        # EPA signals: league avg ~0, good defense ~+0.05 to +0.15
        epa_play  = float(signals["def_epa_per_play"][0]  or 0)
        epa_pass  = float(signals["def_epa_per_pass"][0]  or 0)
        epa_rush  = float(signals["def_epa_per_rush"][0]  or 0)
        sack_r    = float(signals["sack_rate"][0]          or 0.06)
        to_rate   = float(signals["turnover_rate"][0]      or 0.03)
        stop3     = float(signals["third_down_stop_rate"][0] or 0.6)

        # Scale each to approximate 0-100
        epa_score  = np.clip(50 + epa_play  * 300, 0, 100)
        pass_score = np.clip(50 + epa_pass  * 200, 0, 100)
        rush_score = np.clip(50 + epa_rush  * 200, 0, 100)
        sack_score = np.clip(50 + (sack_r - 0.06) * 1000, 0, 100)
        to_score   = np.clip(50 + (to_rate  - 0.03) * 1500, 0, 100)
        stop_score = np.clip(stop3 * 100, 0, 100)

        return round(
            epa_score  * 0.30 +
            pass_score * 0.20 +
            rush_score * 0.15 +
            sack_score * 0.10 +
            to_score   * 0.15 +
            stop_score * 0.10,
            1
        )

    def_metrics["def_quality_score"] = def_metrics.apply(compute_def_quality, axis=1)

    # ── MERGE OFF + DEF ────────────────────────────────────────────
    styles = off.merge(
        def_metrics.rename(columns={"defteam": "posteam"}),
        on=["season", "posteam"], how="outer"
    )
    styles = styles.rename(columns={"posteam": "team"})

    # ── MERGE SITUATIONAL STATS ────────────────────────────────────
    sit_path = RAW / "situational_stats.parquet"
    if sit_path.exists():
        sit = pd.read_parquet(sit_path)
        sit = sit[sit["season"].isin(seasons)]
        styles = styles.merge(
            sit[["season", "team",
                 "pressure_rate_allowed", "sack_rate_allowed",
                 "pressure_rate_gen", "sack_rate_gen",
                 "rz_td_rate", "rz_epa", "rz_success",
                 "rz_td_rate_allowed", "rz_epa_allowed",
                 "two_min_epa", "two_min_success",
                 "fourth_go_rate"]],
            on=["season", "team"], how="left"
        )
        print("Merged situational stats (pressure, red zone, 2-min, 4th down)")
    else:
        print("WARNING: situational_stats.parquet not found -- run fetch_data.py first")

    # ── MERGE FTN DATA (NEW) ───────────────────────────────────────
    # play_action rate, real blitz counts, motion rate, screen rate
    ftn_path = RAW / "ftn_data.parquet"
    if ftn_path.exists():
        ftn = pd.read_parquet(ftn_path)
        ftn = ftn[ftn["season"].isin(seasons)].copy() if "season" in ftn.columns else ftn.copy()

        # Join FTN to PBP via nflverse_game_id + play_id
        ftn_cols_needed = ["nflverse_game_id","season","week",
                           "is_play_action","is_motion","is_screen_pass",
                           "is_no_huddle","n_blitzers","n_pass_rushers",
                           "is_qb_out_of_pocket","is_drop","is_qb_fault_sack"]
        ftn_avail = [c for c in ftn_cols_needed if c in ftn.columns]

        if len(ftn_avail) >= 4 and "nflverse_game_id" in ftn.columns:
            # Merge FTN onto PBP by game_id + play_id
            pbp_ftn = pbp.copy()
            pbp_ftn = pbp_ftn.merge(
                ftn[ftn_avail].rename(columns={"nflverse_game_id": "game_id"}),
                on=["game_id"] + [c for c in ["season","week"] if c in ftn_avail],
                how="left"
            )

            # Team-level FTN aggregations (offensive)
            ftn_off_metrics = {}
            for col in ["is_play_action","is_motion","is_screen_pass","is_no_huddle","is_qb_out_of_pocket"]:
                if col in pbp_ftn.columns:
                    ftn_off_metrics[col] = (col, "mean")

            if ftn_off_metrics:
                pass_plays = pbp_ftn[pbp_ftn["pass_attempt"] == 1]
                ftn_off = pass_plays.groupby(["season","posteam"]).agg(**ftn_off_metrics).reset_index()
                ftn_off = ftn_off.rename(columns={
                    "posteam":          "team",
                    "is_play_action":   "play_action_rate",
                    "is_motion":        "motion_rate",
                    "is_screen_pass":   "screen_pass_rate",
                    "is_no_huddle":     "no_huddle_rate",
                    "is_qb_out_of_pocket": "qb_out_pocket_rate",
                })
                styles = styles.merge(ftn_off, on=["season","team"], how="left")

            # Team-level FTN aggregations (defensive — blitz counts)
            if "n_blitzers" in pbp_ftn.columns:
                ftn_def = pass_plays.groupby(["season","defteam"]).agg(
                    avg_blitzers      = ("n_blitzers",       "mean"),
                    blitz_rate        = ("n_blitzers",       lambda x: (x >= 5).mean()),
                    avg_pass_rushers  = ("n_pass_rushers",   "mean") if "n_pass_rushers" in pass_plays.columns else ("n_blitzers","count"),
                    qb_fault_sack_rate= ("is_qb_fault_sack","mean") if "is_qb_fault_sack" in pass_plays.columns else ("n_blitzers","count"),
                ).reset_index().rename(columns={"defteam": "team"})
                styles = styles.merge(ftn_def, on=["season","team"], how="left")
                print("Merged FTN data (play-action, motion, blitz rates)")
    else:
        print("  SKIP ftn_data.parquet (not found)")

    # ── STYLE LABELS ───────────────────────────────────────────────
    styles = assign_offense_label(styles)
    styles = assign_defense_label(styles)
    styles = assign_matchup_tags(styles)

    print(f"Team styles built: {len(styles)} team-seasons")
    out_path = PROC / "team_styles.parquet"
    styles.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    return styles


def _season_z(df: pd.DataFrame, col: str) -> pd.Series:
    """Within-season z-score of a column (0 where the column is missing or has no variance).
    Labels must be RELATIVE to the league that season — absolute cutoffs collapse everyone into
    one bucket because most teams cluster in the middle on any single metric."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    x = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return x.groupby(df["season"]).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))


def assign_offense_label(df: pd.DataFrame) -> pd.DataFrame:
    """Offensive archetype from where a team sits RELATIVE to the league that season. Each team
    is scored on a few recognizable identities and gets the one it leans into most; teams that
    are genuinely middle-of-the-pack on all of them stay 'Balanced'. (The old version used fixed
    thresholds like pace>=68 / pass_rate>=0.62 that almost no team crosses, so ~27/32 collapsed
    to 'Balanced'.)"""
    zp   = _season_z(df, "pass_rate_overall")   # pass volume
    za   = _season_z(df, "avg_air_yards")       # downfield depth
    zpace = _season_z(df, "pace")               # tempo
    znh  = _season_z(df, "no_huddle_rate")
    zm   = _season_z(df, "motion_rate")
    zsc  = _season_z(df, "screen_pass_rate")
    rl   = (pd.to_numeric(df.get("off_epa_per_rush"), errors="coerce").fillna(0.0)
            - pd.to_numeric(df.get("off_epa_per_pass"), errors="coerce").fillna(0.0))
    zrl  = rl.groupby(df["season"]).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    aff = pd.DataFrame({
        "Air Raid":     0.7 * zp + 0.9 * za,               # pass a lot, throw deep
        "West Coast":   0.8 * zp - 0.7 * za + 0.3 * zsc,   # high-volume short passing + screens
        "Run Heavy":   -0.9 * zp + 0.5 * zrl,              # run-leaning identity
        "RPO / Spread": 0.9 * zpace + 0.6 * znh + 0.3 * zm,# up-tempo, no-huddle, motion
    }, index=df.index)
    best, peak = aff.idxmax(axis=1), aff.max(axis=1)
    df["offense_label"] = best.where(peak >= 0.55, "Balanced")
    return df


def assign_defense_label(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive archetype from within-season z-scores of the signals that ACTUALLY vary in the
    built data (sack rate, blitz volume, pass-EPA-allowed, rush-EPA-allowed, third-down stop).
    The previous version keyed on pressure_rate_gen / stuff_rate / rz_td_rate_allowed — all of
    which are zero or absent in team_styles — so every team fell through to 'Bend Don't Break'."""
    zsk = _season_z(df, "sack_rate")
    zbl = _season_z(df, "avg_blitzers")
    zpd = -_season_z(df, "def_epa_per_pass")     # lower EPA allowed = better → higher z
    zrd = -_season_z(df, "def_epa_per_rush")
    z3  = _season_z(df, "third_down_stop_rate")
    aff = pd.DataFrame({
        "Aggressive Blitz": 0.7 * zsk + 0.6 * zbl,          # brings pressure, with volume
        "Man Coverage":     0.9 * zpd - 0.3 * zbl,          # locks up the pass without blitzing
        "Run Stopper":      0.9 * zrd,                      # wins the run front
        "Cover 2 Zone":     0.7 * z3 + 0.3 * zpd - 0.3 * zbl,  # bend-zone: gets off on third down
    }, index=df.index)
    best, peak = aff.idxmax(axis=1), aff.max(axis=1)
    df["defense_label"] = best.where(peak >= 0.50, "Bend Don't Break")
    return df


def assign_matchup_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean tags used by the matchup engine, now including situational signals."""
    df["run_heavy_off"]    = df["pass_rate_overall"] <= 0.50
    df["pass_heavy_off"]   = df["pass_rate_overall"] >= 0.58
    df["deep_pass_off"]    = df["avg_air_yards"].fillna(7) >= 9.0
    df["short_pass_off"]   = df["avg_air_yards"].fillna(7) <= 6.5
    df["fast_pace"]        = df["pace"].fillna(65) >= 70
    df["slow_pace"]        = df["pace"].fillna(65) <= 60

    # Use real FTN blitz rate if available, fall back to sack rate proxy
    if "blitz_rate" in df.columns and df["blitz_rate"].notna().any():
        df["blitz_heavy_def"] = df["blitz_rate"].fillna(0) >= 0.35
    else:
        df["blitz_heavy_def"] = df["sack_rate"].fillna(0) >= 0.09

    # Play-action offense flag (NEW) — PA rate >= 28% = high usage
    df["high_play_action"] = df.get("play_action_rate", pd.Series(0.20, index=df.index)).fillna(0.20) >= 0.28

    # Motion-heavy offense (NEW) — motion rate >= 55%
    df["motion_heavy_off"] = df.get("motion_rate", pd.Series(0.40, index=df.index)).fillna(0.40) >= 0.55

    # ── QB MOBILITY FLAGS (NEW) ────────────────────────────────────
    # mobile_qb: scramble rate >= 4% of dropbacks (league avg ~2.5%)
    # or avg qb_scramble_yds > 7.0 (QB gains big chunks when scrambling)
    scramble_rate = df.get("scramble_rate", pd.Series(0.025, index=df.index)).fillna(0.025)
    scramble_yds  = df.get("qb_scramble_yds", pd.Series(6.0, index=df.index)).fillna(6.0)
    scramble_epa  = df.get("qb_scramble_epa", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["mobile_qb_offense"] = (scramble_rate >= 0.04) | (scramble_epa >= 0.15)

    # elite_mobile_qb: truly elite mobility (Lamar tier) — top ~5 teams
    df["elite_mobile_qb"] = (scramble_rate >= 0.06) | (scramble_epa >= 0.25)

    # ── DEFENSIVE QB-CONTAIN FLAGS (NEW) ──────────────────────────
    # poor_qb_contain: defense allows a lot of EPA on QB scrambles
    # = scramble EPA allowed is high (bad at containing mobile QBs)
    # League avg scramble EPA ~0.18 (scrambles are generally positive plays)
    # Good contain: < 0.12 | Average: 0.12-0.22 | Poor: > 0.22
    def_scramble_epa = df.get("def_qb_scramble_epa_allowed",
                              pd.Series(0.18, index=df.index)).fillna(0.18)
    def_scramble_yds = df.get("def_scramble_yds_allowed",
                              pd.Series(7.5, index=df.index)).fillna(7.5)
    df["poor_qb_contain"]  = (def_scramble_epa >= 0.22) | (def_scramble_yds >= 9.0)
    df["elite_qb_contain"] = (def_scramble_epa <= 0.12) & (def_scramble_yds <= 6.5)

    def safe_tag(col, threshold, op="gte", default=False):
        if col not in df.columns:
            return pd.Series(default, index=df.index)
        s = df[col].fillna(threshold)
        return s >= threshold if op == "gte" else s <= threshold

    df["fourth_down_aggressive"] = safe_tag("fourth_go_rate",        0.35)
    df["elite_rz_offense"]       = safe_tag("rz_td_rate",            0.65)
    df["elite_rz_defense"]       = safe_tag("rz_td_rate_allowed",    0.45, op="lte")
    df["elite_2min"]             = safe_tag("two_min_epa",           0.08)
    df["leaky_under_pressure"]   = safe_tag("pressure_rate_allowed", 0.35)
    return df


def get_style_clash_score(off_team_style: pd.Series,
                           def_team_style: pd.Series) -> dict:
    """
    Given one offensive team's style row and one defensive team's style row,
    compute clash scores that predict how each style will be amplified or suppressed.

    Returns dict of modifier values (positive = offense benefits, negative = offense hurt).
    """
    clashes = {}

    # ── PASS RATE context ─────────────────────────────────────────
    off_pass_rate = float(off_team_style.get("pass_rate_overall", 0.565) or 0.565)
    is_run_heavy  = off_pass_rate <= 0.50
    is_pass_heavy = off_pass_rate >= 0.58

    # Run heavy offense vs strong run defense = bad for offense
    if off_team_style.get("run_heavy_off") and def_team_style.get("strong_run_def"):
        clashes["run_clash"] = -15
    elif off_team_style.get("run_heavy_off") and not def_team_style.get("strong_run_def"):
        clashes["run_clash"] = +8

    # Pass heavy offense vs strong pass defense
    if off_team_style.get("pass_heavy_off") and def_team_style.get("strong_pass_def"):
        clashes["pass_clash"] = -12
    elif off_team_style.get("pass_heavy_off") and not def_team_style.get("strong_pass_def"):
        clashes["pass_clash"] = +10

    # Deep pass offense vs blitz = dangerous (good and bad)
    if off_team_style.get("deep_pass_off") and def_team_style.get("blitz_heavy_def"):
        clashes["blitz_deep_clash"] = +5

    # Short pass offense vs blitz = bad for offense
    if off_team_style.get("short_pass_off") and def_team_style.get("blitz_heavy_def"):
        clashes["blitz_short_clash"] = -8

    # Pace mismatch
    if off_team_style.get("fast_pace") and def_team_style.get("defense_label") in ["Cover 2 Zone"]:
        clashes["pace_clash"] = +6

    # Run heavy offense vs run stopper defense
    if off_team_style.get("run_heavy_off") and def_team_style.get("defense_label") == "Run Stopper":
        clashes["style_identity_clash"] = -20

    # Elite red zone offense vs elite red zone defense
    if off_team_style.get("elite_rz_offense") and def_team_style.get("elite_rz_defense"):
        clashes["rz_clash"] = -8
    elif off_team_style.get("elite_rz_offense") and not def_team_style.get("elite_rz_defense"):
        clashes["rz_advantage"] = +6

    # Leaky OL vs aggressive blitz — now SCALED by how pass-dependent the offense is
    # A run-heavy team with a leaky OL (rare but exists, e.g. BAL) is hurt less by blitz
    # A pass-heavy team with a leaky OL (e.g. NYJ 2025) is devastated by blitz
    if off_team_style.get("leaky_under_pressure") and def_team_style.get("blitz_heavy_def"):
        # Run heavy: -8  |  Balanced: -12  |  Pass heavy: -16
        pressure_penalty = -8 - (off_pass_rate - 0.50) / (0.62 - 0.50) * 8
        clashes["pressure_clash"] = round(np.clip(pressure_penalty, -18, -6), 1)

    # NEW: Run-heavy offense vs blitz — blitz creates gaps in run lanes
    # A blitz-heavy D sends extra rushers, reducing run-fit defenders in the box
    # This HELPS run-heavy offenses if their OL can handle the pressure
    if is_run_heavy and def_team_style.get("blitz_heavy_def"):
        if not off_team_style.get("leaky_under_pressure"):
            clashes["run_vs_blitz_edge"] = +7  # run exploits vacated boxes
        # If leaky OL: the pressure_clash above already penalises

    # NEW: Play-action offense vs man coverage defense
    # Man coverage = corners on islands = play-action freezes LBs = big gains
    if off_team_style.get("high_play_action") and def_team_style.get("defense_label") == "Man Coverage":
        clashes["pa_man_advantage"] = +8

    # NEW: Play-action offense vs blitz
    # Blitz = fewer coverage players = PA creates easy completions vs exposed secondary
    if off_team_style.get("high_play_action") and def_team_style.get("blitz_heavy_def"):
        clashes["pa_blitz_advantage"] = +6

    # NEW: Motion-heavy offense vs man coverage
    # Pre-snap motion reveals man vs zone and stresses man-coverage assignments
    if off_team_style.get("motion_heavy_off") and def_team_style.get("defense_label") == "Man Coverage":
        clashes["motion_man_advantage"] = +5

    # NEW: Elite 2-minute offense in close games
    if off_team_style.get("elite_2min"):
        clashes["2min_edge"] = +4

    # ── QB MOBILITY MATCHUP CLASHES (NEW) ─────────────────────────
    if off_team_style.get("mobile_qb_offense") and def_team_style.get("poor_qb_contain"):
        clashes["mobile_qb_advantage"] = +6

    if off_team_style.get("elite_mobile_qb") and def_team_style.get("poor_qb_contain"):
        clashes["elite_mobile_qb_advantage"] = +10

    if off_team_style.get("mobile_qb_offense") and def_team_style.get("elite_qb_contain"):
        clashes["mobile_qb_neutralised"] = -5

    if off_team_style.get("mobile_qb_offense") and def_team_style.get("blitz_heavy_def"):
        clashes["mobile_qb_vs_blitz"] = +5

    if (not off_team_style.get("mobile_qb_offense") and
            off_team_style.get("leaky_under_pressure") and
            def_team_style.get("blitz_heavy_def")):
        clashes["pocket_qb_blitz_trap"] = -4

    return clashes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024])
    args = parser.parse_args()
    df = build_team_styles(args.seasons)
    print("\nOffensive style breakdown:")
    print(df.groupby("offense_label")["team"].count().sort_values(ascending=False))
    print("\nDefensive style breakdown:")
    print(df.groupby("defense_label")["team"].count().sort_values(ascending=False))
    print("\nSample style row (KC 2024):")
    row = df[(df["team"] == "KC") & (df["season"] == 2024)]
    if not row.empty:
        print(row[["team","season","offense_label","defense_label",
                   "pass_rate_overall","avg_air_yards","sack_rate",
                   "def_epa_per_pass","def_epa_per_rush"]].to_string(index=False))
