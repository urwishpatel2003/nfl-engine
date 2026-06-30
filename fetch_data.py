"""
NFL Engine -- Data Fetcher
Run this LOCALLY (outside Claude sandbox) to pull all source data.

Usage:
    pip install nfl_data_py pandas pyarrow requests tqdm
    python fetch_data.py               # fetches 2020-2024
    python fetch_data.py --seasons 2022 2023 2024
    python fetch_data.py --seasons 2024 --force  # re-download even if cached

Outputs to ./data/raw/ as parquet files (~500MB total for 5 seasons).
After running, bring the data/raw/ folder back to this project.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

RAW = Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

LOG = []

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {level:5} {msg}"
    print(line)
    LOG.append(line)

def save(df: pd.DataFrame, name: str):
    path = RAW / f"{name}.parquet"
    df.to_parquet(path, index=False)
    mb = path.stat().st_size / 1024 / 1024
    log(f"  saved {name}.parquet  ({len(df):,} rows, {mb:.1f} MB)")
    return path

def already_have(name: str, force: bool) -> bool:
    path = RAW / f"{name}.parquet"
    if path.exists() and not force:
        mb = path.stat().st_size / 1024 / 1024
        log(f"  skip {name}.parquet (cached, {mb:.1f} MB) -- use --force to re-fetch")
        return True
    return False


# -----------------------------------------------------------------
# 1. SCHEDULES
# Columns we care about: game_id, season, week, gameday, gametime,
#   home_team, away_team, home_score, away_score, location,
#   roof, surface, temp, wind, stadium, stadium_id,
#   div_game, playoff, spread_line, total_line,
#   home_rest, away_rest, home_moneyline, away_moneyline
# -----------------------------------------------------------------
def fetch_schedules(seasons, force):
    name = "schedules"
    if already_have(name, force): return
    log("Fetching schedules ...")
    import nfl_data_py as nfl

    # Fetch each season individually and concat — import_schedules with a list
    # may only return the most recent season in some nfl_data_py versions
    frames = []
    for s in seasons:
        try:
            df = nfl.import_schedules([s])
            if len(df) > 0:
                frames.append(df)
                log(f"  season {s}: {len(df)} games")
        except Exception as e:
            log(f"  season {s} schedules failed: {e}", "WARN")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"])
        save(combined, name)
    else:
        log("  No schedule data retrieved", "WARN")


# -----------------------------------------------------------------
# 2. WEEKLY PLAYER STATS
# One row per player per game. Covers:
#   passing: attempts, completions, yards, tds, ints, air_yards, cpoe, epa
#   rushing: carries, yards, tds, yards_after_contact
#   receiving: targets, receptions, yards, tds, air_yards, yac, racr, target_share
#   misc: fantasy_points_ppr, snap counts (offense)
# -----------------------------------------------------------------
def fetch_player_stats(seasons, force):
    name = "player_stats"
    if already_have(name, force): return
    log("Fetching weekly player stats ...")
    import nfl_data_py as nfl

    # Try seasons one at a time — skip any that return 404
    # (nflverse publishes new season data with a ~4 week lag after season end)
    frames = []
    for s in seasons:
        try:
            df = nfl.import_weekly_data([s])
            if len(df) > 0:
                frames.append(df)
                log(f"  season {s}: {len(df):,} rows")
        except Exception as e:
            log(f"  season {s} not available yet ({e}), skipping", "WARN")

    if frames:
        import pandas as pd
        combined = pd.concat(frames, ignore_index=True)
        save(combined, name)
    else:
        log("  No player stats data available -- keeping cached version", "WARN")


def fetch_seasonal_stats(seasons, force):
    name = "seasonal_stats"
    if already_have(name, force): return
    log("Fetching seasonal player stats ...")
    import nfl_data_py as nfl

    frames = []
    for s in seasons:
        try:
            df = nfl.import_seasonal_data([s])
            if len(df) > 0:
                frames.append(df)
        except Exception as e:
            log(f"  season {s} seasonal stats not available ({e}), skipping", "WARN")

    if frames:
        import pandas as pd
        combined = pd.concat(frames, ignore_index=True)
        save(combined, name)
    else:
        log("  No seasonal stats available -- keeping cached version", "WARN")


# -----------------------------------------------------------------
# 4. ROSTERS (with player bio + draft info)
# player_id, player_name, position, depth_chart_position,
# jersey_number, status, birth_date, height, weight,
# college, draft_number, years_exp, headshot_url, team, season
# -----------------------------------------------------------------
def fetch_rosters(seasons, force):
    import nfl_data_py as nfl

    # Weekly rosters (player status + team per week)
    name = "rosters_weekly"
    if not already_have(name, force):
        log("Fetching weekly rosters ...")
        df = nfl.import_weekly_rosters(seasons)
        save(df, name)

    # Seasonal rosters (bio, draft info, measurables)
    name2 = "rosters_seasonal"
    if not already_have(name2, force):
        log("Fetching seasonal rosters ...")
        df2 = nfl.import_seasonal_rosters(seasons)
        save(df2, name2)


# -----------------------------------------------------------------
# 5. DEPTH CHARTS
# depth_team (1=starter), formation_position, gsis_id, season, week, team
# -----------------------------------------------------------------
def fetch_depth_charts(seasons, force):
    name = "depth_charts"
    if already_have(name, force): return
    log("Fetching depth charts ...")
    import nfl_data_py as nfl
    df = nfl.import_depth_charts(seasons)
    save(df, name)


# -----------------------------------------------------------------
# 6. SNAP COUNTS (% of offensive/defensive/ST snaps)
# Essential for usage rates and true role on team
# -----------------------------------------------------------------
def fetch_snap_counts(seasons, force):
    name = "snap_counts"
    if already_have(name, force): return
    log("Fetching snap counts ...")
    import nfl_data_py as nfl
    df = nfl.import_snap_counts(seasons)
    save(df, name)


# -----------------------------------------------------------------
# 7. INJURIES (weekly injury reports)
# practice_status, game_status per player per week
# -----------------------------------------------------------------
def fetch_injuries(seasons, force):
    name = "injuries"
    if already_have(name, force): return
    log("Fetching injuries ...")
    import nfl_data_py as nfl
    df = nfl.import_injuries(seasons)
    save(df, name)


# -----------------------------------------------------------------
# 8. NEXT GEN STATS -- PASSING
# completion_percentage_above_expectation (CPOE), avg_air_distance,
# max_air_distance, avg_time_to_throw, aggressiveness, passer_rating,
# expected_completion_percentage -- per QB per week
# -----------------------------------------------------------------
def fetch_ngs_passing(seasons, force):
    name = "ngs_passing"
    if already_have(name, force): return
    log("Fetching NGS passing ...")
    import nfl_data_py as nfl
    try:
        df = nfl.import_ngs_data("passing", seasons)
        save(df, name)
    except Exception as e:
        log(f"  NGS passing error: {e} -- trying season by season", "WARN")
        import pandas as pd
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.import_ngs_data("passing", [s]))
            except Exception:
                log(f"  NGS passing {s} not available, skipping", "WARN")
        if frames:
            save(pd.concat(frames, ignore_index=True), name)


def fetch_ngs_rushing(seasons, force):
    name = "ngs_rushing"
    if already_have(name, force): return
    log("Fetching NGS rushing ...")
    import nfl_data_py as nfl
    try:
        df = nfl.import_ngs_data("rushing", seasons)
        save(df, name)
    except Exception as e:
        import pandas as pd
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.import_ngs_data("rushing", [s]))
            except Exception:
                log(f"  NGS rushing {s} not available, skipping", "WARN")
        if frames:
            save(pd.concat(frames, ignore_index=True), name)


def fetch_ngs_receiving(seasons, force):
    name = "ngs_receiving"
    if already_have(name, force): return
    log("Fetching NGS receiving ...")
    import nfl_data_py as nfl
    try:
        df = nfl.import_ngs_data("receiving", seasons)
        save(df, name)
    except Exception as e:
        import pandas as pd
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.import_ngs_data("receiving", [s]))
            except Exception:
                log(f"  NGS receiving {s} not available, skipping", "WARN")
        if frames:
            save(pd.concat(frames, ignore_index=True), name)


PBP_COLUMNS = [
    # Core identifiers
    "game_id","play_id","season","week","home_team","away_team",
    "posteam","defteam","game_date","game_seconds_remaining",
    # Situation
    "down","ydstogo","yardline_100","score_differential",
    # Play type
    "play_type","pass_attempt","rush_attempt","qb_scramble",
    # Passing
    "air_yards","yards_after_catch","yards_gained",
    "complete_pass","incomplete_pass","interception","touchdown",
    "sack","penalty","penalty_yards","fumble",
    # Advanced metrics
    "epa","wpa","success","cpoe",
    # Player IDs (scrimmage)
    "passer_player_id","passer_player_name",
    "rusher_player_id","rusher_player_name",
    "receiver_player_id","receiver_player_name",
    "solo_tackle_1_player_id","solo_tackle_2_player_id",
    # Score
    "posteam_score","defteam_score",
    "posteam_score_post","defteam_score_post",
    # Game context
    "roof","surface","temp","wind","home_coach","away_coach",
    "season_type",
    # ── SPECIAL TEAMS ────────────────────────────────────────────
    "field_goal_attempt","field_goal_result","kick_distance",
    "extra_point_attempt","extra_point_result",
    "punt_attempt","punt_blocked",
    "kickoff_attempt","kickoff",
    "return_yards","return_touchdown",
    "kicker_player_id","kicker_player_name",
    "punter_player_id","punter_player_name",
    "kickoff_returner_player_id","kickoff_returner_player_name",
    "punt_returner_player_id","punt_returner_player_name",
]


def fetch_pbp(seasons, force):
    import nfl_data_py as nfl
    for season in seasons:
        name = f"pbp_{season}"
        if already_have(name, force): continue
        log(f"Fetching PBP {season} (large file, may take 30-60s) ...")
        try:
            df = nfl.import_pbp_data([season], downcast=True)
            want = [c for c in PBP_COLUMNS if c in df.columns]
            df = df[want]
            save(df, name)
        except Exception as e:
            log(f"  PBP {season} not available yet: {e}", "WARN")


# -----------------------------------------------------------------
# 12. COMBINE DATA (physical measurables)
# 40_time, bench_reps, broad_jump, cone, shuttle, vert_leap
# Useful for athleticism component of player composite score
# -----------------------------------------------------------------
def fetch_combine(force):
    name = "combine"
    if already_have(name, force): return
    log("Fetching combine data ...")
    import nfl_data_py as nfl
    df = nfl.import_combine_data()
    save(df, name)


# -----------------------------------------------------------------
# 13. OFFICIALS (referee tendencies affect game pace/penalties)
# -----------------------------------------------------------------
def fetch_officials(seasons, force):
    name = "officials"
    if already_have(name, force): return
    log("Fetching officials ...")
    import nfl_data_py as nfl
    try:
        df = nfl.import_officials(seasons)
        save(df, name)
    except Exception as e:
        log(f"  officials not available: {e}", "WARN")


# -----------------------------------------------------------------
# 14. TEAM DESC (abbreviations, full names, conference, division)
# -----------------------------------------------------------------
def fetch_team_info(force):
    name = "team_info"
    if already_have(name, force): return
    log("Fetching team descriptions ...")
    import nfl_data_py as nfl
    df = nfl.import_team_desc()
    save(df, name)


# -----------------------------------------------------------------
# 15. IDs -- cross-reference table (ESPN, PFR, Sleeper, etc.)
# -----------------------------------------------------------------
def fetch_ids(force):
    name = "player_ids"
    if already_have(name, force): return
    log("Fetching player ID crosswalk ...")
    import nfl_data_py as nfl
    df = nfl.import_ids()
    save(df, name)


# -----------------------------------------------------------------
# WEATHER -- Open-Meteo historical (free, no key)
# Fetched per stadium per game date from schedules
# -----------------------------------------------------------------
STADIUM_COORDS = {
    # team_abbr: (lat, lon, is_dome)
    "ARI": (33.5277, -112.2626, True),   # State Farm Stadium (dome)
    "ATL": (33.7554, -84.4010, True),    # Mercedes-Benz Stadium (dome)
    "BAL": (39.2780, -76.6227, False),
    "BUF": (42.7738, -78.7870, False),
    "CAR": (35.2258, -80.8528, False),
    "CHI": (41.8623, -87.6167, False),
    "CIN": (39.0955, -84.5160, False),
    "CLE": (41.5061, -81.6995, False),
    "DAL": (32.7473, -97.0945, True),    # AT&T Stadium (dome)
    "DEN": (39.7439, -105.0201, False),
    "DET": (42.3400, -83.0456, True),    # Ford Field (dome)
    "GB":  (44.5013, -88.0622, False),
    "HOU": (29.6847, -95.4107, True),    # NRG Stadium (retractable)
    "IND": (39.7601, -86.1639, True),    # Lucas Oil (dome)
    "JAX": (30.3239, -81.6373, False),
    "KC":  (39.0489, -94.4839, False),
    "LA":  (33.8644, -118.2611, True),   # SoFi Stadium (dome)
    "LAC": (33.8644, -118.2611, True),
    "LAR": (33.8644, -118.2611, True),
    "LV":  (36.0909, -115.1833, True),   # Allegiant Stadium (dome)
    "MIA": (25.9580, -80.2389, False),
    "MIN": (44.9736, -93.2575, True),    # US Bank Stadium (dome)
    "NE":  (42.0909, -71.2643, False),
    "NO":  (29.9511, -90.0812, True),    # Caesars Superdome (dome)
    "NYG": (40.8135, -74.0745, False),
    "NYJ": (40.8135, -74.0745, False),
    "PHI": (39.9008, -75.1675, False),
    "PIT": (40.4468, -80.0158, False),
    "SEA": (47.5952, -122.3316, False),
    "SF":  (37.4032, -121.9697, False),
    "TB":  (27.9759, -82.5033, False),
    "TEN": (36.1665, -86.7713, False),
    "WAS": (38.9078, -76.8645, False),
    "WSH": (38.9078, -76.8645, False),
}

def fetch_weather_for_games(force):
    """Pull historical weather for every outdoor game in schedules."""
    name = "weather"
    if already_have(name, force): return

    sched_path = RAW / "schedules.parquet"
    if not sched_path.exists():
        log("  schedules.parquet not found -- skipping weather", "WARN")
        return

    sched = pd.read_parquet(sched_path)

    # Filter to outdoor games (schedules has roof column: 'open','closed','dome','retractable')
    outdoor = sched[sched["roof"].isin(["open", "outdoors", "retractable"])].copy()
    outdoor = outdoor.dropna(subset=["gameday", "home_team"])
    outdoor["gameday"] = pd.to_datetime(outdoor["gameday"]).dt.strftime("%Y-%m-%d")

    log(f"Fetching weather for {len(outdoor)} outdoor games ...")

    rows = []
    seen = set()
    base = "https://api.open-meteo.com/v1/history"

    for _, g in outdoor.iterrows():
        team = g["home_team"]
        date = g["gameday"]
        key = (team, date)
        if key in seen:
            continue
        seen.add(key)

        coords = STADIUM_COORDS.get(team)
        if not coords:
            continue
        lat, lon, is_dome = coords

        if is_dome:
            rows.append({
                "game_id": g["game_id"],
                "home_team": team,
                "gameday": date,
                "temp_f": 72.0,
                "wind_mph": 0.0,
                "precip_in": 0.0,
                "is_dome": True,
            })
            continue

        try:
            r = requests.get(base, params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "daily": "temperature_2m_max,wind_speed_10m_max,precipitation_sum",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "America/New_York",
            }, timeout=15)
            r.raise_for_status()
            d = r.json()["daily"]
            rows.append({
                "game_id": g["game_id"],
                "home_team": team,
                "gameday": date,
                "temp_f": d["temperature_2m_max"][0],
                "wind_mph": d["wind_speed_10m_max"][0],
                "precip_in": d["precipitation_sum"][0],
                "is_dome": False,
            })
            time.sleep(0.1)   # respect rate limit
        except Exception as e:
            log(f"  weather failed for {team} {date}: {e}", "WARN")

    if rows:
        save(pd.DataFrame(rows), name)
    else:
        log("  no weather rows collected", "WARN")


# -----------------------------------------------------------------
# WEATHER -- extracted from schedules (no external API)
# nflverse schedules already contain temp, wind, roof, surface
# per game -- we just reshape it into a clean weather table.
# -----------------------------------------------------------------
def fetch_weather_from_schedules(force):
    name = "weather"
    if already_have(name, force): return

    sched_path = RAW / "schedules.parquet"
    if not sched_path.exists():
        log("  schedules.parquet not found -- skipping weather", "WARN")
        return

    log("Extracting weather from schedules ...")
    sched = pd.read_parquet(sched_path)

    weather_cols = ["game_id", "season", "week", "home_team", "away_team",
                    "gameday", "roof", "surface", "temp", "wind"]
    available = [c for c in weather_cols if c in sched.columns]
    df = sched[available].copy()

    # Derive is_dome from roof column
    dome_values = {"dome", "closed", "retractable"}
    if "roof" in df.columns:
        df["is_dome"] = df["roof"].str.lower().isin(dome_values)
        # Fill dome games with neutral conditions
        df.loc[df["is_dome"], "temp"] = df.loc[df["is_dome"], "temp"].fillna(72.0)
        df.loc[df["is_dome"], "wind"] = df.loc[df["is_dome"], "wind"].fillna(0.0)
    else:
        df["is_dome"] = False

    # Rename to consistent column names
    df = df.rename(columns={"temp": "temp_f", "wind": "wind_mph"})

    # Fill remaining nulls with reasonable defaults
    df["temp_f"]   = df["temp_f"].fillna(65.0)
    df["wind_mph"] = df["wind_mph"].fillna(8.0)

    # Add precipitation placeholder (not in schedules -- will be 0 unless PBP added later)
    df["precip_in"] = 0.0

    save(df, name)


# -----------------------------------------------------------------
# NEW SOURCE 1: FTN CHARTING DATA
# route_participation, pocket_time, blitz_count, times_blitzed
# Gives true WR opportunity (routes run) vs just targets
# -----------------------------------------------------------------
def fetch_ftn(seasons, force):
    name = "ftn_data"
    if already_have(name, force): return
    log("Fetching FTN charting data ...")
    import nfl_data_py as nfl
    try:
        # FTN data only available from 2022 onwards
        ftn_seasons = [s for s in seasons if s >= 2022]
        if not ftn_seasons:
            log("  FTN data requires 2022+ seasons -- skipping", "WARN")
            return
        df = nfl.import_ftn_data(ftn_seasons)
        save(df, name)
    except Exception as e:
        log(f"  FTN data not available: {e}", "WARN")


# -----------------------------------------------------------------
# NEW SOURCE 2: PFR WEEKLY PASSING STATS
# pocket_time, pressure_pct, bad_throw_pct, drop_pct, times_sacked
# Splits QB performance: clean pocket vs under pressure
# -----------------------------------------------------------------
def fetch_pfr_passing(seasons, force):
    name = "pfr_passing"
    if already_have(name, force): return
    log("Fetching PFR weekly passing stats ...")
    import nfl_data_py as nfl
    import pandas as pd
    try:
        df = nfl.import_weekly_pfr("pass", seasons)
        save(df, name)
    except Exception:
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.import_weekly_pfr("pass", [s]))
            except Exception:
                log(f"  PFR passing {s} not available, skipping", "WARN")
        if frames:
            save(pd.concat(frames, ignore_index=True), name)


# -----------------------------------------------------------------
# NEW SOURCE 3: PFR WEEKLY DEFENSIVE STATS
# blitz_pct, hurry_pct, pressure_pct, sacks, tackles per player
# Gives individual CB/pass rusher quality beyond team EPA
# -----------------------------------------------------------------
def fetch_pfr_defense(seasons, force):
    name = "pfr_defense"
    if already_have(name, force): return
    log("Fetching PFR weekly defensive stats ...")
    import nfl_data_py as nfl
    import pandas as pd
    try:
        df = nfl.import_weekly_pfr("def", seasons)
        save(df, name)
    except Exception:
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.import_weekly_pfr("def", [s]))
            except Exception:
                log(f"  PFR defense {s} not available, skipping", "WARN")
        if frames:
            save(pd.concat(frames, ignore_index=True), name)


# -----------------------------------------------------------------
# NEW SOURCE 4: PFR WEEKLY RECEIVING STATS
# yac, air_yards, drop_pct, broken_tackles per receiver
# Adds YAC ability and drop rate to WR/TE composite
# -----------------------------------------------------------------
def fetch_pfr_receiving(seasons, force):
    name = "pfr_receiving"
    if already_have(name, force): return
    log("Fetching PFR weekly receiving stats ...")
    import nfl_data_py as nfl
    import pandas as pd
    try:
        df = nfl.import_weekly_pfr("rec", seasons)
        save(df, name)
    except Exception:
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.import_weekly_pfr("rec", [s]))
            except Exception:
                log(f"  PFR receiving {s} not available, skipping", "WARN")
        if frames:
            save(pd.concat(frames, ignore_index=True), name)


# -----------------------------------------------------------------
# NEW SOURCE 5: ESPN QBR (weekly)
# qbr_total, pts_added -- composite QB quality from ESPN
# 3rd QB quality signal alongside dakota and CPOE
# -----------------------------------------------------------------
def fetch_qbr(seasons, force):
    name = "qbr"
    if already_have(name, force): return
    log("Fetching ESPN QBR (weekly) ...")
    import nfl_data_py as nfl
    import pandas as pd
    try:
        df = nfl.import_qbr(seasons, frequency="weekly")
        save(df, name)
    except Exception:
        frames = []
        for s in seasons:
            try:
                frames.append(nfl.import_qbr([s], frequency="weekly"))
            except Exception:
                log(f"  QBR {s} not available, skipping", "WARN")
        if frames:
            save(pd.concat(frames, ignore_index=True), name)
        else:
            log("  QBR not available for any season", "WARN")


# -----------------------------------------------------------------
# NEW SOURCE 6: SITUATIONAL STATS (derived from existing PBP)
# Red zone efficiency, 2-minute drill, 4th down aggression,
# pressure rate, sack rate -- all computed from pbp parquets
# No new download needed -- pure computation on existing data
# -----------------------------------------------------------------
def compute_situational_stats(force):
    name = "situational_stats"
    if already_have(name, force): return

    pbp_files = sorted(RAW.glob("pbp_*.parquet"))
    if not pbp_files:
        log("  No PBP files found -- skipping situational stats", "WARN")
        return

    log(f"Computing situational stats from {len(pbp_files)} PBP seasons ...")
    frames = []

    for path in pbp_files:
        df = pd.read_parquet(path)
        if len(df) == 0:
            continue

        season = int(path.stem.split("_")[1])

        # -- Check which pressure column is available in this PBP version --
        # Our filtered PBP uses qb_hit but some versions name it differently
        has_qb_hit = "qb_hit" in df.columns
        if not has_qb_hit:
            # Derive from qb_hit_1_player_id presence as proxy
            if "qb_hit_1_player_id" in df.columns:
                df["qb_hit"] = df["qb_hit_1_player_id"].notna().astype(float)
                has_qb_hit = True
            else:
                df["qb_hit"] = 0.0

        # Ensure sack column exists
        if "sack" not in df.columns:
            df["sack"] = 0.0

        # -- Pressure / OL quality --
        dropbacks = df[df["pass_attempt"].fillna(0) == 1].copy()
        pressure = (
            dropbacks.groupby(["season", "posteam"])
            .agg(
                dropbacks        =("play_id", "count"),
                qb_hits          =("qb_hit",  "sum"),
                sacks            =("sack",    "sum"),
            )
            .reset_index()
        )
        pressure["pressure_rate_allowed"] = (
            pressure["qb_hits"] / pressure["dropbacks"].replace(0, float("nan"))
        )
        pressure["sack_rate_allowed"] = (
            pressure["sacks"] / pressure["dropbacks"].replace(0, float("nan"))
        )

        # -- Defensive pressure generated --
        def_pressure = (
            dropbacks.groupby(["season", "defteam"])
            .agg(
                dropbacks_faced  =("play_id", "count"),
                qb_hits_gen      =("qb_hit",  "sum"),
                sacks_gen        =("sack",    "sum"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
        )
        def_pressure["pressure_rate_gen"] = (
            def_pressure["qb_hits_gen"] / def_pressure["dropbacks_faced"].replace(0, float("nan"))
        )
        def_pressure["sack_rate_gen"] = (
            def_pressure["sacks_gen"] / def_pressure["dropbacks_faced"].replace(0, float("nan"))
        )

        # -- Red zone efficiency --
        rz = df[(df["yardline_100"].fillna(99) <= 20) & df["play_type"].isin(["pass","run"])]
        rz_off = (
            rz.groupby(["season", "posteam"])
            .agg(
                rz_plays         =("play_id",   "count"),
                rz_tds           =("touchdown", "sum"),
                rz_epa           =("epa",       "mean"),
                rz_success       =("success",   "mean"),
            )
            .reset_index()
        )
        rz_off["rz_td_rate"] = rz_off["rz_tds"] / rz_off["rz_plays"].replace(0, float("nan"))

        rz_def = (
            rz.groupby(["season", "defteam"])
            .agg(
                rz_plays_allowed =("play_id",   "count"),
                rz_tds_allowed   =("touchdown", "sum"),
                rz_epa_allowed   =("epa",       "mean"),
            )
            .reset_index()
            .rename(columns={"defteam": "team"})
        )
        rz_def["rz_td_rate_allowed"] = rz_def["rz_tds_allowed"] / rz_def["rz_plays_allowed"].replace(0, float("nan"))

        # -- 2-minute drill (last 2 min of each half) --
        two_min = df[
            (df["game_seconds_remaining"].fillna(999) <= 120) &
            df["play_type"].isin(["pass", "run"])
        ]
        two_min_off = (
            two_min.groupby(["season", "posteam"])
            .agg(
                two_min_plays    =("play_id", "count"),
                two_min_epa      =("epa",     "mean"),
                two_min_success  =("success", "mean"),
            )
            .reset_index()
        )

        # -- 4th down aggressiveness --
        fourth = df[df["down"].fillna(0) == 4].copy()
        fourth_off = (
            fourth.groupby(["season", "posteam"])
            .agg(
                fourth_attempts  =("play_id",   "count"),
                fourth_goes      =("play_type", lambda x: x.isin(["pass","run"]).sum()),
            )
            .reset_index()
        )
        fourth_off["fourth_go_rate"] = (
            fourth_off["fourth_goes"] / fourth_off["fourth_attempts"].replace(0, float("nan"))
        )

        # -- Merge everything by team + season --
        # Start with pressure (has posteam)
        merged = pressure.rename(columns={"posteam": "team"}).merge(
            def_pressure[["season", "team", "pressure_rate_gen", "sack_rate_gen"]],
            on=["season", "team"], how="outer"
        ).merge(
            rz_off.rename(columns={"posteam": "team"})[
                ["season", "team", "rz_plays", "rz_td_rate", "rz_epa", "rz_success"]],
            on=["season", "team"], how="outer"
        ).merge(
            rz_def[["season", "team", "rz_td_rate_allowed", "rz_epa_allowed"]],
            on=["season", "team"], how="outer"
        ).merge(
            two_min_off.rename(columns={"posteam": "team"})[
                ["season", "team", "two_min_plays", "two_min_epa", "two_min_success"]],
            on=["season", "team"], how="outer"
        ).merge(
            fourth_off.rename(columns={"posteam": "team"})[
                ["season", "team", "fourth_go_rate"]],
            on=["season", "team"], how="outer"
        )

        frames.append(merged)

    if not frames:
        log("  No situational stats computed", "WARN")
        return

    result = pd.concat(frames, ignore_index=True)
    save(result, name)


# -----------------------------------------------------------------
# KICKING STATS (FG, XP, punting — nflverse kicking data)
# fg_made, fg_att, fg_pct, fg_made_0_19..50_, xp_att, xp_made
# punt_att, punt_yards, punt_avg, punt_net_yds, inside_20
# -----------------------------------------------------------------
def fetch_kicking(seasons, force):
    name = "kicking_stats"
    if already_have(name, force): return
    log("Fetching kicking stats (K/P) ...")
    import nfl_data_py as nfl
    import pandas as pd

    # Try 1: dedicated import_kicking function (nfl_data_py >= 0.3)
    try:
        df = nfl.import_kicking(seasons)
        if len(df) > 0:
            save(df, name)
            log(f"  saved kicking_stats.parquet ({len(df)} rows)")
            return
    except Exception as e:
        log(f"  import_kicking not available: {e}", "WARN")

    # Try 2: import_weekly_data with position filter
    try:
        frames = []
        for s in seasons:
            try:
                wk = nfl.import_weekly_data([s])
                if "position" in wk.columns:
                    kp = wk[wk["position"].isin(["K","P","PK","KP"])]
                    if len(kp) > 0:
                        frames.append(kp)
                        log(f"  season {s}: {len(kp)} K/P rows from weekly_data")
            except Exception as e2:
                log(f"  season {s} weekly_data failed: {e2}", "WARN")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            save(combined, name)
            log(f"  saved kicking_stats.parquet via weekly_data ({len(combined)} rows)")
            return
    except Exception as e:
        log(f"  weekly_data fallback failed: {e}", "WARN")

    # Try 3: seasonal kicking from seasonal data
    try:
        frames = []
        for s in seasons:
            try:
                sea = nfl.import_seasonal_data([s])
                if "position" in sea.columns:
                    kp = sea[sea["position"].isin(["K","P","PK"])]
                    if len(kp) > 0:
                        frames.append(kp)
            except Exception:
                pass
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            save(combined, name)
            log(f"  saved kicking_stats.parquet via seasonal_data ({len(combined)} rows)")
            return
    except Exception:
        pass

    log("  No kicking stats available from any source — will use PBP", "WARN")


# -----------------------------------------------------------------
# SCHEMA REPORT
# -----------------------------------------------------------------
def print_schema():
    print("\n" + "="*70)
    print("  DATA INVENTORY")
    print("="*70)
    total_mb = 0
    for path in sorted(RAW.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
            mb = path.stat().st_size / 1024 / 1024
            total_mb += mb
            print(f"\n  [FILE] {path.name}  ({len(df):,} rows x {len(df.columns)} cols, {mb:.1f} MB)")
            cols = list(df.columns)
            for i in range(0, min(len(cols), 30), 6):
                print("     " + "  ".join(f"{c:<22}" for c in cols[i:i+6]))
            if len(cols) > 30:
                print(f"     ... +{len(cols)-30} more columns")
        except Exception as e:
            print(f"  [ERR] {path.name}: {e}")
    print(f"\n  Total: {total_mb:.0f} MB across {len(list(RAW.glob('*.parquet')))} files")
    print("="*70)


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch NFL source data from nflverse + Open-Meteo"
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        default=[2021, 2022, 2023, 2024, 2025],
        help="Which NFL seasons to pull (default: 2021-2025)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if cached parquets exist"
    )
    parser.add_argument(
        "--skip-pbp", action="store_true",
        help="Skip play-by-play (large files: ~400MB/season)"
    )
    parser.add_argument(
        "--schema-only", action="store_true",
        help="Just print schema of existing files, don't fetch"
    )
    args = parser.parse_args()

    if args.schema_only:
        print_schema()
        return

    print("="*70)
    print(f"  NFL ENGINE -- DATA FETCHER")
    print(f"  Seasons: {args.seasons}")
    print(f"  Output:  {RAW.resolve()}")
    print(f"  Force:   {args.force}")
    print("="*70 + "\n")

    # Check nfl_data_py is installed
    try:
        import nfl_data_py as nfl
        log(f"nfl_data_py ready")
    except ImportError:
        log("nfl_data_py not found. Run: pip install nfl_data_py pandas pyarrow", "ERROR")
        sys.exit(1)

    t0 = time.time()

    # -- Fetch in dependency order ------------------------------
    fetch_team_info(args.force)
    fetch_ids(args.force)
    fetch_schedules(args.seasons, args.force)
    fetch_rosters(args.seasons, args.force)
    fetch_depth_charts(args.seasons, args.force)
    fetch_player_stats(args.seasons, args.force)
    fetch_seasonal_stats(args.seasons, args.force)
    fetch_snap_counts(args.seasons, args.force)
    fetch_injuries(args.seasons, args.force)
    fetch_ngs_passing(args.seasons, args.force)
    fetch_ngs_rushing(args.seasons, args.force)
    fetch_ngs_receiving(args.seasons, args.force)
    fetch_combine(args.force)
    fetch_officials(args.seasons, args.force)

    # -- NEW: Tier 1 free sources --------------------------------
    fetch_ftn(args.seasons, args.force)
    fetch_pfr_passing(args.seasons, args.force)
    fetch_pfr_defense(args.seasons, args.force)
    fetch_pfr_receiving(args.seasons, args.force)
    fetch_qbr(args.seasons, args.force)
    fetch_kicking(args.seasons, args.force)

    # Weather -- extracted from schedules (no external API)
    fetch_weather_from_schedules(args.force)

    # PBP last (largest)
    if not args.skip_pbp:
        log("--- Play-by-play (slowest, ~400MB/season) ---")
        fetch_pbp(args.seasons, args.force)
    else:
        log("Skipping PBP (--skip-pbp flag set)")

    # Situational stats -- derived from PBP (run after PBP)
    compute_situational_stats(args.force)

    elapsed = time.time() - t0
    log(f"\nAll done in {elapsed/60:.1f} minutes")

    # Save fetch log (utf-8 explicit to avoid Windows cp1252 issues)
    log_path = RAW / "fetch_log.txt"
    log_path.write_text("\n".join(LOG), encoding="utf-8")
    log(f"Log saved -> {log_path}")

    print_schema()


if __name__ == "__main__":
    main()
