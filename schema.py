# NFL Engine — Data Schema & Column Dictionary
# Every table, every column, what it means and how it feeds the engine.

SCHEMA = {

    # ═══════════════════════════════════════════════════════════════
    # SCHEDULES  →  schedules.parquet
    # One row per game. The backbone — every other table joins here.
    # ═══════════════════════════════════════════════════════════════
    "schedules": {
        "source": "nflverse via nfl_data_py.import_schedules()",
        "grain": "1 row per game",
        "key": "game_id",
        "columns": {
            # Identity
            "game_id":          "Unique game key  (e.g. '2024_01_BAL_KC')",
            "season":           "NFL season year",
            "week":             "Week number (1-22; 19-22 = playoffs)",
            "season_type":      "'REG' or 'POST'",
            "gameday":          "Date of game (YYYY-MM-DD)",
            "gametime":         "Kickoff time local (HH:MM)",
            "weekday":          "Day of week",

            # Teams
            "home_team":        "Home team abbreviation",
            "away_team":        "Away team abbreviation",

            # Outcome (NaN for future games)
            "home_score":       "Final home score",
            "away_score":       "Final away score",
            "result":           "Home margin (home_score - away_score)",
            "total":            "Combined points scored",
            "overtime":         "1 if game went to OT",

            # Coaching (critical for style modeling)
            "home_coach":       "Home head coach name",
            "away_coach":       "Away head coach name",
            "home_qb_id":       "Starting QB player_id (home)",
            "away_qb_id":       "Starting QB player_id (away)",
            "home_qb_name":     "Starting QB name (home)",
            "away_qb_name":     "Starting QB name (away)",

            # Conditions
            "location":         "'Home' or 'Neutral'",
            "roof":             "'dome', 'open', 'closed', 'retractable'",
            "surface":          "Field surface type ('grass','fieldturf','astroturf', etc.)",
            "temp":             "Temperature (°F) at kickoff — NaN for domes",
            "wind":             "Wind speed (mph) at kickoff",
            "stadium":          "Stadium name",
            "stadium_id":       "Stadium ID (joins to coords for weather fetch)",

            # Schedule load (for rest modeling)
            "home_rest":        "Days since home team's last game",
            "away_rest":        "Days since away team's last game",
            "home_moneyline":   "Consensus home moneyline (opening)",
            "away_moneyline":   "Consensus away moneyline",
            "spread_line":      "Consensus spread (home perspective, e.g. -3 = home fav by 3)",
            "total_line":       "Consensus O/U total",
            "div_game":         "1 if divisional matchup",
            "playoff":          "1 if postseason game",

            # TV / misc
            "away_score":       "Final away score",
            "stadium_id":       "Links to stadium coords for altitude/turf lookup",
        },
        "engine_use": [
            "Conditions modifiers (roof→dome, surface, temp, wind)",
            "Rest/schedule modifiers (home_rest, away_rest)",
            "Home field advantage baseline",
            "Coach identity → joins to coaching style profiles",
            "Vegas lines → calibration layer",
            "div_game → rivalry/familiarity modifier",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # WEEKLY PLAYER STATS  →  player_stats.parquet
    # Per player per game. Primary source for player composite scores.
    # ═══════════════════════════════════════════════════════════════
    "player_stats": {
        "source": "nfl_data_py.import_weekly_data()",
        "grain": "1 row per player per game",
        "key": ["player_id", "game_id"],
        "columns": {
            # Identity
            "player_id":                "nflverse player ID (gsis_id)",
            "player_name":              "Display name",
            "player_display_name":      "Full display name",
            "position":                 "QB, RB, WR, TE, K, etc.",
            "position_group":           "Grouped position",
            "recent_team":              "Team abbr this game",
            "season":                   "NFL season",
            "week":                     "Week number",
            "season_type":              "'REG' or 'POST'",
            "opponent_team":            "Opponent abbr",
            "game_id":                  "Links to schedules.game_id",

            # Passing
            "completions":              "Completions",
            "attempts":                 "Pass attempts",
            "passing_yards":            "Passing yards",
            "passing_tds":              "Passing TDs",
            "interceptions":            "Interceptions thrown",
            "sacks":                    "Times sacked",
            "sack_yards":               "Yards lost to sacks",
            "sack_fumbles":             "Fumbles on sacks",
            "passing_air_yards":        "Total air yards (completed + incomplete)",
            "passing_yards_after_catch": "YAC on completions",
            "passing_first_downs":      "First downs via pass",
            "passing_epa":              "Expected points added — passing",
            "passing_2pt_conversions":  "2-pt conversions passing",
            "pacr":                     "Passing air conversion ratio",
            "dakota":                   "Adjusted EPA+CPOE composite (QB quality)",

            # Rushing
            "carries":                  "Rush attempts",
            "rushing_yards":            "Rushing yards",
            "rushing_tds":              "Rushing TDs",
            "rushing_fumbles":          "Rush fumbles",
            "rushing_first_downs":      "First downs via rush",
            "rushing_epa":              "EPA — rushing",
            "rushing_2pt_conversions":  "2-pt conversions rushing",

            # Receiving
            "receptions":               "Receptions",
            "targets":                  "Targets",
            "receiving_yards":          "Receiving yards",
            "receiving_tds":            "Receiving TDs",
            "receiving_fumbles":        "Receiving fumbles",
            "receiving_air_yards":      "Air yards on targets (receiver side)",
            "receiving_yards_after_catch": "YAC",
            "receiving_first_downs":    "First downs via reception",
            "receiving_epa":            "EPA — receiving",
            "target_share":             "Share of team targets (0–1)",
            "air_yards_share":          "Share of team air yards",
            "wopr":                     "Weighted opportunity rating (targets+air yards share blend)",
            "racr":                     "Receiver air conversion ratio (rec_yds / air_yards)",

            # Special teams / misc
            "special_teams_tds":        "ST TDs",
            "fantasy_points":           "Fantasy pts (standard)",
            "fantasy_points_ppr":       "Fantasy pts (PPR)",
        },
        "engine_use": [
            "Rolling averages → player tier/rank per position",
            "EPA columns → efficiency layer of composite score",
            "target_share, air_yards_share, wopr → usage component",
            "dakota (QB) → raw QB quality signal",
            "rushing/receiving EPA → cross-position matchup values",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # SEASONAL STATS  →  seasonal_stats.parquet
    # Full-season totals. Used for season-level rankings and tier
    # assignment when weekly data is sparse (early in season).
    # ═══════════════════════════════════════════════════════════════
    "seasonal_stats": {
        "source": "nfl_data_py.import_seasonal_data()",
        "grain": "1 row per player per season",
        "key": ["player_id", "season"],
        "columns": "Same as player_stats but season-aggregated + games_played",
        "engine_use": [
            "Position rank (1–N) across league per season",
            "Percentile tiers: Elite (top 10%), Above Avg, Avg, Below",
            "Fallback when player has <4 weeks of data in current season",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # ROSTERS  →  rosters.parquet
    # Player bio + status. Needed to know who is on each team each week.
    # ═══════════════════════════════════════════════════════════════
    "rosters": {
        "source": "nfl_data_py.import_rosters()",
        "grain": "1 row per player per season per week (or season snapshot)",
        "key": ["player_id", "season", "week"],
        "columns": {
            "player_id":            "gsis_id (primary key)",
            "gsis_id":              "NFL official ID",
            "espn_id":              "ESPN player ID (for image/headshot lookup)",
            "pff_id":               "PFF player ID (if merging PFF grades)",
            "pfr_id":               "Pro Football Reference ID",
            "player_name":          "Display name",
            "player_display_name":  "Full name",
            "position":             "Canonical position",
            "depth_chart_position": "Position on depth chart (can differ from position)",
            "jersey_number":        "Jersey #",
            "status":               "'ACT','INA','RES','PUP' etc.",
            "birth_date":           "DOB → derive age",
            "height":               "Height in inches",
            "weight":               "Weight in lbs",
            "college":              "College attended",
            "draft_number":         "Overall draft pick #",
            "years_exp":            "Years in NFL",
            "team":                 "Current team abbr",
            "season":               "Season",
            "week":                 "Week (if weekly roster)",
            "headshot_url":         "URL to player headshot image",
        },
        "engine_use": [
            "Determine active roster per team per week",
            "Age → career stage modifier on composite score",
            "years_exp → experience modifier",
            "height/weight → athleticism tier (combined with combine)",
            "status → injury availability filter",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # DEPTH CHARTS  →  depth_charts.parquet
    # Who starts vs. backs up each week per position.
    # ═══════════════════════════════════════════════════════════════
    "depth_charts": {
        "source": "nfl_data_py.import_depth_charts()",
        "grain": "1 row per player per position per team per week",
        "key": ["gsis_id", "season", "week", "team", "formation_position"],
        "columns": {
            "season":               "Season",
            "club_code":            "Team abbr",
            "week":                 "Week",
            "game_type":            "'REG','POST'",
            "depth_team":           "1 = starter, 2 = backup, etc.",
            "last_name":            "Last name",
            "first_name":           "First name",
            "football_name":        "Name as appears on jersey",
            "jersey_number":        "Jersey #",
            "position":             "Canonical position",
            "formation_position":   "Specific formation slot (LT,LG,C,RG,RT,WR1,WR2,etc.)",
            "gsis_id":              "Links to rosters.player_id",
            "full_name":            "Full name",
        },
        "engine_use": [
            "Confirm starter status per position group each week",
            "formation_position → identify specific matchups (LT vs EDGE, WR1 vs CB1)",
            "Depth chart slot → multiplier on player composite (starter vs. backup)",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # SNAP COUNTS  →  snap_counts.parquet
    # Actual snaps played. True usage signal — separates starter from
    # rotational player regardless of depth chart listing.
    # ═══════════════════════════════════════════════════════════════
    "snap_counts": {
        "source": "nfl_data_py.import_snap_counts()",
        "grain": "1 row per player per game",
        "key": ["pfr_player_id", "game_id"],
        "columns": {
            "game_id":              "Links to schedules",
            "season":               "Season",
            "week":                 "Week",
            "team":                 "Team",
            "player":               "Player name",
            "pfr_player_id":        "PFR ID (cross-ref via player_ids.parquet)",
            "position":             "Position",
            "offense_snaps":        "Offensive snaps played",
            "offense_pct":          "% of team offensive snaps (KEY USAGE SIGNAL)",
            "defense_snaps":        "Defensive snaps played",
            "defense_pct":          "% of team defensive snaps",
            "st_snaps":             "Special teams snaps",
            "st_pct":               "% of ST snaps",
        },
        "engine_use": [
            "offense_pct → true usage weight on offensive player score",
            "defense_pct → true usage weight on defensive player score",
            "Snap% > 80% = true starter regardless of depth chart",
            "Combined with target_share → true WR/TE opportunity score",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # INJURIES  →  injuries.parquet
    # Weekly injury report. Affects player availability and composite.
    # ═══════════════════════════════════════════════════════════════
    "injuries": {
        "source": "nfl_data_py.import_injuries()",
        "grain": "1 row per player per week injury report",
        "key": ["gsis_id", "season", "week"],
        "columns": {
            "season":               "Season",
            "week":                 "Week",
            "team":                 "Team",
            "gsis_id":              "Player ID",
            "full_name":            "Player name",
            "game_type":            "'REG' or 'POST'",
            "practice_primary":     "Primary injury type",
            "practice_status":      "'Full', 'Limited', 'Did Not Participate'",
            "report_primary_injury":"Injury label on official report",
            "report_status":        "'Questionable','Doubtful','Out','IR' etc.",
        },
        "engine_use": [
            "Play probability multiplier on composite score:",
            "  Active (no report) → 1.0",
            "  Questionable       → 0.75",
            "  Doubtful           → 0.25",
            "  Out / IR           → 0.0  (exclude from matchup matrix)",
            "practice_status adds secondary signal (Limited = slight downgrade)",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # NEXT GEN STATS — PASSING  →  ngs_passing.parquet
    # Tracking-based QB metrics. Separates scheme from QB talent.
    # ═══════════════════════════════════════════════════════════════
    "ngs_passing": {
        "source": "nfl_data_py.import_ngs_data('passing')",
        "grain": "1 row per QB per week (min snap threshold)",
        "key": ["player_gsis_id", "season", "week"],
        "columns": {
            "player_display_name":          "QB name",
            "player_gsis_id":               "Links to player_id",
            "season":                       "Season",
            "week":                         "Week (0 = season total)",
            "team_abbr":                    "Team",
            "avg_time_to_throw":            "Avg seconds from snap to release",
            "avg_completed_air_yards":      "Avg air yards on completions",
            "avg_intended_air_yards":       "Avg air yards on all attempts (aggressiveness)",
            "avg_air_yards_differential":   "Intended minus completed air yards",
            "aggressiveness":               "% of attempts into tight coverage (<1yd separation)",
            "max_completed_air_distance":   "Longest completed pass (air)",
            "avg_air_yards_to_sticks":      "Avg air yards relative to first down marker",
            "completion_percentage":        "Actual completion %",
            "expected_completion_percentage": "xComp% from tracking",
            "completion_percentage_above_expectation": "CPOE — KEY QB QUALITY METRIC",
            "avg_air_distance":             "Avg total air distance per throw",
            "max_air_distance":             "Longest throw",
            "player_position":              "Should be QB",
        },
        "engine_use": [
            "CPOE → true passing skill beyond weapons/scheme",
            "aggressiveness → offensive style tag (aggressive vs checkdown)",
            "avg_intended_air_yards → deep vs short passing style profile",
            "avg_time_to_throw → pocket mobility style signal",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # NEXT GEN STATS — RUSHING  →  ngs_rushing.parquet
    # Tracking metrics isolate RB skill from OL quality.
    # ═══════════════════════════════════════════════════════════════
    "ngs_rushing": {
        "source": "nfl_data_py.import_ngs_data('rushing')",
        "grain": "1 row per RB per week",
        "key": ["player_gsis_id", "season", "week"],
        "columns": {
            "player_display_name":          "RB name",
            "efficiency":                   "Yards over expectation / attempt — KEY RB SKILL",
            "percent_attempts_gte_8_defenders": "% runs vs stacked box — OL/scheme signal",
            "avg_time_to_los":              "Avg time to reach line of scrimmage",
            "expected_rush_yards":          "Expected yards from tracking data",
            "rush_yards_over_expected":     "RYOE — actual minus expected",
            "rush_yards_over_expected_per_att": "RYOE/attempt",
            "rush_pct_over_expected":       "Percentile RYOE",
        },
        "engine_use": [
            "RYOE/att → true RB skill isolated from OL",
            "% vs stacked box → measure of OL effectiveness / team run identity",
            "efficiency → RB composite score rushing component",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # NEXT GEN STATS — RECEIVING  →  ngs_receiving.parquet
    # Separation, cushion, catch ability beyond scheme.
    # ═══════════════════════════════════════════════════════════════
    "ngs_receiving": {
        "source": "nfl_data_py.import_ngs_data('receiving')",
        "grain": "1 row per receiver per week",
        "key": ["player_gsis_id", "season", "week"],
        "columns": {
            "player_display_name":          "Receiver name",
            "avg_cushion":                  "Avg yards of cushion from nearest defender at snap",
            "avg_separation":               "Avg yards of separation at catch point — KEY WR SKILL",
            "avg_intended_air_yards":       "Avg depth of target",
            "percent_share_of_intended_air_yards": "Air yards share (NGS version)",
            "receptions":                   "Receptions",
            "targets":                      "Targets",
            "catch_percentage":             "Actual catch %",
            "avg_yac":                      "Avg YAC",
            "avg_expected_yac":             "Expected YAC from tracking",
            "avg_yac_above_expectation":    "YAC over expectation — route running / YAC ability",
        },
        "engine_use": [
            "avg_separation → receiver quality score (vs CB coverage quality)",
            "avg_intended_air_yards → route depth → match vs CB alignment",
            "avg_yac_above_expectation → after-catch ability component",
            "avg_cushion → CB respect signal (how much CBs give this WR)",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # PLAY-BY-PLAY  →  pbp_{season}.parquet
    # Most granular source. Used for:
    # - Team offensive/defensive style profiles (run/pass rates, formations)
    # - Down-and-distance tendencies
    # - Situational EPA (red zone, 3rd down, 2-minute)
    # - Coach tendency profiles
    # ═══════════════════════════════════════════════════════════════
    "pbp": {
        "source": "nfl_data_py.import_pbp_data() [filtered columns]",
        "grain": "1 row per play",
        "key": ["game_id", "play_id"],
        "key_columns": {
            "game_id":              "Links to schedules",
            "season":               "Season",
            "week":                 "Week",
            "posteam":              "Team with possession (offense)",
            "defteam":              "Defensive team",
            "play_type":            "'pass','run','punt','field_goal','kickoff', etc.",
            "pass_attempt":         "1 if pass play",
            "rush_attempt":         "1 if run play",
            "down":                 "1,2,3,4",
            "ydstogo":              "Yards to first down",
            "yardline_100":         "Yards from own end zone (100 = own goal line)",
            "score_differential":   "Posteam score minus defteam score at snap",
            "epa":                  "Expected points added on this play — core efficiency metric",
            "success":              "1 if EPA > 0 (binary efficiency)",
            "cpoe":                 "Completion % over expectation (pass plays)",
            "air_yards":            "Depth of target (pass plays)",
            "yards_after_catch":    "YAC",
            "yards_gained":         "Yards gained",
            "touchdown":            "1 if TD",
            "interception":         "1 if INT",
            "sack":                 "1 if sack",
            "roof":                 "Roof type this game",
            "surface":              "Field surface",
            "temp":                 "Temperature",
            "wind":                 "Wind speed",
            "home_coach":           "Home coach (for coach tendency profiling)",
            "away_coach":           "Away coach",
        },
        "engine_use": [
            "Pass rate by down/distance/score → offensive style profile",
            "Run rate in run-heavy situations → identity tag",
            "3rd down conversion rate → efficiency component",
            "Red zone pass% vs run% → scoring style",
            "EPA/play split by pass/run → team efficiency by play type",
            "EPA allowed/play → defensive quality by play type",
            "Coach tendency: early down run%, 3rd-and-short pass%, 2-min drill pass%",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # COMBINE  →  combine.parquet
    # Athletic measurables. Feeds physicality component of composite.
    # ═══════════════════════════════════════════════════════════════
    "combine": {
        "source": "nfl_data_py.import_combine_data()",
        "grain": "1 row per player (draft year)",
        "key": "player_id",
        "columns": {
            "player_name":  "Name",
            "pos":          "Position",
            "draft_year":   "Year drafted",
            "draft_team":   "Team that drafted",
            "draft_round":  "Round",
            "draft_pick":   "Overall pick",
            "height":       "Height (in)",
            "weight":       "Weight (lbs)",
            "forty_yd":     "40-yard dash time (lower = faster)",
            "bench":        "Bench press reps (225 lbs)",
            "broad_jump":   "Broad jump (in)",
            "cone":         "3-cone drill time",
            "shuttle":      "20-yard shuttle time",
            "vert_leap":    "Vertical leap (in)",
            "bmi":          "Body mass index",
            "player_id":    "Links to rosters/player_stats",
        },
        "engine_use": [
            "forty_yd → speed score for WR/CB matchup separation estimates",
            "vert_leap + height → jump ball advantage (WR vs CB)",
            "bench → pass rush power (DL) / run block power (OL)",
            "size/speed score → athleticism tier (0-100 scale per position)",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # WEATHER  →  weather.parquet  (fetched from Open-Meteo)
    # ═══════════════════════════════════════════════════════════════
    "weather": {
        "source": "Open-Meteo historical API (free, no key)",
        "grain": "1 row per game (outdoor games only; dome games = 72°F/0 wind)",
        "key": "game_id",
        "columns": {
            "game_id":      "Links to schedules",
            "home_team":    "Stadium team",
            "gameday":      "Game date",
            "temp_f":       "High temperature (°F) on game day",
            "wind_mph":     "Max wind speed (mph)",
            "precip_in":    "Precipitation (inches)",
            "is_dome":      "True if dome/retractable (weather set to neutral)",
        },
        "engine_use": [
            "wind_mph > 15 → passing penalty (-X% to QB/WR composite in this matchup)",
            "wind_mph > 25 → kicking penalty, major passing suppression",
            "temp_f < 32  → cold weather penalty (visiting warm-weather teams get extra hit)",
            "precip_in > 0.1 → wet field → fumble risk up, footing penalty",
            "is_dome → no weather modifier applied",
        ]
    },

    # ═══════════════════════════════════════════════════════════════
    # TEAM INFO  →  team_info.parquet
    # ═══════════════════════════════════════════════════════════════
    "team_info": {
        "source": "nfl_data_py.import_team_desc()",
        "grain": "1 row per team",
        "columns": {
            "team_abbr":        "3-letter abbreviation (primary key used everywhere)",
            "team_name":        "Full team name",
            "team_id":          "Numeric ID",
            "team_nick":        "Nickname (Chiefs, Eagles, etc.)",
            "team_conf":        "AFC or NFC",
            "team_division":    "e.g. 'NFC East'",
            "team_color":       "Primary hex color",
            "team_color2":      "Secondary hex color",
            "team_logo_wikipedia": "Logo URL",
            "stadium":          "Stadium name",
            "stadium_location": "City, State",
            "team_logo_espn":   "ESPN logo URL",
        },
    },

    # ═══════════════════════════════════════════════════════════════
    # PLAYER IDS  →  player_ids.parquet
    # Cross-reference: links gsis_id → ESPN, PFR, Sleeper, PFF ids
    # ═══════════════════════════════════════════════════════════════
    "player_ids": {
        "source": "nfl_data_py.import_ids()",
        "grain": "1 row per player",
        "key": "gsis_id",
        "columns": {
            "name":         "Player name",
            "position":     "Position",
            "team":         "Current team",
            "gsis_id":      "NFL official ID (primary)",
            "espn_id":      "ESPN ID",
            "pfr_id":       "Pro Football Reference ID",
            "pff_id":       "Pro Football Focus ID",
            "sleeper_id":   "Sleeper fantasy ID",
            "fantasypros_id": "FantasyPros ID",
            "mfl_id":       "MyFantasyLeague ID",
            "rotowire_id":  "Rotowire ID",
            "sportradar_id": "Sportradar ID",
            "yahoo_id":     "Yahoo fantasy ID",
        },
        "engine_use": [
            "snap_counts uses pfr_id → join via this table to gsis_id",
            "If adding PFF grades later → join via pff_id",
            "Future: ESPN API player data → join via espn_id",
        ]
    },
}


# ─────────────────────────────────────────────────────────────────
# ENGINE TABLE MAP
# Shows which parquet files feed each engine component
# ─────────────────────────────────────────────────────────────────
ENGINE_DATA_MAP = {

    "player_composite_score": {
        "description": "0-100 score per player combining rank, tier, efficiency, usage, athleticism",
        "inputs": [
            "player_stats      → rolling EPA, efficiency stats",
            "seasonal_stats    → position rank (1-N), percentile tier",
            "snap_counts       → true usage (offense_pct)",
            "ngs_passing       → CPOE, aggressiveness (QB only)",
            "ngs_rushing       → RYOE/att (RB only)",
            "ngs_receiving     → avg_separation, YAC over expected (WR/TE)",
            "combine           → athleticism tier (speed, size)",
            "depth_charts      → starter multiplier",
            "injuries          → availability multiplier",
            "rosters           → age, experience modifiers",
        ]
    },

    "team_style_profile": {
        "description": "Tags each team as run-heavy/pass-heavy, aggressive/conservative, etc.",
        "inputs": [
            "pbp              → pass_rate by down/distance, EPA splits, formation rates",
            "schedules        → home_coach, away_coach identity",
            "player_stats     → team rushing_yards, passing_yards, target distributions",
            "seasonal_stats   → team-level aggregates",
        ]
    },

    "positional_matchup_matrix": {
        "description": "Head-to-head value: QB vs pass rush, WR1 vs CB1, RB vs run defense, etc.",
        "inputs": [
            "player_composite_score  → both offensive and defensive player scores",
            "depth_charts            → formation_position to identify exact matchup slots",
            "snap_counts             → usage weight per player",
            "ngs_receiving           → separation metric vs CB cushion",
            "pbp                     → historical performance of this matchup type",
        ]
    },

    "conditions_modifier": {
        "description": "Multipliers applied to each matchup based on game environment",
        "inputs": [
            "weather         → temp, wind, precip, is_dome",
            "schedules       → roof, surface, home_rest, away_rest, div_game",
            "team_info       → stadium altitude (Denver = +modifier for home team)",
        ]
    },

    "gameday_prediction": {
        "description": "Predicted offensive/defensive output per team, per game",
        "inputs": [
            "All of the above",
            "schedules       → Vegas lines for calibration",
        ]
    },
}


if __name__ == "__main__":
    import json

    print("\n" + "═"*70)
    print("  NFL ENGINE — DATA SCHEMA")
    print("═"*70)

    for table, info in SCHEMA.items():
        print(f"\n  ┌── {table.upper()}")
        print(f"  │   Source : {info['source']}")
        print(f"  │   Grain  : {info['grain']}")
        if "engine_use" in info:
            print(f"  │   Used for:")
            for u in info["engine_use"]:
                print(f"  │     • {u}")

    print("\n" + "═"*70)
    print("  ENGINE COMPONENT → DATA DEPENDENCIES")
    print("═"*70)
    for component, info in ENGINE_DATA_MAP.items():
        print(f"\n  ► {component}")
        print(f"    {info['description']}")
        for inp in info["inputs"]:
            print(f"    ← {inp}")
