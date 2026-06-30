"""
engine/conditions.py
--------------------
Computes environment modifiers that adjust predicted output up or down.

Modifiers are multiplicative scalars applied to:
  - passing_output_mult  : affects QB and WR/TE projections
  - rushing_output_mult  : affects RB projections
  - scoring_mult         : affects total points predicted
  - home_edge_pts        : flat point addition for home team

Factors:
  1. Weather       (wind, temperature, precipitation, dome)
  2. Field surface (grass vs turf)
  3. Rest / schedule (short week, bye week)
  4. Altitude      (Denver home games)
  5. Home field advantage (crowd noise, familiarity)
  6. Travel burden (cross-country road games)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

# ── Altitude data ──────────────────────────────────────────────────
STADIUM_ALTITUDE = {
    "DEN": 5280,   # Mile High -- significant fatigue factor for visitors
    "ARI": 1083,
    "ATL": 1050,
    "DAL": 430,
    "GB":  669,
    "DET": 600,
    "CHI": 580,
    "MIN": 830,
    "KC":  1014,
    "LAR": 66,
    "LAC": 66,
    "LV":  2001,   # Las Vegas -- notable altitude
    "SF":  52,
    "SEA": 17,
    "PIT": 1370,
}

# ── Turf types ─────────────────────────────────────────────────────
GRASS_SURFACES    = {"grass", "natural grass", "natural_grass"}
TURF_SURFACES     = {"fieldturf", "astroturf", "a_turf", "sportturf",
                     "matrixturf", "shaw sports turf", "artificial"}

# ── Home field advantage by stadium (noise + familiarity) ─────────
HOME_FIELD_PTS = {
    "SEA": 3.2,   # CenturyLink / Lumen -- historically loudest
    "KC":  2.8,
    "NE":  2.5,
    "GB":  2.4,
    "BAL": 2.3,
    "PIT": 2.2,
    "BUF": 2.1,
    "PHI": 2.0,
    "SF":  1.9,
    "CHI": 1.8,
    "DEFAULT": 1.5,
}

# ── Warm-weather team visiting cold-weather stadium ───────────────
WARM_WEATHER_TEAMS = {"MIA", "TB", "LAR", "LAC", "ARI", "NO", "ATL", "CAR"}
COLD_WEATHER_TEAMS = {"GB", "BUF", "NE", "CHI", "CLE", "PIT", "DEN", "MIN"}


def compute_conditions(game: pd.Series) -> dict:
    """
    Given a single game row from schedules + weather, compute all
    condition modifiers.

    Returns dict with all multipliers and adjustments.
    """
    mods = {
        "passing_mult":     1.0,
        "rushing_mult":     1.0,
        "scoring_mult":     1.0,
        "home_edge_pts":    0.0,
        "away_penalty_pts": 0.0,
        "conditions_notes": [],
    }

    home = str(game.get("home_team", ""))
    away = str(game.get("away_team", ""))

    # ── 1. WEATHER ─────────────────────────────────────────────────
    is_dome  = bool(game.get("is_dome", False))
    wind_mph = float(game.get("wind_mph", 0) or 0)
    temp_f   = float(game.get("temp_f", 65) or 65)
    precip   = float(game.get("precip_in", 0) or 0)

    # Also check roof from schedules (dome or retractable closed)
    roof = str(game.get("roof", "")).lower()
    if roof in {"dome", "closed"} or is_dome:
        wind_mph = 0
        temp_f   = 72
        precip   = 0
        mods["conditions_notes"].append("Dome: neutral conditions")

    # Wind penalty (exponential above 15mph)
    if wind_mph >= 25:
        mods["passing_mult"]  *= 0.78
        mods["scoring_mult"]  *= 0.88
        mods["conditions_notes"].append(f"Severe wind {wind_mph:.0f}mph: -22% passing, -12% scoring")
    elif wind_mph >= 20:
        mods["passing_mult"]  *= 0.87
        mods["scoring_mult"]  *= 0.93
        mods["conditions_notes"].append(f"High wind {wind_mph:.0f}mph: -13% passing, -7% scoring")
    elif wind_mph >= 15:
        mods["passing_mult"]  *= 0.94
        mods["scoring_mult"]  *= 0.97
        mods["conditions_notes"].append(f"Moderate wind {wind_mph:.0f}mph: -6% passing")

    # Cold weather penalty (below 32F -- outdoor only)
    if temp_f < 32 and not is_dome:
        mods["passing_mult"]  *= 0.92
        mods["rushing_mult"]  *= 0.95
        mods["scoring_mult"]  *= 0.93
        mods["conditions_notes"].append(f"Freezing temp {temp_f:.0f}F: -8% passing, -7% scoring")
    elif temp_f < 40 and not is_dome:
        mods["passing_mult"]  *= 0.96
        mods["conditions_notes"].append(f"Cold temp {temp_f:.0f}F: -4% passing")

    # Precipitation
    if precip > 0.3:
        mods["passing_mult"]  *= 0.90
        mods["rushing_mult"]  *= 1.05   # rain helps running game slightly
        mods["scoring_mult"]  *= 0.91
        mods["conditions_notes"].append(f"Heavy rain {precip:.2f}in: -10% passing, +5% rushing, -9% scoring")
    elif precip > 0.1:
        mods["passing_mult"]  *= 0.95
        mods["conditions_notes"].append(f"Light rain {precip:.2f}in: -5% passing")

    # Warm-weather team visiting cold stadium
    if away in WARM_WEATHER_TEAMS and temp_f < 35 and not is_dome:
        mods["away_penalty_pts"] += 3.0
        mods["conditions_notes"].append(f"Warm-weather team ({away}) in cold: +3pt home advantage")

    # ── 2. FIELD SURFACE ───────────────────────────────────────────
    surface = str(game.get("surface", "")).lower().replace(" ", "_")
    if surface in GRASS_SURFACES:
        mods["passing_mult"] *= 0.98   # very slight grass penalty for passing
        mods["conditions_notes"].append("Natural grass: neutral-slight rush boost")
    elif surface in TURF_SURFACES:
        mods["passing_mult"] *= 1.02   # turf very slightly boosts passing
        mods["conditions_notes"].append("Artificial turf: slight passing boost")

    # ── 3. REST / SCHEDULE ─────────────────────────────────────────
    home_rest = float(game.get("home_rest", 7) or 7)
    away_rest = float(game.get("away_rest", 7) or 7)

    # Short week (Thursday night or less than 6 days rest)
    if away_rest <= 5:
        mods["away_penalty_pts"] += 2.5
        mods["conditions_notes"].append(f"Away short week ({away_rest:.0f} days rest): +2.5pt home advantage")
    elif away_rest <= 6:
        mods["away_penalty_pts"] += 1.0

    if home_rest <= 5:
        mods["home_edge_pts"] -= 2.0   # home team also hurt by short week
        mods["conditions_notes"].append(f"Home short week ({home_rest:.0f} days rest): -2pt home edge")

    # Bye week advantage (14+ days rest)
    if home_rest >= 14:
        mods["home_edge_pts"] += 2.0
        mods["conditions_notes"].append(f"Home coming off bye: +2pt home edge")
    if away_rest >= 14:
        mods["away_penalty_pts"] -= 1.5   # away bye week reduces disadvantage
        mods["conditions_notes"].append(f"Away coming off bye: -1.5pt disadvantage reduction")

    # ── 4. ALTITUDE ────────────────────────────────────────────────
    home_altitude = STADIUM_ALTITUDE.get(home, 500)
    away_altitude = STADIUM_ALTITUDE.get(away, 500)

    if home_altitude >= 5000:
        mods["home_edge_pts"]    += 2.0
        mods["away_penalty_pts"] += 2.5
        mods["conditions_notes"].append(f"High altitude ({home_altitude}ft): +2pt home, +2.5pt away fatigue")
    elif home_altitude >= 2000:
        mods["home_edge_pts"]    += 0.8
        mods["away_penalty_pts"] += 1.0

    # Away team from high-altitude city visiting sea level (lungs open up)
    if away_altitude >= 5000 and home_altitude < 1000:
        mods["conditions_notes"].append(f"Away team ({away}) from altitude: no penalty at sea level")

    # ── 5. HOME FIELD ADVANTAGE ────────────────────────────────────
    hfa_pts = HOME_FIELD_PTS.get(home, HOME_FIELD_PTS["DEFAULT"])
    mods["home_edge_pts"] += hfa_pts
    mods["conditions_notes"].append(f"Home field ({home}): +{hfa_pts:.1f}pt")

    # ── 6. TRAVEL BURDEN ───────────────────────────────────────────
    # Cross-country road games (East team to West or vice versa)
    EAST_TEAMS = {"NE", "NYJ", "NYG", "BUF", "MIA", "BAL", "PIT", "CLE", "CIN",
                  "PHI", "DAL", "WAS", "ATL", "CAR", "NO", "TB"}
    WEST_TEAMS = {"SEA", "SF", "LAR", "LAC", "ARI", "LV", "DEN", "KC"}

    if (away in EAST_TEAMS and home in WEST_TEAMS) or \
       (away in WEST_TEAMS and home in EAST_TEAMS):
        mods["away_penalty_pts"] += 1.0
        mods["conditions_notes"].append("Cross-country travel: +1pt home advantage")

    # ── Round and summarize ────────────────────────────────────────
    total_home_advantage = round(
        mods["home_edge_pts"] + mods["away_penalty_pts"], 1
    )
    mods["total_home_advantage_pts"] = total_home_advantage
    mods["passing_mult"]  = round(mods["passing_mult"],  3)
    mods["rushing_mult"]  = round(mods["rushing_mult"],  3)
    mods["scoring_mult"]  = round(mods["scoring_mult"],  3)

    return mods


def build_all_conditions(seasons: list = None) -> pd.DataFrame:
    """
    Compute conditions modifiers for all games in schedules + weather.
    """
    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024]

    sched   = pd.read_parquet(RAW / "schedules.parquet")
    weather = pd.read_parquet(RAW / "weather.parquet")

    sched = sched[sched["season"].isin(seasons)]

    # Merge weather into schedules
    games = sched.merge(
        weather[["game_id", "temp_f", "wind_mph", "precip_in", "is_dome"]],
        on="game_id", how="left"
    )

    # Fill weather from schedules own temp/wind columns where available
    if "temp" in games.columns:
        games["temp_f"] = games["temp_f"].fillna(games["temp"])
    if "wind" in games.columns:
        games["wind_mph"] = games["wind_mph"].fillna(games["wind"])

    print(f"Computing conditions for {len(games)} games...")
    results = []
    for _, game in games.iterrows():
        mods = compute_conditions(game)
        results.append({
            "game_id":                  game["game_id"],
            "season":                   game["season"],
            "week":                     game["week"],
            "home_team":                game["home_team"],
            "away_team":                game["away_team"],
            "passing_mult":             mods["passing_mult"],
            "rushing_mult":             mods["rushing_mult"],
            "scoring_mult":             mods["scoring_mult"],
            "home_edge_pts":            mods["home_edge_pts"],
            "away_penalty_pts":         mods["away_penalty_pts"],
            "total_home_advantage_pts": mods["total_home_advantage_pts"],
            "conditions_notes":         " | ".join(mods["conditions_notes"]),
        })

    df = pd.DataFrame(results)
    out_path = PROC / "conditions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}  ({len(df)} games)")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024])
    args = parser.parse_args()

    df = build_all_conditions(args.seasons)
    print("\nTop home-advantaged games:")
    print(df.nlargest(10, "total_home_advantage_pts")[
        ["season", "week", "home_team", "away_team",
         "total_home_advantage_pts", "passing_mult", "scoring_mult"]
    ].to_string(index=False))

    print("\nSample: Bills home game conditions:")
    buf = df[(df["home_team"] == "BUF") & (df["season"] == 2023)]
    if not buf.empty:
        print(buf[[
            "week", "away_team", "temp_f" if "temp_f" in buf.columns else "passing_mult",
            "total_home_advantage_pts", "conditions_notes"
        ]].head(5).to_string(index=False))
