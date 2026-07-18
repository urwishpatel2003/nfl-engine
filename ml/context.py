"""
ml/context.py  —  game-context adjustments for the matchup/schedule model
=========================================================================
The lightweight matchup engine assumes a normal home game. Real slates aren't:
international games are neutral-site (no home-field), teams travel varying distances
across time zones, and weather/altitude shift scoring. This module returns a small,
bounded per-team points adjustment (+ human-readable notes) for a scheduled game.

Signals used (all present in schedules.parquet):
  location  → 'Neutral' marks international / neutral-site games (drop home field)
  stadium   → venue; international venues have coordinates here for travel distance
  roof      → dome/closed = weather-proof; outdoors = exposed
  temp/wind → real weather when the game is near/played (NaN for far-future games)
  week      → for a cold-weather climate prior when no forecast exists yet

Everything is capped so context nudges the line, never dominates it.
"""

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent


def _nfl_coords() -> dict:
    """team -> (lat, lon, is_dome), reused from the data fetcher (no nfl_data_py)."""
    sys.path.insert(0, str(ROOT))
    from fetch_data import STADIUM_COORDS
    return STADIUM_COORDS


# Coordinates for international / neutral venues (2026 slate + regulars).
INTL_VENUES = {
    "Tottenham Hotspur Stadium": (51.6043, -0.0665),
    "Wembley Stadium": (51.5560, -0.2796),
    "Allianz Arena": (48.2188, 11.6247),
    "FC Bayern Munich Stadium": (48.2188, 11.6247),
    "Deutsche Bank Park": (50.0686, 8.6455),
    "Stade de France": (48.9245, 2.3601),
    "Estadio Azteca": (19.3029, -99.1505),
    "Estadio Banorte": (25.6690, -100.2440),
    "Maracana Stadium": (-22.9121, -43.2302),
    "Arena Corinthians": (-23.5453, -46.4742),
    "Santiago Bernabeu": (40.4531, -3.6883),
    "Bernabeu": (40.4531, -3.6883),
    "Melbourne Cricket Ground": (-37.8200, 144.9834),
}

# Cold-weather teams that play OUTDOORS (used only as a late-season prior when no
# forecast exists; dome teams are excluded by the roof check regardless).
COLD_OUTDOOR = {"GB", "BUF", "CHI", "NE", "CLE", "PIT", "CIN", "BAL", "KC",
                "NYJ", "NYG", "PHI", "WAS", "DEN"}


def haversine(a, b) -> float:
    """Great-circle miles between (lat, lon) points."""
    R = 3958.8
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0]); dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def game_context(home: str, away: str, row) -> dict:
    """Per-team points deltas (added to each side's projected score) + notes for a game.
    `row` is a schedules.parquet row (dict-like)."""
    coords = _nfl_coords()
    notes, hd, ad = [], 0.0, 0.0
    loc = str(row.get("location", "Home") or "Home")
    roof = str(row.get("roof", "") or "").lower()
    stadium = row.get("stadium", "") or ""
    neutral = loc.lower() != "home"

    if neutral:
        notes.append(f"Neutral site — {stadium}" if stadium else "Neutral site")
        venue = INTL_VENUES.get(stadium)
        if venue and home in coords and away in coords:
            hdist = haversine(coords[home][:2], venue)
            adist = haversine(coords[away][:2], venue)
            diff = max(-1.0, min(1.0, (adist - hdist) / 3000.0 * 0.6))  # closer team a touch fresher
            hd += diff; ad -= diff
            notes.append(f"Travel: {home} {hdist:,.0f} mi · {away} {adist:,.0f} mi")
    else:
        # domestic: only the visitor travels (to the home city); east-ward trips hurt more
        if home in coords and away in coords:
            hc, ac = coords[home][:2], coords[away][:2]
            dist = haversine(ac, hc)
            # eastward travel (venue east of the visitor's home) hurts the body clock most
            tz_east = max(0.0, (hc[1] - ac[1]) / 15.0)
            pen = min(1.5, dist / 2500.0 * 0.8 + tz_east * 0.3)
            if pen > 0.15:
                ad -= pen
                notes.append(f"{away} travels {dist:,.0f} mi" + (f", {tz_east:.0f} TZ east" if tz_east >= 1 else ""))
        if home == "DEN":                                              # thin air tires the visitor
            ad -= 0.4; notes.append("Altitude (Denver)")

    # weather (suppresses both teams' scoring)
    temp, wind = row.get("temp"), row.get("wind")
    is_dome = roof in ("dome", "closed") or (not neutral and home in coords and coords[home][2])
    wpts = 0.0
    if is_dome:
        pass
    elif pd.notna(temp) or pd.notna(wind):
        t = float(temp) if pd.notna(temp) else 60.0
        w = float(wind) if pd.notna(wind) else 0.0
        if t < 32:
            wpts += 1.5 if t < 20 else 1.0; notes.append(f"Cold {t:.0f}°F")
        if w >= 15:
            wpts += 1.5 if w >= 22 else 1.0; notes.append(f"Wind {w:.0f} mph")
    else:
        wk = int(row.get("week", 1) or 1)
        if roof == "outdoors" and wk >= 14 and home in COLD_OUTDOOR:
            wpts += 0.8; notes.append("Likely cold-weather game")
    hd -= wpts; ad -= wpts

    return {"neutral": neutral, "home_delta": round(hd, 1), "away_delta": round(ad, 1),
            "notes": notes, "stadium": stadium}
