"""
ml/weather.py  —  game weather: the kickoff-window forecast for upcoming games, the archive
for played ones, and what it does to a passing game
======================================================================================
Until 2026-11-02 the only weather the model ever saw was the nflverse schedule's temp/wind
pair, which is EMPTY for every unplayed game (0 of 152 this season) and filled after the fact
for fewer than half of the played ones; precipitation existed nowhere. Falcons at Buccaneers,
2026 week 8: rain every hour of the 1pm window and gusts to 33 mph, Mayfield 0 completions,
the model's props said 200 passing yards. This module closes that gap with Open-Meteo (free,
no key, plain HTTPS — the same kind of call the refresh already makes for nflverse):

  fetch_forecast(season)   the kickoff-window forecast (kickoff hour + 3) for every unplayed
                           outdoor game inside the forecast horizon → data/raw/weather_forecast.parquet
                           (refresh-managed on the server volume, re-pulled by every light and
                           full refresh, rewritten only when a rounded value moved so the
                           cached slate is not invalidated for nothing)
  fetch_archive(seasons)   the same window from the historical archive for every played game →
                           data/raw/weather.parquet (the measurement table; the old daily pull
                           stored 0.0 precipitation for all 929 games — wrong endpoint)
  game_weather(row)        {temp, wind, gust, precip_in, precip_prob, source} for a schedule row:
                           nothing for a dome/closed roof; the forecast before kickoff; the
                           archive / nflverse row after
  pass_multipliers(wx)     what the weather does to a passing game (see the table)

Everything downstream reads through game_weather so the Schedule & Picks card, the Matchup
page, the box score and the props board all see ONE weather for a game.

Conventions: temperature °F, wind and gusts mph, precipitation inches over the window,
precipitation probability 0-100. Kickoff times in the schedule are US/Eastern, so every
request uses America/New_York and the window is matched on the ET hour whatever the venue.
"""

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
FORECAST = RAW / "weather_forecast.parquet"
ARCHIVE = RAW / "weather.parquet"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation,snowfall"
SNOW_TEMP = 34.0          # precipitation at or below this is falling as snow (the archive has no snowfall column)
UNITS = {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
         "timezone": "America/New_York"}
WINDOW_HOURS = 3          # kickoff hour + the next three (a game is ~3 h 15 m)
HORIZON_DAYS = 8          # Open-Meteo forecasts 16 days; past ~8 the skill is weak — pull the week ahead
OUTDOOR = {"outdoors", "open"}          # schedule roof values that leave a game exposed

_FC = None                 # {game_id: row dict} cache of the forecast file
_AR = None                 # {game_id: row dict} cache of the archive file


def clear():
    global _FC, _AR
    _FC = _AR = None


def _coords():
    """team -> (lat, lon, is_dome) from the fetcher's table, plus the international venues."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fetch_data import STADIUM_COORDS
    return STADIUM_COORDS


def _venue(row) -> tuple | None:
    """(lat, lon) the game is played at: a listed international/neutral stadium, else the home
    team's. None when unknown."""
    from ml.context import INTL_VENUES
    stadium = str(row.get("stadium") or "")
    if stadium in INTL_VENUES:
        return INTL_VENUES[stadium]
    c = _coords().get(row.get("home_team"))
    return (c[0], c[1]) if c else None


def _kick_hour(gametime) -> int:
    try:
        return int(str(gametime)[:2])
    except Exception:
        return 13


def _window(hourly: dict, gameday: str, hour: int) -> dict | None:
    """Aggregate the hourly series over the kickoff window on `gameday`."""
    times = hourly.get("time") or []
    idx = [i for i, t in enumerate(times) if t[:10] == str(gameday)[:10] and hour <= int(t[11:13]) < hour + WINDOW_HOURS + 1]
    if not idx:
        return None

    def vals(key):
        v = hourly.get(key) or []
        return [v[i] for i in idx if i < len(v) and v[i] is not None]
    t, w, g, p, pp, sn = (vals("temperature_2m"), vals("wind_speed_10m"), vals("wind_gusts_10m"),
                          vals("precipitation"), vals("precipitation_probability"), vals("snowfall"))
    if not w:
        return None
    return {"temp": round(sum(t) / len(t), 0) if t else None,
            "wind": round(sum(w) / len(w), 0),
            "gust": round(max(g), 0) if g else None,
            "precip_in": round(sum(p), 2) if p else 0.0,
            "precip_prob": round(max(pp), 0) if pp else None,
            "snow_in": round(sum(sn), 1) if sn else None}


def fetch_forecast(season: int, log=print, horizon_days: int = HORIZON_DAYS) -> dict:
    """Pull the kickoff-window forecast for every unplayed outdoor game of `season` that kicks
    off within `horizon_days`. Writes FORECAST only when a rounded value changed. Returns a
    summary dict for the refresh status."""
    import requests
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[(s["season"] == season) & s["home_score"].isna() & s["gameday"].notna()].copy()
    today = datetime.now(timezone.utc).date()
    s["_day"] = pd.to_datetime(s["gameday"]).dt.date
    s = s[(s["_day"] >= today) & ((s["_day"] - today).map(lambda d: d.days) <= horizon_days)]
    old = pd.read_parquet(FORECAST) if FORECAST.exists() else pd.DataFrame()
    rows, failed = [], 0
    for _, g in s.iterrows():
        roof = str(g.get("roof") or "").lower()
        if roof and roof not in OUTDOOR:
            continue                                         # dome / closed: no weather
        venue = _venue(g)
        if venue is None:
            continue
        day = str(g["gameday"])[:10]
        try:
            r = requests.get(FORECAST_URL, params={"latitude": venue[0], "longitude": venue[1],
                                                   "start_date": day, "end_date": day,
                                                   "hourly": HOURLY + ",precipitation_probability", **UNITS}, timeout=20)
            r.raise_for_status()
            wx = _window(r.json().get("hourly", {}), day, _kick_hour(g.get("gametime")))
        except Exception as e:
            failed += 1
            log(f"  forecast failed {g['game_id']}: {str(e)[:80]}", "WARN")
            continue
        if wx is None:
            continue
        rows.append({"game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
                     "home_team": g["home_team"], "away_team": g["away_team"], "gameday": day,
                     "gametime": g.get("gametime"), "stadium": g.get("stadium"), **wx})
        time.sleep(0.05)
    new = pd.DataFrame(rows)
    changed = True
    if len(old) and len(new):
        keep = [c for c in ("game_id", "temp", "wind", "gust", "precip_in", "precip_prob") if c in old.columns and c in new.columns]
        a = old[keep].sort_values("game_id").reset_index(drop=True)
        b = new[keep].sort_values("game_id").reset_index(drop=True)
        changed = not a.equals(b)
    if len(new) and changed:
        new["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new.to_parquet(FORECAST, index=False)
        clear()
    log(f"  weather forecast: {len(new)} outdoor games within {horizon_days} days"
        + (f", {failed} failed" if failed else "") + ("" if changed else " (unchanged)"))
    return {"games": len(new), "failed": failed, "changed": bool(changed)}


def fetch_archive(seasons: list, log=print, sleep: float = 0.12) -> dict:
    """Rebuild ARCHIVE: the kickoff-window weather of every played regular-season game of
    `seasons` from Open-Meteo's historical archive (domes get a fixed indoor row)."""
    import requests
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["season"].isin(seasons) & s["home_score"].notna() & (s["game_type"].astype(str).str.upper() == "REG")]
    rows, failed = [], 0
    for i, (_, g) in enumerate(s.iterrows()):
        roof = str(g.get("roof") or "").lower()
        base = {"game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
                "home_team": g["home_team"], "away_team": g["away_team"], "gameday": str(g["gameday"])[:10],
                "roof": roof, "surface": g.get("surface")}
        if roof and roof not in OUTDOOR:
            rows.append({**base, "is_dome": True, "temp": 70.0, "wind": 0.0, "gust": 0.0, "precip_in": 0.0, "precip_prob": None})
            continue
        venue = _venue(g)
        if venue is None:
            continue
        day = base["gameday"]
        try:
            r = requests.get(ARCHIVE_URL, params={"latitude": venue[0], "longitude": venue[1],
                                                  "start_date": day, "end_date": day, "hourly": HOURLY, **UNITS}, timeout=20)
            r.raise_for_status()
            wx = _window(r.json().get("hourly", {}), day, _kick_hour(g.get("gametime")))
        except Exception as e:
            failed += 1
            log(f"  archive failed {g['game_id']}: {str(e)[:80]}", "WARN")
            continue
        if wx is None:
            continue
        rows.append({**base, "is_dome": False, **wx})
        if i % 100 == 0:
            log(f"  archive: {i}/{len(s)} games")
        time.sleep(sleep)
    df = pd.DataFrame(rows)
    # keep the old column names the measurement scripts used
    df["temp_f"], df["wind_mph"] = df["temp"], df["wind"]
    df.to_parquet(ARCHIVE, index=False)
    clear()
    log(f"  weather archive: {len(df)} games ({failed} failed) -> {ARCHIVE.name}")
    return {"games": len(df), "failed": failed}


def _table(path: Path, which: str) -> dict:
    global _FC, _AR
    cur = _FC if which == "fc" else _AR
    if cur is None:
        cur = {}
        if path.exists():
            try:
                d = pd.read_parquet(path)
                cur = {r["game_id"]: r for r in d.to_dict("records")}
            except Exception:
                cur = {}
        if which == "fc":
            _FC = cur
        else:
            _AR = cur
    return cur


def game_weather(row) -> dict | None:
    """One weather for a schedule row (dict-like). None for a dome/closed roof or when nothing
    is known. Keys: temp, wind, gust, precip_in, precip_prob, source ('forecast' | 'archive'
    | 'nflverse')."""
    roof = str(row.get("roof") or "").lower()
    if roof and roof not in OUTDOOR:
        return None
    gid = row.get("game_id")
    played = not (row.get("home_score") is None or (isinstance(row.get("home_score"), float) and math.isnan(row.get("home_score"))))
    ar = _table(ARCHIVE, "ar").get(gid) if played else None
    if ar is not None and ar.get("wind") is not None and not (isinstance(ar.get("wind"), float) and math.isnan(ar.get("wind"))):
        return {"temp": ar.get("temp"), "wind": ar.get("wind"), "gust": ar.get("gust"),
                "precip_in": ar.get("precip_in"), "precip_prob": None, "snow_in": ar.get("snow_in"), "source": "archive"}
    temp, wind = row.get("temp"), row.get("wind")
    if played and (pd.notna(temp) or pd.notna(wind)):
        return {"temp": None if pd.isna(temp) else float(temp), "wind": None if pd.isna(wind) else float(wind),
                "gust": None, "precip_in": None, "precip_prob": None, "source": "nflverse"}
    fc = _table(FORECAST, "fc").get(gid)
    if fc is not None:
        return {"temp": fc.get("temp"), "wind": fc.get("wind"), "gust": fc.get("gust"),
                "precip_in": fc.get("precip_in"), "precip_prob": fc.get("precip_prob"),
                "snow_in": fc.get("snow_in"), "source": "forecast"}
    return None


def _num(v):
    """float or None (NaN-safe)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def is_snow(wx: dict | None) -> bool:
    """Precipitation falling as snow: the forecast's snowfall column, else precipitation at
    SNOW_TEMP or below (the archive carries no snowfall column)."""
    if not wx:
        return False
    sn = _num(wx.get("snow_in"))
    if sn is not None:
        return sn >= 0.1
    p, t = _num(wx.get("precip_in")), _num(wx.get("temp"))
    return bool(p and p >= 0.05 and t is not None and t <= SNOW_TEMP)


# ── what weather does to a passing game ───────────────────────────────
# Measured 2026-11-02 on 927 outdoor regular-season games 2021-25: the KICKOFF-WINDOW weather
# from fetch_archive (the same 3-hour mean the forecast reports, so the thresholds below mean
# the same thing before and after a game) against the play-by-play box score. Each factor is
# measured with the others held out (rain/cold rows at wind < 15, wind rows on dry games),
# relative to dry, calm (< 10 mph), above-40°F games (n = 631: 71.5 att, 60.5% cmp, 6.13 ypa,
# 438 pass yds, 3.10 pass TD, 204 rush yds):
#
#   factor          n    att    cmp%    yds/att  pass yds  pass TD  rush yds  total vs line
#   wind 10-15    207    -2%   -1.1pt    -4%      -5%      -8%      -1%      -1.5 (over 39%)
#   wind 15-20     45    -2%   -0.3pt     0%      -2%     -15%      -3%      -0.8
#   wind 20+        9   -18%   -9.8pt   -26%     -39%     -30%     +13%      -2.9
#   gusts 30+      29    -7%   -1.2pt    -6%     -11%      -3%      +9%      -3.9 (sustained < 15)
#   rain .05-.15   54    -3%   -1.1pt    -6%      -8%     -15%     +10%      -1.5 (over 35%)
#   rain .15-.35   20    +1%   -1.7pt    -3%      -3%      +3%      -5%      -2.8
#   rain .35+      11    +2%   -3.5pt   -12%      -8%       0%      -6%      -0.7 (fumbles 4.1 vs 2.2)
#   temp 32-40     72    -2%   -1.6pt    -2%      -5%      -7%     +10%      +0.7
#   temp 20-32     42    -6%   -2.0pt    -1%      -7%     +13%     +11%      +2.9
#   temp < 20      14    -3%    0.0pt    -9%     -11%      -5%      -5%      -7.3
#   snow            2    unmeasurable — a snow game gets the RAIN row (precipitation) and the
#                       COLD row (temperature) compounded; no separate snow term until there is data
#
# The tables are the measured numbers rounded toward "do less": the 20+ mph and under-20°F
# rows are 9 and 14 games, so they are shrunk, and every table is monotone. Rushing is left
# alone (it moves up in the cold and the rain, by less than its own noise). The closing total
# already prices some of this (the "total vs line" column: the market moves, not all the way),
# and the slate's score total is anchored to the market — so on the BOARD weather is a note;
# the props and the box score are where the model's own number bends, and that is where these
# are applied (ml.projections._apply_weather).
WIND = [   # (min sustained mph, attempts, completions, yards, touchdowns)
    (20.0, 0.85, 0.85, 0.70, 0.75),
    (15.0, 0.96, 0.97, 0.92, 0.85),
    (10.0, 0.98, 0.98, 0.95, 0.92),
]
GUST = (30.0, 0.93, 0.98, 0.90, 0.97)      # gusts at/above this with sustained wind under 15
RAIN = [   # (min inches over the window, attempts, completions, yards, touchdowns)
    (0.35, 0.98, 0.94, 0.90, 0.88),
    (0.05, 0.98, 0.97, 0.93, 0.88),
]
SNOW = RAIN                                # see the table: no separate snow measurement yet
COLD = [   # (max °F, attempts, completions, yards, touchdowns) — coldest bucket first
    (20.0, 0.95, 0.97, 0.89, 0.95),
    (32.0, 0.95, 0.97, 0.93, 0.95),
    (40.0, 0.98, 0.97, 0.95, 0.95),
]
RAIN_PROB_MIN = 50      # a forecast only counts as rain/snow when the window probability is at least this


def _apply(m: dict, table: list, value: float, note: str, descending: bool = True) -> None:
    for lo, a, c, y, t in table:
        if (value >= lo) if descending else (value <= lo):
            m["att"] *= a; m["cmp"] *= c; m["yds"] *= y; m["td"] *= t
            m["notes"].append(note)
            break


def pass_multipliers(wx: dict | None) -> dict:
    """{att, cmp, yds, td} multipliers for a passing game under this weather (1.0 = none).
    Wind, precipitation (rain or snow) and cold compound — a windy snowstorm is worse than
    any one of them."""
    m = {"att": 1.0, "cmp": 1.0, "yds": 1.0, "td": 1.0, "notes": []}
    if not wx:
        return m
    wind, gust = _num(wx.get("wind")), _num(wx.get("gust"))
    if wind is not None:
        _apply(m, WIND, wind, f"wind {wind:.0f} mph")
        if wind < 15 and gust is not None and gust >= GUST[0]:
            _apply(m, [GUST], gust, f"gusts {gust:.0f} mph")
    rain, prob, temp = _num(wx.get("precip_in")), _num(wx.get("precip_prob")), _num(wx.get("temp"))
    sure = prob is None or prob >= RAIN_PROB_MIN
    if rain is not None and sure:
        tag = f" ({prob:.0f}%)" if prob is not None else ""
        if is_snow(wx):
            _apply(m, SNOW, rain, f"snow {rain:.2f} in" + tag)
        else:
            _apply(m, RAIN, rain, f"rain {rain:.2f} in" + tag)
    if temp is not None:
        _apply(m, COLD, temp, f"{temp:.0f}°F", descending=False)
    return m


def describe(wx: dict | None) -> str | None:
    """Short card text: 'Forecast: 54°F, wind 17 mph (gusts 29), rain 0.20 in (70%)'."""
    if not wx:
        return None
    parts = []
    if wx.get("temp") is not None and not (isinstance(wx.get("temp"), float) and math.isnan(wx.get("temp"))):
        parts.append(f"{float(wx['temp']):.0f}°F")
    if wx.get("wind") is not None and not (isinstance(wx.get("wind"), float) and math.isnan(wx.get("wind"))):
        g = wx.get("gust")
        parts.append(f"wind {float(wx['wind']):.0f} mph" + (f" (gusts {float(g):.0f})" if g is not None and not (isinstance(g, float) and math.isnan(g)) and float(g) >= float(wx["wind"]) + 8 else ""))
    r, p = _num(wx.get("precip_in")), _num(wx.get("precip_prob"))
    if r is not None and r >= 0.05:
        parts.append(("snow" if is_snow(wx) else "rain") + f" {r:.2f} in" + (f" ({p:.0f}%)" if p is not None else ""))
    if not parts:
        return None
    label = {"forecast": "Forecast", "archive": "Weather", "nflverse": "Weather"}.get(wx.get("source"), "Weather")
    return f"{label}: " + ", ".join(parts)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Open-Meteo game weather")
    ap.add_argument("--forecast", type=int, metavar="SEASON", help="pull the week-ahead kickoff forecasts")
    ap.add_argument("--archive", nargs="+", type=int, metavar="SEASON", help="rebuild the historical table")
    a = ap.parse_args()
    if a.forecast:
        print(fetch_forecast(a.forecast, lambda m, *x: print(m)))
    if a.archive:
        print(fetch_archive(a.archive, lambda m, *x: print(m)))
