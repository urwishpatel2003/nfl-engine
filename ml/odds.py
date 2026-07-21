"""
ml/odds.py  —  live player-prop lines from The Odds API (the-odds-api.com)
==========================================================================
Reads ODDS_API_KEY from the environment (NEVER hardcoded / never in the repo). Fetches player
props for a single NFL event and returns a {(player_key, our_market): {...}} map to merge onto
the model projections in ml.props.

Credit-aware by design:
  • only called when the frontend explicitly asks (a "Load live lines" button), not on page load
  • cached per event for _TTL seconds so repeated views don't re-spend credits
  • one request per event pulls all the markets we use at once
The Odds API returns remaining credits in the `x-requests-remaining` header — surfaced in `meta`.

Offseason reality: player props are only posted close to game day, so `event_props` will often
return an empty map with a friendly status until the season is underway. Callers degrade to
model-only.
"""

import os
import re
import statistics as st
import time

import requests

_BASE = "https://api.the-odds-api.com/v4"
_SPORT = "americanfootball_nfl"

# our market key (ml.props) → The Odds API market key
_MKT = {
    "pass_yds": "player_pass_yds", "pass_td": "player_pass_tds", "pass_att": "player_pass_attempts",
    "cmp": "player_pass_completions", "int": "player_pass_interceptions",
    "rush_yds": "player_rush_yds", "carries": "player_rush_attempts",
    "rec": "player_receptions", "rec_yds": "player_reception_yds",
    "rushrec_yds": "player_rush_reception_yds", "anytime_td": "player_anytime_td",
}
_REV = {v: k for k, v in _MKT.items()}

_TTL = 600
_GAME_TTL = 900     # game spreads/totals move slower pre-game; 15-min cache keeps credits sane
_EVENTS = {"t": 0.0, "data": None}
_PROPS = {}         # event_id -> {"t":, "data":, "meta":}
_GAMES = {"t": 0.0, "data": None, "meta": None}


def have_key() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def _namekey(s) -> str:
    k = re.sub(r"[^a-z]", "", str(s).lower())
    return re.sub(r"(iii|iv|ii|jr|sr)$", "", k)


def _get(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def clear():
    global _EVENTS, _PROPS, _GAMES
    _EVENTS = {"t": 0.0, "data": None}
    _PROPS = {}
    _GAMES = {"t": 0.0, "data": None, "meta": None}


def game_lines():
    """Consensus spread + total for every currently-posted NFL game, in ONE bulk request. Returns
    ({(home_key, away_key): {spread, total}}, meta). Spread is converted to nflverse convention
    (POSITIVE = home favored = negative of the home team's odds-API point). Cached _GAME_TTL."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}, {"error": "ODDS_API_KEY not set on the server"}
    if _GAMES["data"] is not None and time.time() - _GAMES["t"] < _GAME_TTL:
        return _GAMES["data"], _GAMES["meta"]
    url = (f"{_BASE}/sports/{_SPORT}/odds?apiKey={key}"
           f"&regions=us&oddsFormat=american&markets=spreads,totals")
    try:
        data, remaining = _get(url)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        return {}, {"error": f"odds fetch failed (HTTP {code})"}
    except Exception as e:
        return {}, {"error": f"odds fetch failed: {str(e)[:100]}"}

    out = {}
    for ev in data:
        home, away = ev.get("home_team"), ev.get("away_team")
        if not home or not away:
            continue
        hk = _namekey(home)
        spreads, totals = [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "spreads":
                    for oc in mk.get("outcomes", []):
                        if _namekey(oc.get("name")) == hk and oc.get("point") is not None:
                            spreads.append(-float(oc["point"]))     # -> nflverse home-favored sign
                elif mk.get("key") == "totals":
                    pt = next((o.get("point") for o in mk.get("outcomes", []) if o.get("point") is not None), None)
                    if pt is not None:
                        totals.append(float(pt))
        rec = {}
        if spreads:
            rec["spread"] = round(st.median(spreads), 1)
        if totals:
            rec["total"] = round(st.median(totals), 1)
        if rec:
            out[(hk, _namekey(away))] = rec
    meta = {"remaining_credits": remaining, "games": len(out)}
    _GAMES.update(t=time.time(), data=out, meta=meta)
    return out, meta


def events():
    """Upcoming NFL events (id, home_team, away_team, commence_time). Cached."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    if _EVENTS["data"] is not None and time.time() - _EVENTS["t"] < _TTL:
        return _EVENTS["data"]
    try:
        data, _ = _get(f"{_BASE}/sports/{_SPORT}/events?apiKey={key}")
    except Exception:
        data = []
    _EVENTS.update(t=time.time(), data=data)
    return data


def _find_event(home_full, away_full):
    hk, ak = _namekey(home_full), _namekey(away_full)
    for e in events():
        if {_namekey(e.get("home_team")), _namekey(e.get("away_team"))} == {hk, ak}:
            return e
    return None


def event_props(home_full, away_full):
    """(props_map, meta) for a matchup. props_map[(player_key, our_market)] = for over/under
    markets {line, over_odds, under_odds, books}; for anytime_td {yes_odds}. Consensus across
    books = median line, median price at that line. Empty map + status if no key/event."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}, {"error": "ODDS_API_KEY not set on the server"}
    ev = _find_event(home_full, away_full)
    if not ev:
        return {}, {"error": "no live event for this matchup yet (player props post near game day)"}
    eid = ev["id"]
    c = _PROPS.get(eid)
    if c and time.time() - c["t"] < _TTL:
        return c["data"], c["meta"]
    markets = ",".join(_MKT.values())
    url = (f"{_BASE}/sports/{_SPORT}/events/{eid}/odds?apiKey={key}"
           f"&regions=us&oddsFormat=american&markets={markets}")
    try:
        data, remaining = _get(url)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        msg = "player props not on your plan/region" if code == 422 else f"HTTP {code}"
        return {}, {"error": f"odds fetch failed ({msg})"}
    except Exception as e:
        return {}, {"error": f"odds fetch failed: {str(e)[:100]}"}

    agg = {}
    for bk in data.get("bookmakers", []):
        for mk in bk.get("markets", []):
            ourm = _REV.get(mk.get("key"))
            if not ourm:
                continue
            for oc in mk.get("outcomes", []):
                if ourm == "anytime_td":
                    pk = (_namekey(oc.get("description") or oc.get("name")), ourm)
                    agg.setdefault(pk, {"yes": []})["yes"].append(oc.get("price"))
                else:
                    player, pt, price = oc.get("description"), oc.get("point"), oc.get("price")
                    side = (oc.get("name") or "").lower()
                    if player is None or pt is None:
                        continue
                    agg.setdefault((_namekey(player), ourm), {"lines": []})["lines"].append((pt, side, price))

    out = {}
    for pk, d in agg.items():
        if pk[1] == "anytime_td":
            ys = [x for x in d.get("yes", []) if x is not None]
            if ys:
                out[pk] = {"yes_odds": int(st.median(ys))}
        else:
            lines = d["lines"]
            pts = [l[0] for l in lines]
            if not pts:
                continue
            med = float(st.median(pts))
            overs = [l[2] for l in lines if l[1] == "over" and abs(l[0] - med) < 1e-6 and l[2] is not None]
            unders = [l[2] for l in lines if l[1] == "under" and abs(l[0] - med) < 1e-6 and l[2] is not None]
            out[pk] = {"line": med,
                       "over_odds": int(st.median(overs)) if overs else None,
                       "under_odds": int(st.median(unders)) if unders else None,
                       "books": len({l[0] for l in lines})}
    meta = {"event_id": eid, "commence": ev.get("commence_time"),
            "remaining_credits": remaining, "markets": len(out)}
    _PROPS[eid] = {"t": time.time(), "data": out, "meta": meta}
    return out, meta
