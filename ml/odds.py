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

_TTL = 1800         # player props: one request per EVENT pulls all 11 markets (~11 credits), and the
                    # props board asks for every game of the week at once, so a 30-min cache
                    # caps a full week at ~150 credits per half hour, not per page view
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


def _imp(odds) -> float:
    """American odds → implied probability (with the book's vig)."""
    o = float(odds)
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def _amer(p: float) -> int:
    """Implied probability → American odds (never inside the ±100 gap)."""
    p = min(max(float(p), 0.001), 0.999)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def consensus_price(prices) -> int | None:
    """The consensus of several books' American prices. American odds are not linear across
    ±100 (−110 and +100 are a hair apart, but their arithmetic mean is −5), so the median is
    taken in PROBABILITY space and converted back. Taking the median of the raw prices gave
    nonsense like −14 or −1 whenever books straddled the gap."""
    ps = [_imp(x) for x in prices if x is not None and abs(float(x)) >= 100]
    if not ps:
        return None
    return _amer(st.median(ps))


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

    # per book as well as the consensus: a single book sitting a point or two off the market
    # is a line-shopping edge in its own right, and the board names the best price per side
    agg = {}
    for bk in data.get("bookmakers", []):
        bname = bk.get("title") or bk.get("key") or "?"
        for mk in bk.get("markets", []):
            ourm = _REV.get(mk.get("key"))
            if not ourm:
                continue
            for oc in mk.get("outcomes", []):
                if ourm == "anytime_td":
                    pk = (_namekey(oc.get("description") or oc.get("name")), ourm)
                    a = agg.setdefault(pk, {"yes": [], "per_book": {}})
                    a["yes"].append(oc.get("price"))
                    if oc.get("price") is not None:
                        a["per_book"][bname] = {"yes": oc.get("price")}
                else:
                    player, pt, price = oc.get("description"), oc.get("point"), oc.get("price")
                    side = (oc.get("name") or "").lower()
                    if player is None or pt is None:
                        continue
                    a = agg.setdefault((_namekey(player), ourm), {"lines": [], "per_book": {}})
                    a["lines"].append((pt, side, price))
                    pb = a["per_book"].setdefault(bname, {"line": pt})
                    if side in ("over", "under") and price is not None:
                        pb["line"] = pt                       # the book's own line (over/under share it)
                        pb[side] = price

    out = {}
    for pk, d in agg.items():
        if pk[1] == "anytime_td":
            yo = consensus_price(d.get("yes", []))
            if yo is not None:
                out[pk] = {"yes_odds": yo, "per_book": d["per_book"]}
        else:
            lines = d["lines"]
            pts = [l[0] for l in lines]
            if not pts:
                continue
            med = float(st.median(pts))
            overs = [l[2] for l in lines if l[1] == "over" and abs(l[0] - med) < 1e-6 and l[2] is not None]
            unders = [l[2] for l in lines if l[1] == "under" and abs(l[0] - med) < 1e-6 and l[2] is not None]
            out[pk] = {"line": med,
                       "over_odds": consensus_price(overs),
                       "under_odds": consensus_price(unders),
                       "books": len([l for l in lines if l[1] == "over"]),
                       "per_book": d["per_book"]}
    meta = {"event_id": eid, "commence": ev.get("commence_time"),
            "remaining_credits": remaining, "markets": len(out)}
    _PROPS[eid] = {"t": time.time(), "data": out, "meta": meta}
    return out, meta
