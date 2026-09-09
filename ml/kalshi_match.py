"""
ml/kalshi_match.py  —  find the Kalshi markets for one of our games, or refuse to guess
=====================================================================================
Inputs : a scheduled game (home, away, gameday) from schedules.parquet
Outputs: the event and its markets — winner (one contract per team), spread strikes
         ("KC wins by over 6.5 points"), totals ("over 47.5 points scored")

Getting this wrong spends real money on the wrong contract, so the module fails closed and
every refusal carries a reason. Verified against live data on 2026-09-09:

    KXNFLGAME-26SEP21NYGLAR-NYG     "New York G wins"          yes_sub_title "New York G"
    KXNFLGAME-26SEP21NYGLAR-LAR     "Los Angeles R wins"
    KXNFLSPREAD-26SEP14DENKC-KC8    "Kansas City wins by over 7.5 points?"  floor_strike 7.5
    KXNFLTOTAL-26SEP14DENKC-64      "Will there be over 63.5 points scored?" floor_strike 63.5

The event ticker is SERIES-YYMONDD + AWAY code + HOME code. The market suffix is the team's
Kalshi code (winner), the code plus a strike index (spread), or a strike index (total).
The strike itself is read from `floor_strike`, never parsed out of the ticker.

Three rules carried over from the tennis port:
1. EXACT series tickers only. Kalshi has dozens of sibling NFL series (KXNFL1H, KXNFL4Q,
   KXNFL2QSPREAD, KXNFL1HTOTAL, KXNFLWINS-…) that a prefix match would sweep in.
2. BOTH teams must resolve inside the SAME event, and the event's date must be the game's
   date (±1 day for the UTC/ET boundary). Teams play once a week, so this is unambiguous.
3. Kalshi's team codes differ from nflverse's (LAR vs LA). The map below lists alternates;
   a wrong guess simply fails to match and the game is skipped with a reason.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from ml import kalshi

SERIES = {"winner": "KXNFLGAME", "spread": "KXNFLSPREAD", "total": "KXNFLTOTAL"}

# nflverse abbreviation → Kalshi codes seen or plausible. First entry is the expected one.
TEAM_CODES = {
    "ARI": ["ARI", "AZ"], "ATL": ["ATL"], "BAL": ["BAL"], "BUF": ["BUF"], "CAR": ["CAR"],
    "CHI": ["CHI"], "CIN": ["CIN"], "CLE": ["CLE"], "DAL": ["DAL"], "DEN": ["DEN"],
    "DET": ["DET"], "GB": ["GB", "GNB"], "HOU": ["HOU"], "IND": ["IND"], "JAX": ["JAX", "JAC"],
    "KC": ["KC", "KAN"], "LA": ["LAR", "LA"], "LAC": ["LAC"], "LV": ["LV", "LVR"],
    "MIA": ["MIA"], "MIN": ["MIN"], "NE": ["NE", "NEP"], "NO": ["NO", "NOS"],
    "NYG": ["NYG"], "NYJ": ["NYJ"], "PHI": ["PHI"], "PIT": ["PIT"], "SEA": ["SEA"],
    "SF": ["SF", "SFO"], "TB": ["TB", "TAM"], "TEN": ["TEN"], "WAS": ["WAS", "WSH"],
}
_MON = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL",
                                    "AUG", "SEP", "OCT", "NOV", "DEC"), start=1)}


def event_date(event_ticker: str) -> date | None:
    """'KXNFLGAME-26SEP21NYGLAR' → 2026-09-21. None if the shape is unfamiliar."""
    m = re.match(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})[A-Z]", str(event_ticker or ""))
    if not m or m.group(2) not in _MON:
        return None
    try:
        return date(2000 + int(m.group(1)), _MON[m.group(2)], int(m.group(3)))
    except ValueError:
        return None


def event_codes(event_ticker: str) -> str:
    """The team-code block after the date: 'KXNFLGAME-26SEP21NYGLAR' → 'NYGLAR'."""
    m = re.match(r"^[A-Z0-9]+-\d{2}[A-Z]{3}\d{2}([A-Z]+)$", str(event_ticker or ""))
    return m.group(1) if m else ""


def market_suffix(ticker: str) -> str:
    """Everything after the event ticker: 'KXNFLSPREAD-26SEP14DENKC-KC8' → 'KC8'."""
    parts = str(ticker or "").split("-")
    return parts[2] if len(parts) >= 3 else ""


def open_events(kind: str) -> dict:
    """Open markets of one series grouped by event ticker (market status re-checked)."""
    series = SERIES.get(kind)
    if not series:
        return {}
    events: dict[str, list] = {}
    for m in kalshi.markets(limit=1000, status="open", series=series):
        # Kalshi's status=open filter is not exact; trust the market's own status.
        if str(m.get("status") or "active").lower() != "active":
            continue
        ev = m.get("event_ticker")
        # exact series only: the event ticker must START with the series and a dash
        if ev and str(ev).startswith(series + "-"):
            events.setdefault(ev, []).append(m)
    return events


def find_event(home: str, away: str, gameday: str, events: dict) -> dict:
    """
    The event for a scheduled game, or {"ok": False, "reason"}.

    Both teams must appear in the event's code block (either order — Kalshi lists
    away+home, but the order is not something to bet on) and the event date must match
    the game date within a day.
    """
    if not events:
        return {"ok": False, "reason": "no open markets in that series"}
    try:
        gd = datetime.strptime(str(gameday), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return {"ok": False, "reason": "game has no date"}
    hc, ac = TEAM_CODES.get(home, [home]), TEAM_CODES.get(away, [away])
    hits = []
    for ev, ms in events.items():
        ed = event_date(ev)
        if ed is None or abs((ed - gd).days) > 1:
            continue
        codes = event_codes(ev)
        h = next((c for c in hc if codes in (c + a for a in ac) or codes in (a + c for a in ac)), None)
        if h is None:
            continue
        a = next((c for c in ac if codes in (h + c, c + h)), None)
        hits.append((ev, ms, h, a))
    if len(hits) == 1:
        ev, ms, h, a = hits[0]
        return {"ok": True, "event": ev, "markets": ms, "home_code": h, "away_code": a}
    if len(hits) > 1:
        return {"ok": False, "reason": f"{len(hits)} events match {away}@{home} on {gameday} — ambiguous"}
    return {"ok": False, "reason": "no Kalshi market for this game"}


def winner_markets(found: dict) -> dict:
    """{'home': market, 'away': market} from a KXNFLGAME event (exactly two contracts)."""
    ms = found.get("markets") or []
    by = {market_suffix(m.get("ticker")): m for m in ms}
    h, a = by.get(found.get("home_code")), by.get(found.get("away_code"))
    if len(ms) != 2 or h is None or a is None:
        return {}
    return {"home": h, "away": a}


def spread_markets(found: dict) -> list:
    """
    [(side, strike, market)] from a KXNFLSPREAD event. side is 'home'/'away' (the team that
    must win by more than `strike`); strike from floor_strike.
    """
    out = []
    for m in found.get("markets") or []:
        suf = market_suffix(m.get("ticker"))
        code = re.sub(r"\d+$", "", suf)
        strike = _num(m.get("floor_strike"))
        if strike is None or not code:
            continue
        if code == found.get("home_code"):
            out.append(("home", strike, m))
        elif code == found.get("away_code"):
            out.append(("away", strike, m))
    return out


def total_markets(found: dict) -> list:
    """[(strike, market)] from a KXNFLTOTAL event: YES = over `strike` total points."""
    out = []
    for m in found.get("markets") or []:
        strike = _num(m.get("floor_strike"))
        if strike is not None:
            out.append((strike, m))
    return out


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ask_price(market: dict) -> float | None:
    """
    The price a taker pays to buy YES, in dollars. Reads `yes_ask_dollars` FIRST (the V2
    fixed-point string); the legacy cent field is the fallback. No ask means nothing to buy.
    """
    p = _num(market.get("yes_ask_dollars"))
    if p is None:
        cents = _num(market.get("yes_ask"))
        p = cents / 100.0 if cents is not None else None
    if p is None:
        return None
    return p if 0.0 < p < 1.0 else None


def ask_size(market: dict) -> float | None:
    """How many contracts are actually offered at the ask."""
    return _num(market.get("yes_ask_size_fp"))


def sport_of(ticker: str) -> str:
    """Series → readable market type; unknown series degrade to themselves, never 'other'."""
    series = str(ticker or "").split("-")[0]
    known = {v: k for k, v in SERIES.items()}
    if series in known:
        return "NFL " + known[series]
    return (series[2:] if series.startswith("KX") else series) or "other"
