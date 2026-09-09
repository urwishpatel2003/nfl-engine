"""
Tests for matching our schedule to Kalshi NFL markets (ml/kalshi_match.py).

    python tests/test_kalshi_match.py

Fixtures are shaped from live Kalshi data captured 2026-09-09. No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.kalshi_match import (SERIES, ask_price, ask_size, event_codes, event_date, find_event,  # noqa: E402
                             market_suffix, sport_of, spread_markets, total_markets, winner_markets)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def mk(ticker, ev, ask="0.5000", size="100.00", strike=None, status="active"):
    m = {"ticker": ticker, "event_ticker": ev, "status": status, "yes_ask_dollars": ask, "yes_ask_size_fp": size}
    if strike is not None:
        m["floor_strike"] = strike
    return m


GAME = {"KXNFLGAME-26SEP21NYGLAR": [mk("KXNFLGAME-26SEP21NYGLAR-NYG", "KXNFLGAME-26SEP21NYGLAR", "0.1900"),
                                    mk("KXNFLGAME-26SEP21NYGLAR-LAR", "KXNFLGAME-26SEP21NYGLAR", "0.8200")],
        "KXNFLGAME-26SEP20INDKC": [mk("KXNFLGAME-26SEP20INDKC-KC", "KXNFLGAME-26SEP20INDKC", "0.7200"),
                                   mk("KXNFLGAME-26SEP20INDKC-IND", "KXNFLGAME-26SEP20INDKC", "0.3000")]}
SPREAD = {"KXNFLSPREAD-26SEP14DENKC": [mk("KXNFLSPREAD-26SEP14DENKC-KC8", "KXNFLSPREAD-26SEP14DENKC", "0.3100", strike=7.5),
                                       mk("KXNFLSPREAD-26SEP14DENKC-KC4", "KXNFLSPREAD-26SEP14DENKC", "0.5200", strike=3.5),
                                       mk("KXNFLSPREAD-26SEP14DENKC-DEN4", "KXNFLSPREAD-26SEP14DENKC", "0.2000", strike=3.5)]}
TOTAL = {"KXNFLTOTAL-26SEP14DENKC": [mk("KXNFLTOTAL-26SEP14DENKC-64", "KXNFLTOTAL-26SEP14DENKC", "0.0800", strike=63.5),
                                     mk("KXNFLTOTAL-26SEP14DENKC-48", "KXNFLTOTAL-26SEP14DENKC", "0.5000", strike=47.5)]}

print("\n1. the series are exact tickers")
check("winner", SERIES["winner"] == "KXNFLGAME")
check("spread", SERIES["spread"] == "KXNFLSPREAD")
check("total", SERIES["total"] == "KXNFLTOTAL")
check("a sibling series is not swept in", not "KXNFL1H".startswith(SERIES["winner"] + "-"))

print("\n2. ticker anatomy")
check("event date is read", str(event_date("KXNFLGAME-26SEP21NYGLAR")) == "2026-09-21")
check("event codes are read", event_codes("KXNFLGAME-26SEP21NYGLAR") == "NYGLAR")
check("market suffix is read", market_suffix("KXNFLSPREAD-26SEP14DENKC-KC8") == "KC8")
check("a strange shape yields nothing", event_date("KXNFLWINS-NO") is None and event_codes("junk") == "")

print("\n3. a game resolves to its own event, with Kalshi's team codes")
f = find_event("LA", "NYG", "2026-09-21", GAME)
check("Rams home vs Giants resolves", f["ok"], str(f))
check("home code is LAR, not LA", f.get("home_code") == "LAR", str(f))
w = winner_markets(f)
check("the winner contracts are split by side", w and w["home"]["ticker"].endswith("-LAR") and w["away"]["ticker"].endswith("-NYG"), str(w))
check("the date guards the match", not find_event("LA", "NYG", "2026-10-21", GAME)["ok"])
check("a game not on Kalshi is refused with a reason", not find_event("KC", "BAL", "2026-09-21", GAME)["ok"])
check("empty events are refused", not find_event("LA", "NYG", "2026-09-21", {})["ok"])
check("a bad date is refused", not find_event("LA", "NYG", None, GAME)["ok"])
two = find_event("KC", "IND", "2026-09-20", GAME)
check("KC vs IND resolves the other event", two["ok"] and two["event"].endswith("INDKC"), str(two))

print("\n4. spread strikes are attributed to the right team")
fs = find_event("KC", "DEN", "2026-09-14", SPREAD)
check("the spread event resolves", fs["ok"], str(fs))
sm = spread_markets(fs)
check("three strikes read", len(sm) == 3, str([(s, k) for s, k, _ in sm]))
check("KC strikes are the home side", all(s == "home" for s, k, m in sm if m["ticker"].split("-")[2].startswith("KC")))
check("DEN strikes are the away side", any(s == "away" and k == 3.5 for s, k, _ in sm))
check("strikes come from floor_strike, not the suffix", any(k == 7.5 for _, k, m in sm if m["ticker"].endswith("KC8")))

print("\n5. totals read their strikes")
ft = find_event("KC", "DEN", "2026-09-14", TOTAL)
tm = total_markets(ft)
check("two total strikes read", sorted(k for k, _ in tm) == [47.5, 63.5], str(tm))

print("\n6. prices and depth")
check("yes_ask_dollars is read", ask_price(mk("t", "e", "0.3700")) == 0.37)
check("the cent field is a fallback", ask_price({"yes_ask": 42}) == 0.42)
check("no ask means no price", ask_price({}) is None)
check("0 and 1 are not prices", ask_price({"yes_ask_dollars": "0.0000"}) is None and ask_price({"yes_ask_dollars": "1.0000"}) is None)
check("junk is not a price", ask_price({"yes_ask_dollars": "n/a"}) is None)
check("ask size is a float", ask_size(mk("t", "e", size="825.81")) == 825.81)

print("\n7. sport labels")
check("winner series", sport_of("KXNFLGAME-26SEP21NYGLAR-NYG") == "NFL winner")
check("spread series", sport_of("KXNFLSPREAD-26SEP14DENKC-KC8") == "NFL spread")
check("unknown series is itself, not 'other'", sport_of("KXWTAMATCH-26AUG27TAUPAR-PAR") == "WTAMATCH")

print(f"\n{'='*54}\n  {PASS} passed, {FAIL} failed\n{'='*54}\n")
sys.exit(1 if FAIL else 0)
