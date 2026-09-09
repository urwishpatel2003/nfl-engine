"""
Tests for the freshness guard on the Kalshi order path (ml/kalshi_order.py).

    python tests/test_kalshi_order_guard.py

A ticket is a snapshot. Between building one and confirming it the price moves, the book
empties and the game kicks off. The start check uses OUR schedule's kickoff, passed in
explicitly — Kalshi keeps a game market ACTIVE through play, so status alone would let an
in-play ticket through. No network: every case passes a market snapshot and a fixed clock.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.kalshi_order import (MAX_PRICE_DRIFT, SELF_TRADE_PREVENTION, TIME_IN_FORCE_MAKER,  # noqa: E402
                             TIME_IN_FORCE_TAKER, build_payload, new_client_order_id,
                             started, verify_market)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


NOW = datetime(2026, 9, 9, 17, 30, tzinfo=timezone.utc)
SOON = (NOW + timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:%SZ")     # tonight's kickoff
GONE = (NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")     # under way
T = "KXNFLGAME-26SEP09NESEA-SEA"


def snap(ask="0.6100", size="500.00", status="active", close=None) -> dict:
    m = {"status": status, "yes_ask_dollars": ask, "yes_ask_size_fp": size}
    if close is not None:
        m["close_time"] = close
    return m


def verify(count, price, market, starts=SOON):
    return verify_market(T, count, price, market=market, now=NOW, starts=starts)


print("\n1. an unchanged market still trades")
r = verify(75, 0.61, snap())
check("the ticket passes", r["ok"], str(r))
check("it reports the ask it saw", r["ask"] == 0.61, str(r))
check("and the depth", r["offered"] == 500.0, str(r))
check("a move inside tolerance is fine", verify(75, 0.61, snap(ask="0.6200"))["ok"])

print("\n2. a price that moved is refused, in either direction")
down = verify(75, 0.61, snap(ask="0.5000"))
check("a stale bid is refused", not down["ok"], str(down))
check("the reason names the direction", "down" in down["reason"], down["reason"])
up = verify(75, 0.61, snap(ask="0.7500"))
check("a move up is refused too", not up["ok"] and "up" in up["reason"], str(up))

print("\n3. the market has to still be there")
check("a closed market is refused", not verify(75, 0.61, snap(status="closed"))["ok"])
check("a settled market is refused", not verify(75, 0.61, snap(status="settled"))["ok"])
check("a vanished market is refused", not verify(75, 0.61, {})["ok"])
empty = verify(75, 0.61, snap(ask=None))
check("an empty book is refused and says so", not empty["ok"] and "offered" in empty["reason"], str(empty))

print("\n4. depth is re-checked, not assumed")
thin = verify(75, 0.61, snap(size="21.00"))
check("more contracts than offered is refused", not thin["ok"], str(thin))
check("the reason gives both numbers", "21" in thin["reason"] and "75" in thin["reason"], thin["reason"])
check("exactly the offered size is allowed", verify(21, 0.61, snap(size="21.00"))["ok"])

print("\n5. a game already under way is refused")
live = verify(75, 0.61, snap(), starts=GONE)
check("an in-play ticket is refused", not live["ok"] and "kicked off" in live["reason"], str(live))
check("an in-play market is still 'active', so status alone would miss it", snap()["status"] == "active")
check("Kalshi's own close_time does not decide", verify(75, 0.61, snap(close=GONE), starts=SOON)["ok"])

print("\n6. the kickoff itself is read correctly")
check("a future kickoff has not begun", started({}, now=NOW, starts=SOON) is False)
check("a past kickoff has begun", started({}, now=NOW, starts=GONE) is True)
check("the moment of kickoff counts as begun", started({}, now=NOW, starts="2026-09-09T17:30:00Z") is True)
check("an ISO offset form is read", started({}, now=NOW, starts="2026-09-10T00:20:00+00:00") is False)
check("a missing kickoff is unknown", started({}, now=NOW) is None)
unknown = verify(75, 0.61, snap(), starts=None)
check("an unknown kickoff is REFUSED, not waved through", not unknown["ok"] and "re-scan" in unknown["reason"], str(unknown))

print("\n7. every refusal explains itself")
for label, m, st in [("closed", snap(status="closed"), SOON), ("no ask", snap(ask=None), SOON),
                     ("moved", snap(ask="0.9000"), SOON), ("thin", snap(size="2.00"), SOON),
                     ("in play", snap(), GONE), ("no kickoff", snap(), None), ("gone", {}, SOON)]:
    res = verify(75, 0.61, m, starts=st)
    check(f"{label} refuses with a reason", not res["ok"] and bool(res["reason"]))
check("the drift tolerance is a sane band", 0.0 < MAX_PRICE_DRIFT < 0.25, str(MAX_PRICE_DRIFT))

print("\n8. the payload carries every field V2 requires, and a taker never rests")
cid = new_client_order_id()
check("the same ticket builds the same order id", build_payload(T, 75, 0.61, cid) == build_payload(T, 75, 0.61, cid))
check("two tickets get different ids", new_client_order_id() != new_client_order_id())
pay = build_payload(T, 109, 0.32, "cid-1")
for field in ("ticker", "side", "count", "price", "client_order_id", "time_in_force", "self_trade_prevention_type"):
    check(f"{field} is present", field in pay, str(pay))
check("count is a fixed-point string", pay["count"] == "109.00", str(pay["count"]))
check("price is a 4dp string", pay["price"] == "0.3200", str(pay["price"]))
check("side is a bid", pay["side"] == "bid")
check("the default order is immediate_or_cancel", pay["time_in_force"] == TIME_IN_FORCE_TAKER)
check("post_only is off by default", pay["post_only"] is False)
maker = build_payload(T, 109, 0.32, "cid-1", post_only=True)
check("a post-only order is allowed to rest", maker["time_in_force"] == TIME_IN_FORCE_MAKER)
check("self-trade prevention is set either way", maker["self_trade_prevention_type"] == SELF_TRADE_PREVENTION)

print(f"\n{'='*54}\n  {PASS} passed, {FAIL} failed\n{'='*54}\n")
sys.exit(1 if FAIL else 0)
