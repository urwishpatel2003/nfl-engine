"""
Tests for Kalshi fees and position sizing (ml/kalshi.py).

    python tests/test_kalshi.py

Money maths, so the assertions are about properties rather than snapshots. No network;
nothing here submits an order, and the module it tests has no code path that does.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.kalshi import KELLY_FRACTION, base_url, fee_dollars, live_mode, size_position  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


print("\n1. real money is opt-in")
check("demo unless KALSHI_LIVE=1", not live_mode())
check("the base url is the demo one", "demo" in base_url(), base_url())

print("\n2. fees follow Kalshi's curve")
check("100 contracts at 50c cost $1.75", fee_dollars(100, 0.50) == 1.75, str(fee_dollars(100, 0.50)))
check("the fee peaks at 50c", fee_dollars(100, 0.50) > fee_dollars(100, 0.25) > fee_dollars(100, 0.10))
check("the curve is symmetric about 50c", abs(fee_dollars(100, 0.30) - fee_dollars(100, 0.70)) < 0.02)
check("a maker pays about a quarter", fee_dollars(100, 0.50, maker=True) < 0.5 * fee_dollars(100, 0.50))
check("no contracts, no fee", fee_dollars(0, 0.50) == 0.0)
check("the fee is rounded UP to the cent", fee_dollars(1, 0.50) >= 0.01, str(fee_dollars(1, 0.50)))
cheap = fee_dollars(100, 0.10) / (100 * 0.10)
rich = fee_dollars(100, 0.90) / (100 * 0.90)
check("a 10c contract pays a far bigger fee share than a 90c one", cheap > 5 * rich, f"{cheap:.3f} vs {rich:.3f}")

print("\n3. sizing respects its limits")
s = size_position(0.60, 0.50, bankroll=1000.0, max_stake_pct=1.0)
check("stake stays inside the per-ticket cap", s["stake"] <= 10.0 + 0.5, str(s["stake"]))
check("a real edge produces contracts", s["contracts"] > 0, str(s))
big = size_position(0.95, 0.50, bankroll=1000.0, max_stake_pct=1.0)
check("even a huge edge is capped", big["stake"] <= 10.0 + 0.5, str(big["stake"]))
check("the Kelly fraction is a sane multiplier", 0.0 < KELLY_FRACTION <= 1.0, str(KELLY_FRACTION))

print("\n4. a bad trade is refused, not sized")
none = size_position(0.40, 0.50, bankroll=1000.0)
check("no edge means no contracts", none["contracts"] == 0, str(none))
check("and it says why", bool(none["reason"]), str(none))
for label, kwargs in [("zero bankroll", dict(prob=0.60, price=0.50, bankroll=0.0)),
                      ("price above range", dict(prob=0.60, price=1.50, bankroll=1000.0)),
                      ("price below range", dict(prob=0.60, price=0.0, bankroll=1000.0)),
                      ("impossible probability", dict(prob=1.5, price=0.50, bankroll=1000.0))]:
    r = size_position(**kwargs)
    check(f"{label} is refused", r["contracts"] == 0 and bool(r["reason"]))

print("\n5. fees are taken out of the answer, not bolted on after")
thin = size_position(0.505, 0.50, bankroll=100000.0, max_stake_pct=100.0)
check("a thin edge is reported net of fees", thin["fee"] > 0, str(thin))
check("EV net of fees is below the gross figure",
      thin["ev"] < thin["contracts"] * (0.505 * 0.5 - 0.495 * 0.5) + 0.01, str(thin))
fat = size_position(0.70, 0.50, bankroll=1000.0, max_stake_pct=5.0)
check("a fat edge still clears its fee", fat["ev"] > 0 and fat["contracts"] > 0, str(fat))
check("EV percentage is relative to the stake", abs(fat["ev_pct"] - 100.0 * fat["ev"] / fat["stake"]) < 0.5, str(fat))

print(f"\n{'='*54}\n  {PASS} passed, {FAIL} failed\n{'='*54}\n")
sys.exit(1 if FAIL else 0)
