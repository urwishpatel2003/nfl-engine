"""
ml/risk.py  —  Kalshi position limits, enforced in code rather than left to discipline
======================================================================================
Ported from the tennis engine (engine/risk.py). Only the ledger location differs: this
project keeps runtime state on the Railway volume at data/, so the exposure ledger lives at
data/processed/kalshi_exposure.jsonl (override with KALSHI_LEDGER_PATH). dashboard/seed.py
treats it as refresh-managed so a deploy never clobbers it — a cap that forgets is not a cap.

Two caps, both refused server-side:

    MAX_TICKET_PCT   the most one ticket may stake, as a % of bankroll
    MAX_DAILY_PCT    the most all tickets may stake in a day, as a % of bankroll

They fail differently. The per-ticket cap stops one confident-looking price taking an
outsized position. The daily cap stops a run of individually reasonable tickets adding up
to a bad day — the failure mode that actually empties accounts, because each step looks
defensible. An NFL Sunday has 13+ games at once, so the daily cap is the one that matters.

What counts as committed: only what a person actually confirmed. Generated-but-untaken
tickets never risked anything and never consume the day's budget.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MAX_TICKET_PCT = float(os.environ.get("KALSHI_MAX_TICKET_PCT", "2.0"))
MAX_DAILY_PCT = float(os.environ.get("KALSHI_MAX_DAILY_PCT", "20.0"))

# Taker unless told otherwise. Makers pay about a quarter of the fee but only if they are
# filled; a resting order that never fills is not a cheaper trade, it is no trade.
MAKER = os.environ.get("KALSHI_MAKER") == "1"

_DEFAULT_LEDGER = Path(__file__).parent.parent / "data" / "processed" / "kalshi_exposure.jsonl"


def _ledger_path() -> Path:
    return Path(os.environ.get("KALSHI_LEDGER_PATH") or _DEFAULT_LEDGER)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _rows() -> list[dict]:
    """Every ledger row. Unreadable lines are skipped: a torn write must not hide exposure."""
    p = _ledger_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _today_rows() -> list[dict]:
    day = _today()
    return [r for r in _rows() if str(r.get("day")) == day]


def staked_today() -> float:
    """Everything confirmed today, settled or not. The gross figure."""
    return round(sum(float(r.get("stake") or 0.0) for r in _today_rows()), 2)


def committed_today(open_tickers: set | None = None) -> float:
    """
    Dollars STILL AT RISK today.

    Pass the tickers with an open position and a stake whose market has since settled is
    released back into the day's allowance. Without the argument this is the gross figure
    and nothing is released — the safe default for a caller that cannot see the account.
    The trade-off: this bounds money AT RISK AT ONCE rather than money staked over the
    day; `staked_today` keeps the gross figure visible so the difference is never hidden.
    """
    rows = _today_rows()
    if open_tickers is not None:
        rows = [r for r in rows if str(r.get("ticker")) in open_tickers]
    return round(sum(float(r.get("stake") or 0.0) for r in rows), 2)


def record_commit(stake: float, ticker: str, contracts: int,
                  client_order_id: str | None = None, label: str | None = None) -> None:
    """
    Record a CONFIRMED commitment. Called after a person presses the button.

    The client_order_id is stored because this ledger is the only record of which orders
    came from THIS application: Kalshi cannot distinguish an order this page sent from one
    placed by hand in the app. `label` is the human description of the ticket (game + side)
    so the report can name it without a market lookup.
    """
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"day": _today(), "stake": round(float(stake), 2),
           "ticker": ticker, "contracts": int(contracts),
           "client_order_id": client_order_id, "label": label,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def budget(bankroll: float, open_tickers: set | None = None) -> dict:
    """
    What is left to risk today, and the cap a single ticket may use.

    `ticket_pct` is the smaller of the per-ticket cap and whatever remains of the day, so
    the daily limit cannot be stepped over by one large ticket.
    """
    if bankroll <= 0:
        return {"bankroll": 0.0, "committed": 0.0, "staked_today": 0.0, "released": 0.0,
                "daily_cap": 0.0, "daily_remaining": 0.0, "ticket_pct": 0.0, "exhausted": True}
    committed = committed_today(open_tickers)
    daily_cap = bankroll * MAX_DAILY_PCT / 100.0
    remaining = max(0.0, daily_cap - committed)
    ticket_pct = min(MAX_TICKET_PCT, 100.0 * remaining / bankroll)
    return {"bankroll": round(bankroll, 2),
            "committed": round(committed, 2),
            "staked_today": staked_today(),
            "released": round(max(0.0, staked_today() - committed), 2),
            "daily_cap": round(daily_cap, 2),
            "daily_remaining": round(remaining, 2),
            "ticket_pct": round(max(ticket_pct, 0.0), 4),
            "exhausted": remaining <= 0.0}


def placed_here() -> dict:
    """
    What this application actually sent: {"tickers": set, "order_ids": set, "labels": dict}.

    Order ids are exact. Tickers are the fallback for fills, positions and settlements,
    which carry a ticker but never our client_order_id.
    """
    out = {"tickers": set(), "order_ids": set(), "labels": {}}
    for row in _rows():
        if row.get("ticker"):
            out["tickers"].add(str(row["ticker"]))
            if row.get("label"):
                out["labels"][str(row["ticker"])] = str(row["label"])
        if row.get("client_order_id"):
            out["order_ids"].add(str(row["client_order_id"]))
    return out
