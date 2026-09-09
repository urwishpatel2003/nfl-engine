"""
ml/kalshi.py  —  Kalshi: read the account, price the market, size a position. NEVER auto-submit.
==============================================================================================
Ported from the tennis engine (engine/kalshi.py) — same auth, fees and sizing; only the
User-Agent changed. Keep the two in step.

Inputs : KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY (PEM text) from the environment
Outputs: bankroll, market prices, and a fully costed order ticket

What this module will and will not do
-------------------------------------
It reads the account and builds a ticket. Submitting an order is a SEPARATE, explicitly
confirmed action taken by a person pressing a button — there is no scheduled path, no retry
loop and no batch mode that places trades on its own. That boundary is the design.

Credentials are read from the environment and never logged, never returned, and never
written to any ledger. The private key is a PEM: set it on the Railway service, never in
this repo (the repo is public GitHub).

Fees are not an afterthought
----------------------------
Kalshi charges ceil(0.07 * C * P * (1-P)) dollars on an order, rounded up to the cent, with
no settlement fee; makers pay about a quarter of that. The curve peaks at 50c, where a taker
pays 1.75c per contract — about 3.5% of the contract's own cost. NFL game markets sit near
50c far more often than tennis does, so the fee bites hardest exactly where our edges live.
Every number here is NET of it.

Sizing
------
Full Kelly by default, overridable with KALSHI_KELLY_FRACTION. Against the 2% per-ticket
cap in ml/risk.py, full and quarter Kelly produce the SAME stake on a strong edge — the caps
bound risk, not the fraction. They diverge only on thin edges, which are exactly where the
model's probability is least reliable (see CLAUDE.md: the engine does not reliably beat NFL
markets out-of-sample). The caps are the safety net; the fraction is not.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.request

# Demo by default. Pointing at real money takes a deliberate env change.
DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"

FEE_RATE = 0.07          # Kalshi's published trading fee coefficient
MAKER_SHARE = 0.25       # makers pay roughly a quarter of the taker fee
KELLY_FRACTION = float(os.environ.get("KALSHI_KELLY_FRACTION", "1.0"))
USER_AGENT = "nfl-engine/0.1"


class KalshiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def live_mode() -> bool:
    """True only when explicitly switched to real money."""
    return os.environ.get("KALSHI_LIVE") == "1"


def base_url() -> str:
    return PROD_BASE if live_mode() else DEMO_BASE


def configured() -> bool:
    return bool(os.environ.get("KALSHI_API_KEY_ID")
                and os.environ.get("KALSHI_PRIVATE_KEY"))


# ──────────────────────────────────────────────────────────────────────────────
# Fees and sizing
# ──────────────────────────────────────────────────────────────────────────────
def fee_dollars(contracts: int, price: float, maker: bool = False) -> float:
    """Kalshi's trading fee in dollars, rounded up to the cent ON THE ORDER (how Kalshi charges it)."""
    if contracts <= 0:
        return 0.0
    raw = FEE_RATE * contracts * price * (1.0 - price)
    if maker:
        raw *= MAKER_SHARE
    # Round to a sane precision BEFORE the ceiling. 0.07*100*0.5*0.5 evaluates to
    # 1.7500000000000002, and ceiling that lands on $1.76 — a cent above Kalshi's
    # published maximum, on every fee that falls exactly on a cent.
    return math.ceil(round(raw * 100.0, 9)) / 100.0


def size_position(prob: float, price: float, bankroll: float,
                  maker: bool = False, max_stake_pct: float = 1.0) -> dict:
    """
    Kelly sizing at a Kalshi price, with the fee taken out of the edge.

    `price` is dollars per contract (0.01-0.99); a winning contract settles at $1.00.
    `max_stake_pct` caps one ticket as a percentage of bankroll.

    Returns contracts = 0 whenever the trade does not survive its own costs. That is the
    common case, and it is not an error.
    """
    out = {"contracts": 0, "stake": 0.0, "fee": 0.0, "ev": 0.0,
           "ev_pct": 0.0, "kelly_fraction": 0.0, "reason": None}
    if not (0.01 <= price <= 0.99) or bankroll <= 0 or not (0.0 < prob < 1.0):
        out["reason"] = "price, probability or bankroll out of range"
        return out

    b = (1.0 - price) / price          # net odds per dollar staked
    edge = prob * b - (1.0 - prob)     # Kelly numerator, before fees
    if edge <= 0:
        out["reason"] = "no edge at this price before fees"
        return out

    frac = min((edge / b) * KELLY_FRACTION, max_stake_pct / 100.0)
    stake = bankroll * frac
    contracts = int(stake // price)
    if contracts < 1:
        out["reason"] = "sized below one contract"
        return out

    cost = contracts * price
    fee = fee_dollars(contracts, price, maker)
    # A winning contract pays $1; the fee is paid either way.
    ev = prob * (contracts * (1.0 - price)) - (1.0 - prob) * cost - fee
    out.update({"contracts": contracts, "stake": round(cost, 2),
                "fee": round(fee, 2), "ev": round(ev, 2),
                "ev_pct": round(100.0 * ev / cost, 2) if cost else 0.0,
                "kelly_fraction": round(frac, 5)})
    if ev <= 0:
        out["reason"] = "positive before fees, negative after them"
    return out


# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────
def _sign(method: str, path: str) -> dict:
    """
    RSA-PSS SHA-256 over timestamp + METHOD + path, per Kalshi's scheme.

    The path is signed WITHOUT its query string and WITH the /trade-api/v2 prefix (callers
    pass it). The key never leaves this function and is never logged.
    """
    key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pem = os.environ.get("KALSHI_PRIVATE_KEY", "")
    if not key_id or not pem:
        raise KalshiError("KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY are not set")
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as e:
        raise KalshiError(f"cryptography is required for Kalshi auth: {e}") from e

    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path.split("?")[0]).encode()
    key = serialization.load_pem_private_key(pem.encode(), password=None)
    sig = key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {"KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


def _get(path: str, auth: bool = True, timeout: int = 30) -> dict:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if auth:
        headers.update(_sign("GET", "/trade-api/v2" + path))
    req = urllib.request.Request(base_url() + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise KalshiError(f"HTTP {e.code}: {body}", e.code) from e
    except Exception as e:
        raise KalshiError(f"{type(e).__name__}: {e}") from e


def balance() -> dict:
    """Account balance in dollars — the bankroll Kelly sizes against. Read live, never configured."""
    d = _get("/portfolio/balance")
    cents = d.get("balance")
    return {"dollars": round(cents / 100.0, 2) if isinstance(cents, (int, float)) else None,
            "live": live_mode()}


def market(ticker: str) -> dict:
    """One market by ticker, read fresh. Public data, so no signature is needed."""
    return (_get(f"/markets/{ticker}", auth=False).get("market") or {})


def markets(limit: int = 200, status: str = "open",
            series: str | None = None) -> list[dict]:
    """Open markets for a series, following the cursor. Public data, so no signature."""
    out, cursor = [], None
    while True:
        path = f"/markets?limit={int(limit)}&status={status}"
        if series:
            path += f"&series_ticker={series}"
        if cursor:
            path += f"&cursor={cursor}"
        d = _get(path, auth=False)
        out.extend(d.get("markets") or [])
        cursor = d.get("cursor")
        if not cursor or len(out) >= 5000:
            return out

# ──── portfolio, read-only ────────────────────────────────────────────────────
# Everything below reads. None of it can open, change or close a position; the
# only code that spends is ml/kalshi_order.py.


def num(value) -> float | None:
    """V2 sends numbers as fixed-point strings. Read both shapes, never raise."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def positions(limit: int = 200) -> list[dict]:
    """Open positions, one per market. `position_fp` is the contract count."""
    d = _get(f"/portfolio/positions?limit={int(limit)}")
    return (d.get("market_positions") or [])


def fills(limit: int = 200) -> list[dict]:
    """Executions. A fill is the only proof an order actually traded."""
    return (_get(f"/portfolio/fills?limit={int(limit)}").get("fills") or [])


def settlements(limit: int = 200) -> list[dict]:
    """Settled markets. `revenue` is in CENTS, unlike the *_dollars fields."""
    return (_get(f"/portfolio/settlements?limit={int(limit)}").get("settlements") or [])


def orders(limit: int = 200, status: str | None = None) -> list[dict]:
    """Orders placed, filled or not. Tries the portfolio path, then the top-level one on 404."""
    q = f"?limit={int(limit)}" + (f"&status={status}" if status else "")
    for path in ("/portfolio/orders", "/orders"):
        try:
            return (_get(path + q).get("orders") or [])
        except KalshiError as e:
            if e.status == 404:
                continue
            raise
    return []
