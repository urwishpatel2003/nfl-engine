"""
ml/props.py  —  player prop projections + over/under probabilities
==================================================================
Turns the per-player mean stat lines from ml.projections.project_matchup (usage × team volume ×
efficiency × game script, opponent-adjusted) into PROP markets: a projected number for each
market PLUS the distribution around it, so a book line becomes P(over) and fair odds.

Markets
  QB   — Pass Yds, Pass TDs, Pass Att, Completions, Interceptions, Rush Yds
  RB   — Rush Yds, Carries, Receptions, Rec Yds, Rush+Rec Yds, Anytime TD
  WR/TE— Receptions, Rec Yds, Rec+Rush Yds, Anytime TD

Distributions (game-to-game box-score variance, tuned to NFL):
  • continuous (yards, attempts, completions) → Normal(mean, sd), sd is a per-market CV of the mean
  • counts (receptions, TDs, INTs)            → Poisson(mean)
  • anytime TD                                → 1 − e^(−expected_TDs)  (a probability, not a line)

The over/under math (normal / Poisson CDF, fair odds) is done client-side so a typed line updates
instantly; this module ships mean + distribution params. prob_over() is here too for the API/tests.

HONEST LIMITS: same as projections.py (2025 usage only, coarse D adjustment, noisy TDs) plus the
variance constants are league-typical, not player-specific. It's a credible model line, not a
market-beating edge — the book's number already prices most of this.
"""

import math

# per-market standard deviation as a function of the projected mean (continuous markets only).
# Values are league-typical single-game coefficients of variation with a floor for low projections.
def _sd(market: str, mean: float) -> float:
    m = max(0.0, float(mean))
    return {
        "pass_yds":    max(40.0, 0.28 * m),
        "pass_att":    max(4.5, 0.16 * m),
        "cmp":         max(3.5, 0.17 * m),
        "rush_yds":    max(16.0, 0.55 * m),
        "carries":     max(3.0, 0.32 * m),
        "rec_yds":     max(14.0, 0.62 * m),
        "rushrec_yds": max(20.0, 0.50 * m),
        "rec_raw_yds": max(14.0, 0.62 * m),
    }.get(market, max(1.0, 0.5 * m))


# market catalog: key → (label, distribution, unit). "prob" = a straight probability (anytime TD).
_MARKETS = {
    "pass_yds":    ("Pass Yds", "normal", "yds"),
    "pass_td":     ("Pass TDs", "poisson", "td"),
    "pass_att":    ("Pass Att", "normal", "att"),
    "cmp":         ("Completions", "normal", "cmp"),
    "int":         ("Interceptions", "poisson", "int"),
    "rush_yds":    ("Rush Yds", "normal", "yds"),
    "carries":     ("Carries", "normal", "car"),
    "rec":         ("Receptions", "poisson", "rec"),
    "rec_yds":     ("Rec Yds", "normal", "yds"),
    "rushrec_yds": ("Rush+Rec Yds", "normal", "yds"),
    "anytime_td":  ("Anytime TD", "prob", "prob"),
}


def _row(market, proj):
    label, dist, unit = _MARKETS[market]
    r = {"market": market, "label": label, "dist": dist, "unit": unit, "proj": round(float(proj), 1)}
    if dist == "normal":
        r["sd"] = round(_sd(market, proj), 1)
    elif dist == "poisson":
        r["lam"] = round(float(proj), 3)
    elif dist == "prob":
        r["proj"] = round(float(proj), 3)          # already a probability
    return r


def _qb_markets(q):
    return [_row("pass_yds", q["pass_yds"]), _row("pass_td", q["pass_td"]),
            _row("pass_att", q["pass_att"]), _row("cmp", q["cmp"]),
            _row("int", q["int"]), _row("rush_yds", q.get("rush_yds", 0))]


def _rb_markets(p):
    rr = (p.get("rush_yds", 0) or 0) + (p.get("rec_yds", 0) or 0)
    atd = 1.0 - math.exp(-((p.get("rush_td", 0) or 0) + (p.get("rec_td", 0) or 0)))
    return [_row("rush_yds", p.get("rush_yds", 0)), _row("carries", p.get("carries", 0)),
            _row("rec", p.get("rec", 0)), _row("rec_yds", p.get("rec_yds", 0)),
            _row("rushrec_yds", rr), _row("anytime_td", atd)]


def _rec_markets(p):
    rr = (p.get("rec_yds", 0) or 0) + (p.get("rush_yds", 0) or 0)
    atd = 1.0 - math.exp(-((p.get("rec_td", 0) or 0) + (p.get("rush_td", 0) or 0)))
    rows = [_row("rec", p.get("rec", 0)), _row("rec_yds", p.get("rec_yds", 0)),
            _row("anytime_td", atd)]
    if (p.get("rush_yds", 0) or 0) > 5:                # WRs with real rush usage get the combo too
        rows.insert(2, _row("rushrec_yds", rr))
    return rows


def player_props(home: str, away: str, neutral: bool = False) -> dict:
    """Per-player prop markets for a matchup, both teams. Each player → projected number + the
    distribution params needed to price any line as over/under."""
    from ml.projections import project_matchup
    m = project_matchup(home, away, neutral=neutral)
    out = {"home": home, "away": away, "pred": m["pred"], "teams": {}}
    for team in (home, away):
        t = m["teams"].get(team, {})
        players = []
        if t.get("qb"):
            q = t["qb"]
            players.append({"name": q["name"], "pos": "QB", "markets": _qb_markets(q)})
        for p in t.get("rush", []):
            players.append({"name": p["name"], "pos": "RB", "markets": _rb_markets(p)})
        for p in t.get("rec", []):
            if p["pos"] == "RB":                          # RBs already covered by the rush loop
                continue
            players.append({"name": p["name"], "pos": p["pos"], "markets": _rec_markets(p)})
        out["teams"][team] = players
    return out


# ── over/under pricing (also mirrored in JS for instant line entry) ──
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pois_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    s, term = 0.0, math.exp(-lam)
    for i in range(0, k + 1):
        if i > 0:
            term *= lam / i
        s += term
    return min(1.0, s)


def prob_over(row: dict, line: float) -> dict:
    """P(over line), P(under), P(push) for a market row from player_props(). Counts push on an
    integer line; continuous markets never push."""
    dist = row["dist"]
    if dist == "prob":
        p = row["proj"]                                   # anytime TD: 'over 0.5' is just the prob
        return {"over": round(p, 3), "under": round(1 - p, 3), "push": 0.0}
    if dist == "normal":
        over = 1.0 - _norm_cdf((line - row["proj"]) / max(1e-6, row["sd"]))
        return {"over": round(over, 3), "under": round(1 - over, 3), "push": 0.0}
    # poisson
    lam = row["lam"]
    is_int = abs(line - round(line)) < 1e-9
    if is_int:
        push = math.exp(-lam) * lam ** line / math.factorial(int(line))
        over = 1.0 - _pois_cdf(int(line), lam)
        return {"over": round(over, 3), "under": round(1 - over - push, 3), "push": round(push, 3)}
    over = 1.0 - _pois_cdf(int(math.floor(line)), lam)
    return {"over": round(over, 3), "under": round(1 - over, 3), "push": 0.0}


def fair_odds(p: float) -> int:
    """Fair American odds for probability p (no vig)."""
    p = min(0.999, max(0.001, p))
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
