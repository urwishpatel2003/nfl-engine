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

Distributions (game-to-game box-score variance, FITTED to 2024-26 game logs — see _SD_FIT):
  • continuous (yards, attempts, completions) → Normal(mean, sd), sd = a·mean^k with a floor
  • counts (TDs, INTs)                        → Poisson(mean)   (var/mean measured 0.9-1.0)
  • receptions                                → NegBin(mean, var = 1.5·mean)  (see REC_PHI)
  • anytime TD                                → 1 − e^(−expected_TDs)  (a probability, not a line)

Pricing against a book line (anchor_to_book):
  The book's line + vig-free over price pin down the mean the MARKET believes in (under our own
  distribution: book_mean solves P(over line | book_mean) = book_over_prob). The number we
  price from is a blend, (1−w)·model + w·book_mean. The book's number is a sharp consensus and
  the model's error against it is the larger one, so w sits well above ½ by default; once the
  calibration log (ml.props_ledger.calib_*) has enough graded rows the weight is MEASURED per
  market (least-squares vs the actual box score) and replaces the default. Edges therefore come
  from where the model disagrees with the market for a reason it can see (injury, role, pace,
  opponent), scaled by how much that disagreement has been worth — not from the raw model.

The over/under math (normal / Poisson CDF, fair odds) is done client-side so a typed line updates
instantly; this module ships mean + distribution params. prob_over() is here too for the API/tests.

HONEST LIMITS: same as projections.py (2025 usage only, coarse D adjustment, noisy TDs). The
spreads are league-wide fits, not player-specific. It's a credible model line, not a
market-beating edge — the book's number already prices most of this.
"""

import math

# ── game-to-game spread per market ────────────────────────────────────
# sd = a · mean^k (floored). Fitted 2026-10-20 on 2024-2026 play-by-play box scores
# (ml.props_ledger.game_boxes), player-seasons with ≥ 6 games: log(sd) ~ log(mean) per market,
# checked against the median sd by volume bucket. QB volume markets are near-constant across
# starters (sd barely moves with the mean), so they use a flat CV with a high floor:
#   pass_yds  starters' median sd 70-73  → 0.30·m, floor 60
#   pass_att  8-9 regardless of volume   → 0.26·m, floor 7.5
#   cmp       5.7-6.2                    → 0.26·m, floor 5
#   rush_yds  3.87·m^0.52  (17→16.9 vs 15.8 observed, 57→31.6 vs 32.6, 81→38.1 vs 38.2)
#   carries   1.6·m^0.42   (var/mean 2.1: overdispersed, so NOT Poisson; 7.8→3.8 vs 4.3, 15→5.0 vs 5.05)
#   rec_yds   3.16·m^0.59  (13→14.6 vs 13.1, 41→28.3 vs 29.3, 57→34.4 vs 36.0)
#   rushrec   4.04·m^0.52  (33→24.9 vs 25.0, 80→39.6 vs 37.7, 113→47.3 vs 49.8)
# The old fixed CVs (pass_att 0.16, cmp 0.17, carries 0.32 …) ran 25-45 % too tight on the volume
# markets, which manufactured 10-20 pt "edges" on attempts/completions/carries that were noise.
#
# A PROJECTION carries its own error on top of the game-to-game spread, so the predictive sd
# is the game sd × k, with k measured by backtest_props.py (section B, walk-forward 2022-25,
# k = median |z| / 0.6745 after removing bias): carries 1.28, rushrec 1.06, pass_yds 1.05,
# cmp 1.05, rec_yds 1.01, rush_yds 0.99, pass_att 0.96 (left at 1.0 — never tighten on k < 1).
# The multipliers below are the game-log coefficients × those k. With the market anchor the
# priced mean is better than the raw one, so this is slightly conservative — on purpose.
_SD_FIT = {
    #  market       (a,     k,    floor)
    "pass_yds":    (0.315, 1.00, 63.0),
    "pass_att":    (0.26, 1.00, 7.5),
    "cmp":         (0.273, 1.00, 5.3),
    "rush_yds":    (3.87, 0.52, 8.0),
    "carries":     (2.05, 0.42, 2.5),
    "rec_yds":     (3.16, 0.59, 8.0),
    "rushrec_yds": (4.26, 0.52, 10.5),
    "rec_raw_yds": (3.16, 0.59, 8.0),
}


def _sd(market: str, mean: float) -> float:
    m = max(0.0, float(mean))
    a, k, floor = _SD_FIT.get(market, (0.5, 1.0, 1.0))
    return max(floor, a * m ** k)


# Receptions: Poisson game-to-game (var/mean 1.08 within a player-season) but against a
# PROJECTION the variance-to-mean ratio is 1.5 (backtest_props.py, 9.5k player-games 2022-25:
# Poisson covered only 44 % of a 50 % band). Negative binomial with var = φ·mean, i.e.
# size r = mean/(φ−1). TDs/INTs stay Poisson (their ratio measured 0.9-1.0).
REC_PHI = 1.5


def _nb_pmf_cdf(k: int, mean: float, phi: float):
    """(pmf(k), cdf(k)) of a negative binomial with the given mean and variance φ·mean."""
    if k < 0:
        return 0.0, 0.0
    m = max(1e-9, float(mean))
    r = m / max(1e-6, phi - 1.0)
    p = m / (r + m)                                   # 'success' prob per the NB2 parameterisation
    term = (1.0 - p) ** r                             # pmf(0)
    s = term
    for i in range(0, k):
        term *= (i + r) / (i + 1) * p                 # pmf(i+1) from pmf(i)
        s += term
    return term, min(1.0, s)


# ── market anchor ─────────────────────────────────────────────────────
# Default weight on the book-implied mean, per market, used until the calibration log can
# measure it (ml.props_ledger.anchor_weights). 0.65 is a prior, not a measurement: the book's
# line is a multi-book consensus that already prices usage, injuries and pace; our mean comes
# from last season's usage with coarse opponent adjustment, so its error is the larger one.
# Anytime TD is anchored harder — TD rates are the noisiest thing the model projects.
ANCHOR_DEFAULT = 0.65
ANCHOR_DEFAULT_BY_MARKET = {"anytime_td": 0.80, "pass_td": 0.75, "int": 0.75}
ANCHOR_MIN, ANCHOR_MAX = 0.30, 0.95        # a measured weight is clipped to this band


def anchor_weight(market: str, measured: dict | None = None) -> float:
    """Weight on the book-implied mean for `market`: the measured one when the calibration log
    has it, else the documented default."""
    if measured and market in measured and measured[market].get("w") is not None:
        return float(min(ANCHOR_MAX, max(ANCHOR_MIN, measured[market]["w"])))
    return ANCHOR_DEFAULT_BY_MARKET.get(market, ANCHOR_DEFAULT)


def book_implied_mean(row: dict, line: float, p_over: float) -> float | None:
    """The mean under this row's distribution at which P(over line) equals the book's vig-free
    over probability — i.e. the number the market is pricing. Normal: closed form via the sd;
    Poisson: bisection on the mean. None when the price is uninformative (p at the rails)."""
    p = float(p_over)
    if not (0.02 < p < 0.98):
        return None
    if row["dist"] == "normal":
        return float(line) + float(row["sd"]) * _norm_ppf(p)
    if row["dist"] in ("poisson", "nbinom"):
        lo, hi = 0.0, max(1.0, float(line)) * 6 + 5
        for _ in range(60):
            mid = (lo + hi) / 2
            over = prob_over({"dist": row["dist"], "lam": mid, "phi": row.get("phi", REC_PHI)}, float(line))["over"]
            if over < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return None


def anchor_to_book(row: dict, line: float, p_over: float, w: float) -> dict:
    """Blend the row's mean toward the book-implied mean with weight w, in place. Keeps the raw
    model number in `model_proj` and the market's in `book_mean`; `proj` (and `lam`) become the
    anchored mean the row is priced from. Returns the row."""
    raw = float(row.get("model_proj", row["proj"]))
    row["model_proj"] = round(raw, 1)
    bm = book_implied_mean(row, line, p_over)
    if bm is None:
        row.update({"book_mean": None, "anchor_w": 0.0})
        return row
    anchored = max(0.0, (1.0 - w) * raw + w * bm)
    row.update({"book_mean": round(bm, 1), "anchor_w": round(w, 2), "proj": round(anchored, 1)})
    if row["dist"] in ("poisson", "nbinom"):
        row["lam"] = round(anchored, 3)
    return row


def _norm_ppf(p: float) -> float:
    """Inverse standard normal (Acklam's rational approximation, |err| < 1.2e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# market catalog: key → (label, distribution, unit). "prob" = a straight probability (anytime TD).
_MARKETS = {
    "pass_yds":    ("Pass Yds", "normal", "yds"),
    "pass_td":     ("Pass TDs", "poisson", "td"),
    "pass_att":    ("Pass Att", "normal", "att"),
    "cmp":         ("Completions", "normal", "cmp"),
    "int":         ("Interceptions", "poisson", "int"),
    "rush_yds":    ("Rush Yds", "normal", "yds"),
    "carries":     ("Carries", "normal", "car"),
    "rec":         ("Receptions", "nbinom", "rec"),
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
    elif dist == "nbinom":
        r["lam"] = round(float(proj), 3)
        r["phi"] = REC_PHI
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
        # player_id (nflverse gsis id) rides along so the props record can grade a row
        # against the play-by-play box score without a name match
        if t.get("qb"):
            q = t["qb"]
            players.append({"name": q["name"], "player_id": q.get("player_id"), "pos": "QB", "markets": _qb_markets(q)})
        for p in t.get("rush", []):
            players.append({"name": p["name"], "player_id": p.get("player_id"), "pos": "RB", "markets": _rb_markets(p)})
        for p in t.get("rec", []):
            if p["pos"] == "RB":                          # RBs already covered by the rush loop
                continue
            players.append({"name": p["name"], "player_id": p.get("player_id"), "pos": p["pos"], "markets": _rec_markets(p)})
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
    # counts: poisson, or negative binomial (receptions)
    lam = row["lam"]
    is_int = abs(line - round(line)) < 1e-9
    if dist == "nbinom":
        pmf, cdf = _nb_pmf_cdf(int(math.floor(line)), lam, row.get("phi", REC_PHI))
    else:
        cdf = _pois_cdf(int(math.floor(line)), lam)
        pmf = math.exp(-lam) * lam ** int(line) / math.factorial(int(line)) if is_int else 0.0
    if is_int:
        over = 1.0 - cdf
        return {"over": round(over, 3), "under": round(1 - over - pmf, 3), "push": round(pmf, 3)}
    over = 1.0 - cdf
    return {"over": round(over, 3), "under": round(1 - over, 3), "push": 0.0}


def fair_odds(p: float) -> int:
    """Fair American odds for probability p (no vig)."""
    p = min(0.999, max(0.001, p))
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
