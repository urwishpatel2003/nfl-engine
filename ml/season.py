"""
ml/season.py  —  season-long projections: team win totals + player season stat totals
=====================================================================================
Two futures-style boards from engines we already have:

  • team_win_totals() — expected regular-season wins for each team. We price every 2026 game with
    the SAME margin model as the matchup engine (roster-talent rating + unit matchup, anchored to
    the rankings), but computed INLINE from one cached squad_ratings()/team_units() pass so all 272
    games price in <1s (vs ~2min looping project_game). Win distribution is the exact
    Poisson-binomial over a team's 17 games → expected wins, a fair over/under line, P(over).

  • player_season_totals(fmt) — full-season stat lines (pass/rush/rec yards, TDs, receptions) +
    fantasy points, from recency-weighted per-game rates (2023-25) × projected games, mapped onto
    each player's current team.

HONEST LIMITS: win totals ignore injuries/schedule-timing/bye luck and treat games as independent;
player totals need real 2025 usage (rookies with no history are omitted). Credible projections,
not a market-beating edge.
"""

import numpy as np
import pandas as pd

RAW = __import__("pathlib").Path(__file__).parent.parent / "data" / "raw"

_WINS = None
_PLAYERS = {}


def clear():
    global _WINS, _PLAYERS
    _WINS = None
    _PLAYERS = {}


# ── team win totals ─────────────────────────────────────────────────
def _game_win_probs():
    """Home win prob for every 2026 REG game, from the matchup-engine margin computed inline off
    cached ratings/units (matches project_game exactly, but O(1) per game)."""
    from ml.squad import squad_ratings, SPREAD_SCALE, HFA
    from ml.matchup_engine import team_units
    out, _ = squad_ratings()
    r = out.set_index("team")["rating"]
    u = team_units()

    def phase(off, deff, sign):
        base = 0.5 * u.loc[off, "pf"] + 0.5 * u.loc[deff, "pa"]
        nudge = 0.8 * ((u.loc[off, "z_off_pass"] + u.loc[deff, "z_def_pass"]) +
                       0.6 * (u.loc[off, "z_off_rush"] + u.loc[deff, "z_def_rush"]))
        return float(base + nudge + u.loc[off, "st"] + 0.4 * u.loc[off, "z_coaching"] + sign * 1.0)

    def wp(home, away):
        if home not in r.index or away not in r.index:
            return None
        rm = float(np.clip((r[home] - r[away]) * SPREAD_SCALE + HFA, -18, 18))
        fm = 0.55 * rm + 0.45 * (phase(home, away, +1) - phase(away, home, -1))
        return float(1 / (1 + np.exp(-fm / 13.5 * np.pi / np.sqrt(3))))

    s = pd.read_parquet(RAW / "schedules.parquet")
    d = s[s["season"] == 2026]
    if "game_type" in d:
        d = d[d["game_type"].fillna("REG").str.upper().eq("REG")]
    games = []
    for _, g in d.iterrows():
        h, a = g.get("home_team"), g.get("away_team")
        if not isinstance(h, str) or not isinstance(a, str):
            continue
        p = wp(h, a)
        if p is not None:
            games.append((h, a, p))
    return games


def _poisson_binomial(ps):
    """Exact distribution of the number of wins given independent per-game win probs."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return dist


def team_win_totals():
    """Per team: expected wins, projected record, a fair win-total line (median) and P(over it)."""
    global _WINS
    if _WINS is not None:
        return _WINS
    games = _game_win_probs()
    by_team = {}
    for h, a, p in games:
        by_team.setdefault(h, []).append(p)
        by_team.setdefault(a, []).append(1 - p)
    rows = []
    for team, ps in by_team.items():
        dist = _poisson_binomial(ps)
        exp = float(sum(ps))
        cum = np.cumsum(dist)
        median = int(np.searchsorted(cum, 0.5))               # fair O/U line
        line = median + 0.5
        p_over = float(dist[median + 1:].sum())               # P(wins > median)  ~ P(over median.5)
        mode = int(np.argmax(dist))
        rows.append({"team": team, "games": len(ps), "proj_wins": round(exp, 1),
                     "record": f"{round(exp)}-{len(ps) - round(exp)}",
                     "win_line": line, "p_over": round(p_over, 3),
                     "mode_wins": mode, "mode_record": f"{mode}-{len(ps) - mode}",
                     "dist": [round(float(x), 3) for x in dist]})
    df = pd.DataFrame(rows).sort_values("proj_wins", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    _WINS = df
    return _WINS


# ── player season stat totals ───────────────────────────────────────
_STAT_COLS = ["pass_yds", "pass_td", "interceptions", "rush_yds", "rush_td",
              "carries", "receptions", "rec_yds", "rec_td", "targets"]
_SEASONS = [2025, 2024, 2023]
_W = {2025: 0.60, 2024: 0.30, 2023: 0.10}


def player_season_totals(fmt: str = "bestball"):
    """Projected full-season stat totals + fantasy points per rostered skill player, from
    recency-weighted per-game rates × projected games."""
    if fmt in _PLAYERS:
        return _PLAYERS[fmt]
    from ml.fantasy import season_stats, _rosters_2026, FORMATS
    rec_pt = 1.0 if FORMATS.get(fmt, FORMATS["bestball"])["scoring"] == "full" else 0.5
    hist = {s: season_stats(s).set_index("pid") for s in _SEASONS}
    roster = _rosters_2026()
    rows = []
    for r in roster.itertuples():
        pid, pos = r.player_id, r.position
        rate = {c: 0.0 for c in _STAT_COLS}
        den = gsum = gden = 0.0
        for s in _SEASONS:
            h = hist[s]
            if pid in h.index and float(h.loc[pid, "games"]) >= 1:
                w, g = _W[s], float(h.loc[pid, "games"])
                row = h.loc[pid]
                for c in _STAT_COLS:
                    rate[c] += w * (float(row.get(c, 0) or 0) / g)
                den += w; gsum += w * g; gden += w
        if den == 0:                                          # no NFL history yet (rookies) → skip
            continue
        games = min(17.0, max(10.0, gsum / gden))
        tot = {c: rate[c] / den * games for c in _STAT_COLS}
        fpts = (tot["pass_yds"] * 0.04 + tot["pass_td"] * 4 - tot["interceptions"] * 1
                + tot["rush_yds"] * 0.1 + tot["rush_td"] * 6
                + tot["receptions"] * rec_pt + tot["rec_yds"] * 0.1 + tot["rec_td"] * 6)
        rows.append({"player": r.player_name, "position": pos, "team": r.team,
                     "games": round(games, 1),
                     "pass_yds": round(tot["pass_yds"]), "pass_td": round(tot["pass_td"], 1),
                     "int": round(tot["interceptions"], 1),
                     "rush_yds": round(tot["rush_yds"]), "rush_td": round(tot["rush_td"], 1),
                     "carries": round(tot["carries"]),
                     "rec": round(tot["receptions"]), "rec_yds": round(tot["rec_yds"]),
                     "rec_td": round(tot["rec_td"], 1),
                     "total_td": round(tot["pass_td"] + tot["rush_td"] + tot["rec_td"], 1),
                     "fpts": round(fpts, 1), "fppg": round(fpts / games, 1)})
    df = pd.DataFrame(rows).sort_values("fpts", ascending=False).reset_index(drop=True)
    df["pos_rank"] = df.groupby("position")["fpts"].rank(ascending=False, method="min").astype(int)
    df["overall_rank"] = range(1, len(df) + 1)
    _PLAYERS[fmt] = df
    return df
