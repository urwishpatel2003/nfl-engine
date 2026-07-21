"""
ml/season.py  —  season-long projections: team win totals + player season stat totals
=====================================================================================
PRECOMPUTED and STORED (data/processed/season_2026.json), not recomputed live:

  build_season()  — the heavy pass. Run ONCE before the season, then WEEKLY during it (wired into
                    ml.refresh). Writes the stored board. As games are played, completed results
                    replace their win probabilities (a played game is a certain win/loss), so the
                    projection folds in reality and only the remaining schedule stays probabilistic.

  team_win_totals() / player_season_totals(fmt) — cheap READERS: load the stored board (fall back
                    to a one-off build if the file is missing).

What's in it:
  • team wins — expected wins over each team's ACTUAL 2026 schedule (real opponents), priced with
    the matchup-engine margin computed inline off cached ratings/units. Exact Poisson-binomial →
    expected wins, projected record, fair O/U line + P(over), full win distribution.
  • player totals — recency-weighted 2023-25 per-game rates × projected games, SCHEDULE-ADJUSTED
    by each team's strength of schedule (mean opponent pass/rush defense), then fantasy points.

HONEST LIMITS: win totals treat games as independent (no injury attrition / bye-timing luck);
player totals need real 2025 usage (rookies omitted). Credible projections, not a market edge.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
STORE = PROC / "season_2026.json"
SEASON = 2026

_CACHE = None            # in-process copy of the stored board


# ── schedule + inline win prob ──────────────────────────────────────
def _schedule():
    s = pd.read_parquet(RAW / "schedules.parquet")
    d = s[s["season"] == SEASON]
    if "game_type" in d:
        d = d[d["game_type"].fillna("REG").str.upper().eq("REG")]
    return d


def _win_prob_fn():
    """Returns wp(home, away) using the matchup-engine margin computed inline off cached
    ratings/units — identical to project_game but O(1) per game."""
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
    return wp


def _poisson_binomial(ps):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return dist


def _compute_wins():
    """Per-team win outcomes over the real schedule: played games contribute a CERTAIN result
    (1/0), unplayed games their projected win prob — so the board folds in results as they land."""
    wp = _win_prob_fn()
    d = _schedule()
    by_team, played = {}, 0
    for _, g in d.iterrows():
        h, a = g.get("home_team"), g.get("away_team")
        if not isinstance(h, str) or not isinstance(a, str):
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if pd.notna(hs) and pd.notna(as_):                 # game is final → certain outcome
            hp = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            played += 1
        else:
            hp = wp(h, a)
            if hp is None:
                continue
        by_team.setdefault(h, []).append(hp)
        by_team.setdefault(a, []).append(1 - hp)
    rows = []
    for team, ps in by_team.items():
        dist = _poisson_binomial(ps)
        exp = float(sum(ps))
        median = int(np.searchsorted(np.cumsum(dist), 0.5))
        rows.append({"team": team, "games": len(ps), "proj_wins": round(exp, 1),
                     "record": f"{round(exp)}-{len(ps) - round(exp)}",
                     "win_line": median + 0.5, "p_over": round(float(dist[median + 1:].sum()), 3),
                     "mode_wins": int(np.argmax(dist)),
                     "dist": [round(float(x), 3) for x in dist]})
    rows.sort(key=lambda x: -x["proj_wins"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows, played


# ── player season stat totals (schedule-adjusted) ───────────────────
_STAT_COLS = ["pass_yds", "pass_td", "interceptions", "rush_yds", "rush_td",
              "carries", "receptions", "rec_yds", "rec_td", "targets"]
_HIST = [2025, 2024, 2023]
_W = {2025: 0.60, 2024: 0.30, 2023: 0.10}


def _team_sos():
    """Per-team strength-of-schedule multipliers for the passing and rushing games: the mean
    opponent defense factor over the team's actual 2026 schedule (>1 = softer slate = more yards)."""
    from ml.matchup_engine import team_units
    u = team_units()
    d = _schedule()

    def fac(opp, col):
        return float(np.clip(1 + 0.14 * u.loc[opp, col], 0.75, 1.30)) if opp in u.index else 1.0

    opps = {}
    for _, g in d.iterrows():
        h, a = g.get("home_team"), g.get("away_team")
        if not isinstance(h, str) or not isinstance(a, str):
            continue
        opps.setdefault(h, []).append(a)
        opps.setdefault(a, []).append(h)
    sos = {}
    for team, ol in opps.items():
        sos[team] = (float(np.mean([fac(o, "z_def_pass") for o in ol])),
                     float(np.mean([fac(o, "z_def_rush") for o in ol])))
    return sos


def _compute_player_stats():
    """Recency-weighted per-game rates × projected games × team strength-of-schedule → season
    stat totals (format-independent; fantasy points are added per-format on read)."""
    from ml.fantasy import season_stats, _rosters_2026
    hist = {s: season_stats(s).set_index("pid") for s in _HIST}
    roster = _rosters_2026()
    sos = _team_sos()
    rows = []
    for r in roster.itertuples():
        pid, pos, team = r.player_id, r.position, r.team
        rate = {c: 0.0 for c in _STAT_COLS}
        den = gsum = gden = 0.0
        for s in _HIST:
            h = hist[s]
            if pid in h.index and float(h.loc[pid, "games"]) >= 1:
                w, g = _W[s], float(h.loc[pid, "games"])
                row = h.loc[pid]
                for c in _STAT_COLS:
                    rate[c] += w * (float(row.get(c, 0) or 0) / g)
                den += w; gsum += w * g; gden += w
        if den == 0:                                       # rookies / no history
            continue
        games = min(17.0, max(10.0, gsum / gden))
        sp, sr = sos.get(team, (1.0, 1.0))                 # schedule adjustment (pass, rush)
        tot = {c: rate[c] / den * games for c in _STAT_COLS}
        for c in ("pass_yds", "pass_td", "rec_yds", "rec_td", "receptions", "targets"):
            tot[c] *= sp
        for c in ("rush_yds", "rush_td"):
            tot[c] *= sr
        rows.append({"player": r.player_name, "position": pos, "team": team,
                     "games": round(games, 1),
                     "pass_yds": round(tot["pass_yds"]), "pass_td": round(tot["pass_td"], 1),
                     "int": round(tot["interceptions"], 1),
                     "rush_yds": round(tot["rush_yds"]), "rush_td": round(tot["rush_td"], 1),
                     "carries": round(tot["carries"]),
                     "rec": round(tot["receptions"]), "rec_yds": round(tot["rec_yds"]),
                     "rec_td": round(tot["rec_td"], 1)})
    return rows


# ── build (write) + read ────────────────────────────────────────────
def build_season() -> dict:
    """Recompute both boards and WRITE the stored file. Run once pre-season, then weekly."""
    global _CACHE
    wins, played = _compute_wins()
    players = _compute_player_stats()
    board = {"season": SEASON, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "games_played": played, "wins": wins, "players": players}
    PROC.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(board))
    _CACHE = board
    return {"generated_at": board["generated_at"], "games_played": played,
            "teams": len(wins), "players": len(players)}


def _board():
    """Load the stored board; build it once if the file is missing."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if STORE.exists():
        _CACHE = json.loads(STORE.read_text())
    else:
        build_season()
    return _CACHE


def clear():
    global _CACHE
    _CACHE = None


def status() -> dict:
    b = _board()
    return {"generated_at": b.get("generated_at"), "games_played": b.get("games_played", 0),
            "stored": STORE.exists()}


def team_win_totals() -> pd.DataFrame:
    return pd.DataFrame(_board()["wins"])


def player_season_totals(fmt: str = "half") -> pd.DataFrame:
    """Read stored stat totals, add fantasy points for the requested scoring, rank."""
    from ml.fantasy import FORMATS
    rec_pt = 1.0 if FORMATS.get(fmt, FORMATS["half"])["scoring"] == "full" else 0.5
    d = pd.DataFrame(_board()["players"])
    if d.empty:
        return d
    d["total_td"] = (d["pass_td"] + d["rush_td"] + d["rec_td"]).round(1)
    d["fpts"] = (d["pass_yds"] * 0.04 + d["pass_td"] * 4 - d["int"] * 1
                 + d["rush_yds"] * 0.1 + d["rush_td"] * 6
                 + d["rec"] * rec_pt + d["rec_yds"] * 0.1 + d["rec_td"] * 6).round(1)
    d["fppg"] = (d["fpts"] / d["games"].clip(lower=1)).round(1)
    d = d.sort_values("fpts", ascending=False).reset_index(drop=True)
    d["pos_rank"] = d.groupby("position")["fpts"].rank(ascending=False, method="min").astype(int)
    d["overall_rank"] = range(1, len(d) + 1)
    return d


if __name__ == "__main__":
    print("Building season projections…")
    print(build_season())
