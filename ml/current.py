"""
ml/current.py  —  the in-season "current" view: this season's results shrunk toward last season
================================================================================================
One week of games is 1/17 of a season. Ranking, labelling or pricing 32 teams off a single
game produces noise dressed as insight, but ignoring the games that HAVE been played is
wrong too. This module defines the single blending rule every in-season view uses:

    w = g / (g + K_GAMES)           g = games this team has played, K_GAMES = 4
    current = w * this_season + (1 - w) * last_season

    week 1 → 20% current · week 3 → 43% · week 4 → 50% · week 6 → 60% · week 12 → 75% · week 17 → 81%

That is the same idea DVOA uses (preseason projection weighted down as games accumulate).
K is "how many games of evidence last season is worth", and it was MEASURED, not guessed:
walk-forward 2021-25 (1,359 games), blending season-to-date point differential with the
prior season's and predicting the next game's margin —

    K            1       2       3       4       5       6       8      15   prior only
    MAE all   10.87   10.63   10.52   10.47   10.47   10.47   10.51   10.68   11.53
    wk 11-18  10.13   10.12   10.14   10.17   10.21   10.25   10.35   10.68   12.13

K = 4-6 are indistinguishable overall (0.01 pts); last season alone is clearly worse, so it
must stay in; and late in the year the optimum falls to K ≈ 2 as the prior goes stale.
K = 4 sits at the optimum's edge, beats K = 6 from week 11 on, and reaches a 75/25 split by
week 12. Re-run the measurement before moving it again; do not set a fixed split.

Three consumers, one rule:
  blended_styles(styles)      team_styles rows for the current season, blended per metric,
                              with labels/tags RE-ASSIGNED on the blended values
  adjusted_units()            opponent-adjusted unit EPA (ml.adjust) blended the same way
  performance_rating()        points-margin-equivalent of this season's adjusted net EPA,
                              plus its weight — folded into the power rankings by the server

Rolling by construction: everything keys off the newest pbp_{season}.parquet and the
schedule's completed games, so the daily refresh moves it with no code change.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

K_GAMES = 4          # games at which season-to-date earns 50% weight vs last season (measured, see above)
PLAYS_PER_GAME = 63  # EPA/play → points/game scale for the performance rating (≈ scrimmage plays)
_CACHE: dict = {}


def clear():
    _CACHE.clear()


def _pbp_seasons() -> list:
    return sorted(int(p.stem.split("_")[1]) for p in RAW.glob("pbp_*.parquet"))


def state() -> dict:
    """{season, prior, games: {team: g}, weeks_played, in_progress}.

    `season` is the newest season with play-by-play; it is IN PROGRESS when its schedule
    still has unplayed regular-season games. Before the first game of a new season there is
    no pbp file for it, so the newest is last season and nothing is blended."""
    if "state" in _CACHE:
        return _CACHE["state"]
    seasons = _pbp_seasons()
    season = seasons[-1] if seasons else 2025
    prior = season - 1 if (season - 1) in seasons else None
    games, weeks, in_progress = {}, 0, False
    sp = RAW / "schedules.parquet"
    if sp.exists():
        s = pd.read_parquet(sp, columns=["season", "game_type", "week", "home_team", "away_team", "home_score"])
        s = s[(s["season"] == season) & (s["game_type"].fillna("REG").str.upper() == "REG")]
        done = s[s["home_score"].notna()]
        in_progress = bool(len(done)) and bool(s["home_score"].isna().any())
        weeks = int(done["week"].max()) if len(done) else 0
        for col in ("home_team", "away_team"):
            for t, n in done[col].value_counts().items():
                games[t] = games.get(t, 0) + int(n)
    _CACHE["state"] = {"season": season, "prior": prior, "games": games,
                       "weeks_played": weeks, "in_progress": in_progress}
    return _CACHE["state"]


def weight(g: float) -> float:
    return float(g) / (float(g) + K_GAMES) if g and g > 0 else 0.0


def weights() -> pd.Series:
    """Per-team blend weight on THIS season (0 when nothing is blended)."""
    st = state()
    if not st["in_progress"] or st["prior"] is None:
        return pd.Series(dtype=float)
    return pd.Series({t: weight(g) for t, g in st["games"].items()})


def league_weight() -> float:
    w = weights()
    return round(float(w.mean()), 3) if len(w) else 0.0


# ── team_styles ───────────────────────────────────────────────────────
def blended_styles(styles: pd.DataFrame) -> pd.DataFrame:
    """Replace the current season's rows with per-metric blends toward last season, then
    re-assign the style labels and matchup tags on the blended numbers (a label from one
    game is the noisiest thing on the site). Returns the full table; untouched when the
    season is not in progress or last season is missing. Adds `blend_w` and keeps
    `games_played` as THIS season's count."""
    st = state()
    w = weights()
    cur, prior = st["season"], st["prior"]
    if styles is None or styles.empty or not len(w) or prior is None:
        return styles
    if not ((styles["season"] == cur).any() and (styles["season"] == prior).any()):
        return styles
    c = styles[styles["season"] == cur].set_index("team")
    p = styles[styles["season"] == prior].set_index("team")
    teams = c.index.intersection(p.index)
    num = [col for col in styles.columns
           if col not in ("team", "season", "games_played", "offense_label", "defense_label")
           and pd.api.types.is_numeric_dtype(styles[col]) and styles[col].dtype != bool]
    b = c.loc[teams].copy()
    wv = w.reindex(teams).fillna(0.0)
    for col in num:
        cv = pd.to_numeric(c.loc[teams, col], errors="coerce")
        pv = pd.to_numeric(p.loc[teams, col], errors="coerce")
        # a metric missing this season (e.g. no FTN data yet) falls back to last season's value
        b[col] = np.where(cv.notna(), wv * cv.fillna(0) + (1 - wv) * pv.fillna(cv.fillna(0)), pv)
    b["blend_w"] = wv.values
    b["season"] = cur
    b = b.reset_index()
    try:
        from engine.styles import assign_offense_label, assign_defense_label, assign_matchup_tags
        b = assign_matchup_tags(assign_defense_label(assign_offense_label(b)))
    except Exception:
        pass
    rest = styles[styles["season"] != cur]
    if "blend_w" not in rest.columns:
        rest = rest.assign(blend_w=np.nan)
    return pd.concat([rest, b], ignore_index=True)


# ── opponent-adjusted units ───────────────────────────────────────────
def adjusted_units() -> dict:
    """{team: {off_pass, off_rush, def_pass, def_rush}} for the CURRENT view: this season's
    ml.adjust ratings blended toward last season's by each team's games played. Same sign
    conventions as ml.adjust (offense + = good, defense − = good)."""
    if "adj" in _CACHE:
        return _CACHE["adj"]
    from ml.adjust import adjusted_unit_epa
    st = state()
    w = weights()
    cur = adjusted_unit_epa(st["season"])
    if not len(w) or st["prior"] is None:
        _CACHE["adj"] = cur
        return cur
    pri = adjusted_unit_epa(st["prior"])
    out = {}
    for t in sorted(set(cur) | set(pri)):
        wt = float(w.get(t, 0.0))
        rec = {}
        for k in ("off_pass", "off_rush", "def_pass", "def_rush"):
            cv, pv = cur.get(t, {}).get(k), pri.get(t, {}).get(k)
            if cv is None and pv is None:
                continue
            if cv is None:
                rec[k] = pv
            elif pv is None:
                rec[k] = cv
            else:
                rec[k] = round(wt * cv + (1 - wt) * pv, 4)
        out[t] = rec
    _CACHE["adj"] = out
    return out


# ── performance rating for the power rankings ─────────────────────────
def performance_rating() -> tuple:
    """(rating: Series[team] in points-margin units, weight: Series[team]).

    This season's opponent-adjusted net EPA/play — (off_pass + off_rush)/2 − (def_pass +
    def_rush)/2 — scaled by ~63 plays a game to the same points-per-game scale as the
    roster-talent rating, so the two can be averaged. Weight is the shrinkage weight; the
    server does rating = (1 − w)·talent + w·performance. Empty before the first game."""
    st = state()
    w = weights()
    if not len(w):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    from ml.adjust import adjusted_unit_epa
    cur = adjusted_unit_epa(st["season"])
    rows = {}
    for t, u in cur.items():
        off = np.nanmean([u.get("off_pass", np.nan), u.get("off_rush", np.nan)])
        de = np.nanmean([u.get("def_pass", np.nan), u.get("def_rush", np.nan)])
        if np.isnan(off) or np.isnan(de):
            continue
        rows[t] = (off - de) * PLAYS_PER_GAME
    r = pd.Series(rows)
    return r, w.reindex(r.index).fillna(0.0)
