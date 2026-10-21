"""
backtest_props.py  —  walk-forward backtest of the player-prop model, 2022-2025
================================================================================
What it proves, and what it cannot.

The live props model (ml/projections.py → ml/props.py) is built on the CURRENT season's files
(rosters_2026, depth_2026_current, injuries) and cannot be run as-of a past week. This script
re-implements its core from play-by-play alone, strictly AS-OF each game (nothing from the
game itself or later):
    • per-game usage = season-to-date blended with last season by w = g/(g+K_GAMES), last
      season shrunk toward a depth-rank role prior (rank by last season's usage on the team)
    • efficiency (yds/att, catch rate, TD rates…) shrunk toward replacement priors by sample
      size with the live K_PASS_ATT / K_CARRIES / K_TARGETS
    • an opponent pace factor (attempts faced per game vs league, half weight)
and projects every regular-season player-game 2022-2025 for the markets the board prices.
The spread rules and the market anchor are then tested on those projections.

Sections
  A. mean quality        MAE of the as-of projection vs the actual box score, against naive
                         baselines (last-season average, season-to-date average, their blend).
  B. spread calibration  Coverage of the central 50 % and 68 % bands per market under the OLD
                         fixed-CV spreads and the FITTED ones (ml.props._SD_FIT). The fit used
                         2024-26, so 2022-23 are out of sample.
  C. pick rule vs a      Lines set at the naive blend (half-point, −110 each side); hit rate
     naive market        and units by edge bucket for the raw model and the anchored one. A
                         naive market is softer than a real book, so these are UPPER bounds.
  D. real book lines     Only when data/raw/props_history.parquet exists (fetch_prop_history.py):
                         hit rate / units by edge bucket and market against the closing
                         consensus, anchor weights fitted walk-forward on EARLIER seasons only,
                         MAE of model vs book vs blend.

Usage
  python backtest_props.py                          # 2022-2025
  python backtest_props.py --seasons 2024 2025 --out data/processed/backtest_props_rows.parquet
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")       # Windows consoles default to cp1252
except Exception:
    pass
from ml.props import _sd, prob_over, book_implied_mean, ANCHOR_DEFAULT, ANCHOR_DEFAULT_BY_MARKET, ANCHOR_MIN, ANCHOR_MAX  # noqa: E402
from ml.projections import ROOKIE_QB, SKILL_PRIOR, K_PASS_ATT, K_CARRIES, K_TARGETS   # noqa: E402
from ml.current import K_GAMES                                                        # noqa: E402
from ml.ledger import units                                                           # noqa: E402

RAW = Path(__file__).parent / "data" / "raw"

# the spread rules in force before 2026-10-20 (fixed CV with a floor), kept for the comparison
_OLD_SD = {"pass_yds": (40, .28), "pass_att": (4.5, .16), "cmp": (3.5, .17), "rush_yds": (16, .55),
           "carries": (3, .32), "rec_yds": (14, .62), "rushrec_yds": (20, .50)}


def old_sd(mk, m):
    f, cv = _OLD_SD[mk]
    return max(f, cv * max(0.0, m))


STATS = ["pass_att", "cmp", "pass_yds", "pass_td", "int", "carries", "rush_yds", "rush_td",
         "targets", "rec", "rec_yds", "rec_td"]
NORMAL = {"pass_yds", "pass_att", "cmp", "rush_yds", "carries", "rec_yds", "rushrec_yds"}
POISSON = {"pass_td", "int"}
NBINOM = {"rec"}                      # receptions: negative binomial, var = REC_PHI × mean (ml.props)
COUNTS = POISSON | NBINOM
MARKETS_BY_POS = {"QB": ["pass_yds", "pass_td", "pass_att", "cmp", "int", "rush_yds"],
                  "RB": ["rush_yds", "carries", "rec", "rec_yds", "rushrec_yds", "anytime_td"],
                  "WR": ["rec", "rec_yds", "anytime_td"], "TE": ["rec", "rec_yds", "anytime_td"]}
PBP_COLS = ["game_id", "week", "play_id", "season_type", "posteam", "defteam", "down", "play_type",
            "touchdown", "return_touchdown", "complete_pass", "incomplete_pass", "interception",
            "yards_gained", "passer_player_id", "rusher_player_id", "receiver_player_id", "rush_attempt"]


# ── box scores, per (game, team, player), plus team volume per game ───────────────
def boxes(season: int):
    import pyarrow.parquet as pq
    have = set(pq.read_schema(RAW / f"pbp_{season}.parquet").names)      # older pulls lack a few columns
    p = pd.read_parquet(RAW / f"pbp_{season}.parquet", columns=[c for c in PBP_COLS if c in have])
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    p = p[(p["week"] <= 18) & p["down"].notna()].copy()
    ret = p["return_touchdown"].fillna(0) if "return_touchdown" in p.columns else 0
    p["is_td"] = (p["touchdown"].fillna(0) == 1) & (ret != 1)
    key = ["game_id", "week", "posteam", "defteam"]

    pa = p[p["play_type"].isin(["pass", "qb_spike"])].copy()
    pa["att"] = ((pa["complete_pass"] == 1) | (pa["incomplete_pass"] == 1) | (pa["interception"] == 1)).astype(int)
    pa["cmp_yds"] = pa["yards_gained"].fillna(0) * pa["complete_pass"].fillna(0)
    pa["ptd"] = (pa["is_td"] & (pa["complete_pass"] == 1)).astype(int)
    passing = pa.dropna(subset=["passer_player_id"]).groupby(key + ["passer_player_id"]).agg(
        pass_att=("att", "sum"), cmp=("complete_pass", "sum"), pass_yds=("cmp_yds", "sum"),
        pass_td=("ptd", "sum"), int=("interception", "sum")).reset_index().rename(columns={"passer_player_id": "player_id"})
    # the starter = the first passer of the game for that team (a pre-game fact, no hindsight on injuries)
    first = pa.dropna(subset=["passer_player_id"]).sort_values("play_id").groupby(["game_id", "posteam"]).head(1)
    starters = set(zip(first["game_id"], first["posteam"], first["passer_player_id"]))

    ru = p[p["play_type"].isin(["run", "qb_kneel"])].copy()
    ru["rtd"] = ru["is_td"].astype(int)
    rushing = ru.dropna(subset=["rusher_player_id"]).groupby(key + ["rusher_player_id"]).agg(
        carries=("rush_attempt", "sum"), rush_yds=("yards_gained", "sum"), rush_td=("rtd", "sum")).reset_index() \
        .rename(columns={"rusher_player_id": "player_id"})

    rc = pa[pa["receiver_player_id"].notna()].copy()
    rc["rectd"] = (rc["is_td"] & (rc["complete_pass"] == 1)).astype(int)
    receiving = rc.groupby(key + ["receiver_player_id"]).agg(
        targets=("att", "sum"), rec=("complete_pass", "sum"), rec_yds=("cmp_yds", "sum"),
        rec_td=("rectd", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"})

    box = passing.merge(rushing, on=key + ["player_id"], how="outer").merge(receiving, on=key + ["player_id"], how="outer")
    box[STATS] = box[STATS].fillna(0).astype(float)
    box["qb_start"] = [(g, t, pid) in starters for g, t, pid in zip(box["game_id"], box["posteam"], box["player_id"])]

    teams = pa.groupby(key)["att"].sum().rename("team_pa").reset_index() \
        .merge(ru.groupby(key)["rush_attempt"].sum().rename("team_ra").reset_index(), on=key, how="outer").fillna(0)
    return box, teams


def positions(season: int) -> pd.Series:
    r = pd.read_parquet(RAW / "rosters_weekly.parquet", columns=["season", "player_id", "position"])
    r = r[(r["season"] == season) & r["player_id"].notna()]
    pos = r.groupby("player_id")["position"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
    return pos.replace({"FB": "RB", "HB": "RB"})


def player_names(seasons) -> pd.Series:
    r = pd.read_parquet(RAW / "rosters_weekly.parquet", columns=["season", "player_id", "player_name"])
    r = r[r["season"].isin(list(seasons) + [max(seasons) + 1])].dropna()
    return r.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]


# ── as-of projections for one season ─────────────────────────────────────────────
def season_rows(S: int, prev_box: pd.DataFrame, prev_teams: pd.DataFrame) -> pd.DataFrame:
    box, teams = boxes(S)
    pos = positions(S)
    box["pos"] = box["player_id"].map(pos)
    # usage-inferred position where the roster has none (practice-squad call-ups etc.)
    inf = np.where(box["pass_att"] >= 5, "QB", np.where(box["carries"] > box["targets"], "RB", "WR"))
    box["pos"] = box["pos"].fillna(pd.Series(inf, index=box.index))
    box = box[box["pos"].isin(["QB", "RB", "WR", "TE"])].copy()

    # last season per player: totals + games (regular season)
    P = prev_box.groupby("player_id")[STATS].sum()
    P["g1"] = prev_box.groupby("player_id").size()
    for c in STATS:
        box[c + "_p"] = box["player_id"].map(P[c]).fillna(0.0)
    box["g1"] = box["player_id"].map(P["g1"]).fillna(0.0)

    # season-to-date per player, BEFORE this game
    box = box.sort_values(["player_id", "week"]).reset_index(drop=True)
    cum = box.groupby("player_id")[STATS].cumsum() - box[STATS]
    for c in STATS:
        box[c + "_s"] = cum[c].values
    box["g2p"] = box.groupby("player_id").cumcount().astype(float)

    # team games before this one, and the team's / opponent's pace so far
    teams = teams.sort_values(["posteam", "week"]).reset_index(drop=True)
    teams["tg"] = teams.groupby("posteam").cumcount().astype(float)
    tcum = teams.groupby("posteam")[["team_pa", "team_ra"]].cumsum() - teams[["team_pa", "team_ra"]]
    teams["pa_s"], teams["ra_s"] = tcum["team_pa"], tcum["team_ra"]
    d = teams.sort_values(["defteam", "week"]).reset_index(drop=True)
    d["dg"] = d.groupby("defteam").cumcount().astype(float)
    dcum = d.groupby("defteam")[["team_pa", "team_ra"]].cumsum() - d[["team_pa", "team_ra"]]
    d["dpa_s"], d["dra_s"] = dcum["team_pa"], dcum["team_ra"]
    teams = teams.merge(d[["game_id", "posteam", "dg", "dpa_s", "dra_s"]], on=["game_id", "posteam"])
    box = box.merge(teams[["game_id", "posteam", "tg", "dg", "dpa_s", "dra_s"]], on=["game_id", "posteam"], how="left")
    tpg1 = prev_teams.groupby("posteam")[["team_pa", "team_ra"]].mean()
    dpg1 = prev_teams.groupby("defteam")[["team_pa", "team_ra"]].mean()
    lg_pa, lg_ra = float(tpg1["team_pa"].mean()), float(tpg1["team_ra"].mean())

    # opponent pace factor: attempts faced per game vs league, prior season blended with this one
    wd = box["dg"] / (box["dg"] + K_GAMES)
    opp_pa = wd * (box["dpa_s"] / box["dg"].clip(lower=1)).where(box["dg"] > 0, 0) + (1 - wd) * box["defteam"].map(dpg1["team_pa"]).fillna(lg_pa)
    opp_ra = wd * (box["dra_s"] / box["dg"].clip(lower=1)).where(box["dg"] > 0, 0) + (1 - wd) * box["defteam"].map(dpg1["team_ra"]).fillna(lg_ra)
    f_pass = 1 + 0.5 * (opp_pa / lg_pa - 1)
    f_rush = 1 + 0.5 * (opp_ra / lg_ra - 1)

    # role prior: last season's per-game usage at the position by depth rank (rank = the
    # player's last-season usage among his CURRENT team's players at the position; the
    # starting QB is rank 1 by definition, rookies/new faces rank last)
    q_of_rank = {1: 0.65, 2: 0.30, 3: 0.15, 4: 0.05}
    base = P[P["g1"] >= 4].copy()
    base["pos"] = base.index.map(pos)
    vol_of = {"QB": "pass_att", "RB": "carries", "WR": "targets", "TE": "targets"}
    w1 = box["g1"] / (box["g1"] + K_GAMES)
    w2 = box["tg"] / (box["tg"] + K_GAMES)
    for v in ("pass_att", "carries", "targets"):
        pg1 = (box[v + "_p"] / box["g1"].clip(lower=1)).where(box["g1"] > 0, 0.0)
        box["_rk"] = box.groupby(["game_id", "posteam", "pos"])[v + "_p"].rank(ascending=False, method="first")
        box.loc[box["qb_start"] & (box["pos"] == "QB"), "_rk"] = 1
        rk = box["_rk"].fillna(4).clip(1, 4).astype(int)
        prior = pd.Series(0.0, index=box.index)
        for p_ in ("QB", "RB", "WR", "TE"):
            sub = base[base["pos"] == p_]
            if not len(sub):
                continue
            bpg = sub[v] / sub["g1"]
            for r_, q_ in q_of_rank.items():
                m = (box["pos"] == p_) & (rk == r_)
                prior[m] = float(bpg.quantile(q_))
        last = w1 * pg1 + (1 - w1) * prior
        pg2 = (box[v + "_s"] / box["tg"].clip(lower=1)).where(box["tg"] > 0, 0.0)
        box["vol_" + v] = w2 * pg2 + (1 - w2) * last
    # the starting QB throws the TEAM's passes (as the live _distribute does: team pace, not his
    # own history — a backup promoted to starter is not projected off his mop-up games);
    # team pace = last season blended with this one by games played
    wt = box["tg"] / (box["tg"] + K_GAMES)
    tp = teams.set_index(["game_id", "posteam"]).loc[list(zip(box["game_id"], box["posteam"])), "pa_s"].values
    team_pa_now = pd.Series(tp, index=box.index) / box["tg"].clip(lower=1)
    team_pa_prior = box["posteam"].map(tpg1["team_pa"]).fillna(lg_pa)
    team_pa = wt * team_pa_now.where(box["tg"] > 0, 0.0) + (1 - wt) * team_pa_prior
    is_starter = box["qb_start"] & (box["pos"] == "QB")
    att = (box["vol_pass_att"] * f_pass).where(~is_starter, team_pa * f_pass)
    car = box["vol_carries"] * f_rush
    tgt = box["vol_targets"] * f_pass

    # efficiency, shrunk toward the live priors by sample size (last season + season to date)
    def shrink(num, den, prior, k):
        return (num + k * prior) / (den + k)
    T = {c: box[c + "_p"] + box[c + "_s"] for c in STATS}
    cmp_pct = shrink(T["cmp"], T["pass_att"], ROOKIE_QB["cmp_pct"], K_PASS_ATT)
    ypa = shrink(T["pass_yds"], T["pass_att"], ROOKIE_QB["ypa"], K_PASS_ATT)
    ptd = shrink(T["pass_td"], T["pass_att"], ROOKIE_QB["ptd_pa"], K_PASS_ATT)
    ipa = shrink(T["int"], T["pass_att"], ROOKIE_QB["int_pa"], K_PASS_ATT)
    ypc = shrink(T["rush_yds"], T["carries"], SKILL_PRIOR["ypc"], K_CARRIES)
    rtd = shrink(T["rush_td"], T["carries"], SKILL_PRIOR["rtd_carry"], K_CARRIES)
    catch = shrink(T["rec"], T["targets"], SKILL_PRIOR["catch_pct"], K_TARGETS)
    ypt = shrink(T["rec_yds"], T["targets"], SKILL_PRIOR["ypt"], K_TARGETS)
    rectd = shrink(T["rec_td"], T["targets"], SKILL_PRIOR["rectd_tgt"], K_TARGETS)

    mean = {"pass_att": att, "cmp": att * cmp_pct, "pass_yds": att * ypa, "pass_td": att * ptd, "int": att * ipa,
            "carries": car, "rush_yds": car * ypc, "rec": tgt * catch, "rec_yds": tgt * ypt}
    mean["rushrec_yds"] = mean["rush_yds"] + mean["rec_yds"]
    mean["anytime_td"] = 1 - np.exp(-(car * rtd + tgt * rectd))

    # naive baselines per stat: last-season per-game avg, season-to-date per-game avg, their blend
    wn = box["g2p"] / (box["g2p"] + K_GAMES)
    naive = {}
    for c in STATS:
        pg1 = (box[c + "_p"] / box["g1"].clip(lower=1)).where(box["g1"] > 0, np.nan)
        pg2 = (box[c + "_s"] / box["g2p"].clip(lower=1)).where(box["g2p"] > 0, np.nan)
        blend = (wn * pg2.fillna(0) + (1 - wn) * pg1.fillna(pg2)).where(pg1.notna() | pg2.notna(), np.nan)
        naive[c] = (pg1, pg2, blend)
    naive["rushrec_yds"] = tuple(naive["rush_yds"][i] + naive["rec_yds"][i] for i in range(3))
    naive["anytime_td"] = tuple(1 - np.exp(-(naive["rush_td"][i].fillna(0) + naive["rec_td"][i].fillna(0)))
                                .where(naive["rush_td"][i].notna() | naive["rec_td"][i].notna(), np.nan) for i in range(3))

    # the board: the starting QB, and skill players with a real projected role
    on = ((box["pos"] == "QB") & box["qb_start"] & (att >= 15)) | \
         ((box["pos"] == "RB") & ((car >= 6) | (tgt >= 3))) | \
         (box["pos"].isin(["WR", "TE"]) & (tgt >= 3))
    actual = {c: box[c] for c in STATS}
    actual["rushrec_yds"] = box["rush_yds"] + box["rec_yds"]
    actual["anytime_td"] = ((box["rush_td"] + box["rec_td"]) >= 1).astype(float)
    rows = []
    for p_, mks in MARKETS_BY_POS.items():
        sel = on & (box["pos"] == p_)
        if not sel.any():
            continue
        for mk in mks:
            sub = pd.DataFrame({"season": S, "week": box.loc[sel, "week"].values, "game_id": box.loc[sel, "game_id"].values,
                                "team": box.loc[sel, "posteam"].values, "player_id": box.loc[sel, "player_id"].values,
                                "pos": p_, "market": mk, "mean": mean[mk][sel].values, "actual": actual[mk][sel].values,
                                "g1": box.loc[sel, "g1"].values, "tg": box.loc[sel, "tg"].values,
                                "naive_prior": naive[mk][0][sel].values, "naive_std": naive[mk][1][sel].values,
                                "naive": naive[mk][2][sel].values})
            rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["dist"] = np.where(out["market"].isin(NORMAL), "normal",
                           np.where(out["market"].isin(POISSON), "poisson", np.where(out["market"].isin(NBINOM), "nbinom", "prob")))
    out["sd_new"] = [(_sd(mk, m) if mk in NORMAL else np.nan) for mk, m in zip(out["market"], out["mean"])]
    out["sd_old"] = [(old_sd(mk, m) if mk in NORMAL else np.nan) for mk, m in zip(out["market"], out["mean"])]
    return out, box, teams


# ── probability helpers (vectorised where it matters) ─────────────────────────
def _count_cdf_vec(k: np.ndarray, mean: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """CDF at k for count rows: Poisson, or the model's negative binomial (ml.props)."""
    from ml.props import _pois_cdf, _nb_pmf_cdf, REC_PHI
    out = np.zeros(len(k))
    for i, (kk, mm, dd) in enumerate(zip(k, mean, dist)):
        kk = int(kk)
        if kk < 0:
            continue
        out[i] = _nb_pmf_cdf(kk, mm, REC_PHI)[1] if dd == "nbinom" else _pois_cdf(kk, max(1e-6, mm))
    return out


def p_over(df: pd.DataFrame, mean_col: str, sd_col: str, line: np.ndarray) -> np.ndarray:
    """P(actual > line) under the row's distribution (half-point lines: no pushes)."""
    from math import erf, sqrt
    m = df[mean_col].to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    nm = (df["dist"] == "normal").to_numpy()
    sd = df[sd_col].to_numpy(dtype=float)
    z = (line[nm] - m[nm]) / np.maximum(1e-6, sd[nm])
    out[nm] = 1 - 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
    po = df["dist"].isin(["poisson", "nbinom"]).to_numpy()
    out[po] = 1 - _count_cdf_vec(np.floor(line[po]), m[po], df["dist"].to_numpy()[po])
    pr = (df["dist"] == "prob").to_numpy()
    out[pr] = m[pr]
    return out


def pit(df: pd.DataFrame, sd_col: str, rng: np.random.Generator, counts_as: str | None = None) -> np.ndarray:
    """Probability integral transform of the actual under the row's distribution (randomised
    for counts): calibrated spreads give u ~ Uniform(0, 1). `counts_as="poisson"` forces the
    old Poisson on every count market (the before/after comparison)."""
    from math import erf, sqrt
    m = df["mean"].to_numpy(dtype=float)
    a = df["actual"].to_numpy(dtype=float)
    u = np.full(len(df), np.nan)
    nm = (df["dist"] == "normal").to_numpy()
    sd = df[sd_col].to_numpy(dtype=float)
    u[nm] = 0.5 * (1 + np.vectorize(erf)(((a[nm] - m[nm]) / np.maximum(1e-6, sd[nm])) / sqrt(2)))
    po = df["dist"].isin(["poisson", "nbinom"]).to_numpy()
    dist = df["dist"].to_numpy()[po] if counts_as is None else np.full(po.sum(), counts_as)
    hi = _count_cdf_vec(a[po], m[po], dist)
    lo = _count_cdf_vec(a[po] - 1, m[po], dist)
    u[po] = lo + rng.random(po.sum()) * (hi - lo)
    return u


def bucket(edge: pd.Series) -> pd.Series:
    return pd.cut(edge, [0.05, 0.10, 0.20, 1.0], right=False, labels=["5-10", "10-20", "20+"])


def tally(df: pd.DataFrame, by):
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({"n": g.size(), "hit%": (g["won"].mean() * 100).round(1), "units": g["units"].sum().round(1)})
    t["roi%"] = (t["units"] / t["n"] * 100).round(1)
    return t


# ── sections ───────────────────────────────────────────────────────────────────
def section_a(df: pd.DataFrame):
    print("\n" + "=" * 100 + "\nA. MEAN QUALITY — mean absolute error vs the actual box score (lower is better)\n" + "=" * 100)
    rows = []
    for mk, g in df.groupby("market"):
        g = g.dropna(subset=["naive"])
        rows.append({"market": mk, "n": len(g), "mean actual": round(g["actual"].mean(), 2),
                     "bias (model−actual)": round((g["mean"] - g["actual"]).mean(), 2),
                     "MAE model": round((g["mean"] - g["actual"]).abs().mean(), 2),
                     "MAE last-season avg": round((g["naive_prior"] - g["actual"]).abs().mean(), 2),
                     "MAE season-to-date": round((g["naive_std"] - g["actual"]).abs().mean(), 2),
                     "MAE naive blend": round((g["naive"] - g["actual"]).abs().mean(), 2)})
    print(pd.DataFrame(rows).set_index("market").to_string())
    print("\nby season, MAE model / naive blend:")
    t = df.dropna(subset=["naive"]).groupby(["market", "season"]).apply(
        lambda g: pd.Series({"model": (g["mean"] - g["actual"]).abs().mean(), "naive": (g["naive"] - g["actual"]).abs().mean()}),
        include_groups=False).round(2).unstack("season")
    print(t.to_string())


def section_b(df: pd.DataFrame, rng):
    print("\n" + "=" * 100 + "\nB. SPREAD CALIBRATION — share of actuals inside the central 50 % / 68 % bands (targets 50 / 68)\n" + "=" * 100)
    d = df[df["dist"] != "prob"].copy()
    d["u_new"] = pit(d, "sd_new", rng)
    d["u_old"] = pit(d, "sd_old", rng, counts_as="poisson")      # every count market was Poisson before

    def cov(u):
        return pd.Series({"50% band": round(((u >= .25) & (u <= .75)).mean() * 100, 1),
                          "68% band": round(((u >= .16) & (u <= .84)).mean() * 100, 1),
                          "mean u (0.5 = unbiased)": round(u.mean(), 3)})
    rows = []
    for mk, g in d.groupby("market"):
        o, n = cov(g["u_old"]), cov(g["u_new"])
        rows.append({"market": mk, "n": len(g), "old 50%": o["50% band"], "new 50%": n["50% band"],
                     "old 68%": o["68% band"], "new 68%": n["68% band"], "mean u new": n["mean u (0.5 = unbiased)"]})
    print(pd.DataFrame(rows).set_index("market").to_string())
    # the scale that WOULD calibrate each market: a projection carries its own error on top of
    # the game-to-game spread, so the predictive sd = game sd × k; k = median |z| / 0.6745
    print("\nscale k that calibrates the 50 % band (predictive sd = fitted game sd × k), normal markets:")
    rows = []
    for mk, g in d[d["dist"] == "normal"].groupby("market"):
        z = (g["actual"] - g["mean"]) / g["sd_new"]
        zc = z - z.median()                                   # k for spread, separate from bias
        rows.append({"market": mk, "n": len(g), "bias (sd units)": round(z.median(), 3),
                     "k": round(float(zc.abs().median() / 0.6745), 3),
                     "k (2022-23)": round(float((zc[g["season"] <= 2023]).abs().median() / 0.6745), 3),
                     "k (2024-25)": round(float((zc[g["season"] >= 2024]).abs().median() / 0.6745), 3)})
    print(pd.DataFrame(rows).set_index("market").to_string())
    print("\nQB bias by evidence (median z, by last-season games g1):")
    q = d[d["market"].isin(["pass_att", "pass_yds", "cmp"])].copy()
    q["g1b"] = pd.cut(q["g1"], [-1, 0, 5, 11, 20], labels=["none", "1-5", "6-11", "12+"])
    print(q.groupby(["market", "g1b"], observed=True).apply(
        lambda g: pd.Series({"n": len(g), "median z": ((g["actual"] - g["mean"]) / g["sd_new"]).median().round(3),
                             "bias": (g["mean"] - g["actual"]).mean().round(2)}), include_groups=False).unstack("g1b").to_string())
    print("\nout-of-sample seasons only (2022-23; the fit used 2024-26), new spreads:")
    oos = d[d["season"] <= 2023]
    if len(oos):
        rows = []
        for mk, g in oos.groupby("market"):
            n = cov(g["u_new"])
            rows.append({"market": mk, "n": len(g), "50% band": n["50% band"], "68% band": n["68% band"], "mean u": n["mean u (0.5 = unbiased)"]})
        print(pd.DataFrame(rows).set_index("market").to_string())


def section_c(df: pd.DataFrame):
    print("\n" + "=" * 100 + "\nC. PICK RULE vs a NAIVE MARKET — line = naive blend at a half point, −110 both sides (UPPER bound)\n" + "=" * 100)
    d = df.dropna(subset=["naive"]).copy()
    d = d[d["naive"] > 0]
    ln = np.where(d["dist"] == "prob", 0.5, np.floor(d["naive"]) + 0.5)
    d["line"] = ln
    # the naive book's number: for a −110/−110 half-point line the market-implied mean IS the line
    d["book_mean"] = np.where(d["dist"] == "prob", d["naive"], d["line"])
    d["book_p"] = np.where(d["dist"] == "prob", d["naive"], 0.5)
    w = d["market"].map(lambda m: ANCHOR_DEFAULT_BY_MARKET.get(m, ANCHOR_DEFAULT)).to_numpy()
    d["mean_anch"] = (1 - w) * d["mean"] + w * d["book_mean"]
    variants = {"raw model, old spreads": ("mean", "sd_old"), "raw model, fitted spreads": ("mean", "sd_new"),
                "anchored, fitted spreads": ("mean_anch", "sd_new")}
    won_over = (d["actual"] > d["line"]).to_numpy()
    for name, (mc, sc) in variants.items():
        po = p_over(d, mc, sc, d["line"].to_numpy(dtype=float))
        over_side = po >= (1 - po) if True else None
        prob = np.where(po >= 0.5, po, 1 - po)
        book = np.where(po >= 0.5, d["book_p"], 1 - d["book_p"])
        e = d.assign(edge=prob - book, side_over=po >= 0.5)
        e["won"] = np.where(e["side_over"], won_over, ~won_over)
        e = e[(e["dist"] != "prob") | e["side_over"]]           # anytime TD: YES bets only, like the board
        e = e[e["edge"] >= 0.05]
        odds = np.where(e["dist"] == "prob", np.vectorize(lambda p: -100 * p / (1 - p) if p >= .5 else 100 * (1 - p) / p)(e["book_p"].clip(.02, .98)), -110.0)
        e["units"] = [units(o, bool(wn)) for o, wn in zip(odds, e["won"])]
        e["bucket"] = bucket(e["edge"])
        print(f"\n— {name} —  ({len(e)} picks at edge ≥ 5 pts out of {len(d)} markets)")
        print(tally(e, "bucket").to_string())
        print(tally(e, "market").to_string())


def section_d(df: pd.DataFrame, names: pd.Series):
    path = RAW / "props_history.parquet"
    if not path.exists():
        print("\n" + "=" * 100 + "\nD. REAL BOOK LINES — skipped: data/raw/props_history.parquet not found (run fetch_prop_history.py)\n" + "=" * 100)
        return
    from ml.odds import consensus_price, _namekey
    print("\n" + "=" * 100 + "\nD. REAL BOOK LINES — closing consensus, anchor weight fitted on EARLIER seasons only\n" + "=" * 100)
    h = pd.read_parquet(path)
    h["pk"] = h["player"].map(_namekey)
    rows = []
    for (gid, mk, pk), g in h.groupby(["game_id", "market", "pk"]):
        if mk == "anytime_td":
            yo = consensus_price([p for p in g["price"] if p is not None])
            if yo is not None:
                rows.append({"game_id": gid, "market": mk, "pk": pk, "line": 0.5, "over_odds": yo, "under_odds": None})
            continue
        pts = g["line"].dropna()
        if not len(pts):
            continue
        med = float(pts.median())
        at = g[(g["line"] - med).abs() < 1e-6]
        o = consensus_price([p for p in at.loc[at["side"] == "over", "price"] if p is not None])
        u = consensus_price([p for p in at.loc[at["side"] == "under", "price"] if p is not None])
        if o is None or u is None:
            continue
        rows.append({"game_id": gid, "market": mk, "pk": pk, "line": med, "over_odds": o, "under_odds": u})
    book = pd.DataFrame(rows)
    d = df.copy()
    d["pk"] = d["player_id"].map(names).map(_namekey)
    d = d.merge(book, on=["game_id", "market", "pk"], how="inner")
    print(f"{len(d):,} model rows matched to a closing line ({d['season'].min()}-{d['season'].max()})")

    def imp(o):
        o = float(o)
        return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)
    d["p_o_book"] = [imp(o) for o in d["over_odds"]]
    d["p_u_book"] = [imp(u) if u is not None and not pd.isna(u) else None for u in d["under_odds"]]
    nv = np.where(d["dist"] == "prob", d["p_o_book"], d["p_o_book"] / (d["p_o_book"] + d["p_u_book"].fillna(0.5)))
    d["novig_over"] = nv
    d["book_mean"] = [book_implied_mean({"dist": ds, "sd": sd}, ln, p) if ds != "prob" else p
                      for ds, sd, ln, p in zip(d["dist"], d["sd_new"], d["line"], d["novig_over"])]
    d = d.dropna(subset=["book_mean"])
    # walk-forward anchor weight per market: least squares on the seasons BEFORE the one priced
    d["w"] = np.nan
    fitted = {}
    for S in sorted(d["season"].unique()):
        prev = d[d["season"] < S]
        for mk, g in d[d["season"] == S].groupby("market"):
            pg = prev[prev["market"] == mk]
            default = ANCHOR_DEFAULT_BY_MARKET.get(mk, ANCHOR_DEFAULT)
            w = default
            if len(pg) >= 100:
                dd = (pg["book_mean"] - pg["mean"]).to_numpy(dtype=float)
                rr = (pg["actual"] - pg["mean"]).to_numpy(dtype=float)
                if (dd * dd).sum() > 0:
                    w = float(min(ANCHOR_MAX, max(ANCHOR_MIN, (dd * rr).sum() / (dd * dd).sum())))
            fitted[(S, mk)] = w
            d.loc[(d["season"] == S) & (d["market"] == mk), "w"] = w
    d["mean_anch"] = (1 - d["w"]) * d["mean"] + d["w"] * d["book_mean"]
    print("\nmeasured anchor weight per market (all matched seasons pooled) and MAE model / book / blend:")
    rows = []
    for mk, g in d.groupby("market"):
        dd = (g["book_mean"] - g["mean"]).to_numpy(dtype=float)
        rr = (g["actual"] - g["mean"]).to_numpy(dtype=float)
        w = (dd * rr).sum() / (dd * dd).sum() if (dd * dd).sum() > 0 else np.nan
        blend = (1 - min(ANCHOR_MAX, max(ANCHOR_MIN, w))) * g["mean"] + min(ANCHOR_MAX, max(ANCHOR_MIN, w)) * g["book_mean"]
        rows.append({"market": mk, "n": len(g), "w (LS)": round(w, 2),
                     "MAE model": round((g["mean"] - g["actual"]).abs().mean(), 2),
                     "MAE book": round((g["book_mean"] - g["actual"]).abs().mean(), 2),
                     "MAE blend": round((blend - g["actual"]).abs().mean(), 2)})
    print(pd.DataFrame(rows).set_index("market").to_string())
    won_over = (d["actual"] > d["line"]).to_numpy()
    push = (d["actual"] == d["line"]).to_numpy()
    for name, mc in (("raw model (fitted spreads)", "mean"), ("anchored, walk-forward weight", "mean_anch")):
        po = p_over(d, mc, "sd_new", d["line"].to_numpy(dtype=float))
        side_over = po >= 0.5
        prob = np.where(side_over, po, 1 - po)
        bookp = np.where(side_over, d["novig_over"], 1 - d["novig_over"])
        e = d.assign(edge=prob - bookp, side_over=side_over, push=push)
        e["won"] = np.where(e["side_over"], won_over, ~won_over)
        e = e[(e["dist"] != "prob") | e["side_over"]]
        e = e[(e["edge"] >= 0.05) & ~e["push"]]
        odds = np.where(e["side_over"], e["over_odds"], e["under_odds"].fillna(-110))
        e["units"] = [units(float(o), bool(wn)) for o, wn in zip(odds, e["won"])]
        e["bucket"] = bucket(e["edge"])
        print(f"\n— {name} —  ({len(e)} picks at edge ≥ 5 pts out of {len(d)} lined markets)")
        print(tally(e, "bucket").to_string())
        print(tally(e, "market").to_string())
        print(tally(e, "season").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--out", default=None, help="write the per-row frame (parquet) for further analysis")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    frames = []
    prev = None
    for S in sorted(a.seasons):
        if prev is None or prev[0] != S - 1:
            prev = (S - 1,) + boxes(S - 1)
        rows, box, teams = season_rows(S, prev[1], prev[2])
        print(f"{S}: {len(rows):,} market rows from {rows['game_id'].nunique()} games")
        frames.append(rows)
        prev = (S, box, teams)
    df = pd.concat(frames, ignore_index=True)
    if a.out:
        df.to_parquet(a.out, index=False)
    section_a(df)
    section_b(df, rng)
    section_c(df)
    section_d(df, player_names(a.seasons))


if __name__ == "__main__":
    main()
