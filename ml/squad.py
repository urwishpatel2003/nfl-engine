"""
ml/squad.py  —  roster-talent power ratings for the upcoming season
====================================================================
Rates each team by the talent CURRENTLY on its roster + coaching, NOT by last
year's results (offseason roster/coaching turnover makes prior results stale;
2025->2026 strength persistence is only r=0.35). This is the preseason product
ranking the user asked for ("based on squad and coaching strength").

Method — map the current roster to each player's 2025 performance, z-score within
position group, aggregate the group talents with weights, add coaching:

  QB        composite (gsis; name fallback)     weight 0.34   -- dominates
  WR/TE/RB  composite (top starters)            weight 0.18
  OL        team 2025 pass-block/run-block      weight 0.12   -- team-level (continuity)
  pass rush current-roster DL 2025 pressure     weight 0.14   -- name-matched
  coverage  current-roster DB 2025 (pfr)        weight 0.14   -- name-matched
  coaching  2025 coaching_score (neutral=new)   weight 0.08

Data caveats: offense maps cleanly; defense is name-matched (noisier); OL is only
team-level; rookies/no-2025 default to replacement. Honest about being a projection.

Usage:
  python -m ml.squad                 # 2026 roster-talent ranking
  python -m ml.squad --breakdown     # show per-team component scores
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

# QB is the dominant position in the NFL, so it carries the most weight. Coaching is
# down-weighted: it's noisy, partly results-derived, and blind to 2026 staff changes.
# Team-quality weights. Offense (0.59) = QB + skill + OL, with OL weighted heavily (0.14) because a
# QB/RB only produce behind protection. Defense (0.34) is a deliberate ROSTER + PERFORMANCE blend:
# team-EPA performance (def_team, 0.14 — includes run defense, which isn't cleanly measurable per
# player) PLUS current-roster individual defenders (pass rush 0.09 + coverage 0.11 = 0.20, given the
# larger share). Coaching 0.07. (Base was calibrated to ESPN FPI; this rebalances OL up + defense
# toward roster per design intent, so the FPI correlation loosens slightly by choice.)
WEIGHTS = {"qb": 0.33, "skill": 0.12, "ol": 0.14,
           "def_team": 0.14, "rush": 0.09, "cover": 0.11, "coach": 0.07}
RATING_SCALE = 9.0   # maps the blended z-score to ~points; sets the spread of the ranking


def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def _key(name):
    """Robust match key: lastname + first initial (handles Greg/Gregory, suffixes)."""
    parts = [p for p in re.sub(r"[^a-z ]", "", str(name).lower()).split() if p not in _SUFFIX]
    if not parts:
        return ""
    return parts[-1] + (parts[0][0] if parts[0] else "")


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0


# ── player-quality tables (2025) ────────────────────────────────────
def _composite_2025():
    # Use season-MEAN composite_score (pure talent) not last-week adjusted_score, which
    # multiplies in end-of-season injury/depth status (e.g. it zeroed out Jayden Daniels).
    c = pd.read_parquet(PROC / "composite_scores.parquet")
    c = c[c.season == 2025]
    talent_col = "composite_score" if "composite_score" in c.columns else "adjusted_score"
    agg = c.groupby("player_id").agg(adjusted_score=(talent_col, "mean"),
                                     name=("player_display_name", "last"),
                                     position=("position", "last")).reset_index()
    agg["nm"] = agg["name"].map(_norm)
    agg["k"] = agg["name"].map(_key)
    return agg[["player_id", "nm", "k", "position", "adjusted_score"]]


# ── play-by-play player values ──────────────────────────────────────
# seasonal_stats.parquet lags a year behind (no current season) and the composite's
# efficiency/usage components are flat, so QBs missed their latest season and volume RBs
# were buried. These value tables are computed straight from PBP (which DOES include the
# current season) — recency-weighted passing+rushing EPA for QBs, and real production
# (yards + TDs + touches + EPA) for skill players.
_PBP_W = {2025: 0.32, 2024: 0.24, 2023: 0.18, 2022: 0.14, 2021: 0.12}   # 5-yr recency weights
_COV_SHRINK = 35.0   # targets of regression toward league-average coverage (tames small samples)
_PBP_AGG = None
_SKILL_CACHE = None


def _pbp_agg():
    """Per-player rushing / receiving / passing aggregates for each weighted season."""
    global _PBP_AGG
    if _PBP_AGG is not None:
        return _PBP_AGG
    rush, rec, pas, names = [], [], [], {}
    for s, w in _PBP_W.items():
        p = RAW / f"pbp_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if "season" in d.columns:
            d = d[d["season"] == s]
        if "week" in d.columns:
            d = d[d["week"] <= 18]
        ru = d[d["rush_attempt"] == 1]
        gr = ru.groupby("rusher_player_id").agg(
            rush_yds=("yards_gained", "sum"), rush_td=("touchdown", "sum"),
            carries=("play_id", "count"), rush_epa=("epa", "sum")).reset_index()
        gr = gr.rename(columns={"rusher_player_id": "pid"}); gr["w"] = w; rush.append(gr)
        names.update(dict(zip(ru["rusher_player_id"], ru["rusher_player_name"])))
        tg = d[(d["pass_attempt"] == 1) & d["receiver_player_id"].notna()]
        gc = tg.groupby("receiver_player_id").agg(
            targets=("play_id", "count"), rec=("complete_pass", "sum"),
            rec_yds=("yards_gained", "sum"), rec_td=("touchdown", "sum"),
            rec_epa=("epa", "sum")).reset_index()
        gc = gc.rename(columns={"receiver_player_id": "pid"}); gc["w"] = w; rec.append(gc)
        names.update(dict(zip(tg["receiver_player_id"], tg["receiver_player_name"])))
        ps = d[d["pass_attempt"] == 1]
        gp = ps.groupby("passer_player_id").agg(pass_epa=("epa", "sum"), pass_n=("play_id", "count")).reset_index()
        gp = gp.rename(columns={"passer_player_id": "pid"}); gp["w"] = w; pas.append(gp)
        names.update(dict(zip(ps["passer_player_id"], ps["passer_player_name"])))
    cat = lambda fr: pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
    _PBP_AGG = (cat(rush), cat(rec), cat(pas), names)
    return _PBP_AGG


def _wavg(df, metrics):
    """Recency-weighted per-season average per player: sum(metric*w)/sum(w)."""
    if df.empty:
        return pd.DataFrame(columns=["pid"] + metrics)
    return df.groupby("pid").apply(
        lambda x: pd.Series({m: (x[m] * x["w"]).sum() / x["w"].sum() for m in metrics}),
        include_groups=False).reset_index()


def _pos_map():
    """player_id → position (RB/WR/TE/QB/…), from 2025 composite then 2026 rosters."""
    m = {}
    cp = PROC / "composite_scores.parquet"
    if cp.exists():
        c = pd.read_parquet(cp); c = c[c.season == 2025].drop_duplicates("player_id")
        m.update(dict(zip(c["player_id"], c["position"])))
    r = pd.read_parquet(RAW / "rosters_2026.parquet").dropna(subset=["player_id"]).drop_duplicates("player_id")
    for pid, pos in zip(r["player_id"], r["position"]):
        m.setdefault(pid, pos)
    return m


def _skill_value():
    """RB/WR/TE value from real production (yards + TDs + touches + EPA), recency-weighted
    and z-scored within position. Same schema as _composite_2025 so it drops into the
    offense-group and per-player-percentile code. Correctly credits volume workhorses."""
    global _SKILL_CACHE
    if _SKILL_CACHE is not None:
        return _SKILL_CACHE
    rush, rec, _, names = _pbp_agg()
    rw = _wavg(rush, ["rush_yds", "rush_td", "carries", "rush_epa"])
    cw = _wavg(rec, ["targets", "rec", "rec_yds", "rec_td", "rec_epa"])
    m = rw.merge(cw, on="pid", how="outer").fillna(0.0)
    m["position"] = m["pid"].map(_pos_map())
    m = m[m["position"].isin(["RB", "WR", "TE", "FB"])].copy()
    m["grp"] = m["position"].replace({"FB": "RB"})
    m["scrim_yds"] = m["rush_yds"] + m["rec_yds"]
    m["tds"] = m["rush_td"] + m["rec_td"]
    m["touches"] = m["carries"] + m["rec"]
    m["epa"] = m["rush_epa"] + m["rec_epa"]

    def _zc(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0

    m["adjusted_score"] = 0.0
    for _, sub in m.groupby("grp"):
        m.loc[sub.index, "adjusted_score"] = (
            0.42 * _zc(sub["scrim_yds"]) + 0.25 * _zc(sub["tds"]) +
            0.20 * _zc(sub["epa"]) + 0.13 * _zc(sub["touches"]))
    m["name"] = m["pid"].map(names)
    m["nm"] = m["name"].map(_norm)
    m["k"] = m["name"].map(_key)
    _SKILL_CACHE = m.rename(columns={"pid": "player_id"})[
        ["player_id", "nm", "k", "position", "adjusted_score"]]
    return _SKILL_CACHE


# QB rating is recency-weighted over five seasons and regressed toward the mean by volume.
# Steepened toward recent play: QB efficiency is volatile and a 3-year-old season shouldn't anchor
# a developing QB (e.g. a rookie disaster over-weighting a since-improved starter). The current
# year now carries ~2.6x an early-window season, up from ~2.3x.
_QB_W = {2025: 0.40, 2024: 0.26, 2023: 0.16, 2022: 0.11, 2021: 0.07}
_QB_SHRINK = 650.0        # attempts of regression toward the draft-capital prior (tames small samples)
# Draft-pedigree prior + youth regression: young high picks are graded partly on outlook, not just
# their limited/rough early production. Prior (EPA-equiv shrink target) is high for early picks; the
# youth multiplier adds extra regression so a #1-pick's first few years lean on pedigree.
_QB_PRIOR_CAP = 0.055     # prior for a #1 overall pick
_QB_PRIOR_SLOPE = 0.0011  # decay per draft slot
_QB_PRIOR_FLOOR = -0.04   # late/undrafted prior floor
_QB_YOUTH_K = {0: 2.2, 1: 2.2, 2: 1.7, 3: 1.3}   # years_exp -> shrinkage multiplier (else 1.0)


def _qb_value_table():
    """QB EFFICIENCY value: recency+volume-weighted passing+rushing EPA/play over up to five
    seasons of PBP, regressed toward the league mean by attempts so small samples (e.g. a
    rookie's 900 snaps) don't spike to the top. This is an efficiency/production rating —
    scheme- and supporting-cast-influenced — not an isolated talent grade; a great-scheme QB
    legitimately grades near the top by these numbers."""
    rows = []
    for s, w in _QB_W.items():
        p = RAW / f"pbp_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if "season" in d.columns:
            d = d[d["season"] == s]
        if "week" in d.columns:
            d = d[d["week"] <= 18]
        pa = d[d["pass_attempt"] == 1].groupby("passer_player_id").agg(pe=("epa", "sum"), pn=("play_id", "count"))
        ru = d[d["rush_attempt"] == 1].groupby("rusher_player_id").agg(re=("epa", "sum"), rn=("play_id", "count"))
        j = pa.join(ru, how="left").fillna(0.0)
        j = j[j["pn"] >= 60]
        j["wepa"] = (j["pe"] + j["re"]) * w                 # weighted EPA total
        j["wpl"] = (j["pn"] + j["rn"]) * w                  # weighted play total
        j["att"] = j["pn"]
        rows.append(j.reset_index().rename(columns={"passer_player_id": "pid"})[["pid", "wepa", "wpl", "att"]])
    if not rows:
        return pd.Series(dtype=float)
    g = pd.concat(rows, ignore_index=True).groupby("pid").agg(
        wepa=("wepa", "sum"), wpl=("wpl", "sum"), att=("att", "sum"))
    g = g[g["att"] >= 300]                                  # qualified starters
    g["epa_play"] = g["wepa"] / g["wpl"].clip(lower=1)
    mean = float(g["epa_play"].mean())
    # Regress toward a DRAFT-CAPITAL prior instead of the flat league mean, and shrink young QBs
    # harder: a promising high pick's small or rough early sample (e.g. a #1 overall rookie's bad
    # year) shouldn't be graded as a finished product. High picks get a slightly-above-mean prior,
    # late/undrafted a below-mean one; rookies/2nd-years carry extra regression so one year moves
    # them less. Veterans (large samples) are barely affected — their EPA still rules.
    try:
        rm = pd.read_parquet(RAW / "rosters_2026.parquet").drop_duplicates("player_id").set_index("player_id")
        dn, yx = rm.get("draft_number"), rm.get("years_exp")
    except Exception:
        dn = yx = None

    def _prior(pid):
        d = None if dn is None else dn.get(pid)
        if d is None or pd.isna(d):
            return mean - 0.02                                  # undrafted → a touch below average
        return float(np.clip(_QB_PRIOR_CAP - _QB_PRIOR_SLOPE * float(d), _QB_PRIOR_FLOOR, _QB_PRIOR_CAP))

    def _kadj(pid):
        e = None if yx is None else yx.get(pid)
        e = 9.0 if e is None or pd.isna(e) else float(e)
        return _QB_SHRINK * _QB_YOUTH_K.get(int(e), 1.0)

    prior = g.index.to_series().map(_prior)
    kadj = g.index.to_series().map(_kadj)
    a = g["att"]
    g["value"] = (a / (a + kadj)) * g["epa_play"] + (kadj / (a + kadj)) * prior
    return g["value"]


def _qb_starter():
    """Per-team QB1 (from the current depth chart) rated by multi-year EPA/play.
    Rookies / no-history default to the 10th-percentile (replacement) value."""
    dc = pd.read_parquet(RAW / "depth_2026_current.parquet")
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    qb1 = dc[(dc.pos_abb == "QB") & (dc.pos_rank == 1)].drop_duplicates("team")[["team", "gsis_id"]]
    val = _qb_value_table()
    qb1["qb"] = qb1["gsis_id"].map(val).fillna(float(val.quantile(0.10)))
    return qb1.set_index("team")["qb"]


def _team_rosters_2026():
    r = pd.read_parquet(RAW / "rosters_2026.parquet")
    r["nm"] = r["player_name"].map(_norm)
    r["nm_full"] = r["player_name"]
    return r[["team", "player_id", "nm", "nm_full", "position"]]


def _offense_group(roster, comp, positions, topn):
    """Best `topn` current-roster players at `positions`, by 2025 composite (id then name)."""
    sub = roster[roster.position.isin(positions)].copy()
    m = sub.merge(comp[["player_id", "adjusted_score"]], on="player_id", how="left")
    # name fallback for unmatched (id mismatches)
    miss = m["adjusted_score"].isna()
    if miss.any():
        byname = comp.dropna(subset=["nm"]).drop_duplicates("nm").set_index("nm")["adjusted_score"]
        m.loc[miss, "adjusted_score"] = m.loc[miss, "nm"].map(byname)
    m["adjusted_score"] = m["adjusted_score"].fillna(comp["adjusted_score"].quantile(0.15))
    return m.sort_values("adjusted_score", ascending=False).groupby("team").head(topn) \
            .groupby("team")["adjusted_score"].mean()


def _pfr_to_gsis():
    """Crosswalk PFR's pfr_player_id → gsis id via the 2026 roster's pfr_id column. Lets us match
    PFR stats to depth-chart players by ID instead of name, so nickname/spelling differences (e.g.
    depth 'Dru Phillips' vs PFR 'Andru Phillips') no longer drop a real starter to a depth default."""
    global _PFR2GSIS
    if _PFR2GSIS is not None:
        return _PFR2GSIS
    try:
        r = pd.read_parquet(RAW / "rosters_2026.parquet").dropna(subset=["pfr_id", "player_id"]).drop_duplicates("pfr_id")
        _PFR2GSIS = dict(zip(r["pfr_id"], r["player_id"]))
    except Exception:
        _PFR2GSIS = {}
    return _PFR2GSIS


def _pass_rush_players():
    """Per-player pass-rush table: MULTI-YEAR pressures+½·sacks PER GAME (recency-weighted, 2023-25).
    Per-game + game-weighting means a star's injured season contributes little, so his healthy-year
    disruption carries — e.g. Bosa (2 games in 2025) keeps his 2023-24 form. Grouped by pfr_player_id
    (no name collisions) and carries both a gsis id and a name-key so callers can match by ID first,
    name second. SHARED by the team metric and the per-player card ratings so both tell one story."""
    pf = pd.read_parquet(RAW / "pfr_defense.parquet")
    frames = []
    for yr, w in _DEF_W.items():
        d = pf[pf["season"] == yr]
        if d.empty:
            continue
        s = d.groupby("pfr_player_id").agg(pr=("def_pressures", "sum"), sk=("def_sacks", "sum"),
                                           gm=("def_pressures", "size"), nm=("pfr_player_name", "first"))
        s["num"], s["den"] = (s["pr"] + 0.5 * s["sk"]) * w, s["gm"] * w
        frames.append(s[["num", "den", "nm"]])
    if not frames:
        return pd.DataFrame(columns=["ppg", "gsis", "k"])
    a = pd.concat(frames).groupby(level=0).agg(num=("num", "sum"), den=("den", "sum"), nm=("nm", "first"))
    a["ppg"] = a["num"] / a["den"].clip(lower=1)
    a["gsis"] = a.index.to_series().map(_pfr_to_gsis())
    a["k"] = a["nm"].map(_key)
    return a[["ppg", "gsis", "k"]]


def _id_name_lookup(df, valcol):
    """From a per-player table with columns [valcol, 'gsis', 'k'], return (by_gsis, by_name) Series
    so a caller can resolve a player by ID first (robust) then by name-key (fallback)."""
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    by_gsis = df.dropna(subset=["gsis"]).groupby("gsis")[valcol].max()
    by_name = df.groupby("k")[valcol].max()
    return by_gsis, by_name


def _pass_rush(roster):
    """Team pass rush = top-5 mean of the current roster's per-player rush production, matched by
    gsis id first then name (edge rushers often listed as LB)."""
    g, n = _id_name_lookup(_pass_rush_players(), "ppg")
    sub = roster[roster.position.isin(["DL", "LB"])].copy()
    sub["prod"] = sub["player_id"].map(g)
    miss = sub["prod"].isna()
    sub.loc[miss, "prod"] = sub.loc[miss, "nm_full"].map(_key).map(n)
    sub["prod"] = sub["prod"].fillna(0.0)
    return sub.sort_values("prod", ascending=False).groupby("team").head(5).groupby("team")["prod"].mean()


def _coverage_players():
    """Per-player coverage table: MULTI-YEAR, TARGET-WEIGHTED passer rating allowed (recency-weighted),
    negated so higher = better. Target-weighting stops a 4-target blowup game (rating 150+) from
    sinking an elite low-target corner; multi-year keeps an injured/limited 2025 (Sauce) from erasing
    his healthy 2023-24 body of work; then shrink low-target players toward the league rate. Grouped
    by pfr_player_id with gsis + name-key so cards/team metric can match by ID first. SHARED."""
    pf = pd.read_parquet(RAW / "pfr_defense.parquet")
    frames = []
    for yr, w in _DEF_W.items():
        d = pf[pf["season"] == yr]
        if d.empty:
            continue
        d = d.copy(); d["wr"] = d["def_passer_rating_allowed"] * d["def_targets"]
        s = d.groupby("pfr_player_id").agg(tgt=("def_targets", "sum"), wr=("wr", "sum"),
                                           nm=("pfr_player_name", "first"))
        s["wtgt"], s["wwr"] = s["tgt"] * w, s["wr"] * w
        frames.append(s[["wtgt", "wwr", "nm"]])
    if not frames:
        return pd.DataFrame(columns=["cov", "gsis", "k"])
    agg = pd.concat(frames).groupby(level=0).agg(wtgt=("wtgt", "sum"), wwr=("wwr", "sum"), nm=("nm", "first"))
    agg = agg[agg["wtgt"] >= 12]
    lg = float(agg["wwr"].sum() / max(1.0, agg["wtgt"].sum())); K = _COV_SHRINK
    agg["rate"] = agg["wwr"] / agg["wtgt"].clip(lower=1)
    agg["rate"] = (agg["wtgt"] / (agg["wtgt"] + K)) * agg["rate"] + (K / (agg["wtgt"] + K)) * lg
    agg["cov"] = -agg["rate"]                      # lower passer rating allowed = better
    agg["gsis"] = agg.index.to_series().map(_pfr_to_gsis())
    agg["k"] = agg["nm"].map(_key)
    return agg[["cov", "gsis", "k"]]


def _coverage(roster):
    """Team coverage = top-5 mean of the current roster's per-player coverage grade (id-first match)."""
    g, n = _id_name_lookup(_coverage_players(), "cov")
    sub = roster[roster.position.isin(["DB", "LB"])].copy()
    sub["cov"] = sub["player_id"].map(g)
    miss = sub["cov"].isna()
    sub.loc[miss, "cov"] = sub.loc[miss, "nm_full"].map(_key).map(n)
    good = sub.dropna(subset=["cov"])
    return good.sort_values("cov", ascending=False).groupby("team").head(5).groupby("team")["cov"].mean()


# Recency weights for the MULTI-YEAR defensive/OL metrics. Multi-year (weighted by playing time)
# makes them injury-robust: a star who missed 2025 (Bosa/Sauce/Warner) or a line whose tackles were
# hurt (LAC) keep credit from healthy 2023-24 instead of being judged on a lost season.
_DEF_W = {2025: 0.5, 2024: 0.3, 2023: 0.2}


def _recency_damp(by_year, damp=0.4, thresh=0.03):
    """Recency-weighted average across years, but DAMPEN a current-year value that's anomalously
    WORSE (higher) than the prior-year baseline — a one-year pressure-rate spike is usually an
    injury-wrecked line (LAC/ARI/CLE/MIA/JAX in 2025), not a true collapse. Only the excess beyond
    `thresh` past baseline is dampened; genuine year-over-year change within threshold passes through."""
    cur = max(by_year)
    prior = [y for y in by_year if y < cur]
    yb = dict(by_year)
    if prior and cur in by_year:
        base = pd.concat([by_year[y] for y in prior], axis=1).mean(axis=1)
        common = by_year[cur].index.union(base.index)
        cv, bs = by_year[cur].reindex(common), base.reindex(common)
        excess = cv - bs
        damped = bs + np.where(excess > thresh, thresh + damp * (excess - thresh), excess)
        out = pd.Series(damped, index=common)
        out[bs.isna()] = cv[bs.isna()]                         # no baseline → keep current value
        yb[cur] = out
    df = pd.DataFrame(yb)
    wts = np.array([_DEF_W.get(c, 0.0) for c in df.columns])
    wmat = df.notna().to_numpy() * wts
    return pd.Series((df.fillna(0).to_numpy() * wts).sum(1) / wmat.sum(1), index=df.index)


def _ol():
    """Team O-line grade from raw data, MULTI-YEAR (recency-weighted, weighted by dropbacks/carries
    so a degraded/injured season counts less). Pass protection dominates via PFR PRESSURE RATE
    allowed (less QB-dependent than raw sacks); a smaller RB-ONLY run-block signal excludes QB
    sneaks/kneels. Higher = better. (LAC's 2025 was wrecked by hurt tackles — 2023-24 lifts it back.)"""
    try:
        pf = pd.read_parquet(RAW / "pfr_passing.parquet")
        ros_all = pd.read_parquet(RAW / "rosters_seasonal.parquet")
        pr_by, sk_by, run = {}, {}, []
        for yr, w in _DEF_W.items():
            d = pf[pf["season"] == yr]
            pbpf = RAW / f"pbp_{yr}.parquet"
            if d.empty or not pbpf.exists():
                continue
            pbp = pd.read_parquet(pbpf)
            db = pbp[pbp["pass_attempt"] == 1].groupby("posteam").size()
            g = d.groupby("team").agg(press=("times_pressured", "sum"), sk=("times_sacked", "sum"))
            dbg = db.reindex(g.index).clip(lower=1)
            pr_by[yr] = g["press"] / dbg                          # per-year team pressure rate
            sk_by[yr] = g["sk"] / dbg
            pos = ros_all[ros_all["season"] == yr].drop_duplicates("player_id").set_index("player_id")["position"]
            runs = pbp[pbp["rush_attempt"] == 1].copy()
            runs["rp"] = runs["rusher_player_id"].map(pos)
            rb = runs[runs["rp"].isin(["RB", "FB"])]                # exclude QB sneaks/kneels/scrambles
            r = rb.groupby("posteam").agg(car=("play_id", "count"), yds=("yards_gained", "sum"),
                                          stuff=("yards_gained", lambda x: (x <= 0).sum()))
            r["wc"], r["wy"], r["ws"] = r["car"] * w, r["yds"] * w, r["stuff"] * w
            run.append(r[["wc", "wy", "ws"]])
        # pass-pro: dampen an anomalously-bad 2025 (likely injured line) toward the healthy baseline
        press_rate, sack_rate = _recency_damp(pr_by), _recency_damp(sk_by)
        R = pd.concat(run).groupby(level=0).sum()
        R["ypc"], R["stuff_rate"] = R["wy"] / R["wc"].clip(lower=1), R["ws"] / R["wc"].clip(lower=1)
        df = pd.DataFrame({"press_rate": press_rate, "sack_rate": sack_rate}).join(R[["ypc", "stuff_rate"]])
        passpro = -(0.7 * _z(df["press_rate"]) + 0.3 * _z(df["sack_rate"]))
        runblk = _z(df["ypc"]) - 0.25 * _z(df["stuff_rate"])
        return (0.7 * passpro + 0.3 * runblk).dropna()
    except Exception:                                              # fallback: old composite CSV
        ol = pd.read_csv(PROC / "ol_rankings_2025.csv").set_index("team")
        return -_z(ol["sack_rate_allowed"]) + _z(ol.get("ypc", pd.Series(0, index=ol.index)))


def _def_team():
    """Opponent-adjusted TEAM defense (run + pass EPA suppressed), higher = better. A complete,
    schedule-adjusted defensive signal — much stronger than name-matched individual stats."""
    from ml.adjust import adjusted_unit_epa
    adj = adjusted_unit_epa(2025)
    if adj:
        return pd.Series({t: -(d.get("def_pass", 0.0) + d.get("def_rush", 0.0)) for t, d in adj.items()})
    p = PROC / "team_styles.parquet"                               # fallback: raw def EPA
    if p.exists():
        s = pd.read_parquet(p); s = s[s["season"] == s["season"].max()]
        if "def_epa_per_play" in s.columns:
            return -s.set_index("team")["def_epa_per_play"]
    return pd.Series(dtype=float)


def _coaching():
    # multi-year (2023-2025) mean, so a single strong season doesn't spike a team
    c = pd.read_parquet(PROC / "coaching_scores.parquet")
    c = c[c.season.isin([2023, 2024, 2025])]
    return c.groupby("team")["coaching_score"].mean()


# ── assemble ────────────────────────────────────────────────────────
def squad_ratings(breakdown: bool = False) -> pd.DataFrame:
    roster = _team_rosters_2026()
    skill_val = _skill_value()                                      # PBP production, not the composite
    teams = sorted(roster.team.unique())

    g = pd.DataFrame(index=teams)
    g["qb"]    = _qb_starter()                                      # depth-chart starter, EPA/play
    g["skill"] = _offense_group(roster, skill_val, ["WR", "RB", "TE"], 5)
    g["ol"]    = _ol()
    g["def_team"] = _def_team()                                    # opponent-adjusted team defense (run+pass)
    g["rush"]  = _pass_rush(roster)
    g["cover"] = _coverage(roster)
    g["coach"] = _coaching()
    # replacement-fill any team missing a group, then z-score each group
    for c in g.columns:
        g[c] = g[c].fillna(g[c].median())
    z = g.apply(_z)

    z["blend"] = sum(z[c] * w for c, w in WEIGHTS.items())
    z["rating"] = (z["blend"] * RATING_SCALE).round(1)
    out = z.sort_values("rating", ascending=False).reset_index().rename(columns={"index": "team"})
    out.insert(0, "rank", out.index + 1)
    return (out if breakdown else out[["rank", "team", "rating"]]), g


HFA = 2.0            # home-field points
SPREAD_SCALE = 0.9   # map roster-talent rating difference to a point spread


def predict_matchup(home: str, away: str, neutral: bool = False) -> dict:
    """Predict a matchup from the SAME roster-talent ratings as the power rankings,
    so a #1 team is favored over a #16 team (previously it used a separate SRS model)."""
    out, _ = squad_ratings()
    r = out.set_index("team")["rating"]
    if home not in r.index or away not in r.index:
        return {"error": "unknown team(s)"}
    hfa = 0.0 if neutral else HFA
    margin = float(np.clip((r[home] - r[away]) * SPREAD_SCALE + hfa, -18, 18))
    total = 44.0        # league-average total; scores split around the margin
    home_pts, away_pts = (total + margin) / 2, (total - margin) / 2
    wp = float(1 / (1 + np.exp(-margin / 13.5 * np.pi / np.sqrt(3))))
    return {
        "home": home, "away": away,
        "pred_home_score": round(home_pts, 1), "pred_away_score": round(away_pts, 1),
        "pred_margin": round(margin, 1), "pred_total": round(total, 1),
        "home_win_prob": round(wp, 3), "away_win_prob": round(1 - wp, 3),
    }


# ── team depth chart + per-player ratings (for the dashboard team page) ──
# The depth chart uses granular slots (LT, RG, LDE, RCB, ...); map them to groups.
GROUP_MAP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
             "C": "OL", "LG": "OL", "RG": "OL", "LT": "OL", "RT": "OL",
             "LDE": "DL", "RDE": "DL", "LDT": "DL", "RDT": "DL", "NT": "DL",
             "MLB": "LB", "LILB": "LB", "RILB": "LB", "SLB": "LB", "WLB": "LB",
             "LCB": "DB", "RCB": "DB", "NB": "DB", "FS": "DB", "SS": "DB",
             "PK": "K", "P": "P", "LS": "LS"}
POS_ORDER = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P", "LS"]
POS_LABEL = {"QB": "Quarterback", "RB": "Running Back", "WR": "Wide Receiver",
             "TE": "Tight End", "OL": "Offensive Line", "DL": "Defensive Line",
             "LB": "Linebacker", "DB": "Defensive Back", "K": "Kicker",
             "P": "Punter", "LS": "Long Snapper"}


def _ol_players():
    """Per-player individual O-line quality (0-100), keyed by gsis and name. There is no clean
    per-lineman performance grade in free data, so blend two honest signals: a PROVEN-STARTER score
    (recency-weighted recent offensive snaps, 2023-25 — teams don't keep starting bad linemen) and
    DRAFT PEDIGREE, trusting snaps more as a player accrues them and the draft prior more for young/
    unproven linemen. Lets OL cards show studs above replacement teammates instead of one flat number."""
    try:
        sc = pd.read_parquet(RAW / "snap_counts.parquet")
    except Exception:
        return pd.DataFrame(columns=["q", "gsis", "k"])
    sc = sc[sc["position"].isin(["T", "G", "C"])]
    frames = []
    for yr, w in _DEF_W.items():
        d = sc[sc["season"] == yr]
        if d.empty:
            continue
        s = d.groupby("pfr_player_id").agg(mp=("offense_pct", "mean"), g=("offense_pct", "size"),
                                           nm=("player", "first"))
        s["wnum"], s["wden"] = s["mp"] * s["g"] * w, s["g"] * w        # games- & recency-weighted share
        frames.append(s[["wnum", "wden", "nm"]])
    if not frames:
        return pd.DataFrame(columns=["q", "gsis", "k"])
    a = pd.concat(frames).groupby(level=0).agg(wnum=("wnum", "sum"), wden=("wden", "sum"), nm=("nm", "first"))
    # SNAP SHARE (per game), not total snaps — a proven full-time starter stays high even if injury cost
    # him games (Slater), so the OL individual grade is injury-robust like the defensive metrics.
    a["share"] = a["wnum"] / a["wden"].clip(lower=0.1)
    a["gsis"] = a.index.to_series().map(_pfr_to_gsis())
    a["k"] = a["nm"].map(_key)
    a["snap_score"] = a["share"].rank(pct=True) * 100                 # proven-starter percentile
    try:
        rm = pd.read_parquet(RAW / "rosters_2026.parquet").drop_duplicates("player_id").set_index("player_id")
        dn = pd.to_numeric(rm.get("draft_number"), errors="coerce")
    except Exception:
        dn = None
    a["draft_score"] = a["gsis"].map(dn).apply(lambda p: _draft_rating(p) if pd.notna(p) else 42.0) if dn is not None else 42.0
    a["q"] = 0.6 * a["snap_score"] + 0.4 * a["draft_score"]           # proven starter + draft pedigree
    return a[["q", "gsis", "k"]]


def _kicker_players():
    """Kicker rating (0-100) keyed by gsis from the season kicker composite (FG% / long-FG% / XP%).
    True multi-year FG data isn't in the offline dataset — the raw PBP is trimmed to engine columns and
    only the latest-season kicker rankings CSV exists — so this is that season's grade. A team's
    established KICKER1 who has no line that season is given a neutral veteran rating at the card layer
    (see the K branch) rather than a replacement default, so a hurt/limited season doesn't read as bad."""
    kp = PROC / "kicker_rankings_2025.csv"
    if not kp.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(kp).dropna(subset=["player_id"]).drop_duplicates("player_id")
    return d.set_index("player_id")["composite"].rank(pct=True) * 100


_PCT_CACHE = None
_META_CACHE = None
_PFR2GSIS = None


def _player_pct():
    """Per-player 0-100 rating = percentile within position (2025). Comparable across
    positions: 90+=elite, ~60-80=solid starter, <40=depth. Indexed by BOTH gsis id and
    normalized name so id mismatches still resolve. The composite covers most positions
    (incl. OL guards/tackles, DL, DB, LB, P), with specialized QB/pass-rush/coverage on top."""
    global _PCT_CACHE
    if _PCT_CACHE is not None:
        return _PCT_CACHE
    comp = _composite_2025()
    comp["pct"] = comp.groupby("position")["adjusted_score"].rank(pct=True) * 100
    skill = comp.set_index("player_id")["pct"]                       # composite %ile (all positions)
    skill_nm = comp.dropna(subset=["nm"]).drop_duplicates("nm").set_index("nm")["pct"]
    # RB/WR/TE rated on real PBP production (fixes volume backs), %ile within position
    sv = _skill_value().copy()
    sv["pct"] = sv.groupby("position")["adjusted_score"].rank(pct=True) * 100
    prod = sv.set_index("player_id")["pct"]
    prod_nm = sv.dropna(subset=["nm"]).drop_duplicates("nm").set_index("nm")["pct"]
    qb = _qb_value_table(); qb_pct = qb.rank(pct=True) * 100         # QB by multi-year EPA + draft prior
    # DL and DB/LB cards read the SAME multi-year, injury-robust per-player signals that feed the team
    # pass-rush / coverage grades — so a card can't disagree with how the team rating treats that
    # player. Single-year 2025 CSVs (which buried Bosa/Sauce's injured seasons) are no longer used here.
    # Rank each per-player table into a within-metric percentile ONCE, then expose id- and name-keyed
    # lookups (match by gsis first, name second) so nickname mismatches can't bury a real starter.
    prdf = _pass_rush_players().copy(); prdf["pct"] = prdf["ppg"].rank(pct=True) * 100
    dl_id, dl_nm = _id_name_lookup(prdf, "pct")                      # DL by MULTI-YEAR pass-rush
    covdf = _coverage_players().copy(); covdf["pct"] = covdf["cov"].rank(pct=True) * 100
    cov_id, cov_nm = _id_name_lookup(covdf, "pct")                   # DB/LB by MULTI-YEAR coverage
    ol_id, ol_nm = _id_name_lookup(_ol_players(), "q")              # OL individual quality (snaps+draft)
    k_id = _kicker_players()                                         # K by MULTI-YEAR FG/long/XP (gsis)
    _PCT_CACHE = (skill, skill_nm, prod, prod_nm, qb_pct, dl_id, dl_nm, cov_id, cov_nm, ol_id, ol_nm, k_id)
    return _PCT_CACHE


def _roster_meta():
    """For players with NO 2025 stat line: draft pick + experience (rookie estimate),
    plus the team O-line %ile and kicker %ile. Cached; reset by the server on refresh."""
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE
    r = pd.read_parquet(RAW / "rosters_2026.parquet").dropna(subset=["player_id"]).copy()
    r["nm"] = r["player_name"].map(_norm)
    r["dn"] = pd.to_numeric(r.get("draft_number"), errors="coerce")
    r["yx"] = pd.to_numeric(r.get("years_exp"), errors="coerce")
    by_id, by_nm = r.drop_duplicates("player_id").set_index("player_id"), r.drop_duplicates("nm").set_index("nm")
    draft = {"id": by_id["dn"], "nm": by_nm["dn"]}
    exp = {"id": by_id["yx"], "nm": by_nm["yx"]}
    # OL cards show the team grade adjusted for depth. Use the SAME multi-year, injury-dampened team
    # grade (_ol) that the power ranking uses — not the single-year 2025 CSV that buried injured lines
    # like LAC — so a lineman's card matches how his unit is rated everywhere else.
    ol_grade = _ol()
    ol_pct = (ol_grade.rank(pct=True) * 100) if len(ol_grade) else pd.Series(dtype=float)
    kp = PROC / "kicker_rankings_2025.csv"
    # drop_duplicates so a player who appears twice (traded mid-season → two team rows) doesn't
    # give a non-unique index; k_pct.get(id) must return a scalar, not a Series (see _first).
    k_pct = (pd.read_csv(kp).dropna(subset=["player_id"]).drop_duplicates("player_id")
             .set_index("player_id")["composite"].rank(pct=True) * 100) if kp.exists() else pd.Series(dtype=float)
    _META_CACHE = (draft, exp, ol_pct, k_pct)
    return _META_CACHE


def _draft_rating(pick):
    """Rookie estimate from overall draft slot: high picks project higher than late/UDFA."""
    if pick is None or pd.isna(pick):
        return None
    p = float(pick)
    for lim, val in [(10, 70), (32, 64), (64, 58), (105, 53), (150, 48), (200, 44)]:
        if p <= lim:
            return val
    return 40


_DEPTH_RT = {1: 48, 2: 42, 3: 37, 4: 33}   # replacement rating by depth-chart rank


def team_depth_chart(team: str) -> list:
    """Every depth-chart player gets a 0-100 rating via a waterfall, tagged with its source:
       measured (2025 production) → team (O-line grade) → rookie (draft slot) → proj (depth)."""
    dc = pd.read_parquet(RAW / "depth_2026_current.parquet").copy()
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    dc = dc[(dc.team == team) & dc.player_name.notna()].copy()
    dc["grp"] = dc["pos_abb"].map(GROUP_MAP)
    skill, skill_nm, prod, prod_nm, qb_pct, dl_id, dl_nm, cov_id, cov_nm, ol_id, ol_nm, k_id = _player_pct()
    draft, exp, ol_pct, _ = _roster_meta()
    team_ol = float(ol_pct.get(team, 50.0)) if len(ol_pct) else 50.0

    def _first(*vals):
        for v in vals:
            if v is None:
                continue
            if isinstance(v, pd.Series):        # a non-unique lookup index returned multiple rows
                v = v.dropna()                  # → collapse to the first valid value, don't crash
                if v.empty:
                    continue
                v = v.iloc[0]
            if not pd.isna(v):
                return v
        return None

    def rate(grp, gid, name, pos_rank):
        k, nm = _key(name), _norm(name)
        pr = int(pos_rank) if pos_rank and not pd.isna(pos_rank) else 3
        # O-line: blend the team unit grade with the player's INDIVIDUAL quality (snaps+draft), then
        # depth-adjust — so a stud (Sewell) or an injured team's proven starter (Slater) reads above a
        # replacement teammate instead of every lineman showing one flat team number.
        if grp == "OL":
            iq = _first(ol_id.get(gid), ol_nm.get(k))
            src = "measured" if iq is not None else "team"
            if iq is None:
                iq = team_ol                                  # no snap/draft history → fall to team level
            base = 0.4 * team_ol + 0.6 * iq                   # individual leads; team unit is light context
            return int(round(max(20.0, base + {1: 0, 2: -6, 3: -12}.get(pr, -16)))), src
        # 1. best measured signal for the group, then the universal composite (id or name)
        if grp == "QB":
            v = _first(qb_pct.get(gid), skill.get(gid), skill_nm.get(nm))
        elif grp == "DL":
            v = _first(dl_id.get(gid), dl_nm.get(k), skill.get(gid), skill_nm.get(nm))
        elif grp == "LB":
            # LB spans edge rushers (listed as LB) and off-ball coverage LBs — rate on whichever
            # skill the player actually shows, so an edge like Bosa gets his pass-rush grade instead
            # of falling through to a depth default for lacking coverage targets. Match by id first.
            cands = [dl_id.get(gid), dl_nm.get(k), cov_id.get(gid), cov_nm.get(k)]
            best = max([x for x in cands if x is not None and not pd.isna(x)], default=None)
            v = _first(best, skill.get(gid), skill_nm.get(nm))
        elif grp == "DB":
            v = _first(cov_id.get(gid), cov_nm.get(k), skill.get(gid), skill_nm.get(nm))
        elif grp == "K":
            v = _first(k_id.get(gid), skill.get(gid), skill_nm.get(nm))
            if v is None and pr == 1:
                return 52, "team"    # established starting kicker, no recent line → neutral (data-limited)
        elif grp in ("WR", "RB", "TE"):                  # real PBP production first, composite as backup
            v = _first(prod.get(gid), prod_nm.get(nm), skill.get(gid), skill_nm.get(nm))
        else:                                            # P/LS — composite covers punters
            v = _first(skill.get(gid), skill_nm.get(nm))
        if v is not None:
            return int(round(v)), "measured"
        # 2. rookies / 2nd-year with no snaps → draft-slot estimate
        ex = _first(exp["id"].get(gid), exp["nm"].get(nm))
        if ex is not None and ex <= 1:
            dv = _draft_rating(_first(draft["id"].get(gid), draft["nm"].get(nm)))
            if dv is not None:
                return int(round(dv)), "rookie"
        # 3. everyone else → depth-based replacement level
        return _DEPTH_RT.get(pr, 30), "proj"

    groups = []
    for grp in POS_ORDER:
        # starters first (rank 1 of each slot), then depth; group like slots together
        sub = dc[dc.grp == grp].sort_values(["pos_rank", "pos_abb"])
        if sub.empty:
            continue
        players = []
        for r in sub.itertuples():
            rt, src = rate(grp, r.gsis_id, r.player_name, r.pos_rank)
            players.append({"name": r.player_name, "slot": r.pos_abb, "gsis": r.gsis_id,
                            "rank": int(r.pos_rank) if pd.notna(r.pos_rank) else None,
                            "rating": rt, "source": src})
        groups.append({"pos": grp, "label": POS_LABEL.get(grp, grp), "players": players})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", action="store_true")
    args = ap.parse_args()
    out, raw = squad_ratings(breakdown=args.breakdown)
    print(f"\n  2026 ROSTER-TALENT RANKINGS  (current squad + coaching, not 2025 results)")
    print(f"  {'-'*52}")
    for _, x in out.iterrows():
        bar = "#" * max(0, int(round(x["rating"] + 12)))
        print(f"  {x['rank']:2d}. {x['team']:<3} {x['rating']:+5.1f}  {bar}")
    print()
    if args.breakdown:
        print(raw.round(1).to_string())


if __name__ == "__main__":
    main()
