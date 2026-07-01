"""
ml/features.py  —  leak-free per-game feature matrix
=====================================================
Builds one row per game with ONLY information available before kickoff, so a
model trained on it has no future leakage (the failure mode that faked every
earlier "edge" in this repo).

Feature groups
  market      : spread_line, total_line, moneyline implied prob  (the strong prior)
  team form   : current-season-TO-DATE EPA/success/pace (expanding, shifted 1 week)
  prior form  : previous completed season's team EPA (established quality baseline)
  qb/roster   : starting-QB + team composite quality (from de-leaked composite_scores)
  situational : rest differential, HFA, division, dome/turf, temp, wind, week

Target
  home_margin = home_score - away_score
  total       = home_score + away_score

LEAK RULES (enforced here):
  * team form uses weeks 1..W-1 of the CURRENT season only (expanding mean, shift 1)
  * prior form uses seasons strictly < the game's season
  * composite scores are already prior-season rank + rolling (shift 1) efficiency

Usage:
  python -m ml.features                 # build for all seasons -> data/processed/game_features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

PASS_RUN = ("pass", "run")


# ── 1. team-week EPA panel from PBP ─────────────────────────────────
def team_week_panel(seasons: list) -> pd.DataFrame:
    """Per (season, team, week): offensive & defensive EPA/success/pace + points."""
    frames = []
    for s in seasons:
        p = RAW / f"pbp_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=[
                "season", "week", "posteam", "defteam", "play_type",
                "epa", "success", "game_id"]))
    pbp = pd.concat(frames, ignore_index=True)
    plays = pbp[pbp["play_type"].isin(PASS_RUN) & pbp["epa"].notna()].copy()
    plays["is_pass"] = (plays["play_type"] == "pass").astype(float)

    def side(team_col, prefix):
        g = plays.groupby(["season", "week", team_col])
        out = g.agg(
            **{f"{prefix}_epa":      ("epa", "mean"),
               f"{prefix}_success":  ("success", "mean"),
               f"{prefix}_plays":    ("epa", "size"),
               f"{prefix}_pass_rate":("is_pass", "mean")}
        ).reset_index().rename(columns={team_col: "team"})
        # pass/rush EPA splits
        pe = (plays[plays["play_type"] == "pass"].groupby(["season", "week", team_col])["epa"]
              .mean().reset_index().rename(columns={team_col: "team", "epa": f"{prefix}_pass_epa"}))
        re = (plays[plays["play_type"] == "run"].groupby(["season", "week", team_col])["epa"]
              .mean().reset_index().rename(columns={team_col: "team", "epa": f"{prefix}_rush_epa"}))
        out = out.merge(pe, on=["season", "week", "team"], how="left")
        out = out.merge(re, on=["season", "week", "team"], how="left")
        return out

    off = side("posteam", "off")
    deff = side("defteam", "def")
    panel = off.merge(deff, on=["season", "week", "team"], how="outer")
    return panel.sort_values(["team", "season", "week"]).reset_index(drop=True)


def points_panel(sched: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team): points for / against, from final scores in schedules."""
    s = sched[sched["home_score"].notna()][
        ["season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    hp = s.rename(columns={"home_team": "team", "home_score": "pts_for",
                           "away_score": "pts_against"})[["season", "week", "team", "pts_for", "pts_against"]]
    ap = s.rename(columns={"away_team": "team", "away_score": "pts_for",
                           "home_score": "pts_against"})[["season", "week", "team", "pts_for", "pts_against"]]
    return pd.concat([hp, ap], ignore_index=True)


# ── 2. to-date (current season) + prior-season aggregates ───────────
METRICS = ["off_epa", "off_success", "off_pass_epa", "off_rush_epa", "off_pass_rate",
           "def_epa", "def_success", "def_pass_epa", "def_rush_epa",
           "pts_for", "pts_against"]


def add_todate(panel: pd.DataFrame) -> pd.DataFrame:
    """Expanding mean of weeks 1..W-1 within each (team, season) — leak-free."""
    panel = panel.sort_values(["team", "season", "week"]).copy()
    g = panel.groupby(["team", "season"])
    for m in METRICS:
        panel[f"td_{m}"] = g[m].transform(lambda x: x.shift(1).expanding().mean())
    panel["td_games"] = g.cumcount()  # games already played this season entering this week
    return panel


def prior_season_aggregates(panel: pd.DataFrame) -> pd.DataFrame:
    """Full prior-season mean per (team, season) -> to merge as season N-1 baseline."""
    agg = panel.groupby(["team", "season"])[METRICS].mean().reset_index()
    agg = agg.rename(columns={m: f"prior_{m}" for m in METRICS})
    agg["season"] = agg["season"] + 1   # season N gets season N-1's aggregates
    return agg


# ── 3. per-game feature matrix ──────────────────────────────────────
def _implied_prob(ml):
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(ml) or abs(ml) < 100:
        return np.nan
    return 100 / (ml + 100) if ml > 0 else (-ml) / (-ml + 100)


def build_game_features(seasons: list = None) -> pd.DataFrame:
    if seasons is None:
        seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    sched = pd.read_parquet(RAW / "schedules.parquet")
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["game_type"] = sched["game_type"].str.upper().str.strip()

    panel = team_week_panel(seasons)
    panel = panel.merge(points_panel(sched), on=["season", "week", "team"], how="left")
    panel = add_todate(panel)
    prior = prior_season_aggregates(panel)

    td_cols = [f"td_{m}" for m in METRICS] + ["td_games"]
    prior_cols = [f"prior_{m}" for m in METRICS]
    g = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()
              & sched["spread_line"].notna() & sched["season"].isin(seasons)].copy()

    # composite team quality (de-leaked) — best available adjusted_score per team/season/week
    comp = None
    cp = PROC / "composite_scores.parquet"
    if cp.exists():
        comp = pd.read_parquet(cp, columns=["recent_team", "season", "week", "position",
                                            "adjusted_score", "player_id"])

    def team_features(df, team_col, prefix):
        m = df.merge(panel[["team", "season", "week"] + td_cols],
                     left_on=[team_col, "season", "week"], right_on=["team", "season", "week"],
                     how="left").drop(columns=["team"])
        m = m.merge(prior[["team", "season"] + prior_cols],
                    left_on=[team_col, "season"], right_on=["team", "season"],
                    how="left").drop(columns=["team"])
        return m.rename(columns={c: f"{prefix}_{c}" for c in td_cols + prior_cols})

    g = team_features(g, "home_team", "h")
    g = team_features(g, "away_team", "a")

    # composite: team offense quality = mean of top skill adjusted_score to date; QB = max QB
    if comp is not None:
        def comp_quality(row, team):
            sub = comp[(comp["recent_team"] == team) & (comp["season"] == row["season"])
                       & (comp["week"] <= row["week"])]
            if sub.empty:
                return pd.Series({"qb": np.nan, "off": np.nan})
            latest = sub.sort_values("week").drop_duplicates("player_id", keep="last")
            qb = latest[latest["position"] == "QB"]["adjusted_score"].max()
            off = latest[latest["position"].isin(["QB", "WR", "RB", "TE"])]["adjusted_score"].nlargest(6).mean()
            return pd.Series({"qb": qb, "off": off})
        # vectorized-ish: compute per unique (team,season,week) to avoid recompute
        keys = pd.concat([
            g[["home_team", "season", "week"]].rename(columns={"home_team": "team"}),
            g[["away_team", "season", "week"]].rename(columns={"away_team": "team"}),
        ]).drop_duplicates()
        qmap = {}
        for _, k in keys.iterrows():
            sub = comp[(comp["recent_team"] == k["team"]) & (comp["season"] == k["season"])
                       & (comp["week"] <= k["week"])]
            if sub.empty:
                qmap[(k["team"], k["season"], k["week"])] = (np.nan, np.nan)
                continue
            latest = sub.sort_values("week").drop_duplicates("player_id", keep="last")
            qb = latest[latest["position"] == "QB"]["adjusted_score"].max()
            off = latest[latest["position"].isin(["QB", "WR", "RB", "TE"])]["adjusted_score"].nlargest(6).mean()
            qmap[(k["team"], k["season"], k["week"])] = (qb, off)
        for side_, tcol in [("h", "home_team"), ("a", "away_team")]:
            g[f"{side_}_qb_comp"] = g.apply(lambda r: qmap.get((r[tcol], r["season"], r["week"]), (np.nan, np.nan))[0], axis=1)
            g[f"{side_}_off_comp"] = g.apply(lambda r: qmap.get((r[tcol], r["season"], r["week"]), (np.nan, np.nan))[1], axis=1)

    # market + situational + targets
    g["mkt_spread"] = g["spread_line"].astype(float)          # home margin (positive=home fav)
    g["mkt_total"]  = pd.to_numeric(g["total_line"], errors="coerce")
    g["mkt_home_impl"] = g["home_moneyline"].map(_implied_prob)
    g["rest_diff"] = pd.to_numeric(g["home_rest"], errors="coerce") - pd.to_numeric(g["away_rest"], errors="coerce")
    g["is_div"] = pd.to_numeric(g.get("div_game", 0), errors="coerce").fillna(0)
    roof = g.get("roof", pd.Series("", index=g.index)).astype(str).str.lower()
    g["is_dome"] = roof.str.contains("dome|closed|retract", regex=True).astype(float)
    surface = g.get("surface", pd.Series("", index=g.index)).astype(str).str.lower()
    g["is_turf"] = surface.str.contains("turf|artificial|astro", regex=True).astype(float)
    g["temp"] = pd.to_numeric(g.get("temp"), errors="coerce")
    g["wind"] = pd.to_numeric(g.get("wind"), errors="coerce")
    g["week_num"] = g["week"].astype(float)

    g["home_margin"] = g["home_score"].astype(float) - g["away_score"].astype(float)
    g["total"]       = g["home_score"].astype(float) + g["away_score"].astype(float)

    # difference features (home - away) for the to-date/prior metrics
    for m in METRICS:
        g[f"d_td_{m}"]    = g[f"h_td_{m}"]    - g[f"a_td_{m}"]
        g[f"d_prior_{m}"] = g[f"h_prior_{m}"] - g[f"a_prior_{m}"]
    if comp is not None:
        g["d_qb_comp"]  = g["h_qb_comp"]  - g["a_qb_comp"]
        g["d_off_comp"] = g["h_off_comp"] - g["a_off_comp"]

    keep_meta = ["game_id", "season", "week", "home_team", "away_team",
                 "home_margin", "total", "mkt_spread", "mkt_total", "mkt_home_impl"]
    feat_cols = [c for c in g.columns if c.startswith(("h_td_", "a_td_", "h_prior_",
                 "a_prior_", "d_td_", "d_prior_", "d_qb_comp", "d_off_comp",
                 "h_qb_comp", "a_qb_comp", "h_off_comp", "a_off_comp"))]
    situ = ["rest_diff", "is_div", "is_dome", "is_turf", "temp", "wind", "week_num"]
    out = g[keep_meta + situ + feat_cols].copy()
    return out


if __name__ == "__main__":
    df = build_game_features()
    out = PROC / "game_features.parquet"
    df.to_parquet(out, index=False)
    print(f"Built {len(df)} games x {df.shape[1]} cols -> {out.name}")
    print(f"seasons: {sorted(df.season.unique())}")
    print(f"feature cols: {df.shape[1] - 10}")
    print(f"target home_margin: mean={df.home_margin.mean():.2f} std={df.home_margin.std():.2f}")
