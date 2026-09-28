"""
backtest_factors.py  —  do situational factors carry signal AGAINST THE CLOSING LINE?
====================================================================================
Run:  python backtest_factors.py        (2021-2025 regular season, ~1 min)

Every candidate is regressed on cover margin = (home - away) - spread_line, using only
information available before kickoff (season-to-date stats use prior weeks only). A
factor earns its way into the score only if it is significant (|t| > 2) AND the sign
holds in both halves (2021-23 vs 2024-25); everything else stays display-only, per the
project's evidence standard.

Result on 2026-09-28 (the owner asked for divisional, home-ground, weather, travel and
desperation factors):
  divisional game .......... t=+0.12   nothing
  per-team home advantage .. t=+0.12   real for PREDICTION (already in ml/squad.team_hfa),
                                       fully priced by the market
  rest differential ........ t=+0.38   nothing (short week either side t=-1.2, unstable)
  wind (cover) ............. t=-0.07   nothing;  wind (TOTALS) t=-2.18 STABLE, -0.14 pts/mph
                                       -> ml/context already takes ~2-3 pts off the total
  cold / dome .............. t=+1.0 / -0.45   nothing on cover
  west-coast early kickoff . t=-0.51   nothing; time-zones crossed t=-1.92 (sign stable,
                                       -0.65 pts/hour) -> ml/context already penalises ~0.3/hr
                                       + distance, about the same size
  turnover / EPA luck ...... |t|<0.9   nothing (the market already regresses them)
  record, last result, streaks |t|<1.2 nothing
  home team eliminated late  t=-2.02   borderline (n=536, -2.6 pts, one of ~40 tests);
                                       candidate to re-test after 2026 — NOT wired
  primetime ................ t=-1.71   home teams under-cover, both halves negative,
                                       not significant — watch
Conclusion: the closing line already prices these; the engine's job is prediction.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
RAW = str(Path(__file__).parent / "data" / "raw")
TZ = {"ARI": -7, "ATL": -5, "BAL": -5, "BUF": -5, "CAR": -5, "CHI": -6, "CIN": -5, "CLE": -5, "DAL": -6, "DEN": -7,
      "DET": -5, "GB": -6, "HOU": -6, "IND": -5, "JAX": -5, "KC": -6, "LA": -8, "LAC": -8, "LV": -8, "MIA": -5,
      "MIN": -6, "NE": -5, "NO": -6, "NYG": -5, "NYJ": -5, "PHI": -5, "PIT": -5, "SEA": -8, "SF": -8, "TB": -5,
      "TEN": -6, "WAS": -5}


def _ols(x, y):
    x = x - x.mean()
    if (x ** 2).sum() == 0:
        return 0.0, 0.0
    b = (x * (y - y.mean())).sum() / (x ** 2).sum()
    res = (y - y.mean()) - b * x
    se = np.sqrt((res ** 2).sum() / (len(x) - 2) / (x ** 2).sum())
    return b, b / se


def report(df, feats, target="cover"):
    X = df[feats].fillna(0).astype(float); y = df[target].astype(float)
    print(f"\n{target}: n={len(df)}  mean {y.mean():+.2f}")
    print(f"  {'feature':12} {'coef':>7} {'t':>6} | {'2021-23':>8} {'2024-25':>8}  (per-half coef)")
    for f in feats:
        x = X[f]
        if x.std() == 0:
            continue
        b, t = _ols(x, y)
        b1, _ = _ols(x[df.season <= 2023], y[df.season <= 2023]); b2, _ = _ols(x[df.season >= 2024], y[df.season >= 2024])
        flag = " <-- stable" if abs(t) > 2 and np.sign(b1) == np.sign(b2) else ""
        print(f"  {f:12} {b:+7.2f} {t:+6.2f} | {b1:+8.2f} {b2:+8.2f}{flag}")


def main():
    s = pd.read_parquet(RAW + "/schedules.parquet")
    s = s[(s.game_type == "REG") & s.home_score.notna() & s.spread_line.notna() & (s.season.between(2021, 2025))].copy()
    s["margin"] = s.home_score - s.away_score
    s["cover"] = s.margin - s.spread_line
    s["total_res"] = (s.home_score + s.away_score) - s.total_line

    # ── season-to-date team features from PBP (prior weeks only) ──
    rows = []
    for yr in range(2021, 2026):
        p = pd.read_parquet(f"{RAW}/pbp_{yr}.parquet")
        p = p[(p.week <= 18) & p.posteam.notna() & p.defteam.notna()]
        g = p.groupby(["game_id", "week", "posteam", "defteam"]).agg(
            epa=("epa", "sum"), fum_lost=("fumble", "sum"),           # slim PBP has no fumble_lost
            intc=("interception", "sum"), plays=("play_id", "count")).reset_index()
        sc = s[s.season == yr][["game_id", "home_team", "away_team", "home_score", "away_score"]]
        g = g.merge(sc, on="game_id")
        g["pts"] = np.where(g.posteam == g.home_team, g.home_score, g.away_score)
        g["opp_pts"] = np.where(g.posteam == g.home_team, g.away_score, g.home_score)
        g["season"] = yr
        rows.append(g)
    tg = pd.concat(rows)
    give = tg[["season", "game_id", "posteam", "fum_lost", "intc", "epa"]].rename(columns={"posteam": "t", "fum_lost": "gv_f", "intc": "gv_i", "epa": "off_epa"})
    take = tg[["season", "game_id", "defteam", "fum_lost", "intc", "epa"]].rename(columns={"defteam": "t", "fum_lost": "tk_f", "intc": "tk_i", "epa": "def_epa_allowed"})
    tm = give.merge(take, on=["season", "game_id", "t"]).merge(
        tg[["season", "game_id", "posteam", "week", "pts", "opp_pts"]].rename(columns={"posteam": "t"}), on=["season", "game_id", "t"])
    tm["to_margin"] = (tm.tk_f + tm.tk_i) - (tm.gv_f + tm.gv_i)
    tm["net_epa"] = tm.off_epa - tm.def_epa_allowed
    tm["pd"] = tm.pts - tm.opp_pts
    tm = tm.sort_values(["season", "t", "week"])
    for c in ("to_margin", "net_epa", "pd"):
        tm[c + "_std"] = tm.groupby(["season", "t"])[c].transform(lambda x: x.shift(1).expanding().mean())
    tm["g_std"] = tm.groupby(["season", "t"]).cumcount()
    tm["luck_std"] = tm.pd_std - tm.net_epa_std                      # points vs what net EPA "should" give
    tm["win"] = (tm.pd > 0).astype(float)
    grp = tm.groupby(["season", "t"])
    tm["wpct_std"] = grp["win"].transform(lambda x: x.shift(1).expanding().mean())
    tm["lost_prev"] = grp["win"].transform(lambda x: 1 - x.shift(1))
    tm["lstreak"] = grp["win"].transform(lambda x: (1 - x).shift(1).rolling(3, min_periods=1).sum())
    tm["wstreak"] = grp["win"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    tm["wins_std"] = grp["win"].transform(lambda x: x.shift(1).cumsum())
    feat = tm[["season", "game_id", "t", "to_margin_std", "luck_std", "g_std", "wpct_std", "lost_prev", "lstreak", "wstreak", "wins_std"]]
    h = feat.rename(columns={"t": "home_team", "to_margin_std": "h_to", "luck_std": "h_luck", "g_std": "h_g", "wpct_std": "h_wp",
                             "lost_prev": "h_lp", "lstreak": "h_ls", "wstreak": "h_ws", "wins_std": "h_w"})
    a = feat.rename(columns={"t": "away_team", "to_margin_std": "a_to", "luck_std": "a_luck", "g_std": "a_g", "wpct_std": "a_wp",
                             "lost_prev": "a_lp", "lstreak": "a_ls", "wstreak": "a_ws", "wins_std": "a_w"})
    d = s.merge(h, on=["season", "game_id", "home_team"], how="left").merge(a, on=["season", "game_id", "away_team"], how="left")

    # ── exogenous features ──
    d["rest_diff"] = d.home_rest - d.away_rest
    d["home_short"] = (d.home_rest <= 5).astype(int); d["away_short"] = (d.away_rest <= 5).astype(int)
    d["home_bye"] = (d.home_rest >= 13).astype(int); d["away_bye"] = (d.away_rest >= 13).astype(int)
    d["div"] = d.div_game.fillna(0).astype(int)
    d["dome"] = d.roof.isin(["dome", "closed"]).astype(int)
    d["wind"] = np.where(d.dome == 1, 0, d.wind.fillna(0))
    d["cold"] = np.where(d.dome == 1, 0, (d.temp.fillna(60) < 35).astype(int))
    hr = d.gametime.fillna("13:00").str.slice(0, 2).astype(int)
    d["west_early"] = ((d.away_team.map(TZ) <= -7) & (d.home_team.map(TZ) >= -6) & (hr <= 13)).astype(int)
    d["east_late"] = ((d.away_team.map(TZ) >= -5) & (d.home_team.map(TZ) <= -8) & (hr >= 20)).astype(int)
    d["tz_diff"] = (d.home_team.map(TZ) - d.away_team.map(TZ)).abs()
    d["to_diff"] = d.h_to - d.a_to
    d["luck_diff"] = d.h_luck - d.a_luck
    d["big_fav"] = (d.spread_line.abs() >= 7).astype(int) * np.sign(d.spread_line)
    d["home_dog"] = (d.spread_line < 0).astype(int)
    d["primetime"] = (hr >= 20).astype(int)
    allg = pd.read_parquet(RAW + "/schedules.parquet")
    allg = allg[(allg.game_type == "REG") & allg.home_score.notna() & (allg.season >= 2019)]
    allg["hm"] = allg.home_score - allg.away_score
    hfa = {}
    for yr in range(2021, 2026):                                    # prior seasons only
        pr = allg[allg.season < yr]; lg = pr.hm.mean()
        hfa[yr] = (pr.groupby("home_team").hm.mean() - lg).to_dict()
    d["hfa_dev"] = [hfa[y].get(t, 0.0) for y, t in zip(d.season, d.home_team)]
    d["wp_diff"] = d.h_wp - d.a_wp
    d["lp_diff"] = d.h_lp - d.a_lp
    d["ls_diff"] = d.h_ls - d.a_ls
    d["ws_diff"] = d.h_ws - d.a_ws
    late = d[d.week >= 12].copy()
    late["h_elim"] = (late.h_w <= (late.week - 1) * 0.30).astype(int); late["a_elim"] = (late.a_w <= (late.week - 1) * 0.30).astype(int)
    late["elim_diff"] = late.h_elim - late.a_elim
    late["h_race"] = ((late.h_w >= (late.week - 1) * 0.5) & (late.h_w <= (late.week - 1) * 0.65)).astype(int)
    late["a_race"] = ((late.a_w >= (late.week - 1) * 0.5) & (late.a_w <= (late.week - 1) * 0.65)).astype(int)
    late["race_diff"] = late.h_race - late.a_race

    report(d, ["rest_diff", "home_short", "away_short", "home_bye", "away_bye", "div", "dome", "wind", "cold",
               "west_early", "east_late", "tz_diff", "big_fav", "home_dog", "primetime", "hfa_dev"], "cover")
    report(d[d.week >= 4], ["to_diff", "luck_diff", "h_to", "a_to", "h_luck", "a_luck"], "cover")
    report(d[d.week >= 4], ["wp_diff", "lp_diff", "ls_diff", "ws_diff", "h_lp", "a_lp", "h_ls", "a_ls"], "cover")
    report(late, ["elim_diff", "h_elim", "a_elim", "race_diff", "h_race", "a_race"], "cover")
    report(d, ["wind", "cold", "dome", "primetime", "div"], "total_res")


if __name__ == "__main__":
    main()
