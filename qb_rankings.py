"""
qb_rankings.py  —  2025 QB Rankings
Uses: NGS (CPOE, air yards, aggressiveness, time-to-throw)
      PFR (pressures, blitzes, hurries, hits, bad throws, sacks)
      Derives pressure performance rating

Run: python qb_rankings.py --season 2025
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"


def load(name):
    p = RAW / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def norm(series, invert=False, lo=None, hi=None):
    s = series.copy().astype(float)
    mn = lo if lo is not None else s.min()
    mx = hi if hi is not None else s.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    out = (s - mn) / (mx - mn) * 100
    return (100 - out).clip(0, 100) if invert else out.clip(0, 100)


def rank_qbs(season: int, min_dropbacks: int = 200):

    # ── NGS passing ──────────────────────────────────────────────────
    ngs = load("ngs_passing")
    ngs_s = ngs[
        (ngs["season"] == season) &
        (ngs["season_type"] == "REG")
    ].copy() if not ngs.empty else pd.DataFrame()

    if ngs_s.empty:
        print(f"No NGS data for {season}. Available: {sorted(ngs['season'].unique()) if not ngs.empty else []}")
        return

    ngs_agg = ngs_s.groupby(["player_display_name", "team_abbr"]).agg(
        dropbacks           = ("attempts",                                    "sum"),
        cpoe                = ("completion_percentage_above_expectation",     "mean"),
        avg_air_yards       = ("avg_intended_air_yards",                      "mean"),
        avg_completed_air   = ("avg_completed_air_yards",                     "mean"),
        air_diff            = ("avg_air_yards_differential",                  "mean"),
        aggressiveness      = ("aggressiveness",                              "mean"),
        time_to_throw       = ("avg_time_to_throw",                           "mean"),
        passer_rating       = ("passer_rating",                               "mean"),
        comp_pct            = ("completion_percentage",                       "mean"),
        pass_yards          = ("pass_yards",                                  "sum"),
        pass_tds            = ("pass_touchdowns",                             "sum"),
        interceptions       = ("interceptions",                               "sum"),
        weeks               = ("week",                                        "count"),
    ).reset_index().rename(columns={
        "player_display_name": "name",
        "team_abbr":           "team",
    })

    # Filter volume
    df = ngs_agg[ngs_agg["dropbacks"] >= min_dropbacks].copy()
    df["ypa"]     = df["pass_yards"] / df["dropbacks"].clip(lower=1)
    df["td_rate"] = df["pass_tds"]   / df["dropbacks"].clip(lower=1) * 100
    df["int_rate"]= df["interceptions"] / df["dropbacks"].clip(lower=1) * 100
    df["name_last"] = df["name"].str.split().str[-1].str.upper()

    # ── PFR pressure ─────────────────────────────────────────────────
    pfr = load("pfr_passing")
    pfr_s = pfr[
        (pfr["season"] == season) &
        (pfr["game_type"] == "REG")
    ].copy() if not pfr.empty else pd.DataFrame()

    pfr_agg = pd.DataFrame()
    if not pfr_s.empty:
        pfr_agg = pfr_s.groupby(["pfr_player_name", "team"]).agg(
            times_pressured     = ("times_pressured",         "sum"),
            times_blitzed       = ("times_blitzed",           "sum"),
            times_hurried       = ("times_hurried",           "sum"),
            times_hit           = ("times_hit",               "sum"),
            bad_throws          = ("passing_bad_throws",      "sum"),
            bad_throw_pct       = ("passing_bad_throw_pct",   "mean"),
            drops               = ("passing_drops",           "sum"),
            times_sacked        = ("times_sacked",            "sum"),
            pressure_pct        = ("times_pressured_pct",     "mean"),
        ).reset_index()
        # PFR name format: "Last, First" → extract last name
        pfr_agg["name_last"] = pfr_agg["pfr_player_name"].str.split(",").str[0].str.strip().str.upper()

        # Merge
        df = df.merge(pfr_agg[[
            "name_last","times_pressured","times_blitzed","times_hurried",
            "times_hit","bad_throws","bad_throw_pct","drops","times_sacked","pressure_pct"
        ]], on="name_last", how="left")

        # Pressure efficiency: bad throw rate under pressure (lower = better)
        # Also derive: sack rate = sacks / (dropbacks + sacks)
        df["sack_rate"] = df["times_sacked"] / (df["dropbacks"] + df["times_sacked"]).clip(lower=1) * 100

    # ── Composite score ───────────────────────────────────────────────
    # Primary metrics (what we actually have)
    df["s_cpoe"]        = norm(df["cpoe"])                                  # CPOE: king metric
    df["s_passer_rat"]  = norm(df["passer_rating"])                         # Traditional efficiency
    df["s_air_yards"]   = norm(df["avg_air_yards"])                         # Depth of target (ambition)
    df["s_aggress"]     = norm(df["aggressiveness"])                        # Willingness to push ball
    df["s_td_int"]      = norm(df["td_rate"] - df["int_rate"] * 2)         # TD:INT quality

    if "bad_throw_pct" in df.columns and df["bad_throw_pct"].notna().any():
        df["s_accuracy"] = norm(df["bad_throw_pct"], invert=True)           # Bad throw % (lower better)
        df["s_sack"]     = norm(df["sack_rate"],     invert=True)           # Sack rate (lower better)
        weights = {
            "s_cpoe":       0.30,   # CPOE: best single metric, difficulty-adjusted
            "s_passer_rat": 0.18,   # Passer rating: volume efficiency
            "s_accuracy":   0.17,   # Bad throw %: pure accuracy under any situation
            "s_td_int":     0.15,   # TD vs turnover quality
            "s_sack":       0.10,   # Avoiding sacks = decision speed + mobility
            "s_air_yards":  0.05,   # Aggressiveness (depth of target)
            "s_aggress":    0.05,   # Willingness to push downfield
        }
    else:
        weights = {
            "s_cpoe":       0.40,
            "s_passer_rat": 0.25,
            "s_td_int":     0.20,
            "s_air_yards":  0.08,
            "s_aggress":    0.07,
        }

    df["composite"] = sum(df[col] * w for col, w in weights.items() if col in df.columns)
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # ── MAIN TABLE ────────────────────────────────────────────────────
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  {season} QB RANKINGS  |  NGS + PFR Data  |  Min {min_dropbacks} dropbacks")
    print(f"{sep}")
    print(f"  {'#':<3} {'QB':<22} {'Team':<5} {'DBs':<5}  {'CPOE':>6}  {'PsrRtg':>6}  {'AirYds':>6}  {'Aggress':>7}  {'BadThw%':>7}  {'SackRt':>6}  {'SCORE':>5}")
    print(f"  {'-'*97}")

    for _, r in df.iterrows():
        cpoe   = f"{r['cpoe']:+.1f}%"
        prat   = f"{r['passer_rating']:.1f}"     if pd.notna(r.get('passer_rating'))  else "—"
        airy   = f"{r['avg_air_yards']:.1f}"     if pd.notna(r.get('avg_air_yards'))  else "—"
        agg    = f"{r['aggressiveness']:.1f}%"   if pd.notna(r.get('aggressiveness')) else "—"
        btp    = f"{r['bad_throw_pct']:.1f}%"    if pd.notna(r.get('bad_throw_pct')) and r.get('bad_throw_pct') else "—"
        sackr  = f"{r['sack_rate']:.1f}%"        if pd.notna(r.get('sack_rate'))      else "—"
        print(f"  {int(r['rank']):<3} {r['name']:<22} {r['team']:<5} {int(r['dropbacks']):<5}  "
              f"{cpoe:>6}  {prat:>6}  {airy:>6}  {agg:>7}  {btp:>7}  {sackr:>6}  {r['composite']:>5.1f}")

    # ── PRESSURE TABLE ────────────────────────────────────────────────
    if "times_pressured" in df.columns and df["times_pressured"].notna().any():
        print(f"\n{'='*95}")
        print(f"  UNDER PRESSURE  —  {season}")
        print(f"  (ranked by composite above, showing pressure volume and accuracy metrics)")
        print(f"{'='*95}")
        print(f"  {'#':<3} {'QB':<22} {'Team':<5}  {'Pressured':>9}  {'Blitzed':>7}  {'Hurried':>7}  {'Hit':>5}  {'Sacked':>6}  {'BadThrow':>8}  {'Press%':>6}")
        print(f"  {'-'*92}")

        for _, r in df.sort_values("rank").iterrows():
            if pd.isna(r.get("times_pressured")):
                continue
            print(f"  {int(r['rank']):<3} {r['name']:<22} {r['team']:<5}  "
                  f"{int(r['times_pressured']):>9}  "
                  f"{int(r['times_blitzed']):>7}  "
                  f"{int(r['times_hurried']):>7}  "
                  f"{int(r['times_hit']):>5}  "
                  f"{int(r['times_sacked']):>6}  "
                  f"{int(r['bad_throws']):>8}  "
                  f"{r['pressure_pct']:>5.1f}%")

    # ── TIER SUMMARY ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TIER BREAKDOWN  —  {season}")
    print(f"{'='*60}")
    tiers = [("ELITE",     df["composite"] >= 70),
             ("GOOD",      (df["composite"] >= 55) & (df["composite"] < 70)),
             ("AVERAGE",   (df["composite"] >= 42) & (df["composite"] < 55)),
             ("BELOW AVG", df["composite"] < 42)]
    for label, mask in tiers:
        names = df[mask]["name"].tolist()
        if names:
            print(f"\n  {label}:")
            for n in names:
                row = df[df["name"]==n].iloc[0]
                print(f"    #{int(row['rank']):<2} {n:<22} {row['team']:<5}  CPOE {row['cpoe']:+.1f}%  Score {row['composite']:.0f}")

    # ── KEY NOTES ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  METRIC GUIDE")
    print(f"{'='*60}")
    print(f"  CPOE    Completion % Over Expectation (NGS)")
    print(f"          Adjusts for throw difficulty — best single QB metric")
    print(f"  PsrRtg  Traditional passer rating (comp, yds, TD, INT)")
    print(f"  AirYds  Avg intended air yards per attempt (depth of target)")
    print(f"  Aggress % of throws into tight windows (aggressiveness)")
    print(f"  BadThw% Bad throw rate (PFR, lower = more accurate)")
    print(f"  SackRt  Sack rate (lower = faster decisions / better mobility)")
    print(f"  SCORE   Composite 0-100 (CPOE 30%, PsrRtg 18%, Accuracy 17%,")
    print(f"          TD:INT 15%, SackRt 10%, AirYds 5%, Aggress 5%)")

    out = PROC / f"qb_rankings_{season}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",        type=int, default=2025)
    parser.add_argument("--min-dropbacks", type=int, default=200)
    args = parser.parse_args()
    rank_qbs(args.season, args.min_dropbacks)
