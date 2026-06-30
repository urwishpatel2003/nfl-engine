"""
skill_rankings.py  —  2025 WR and RB Rankings
Uses: NGS receiving/rushing, player_stats, PFR receiving, snap counts

Run: python skill_rankings.py --season 2025
     python skill_rankings.py --season 2025 --pos WR
     python skill_rankings.py --season 2025 --pos RB
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"


def load(name):
    p = RAW / f"{name}.parquet"
    if not p.exists():
        print(f"  WARN: {name}.parquet not found")
        return pd.DataFrame()
    return pd.read_parquet(p)


def norm(series, invert=False):
    s = series.copy().astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    out = (s - mn) / (mx - mn) * 100
    return (100 - out).clip(0, 100) if invert else out.clip(0, 100)


def rank_wr(season, min_targets=50):
    print(f"\n{'='*85}")
    print(f"  {season} WR RANKINGS  |  NGS + PFR + Player Stats  |  Min {min_targets} targets")
    print(f"{'='*85}")

    # NGS receiving
    ngs = load("ngs_receiving")
    ngs_s = ngs[
        (ngs["season"] == season) &
        (ngs["season_type"] == "REG")
    ].copy() if not ngs.empty else pd.DataFrame()

    if ngs_s.empty:
        print(f"No NGS receiving data for {season}. Available: {sorted(ngs['season'].unique()) if not ngs.empty else []}")
        return pd.DataFrame()

    # Filter to WR/TE position
    if "player_position" in ngs_s.columns:
        ngs_wr = ngs_s[ngs_s["player_position"].isin(["WR","TE","SLOT"])].copy()
    else:
        ngs_wr = ngs_s.copy()

    name_col = "player_display_name" if "player_display_name" in ngs_wr.columns else "player_gsis_id"
    team_col = "team_abbr" if "team_abbr" in ngs_wr.columns else "team"
    pos_col  = "player_position" if "player_position" in ngs_wr.columns else None

    agg_dict = {}
    for col, agg in [
        ("targets",            "sum"),
        ("receptions",         "sum"),
        ("yards",              "sum"),
        ("rec_touchdowns",     "sum"),
        ("avg_cushion",        "mean"),
        ("avg_separation",     "mean"),
        ("avg_intended_air_yards","mean"),
        ("avg_yac",            "mean"),
        ("avg_expected_yac",   "mean"),
        ("avg_yac_above_expectation","mean"),
        ("catch_percentage",   "mean"),
        ("percent_share_of_intended_air_yards","mean"),
    ]:
        if col in ngs_wr.columns:
            agg_dict[col] = (col, agg)

    gb_cols = [name_col, team_col] + ([pos_col] if pos_col else [])
    df = ngs_wr.groupby(gb_cols).agg(**agg_dict).reset_index()
    df = df.rename(columns={name_col:"name", team_col:"team"})
    if pos_col:
        df = df.rename(columns={pos_col:"pos"})

    # Derived
    df["catch_pct"]   = df["receptions"] / df["targets"].clip(lower=1) * 100
    df["ypr"]         = df["yards"] / df["receptions"].clip(lower=1)
    df["ypt"]         = df["yards"] / df["targets"].clip(lower=1)
    df["td_rate"]     = df["rec_touchdowns"] / df["targets"].clip(lower=1) * 100
    df["yac_oe"]      = df.get("avg_yac_above_expectation", pd.Series(0, index=df.index))
    df["name_last"]   = df["name"].str.split().str[-1].str.upper()

    # Volume filter
    if "targets" in df.columns:
        df = df[df["targets"] >= min_targets].copy()

    # PFR receiving (broken tackles, drops)
    pfr = load("pfr_receiving")
    pfr_s = pfr[
        (pfr["season"] == season) &
        (pfr["game_type"] == "REG")
    ].copy() if not pfr.empty else pd.DataFrame()

    if not pfr_s.empty:
        pfr_agg_dict = {}
        for col, agg in [
            ("receiving_broken_tackles","sum"),
            ("receiving_drop",          "sum"),
            ("receiving_drop_pct",      "mean"),
            ("receiving_int",           "sum"),
        ]:
            if col in pfr_s.columns:
                pfr_agg_dict[col] = (col, agg)

        if pfr_agg_dict:
            pfr_agg = pfr_s.groupby(["pfr_player_name","team"]).agg(**pfr_agg_dict).reset_index()
            pfr_agg["name_last"] = pfr_agg["pfr_player_name"].str.split(",").str[0].str.strip().str.upper()
            df = df.merge(pfr_agg[["name_last"] + list(pfr_agg_dict.keys())], on="name_last", how="left")

    # Player stats (season totals for volume)
    pstats = load("player_stats")
    if not pstats.empty:
        ps = pstats[
            (pstats["season"] == season) &
            (pstats["position"].isin(["WR","TE"]))
        ].copy()
        if "season_type" in ps.columns:
            reg = [v for v in ps["season_type"].unique() if "reg" in str(v).lower()]
            if reg:
                ps = ps[ps["season_type"].isin(reg)]
        name_c = "player_display_name" if "player_display_name" in ps.columns else "player_name"
        team_c = "recent_team" if "recent_team" in ps.columns else "team"
        ps_agg = ps.groupby([name_c, team_c]).agg(
            receiving_yards=("receiving_yards","sum"),
            receiving_tds  =("receiving_tds",  "sum"),
            total_targets  =("targets",        "sum"),
            reception_count=("receptions",     "sum"),
            receiving_epa  =("receiving_epa",  "sum") if "receiving_epa" in ps.columns else ("receiving_yards","sum"),
            air_yards_total=("receiving_air_yards","sum") if "receiving_air_yards" in ps.columns else ("receiving_yards","sum"),
        ).reset_index()
        ps_agg["name_last"] = ps_agg[name_c].str.split().str[-1].str.upper()
        df = df.merge(ps_agg[["name_last","receiving_epa","air_yards_total"]], on="name_last", how="left")

    # Composite score
    score_map = {
        "s_sep":    ("avg_separation",   False, 0.22),
        "s_yac_oe": ("yac_oe",           False, 0.18),
        "s_ypt":    ("ypt",              False, 0.15),
        "s_catch":  ("catch_pct",        False, 0.13),
        "s_air":    ("avg_intended_air_yards", False, 0.10),
        "s_cushion":("avg_cushion",      False, 0.08),
        "s_drop":   ("receiving_drop_pct",True,  0.08) if "receiving_drop_pct" in df.columns else None,
        "s_btk":    ("receiving_broken_tackles",False,0.06) if "receiving_broken_tackles" in df.columns else None,
    }
    score_map = {k:v for k,v in score_map.items() if v}
    total_w = sum(v[2] for v in score_map.values())

    for sname, (col, inv, w) in score_map.items():
        if col in df.columns and df[col].notna().any():
            df[sname] = norm(df[col], invert=inv)
        else:
            df[sname] = 50.0

    df["composite"] = sum(df[s] * (w/total_w) for s,(c,i,w) in score_map.items())
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Print
    has_sep  = "avg_separation" in df.columns and df["avg_separation"].notna().any()
    has_drop = "receiving_drop_pct" in df.columns and df["receiving_drop_pct"].notna().any()
    has_btk  = "receiving_broken_tackles" in df.columns and df["receiving_broken_tackles"].notna().any()

    hdr = f"  {'#':<3} {'WR':<24} {'Team':<5} {'Tgt':<4} {'Rec':<4} {'Yds':<5} {'Sep':>5} {'YAC+':>5} {'Catch%':>7} {'YPT':>5}"
    if has_drop: hdr += f" {'Drop%':>6}"
    if has_btk:  hdr += f" {'BTkl':>5}"
    hdr += f" {'SCORE':>5}"
    print(hdr)
    print(f"  {'-'*95}")

    for _, r in df.iterrows():
        sep  = f"{r['avg_separation']:.1f}" if pd.notna(r.get('avg_separation')) else "—"
        yacoe= f"{r['yac_oe']:+.1f}"       if pd.notna(r.get('yac_oe'))         else "—"
        cp   = f"{r['catch_pct']:.1f}%"    if pd.notna(r.get('catch_pct'))       else "—"
        ypt  = f"{r['ypt']:.1f}"           if pd.notna(r.get('ypt'))             else "—"
        drp  = f"{r['receiving_drop_pct']:.1f}%" if has_drop and pd.notna(r.get('receiving_drop_pct')) else ""
        btk  = f"{int(r['receiving_broken_tackles'])}" if has_btk and pd.notna(r.get('receiving_broken_tackles')) else ""
        pos_tag = f"({r['pos']})" if "pos" in r and pd.notna(r.get("pos")) else ""

        line = (f"  {int(r['rank']):<3} {r['name']:<22} {pos_tag:<3} {r['team']:<5} "
                f"{int(r.get('targets',0)):<4} {int(r.get('receptions',0)):<4} "
                f"{int(r.get('yards',0)):<5} {sep:>5} {yacoe:>5} {cp:>7} {ypt:>5}")
        if has_drop: line += f" {drp:>6}"
        if has_btk:  line += f" {btk:>5}"
        line += f" {r['composite']:>5.1f}"
        print(line)

    # Save
    out = PROC / f"wr_rankings_{season}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df


def rank_rb(season, min_carries=75):
    print(f"\n{'='*85}")
    print(f"  {season} RB RANKINGS  |  NGS + PFR + Player Stats  |  Min {min_carries} carries")
    print(f"{'='*85}")

    # NGS rushing
    ngs = load("ngs_rushing")
    ngs_s = ngs[
        (ngs["season"] == season) &
        (ngs["season_type"] == "REG")
    ].copy() if not ngs.empty else pd.DataFrame()

    if ngs_s.empty:
        print(f"No NGS rushing data for {season}. Available: {sorted(ngs['season'].unique()) if not ngs.empty else []}")
        return pd.DataFrame()

    name_col = "player_display_name" if "player_display_name" in ngs_s.columns else "player_gsis_id"
    team_col = "team_abbr" if "team_abbr" in ngs_s.columns else "team"

    agg_dict = {}
    for col, agg in [
        ("rush_attempts",       "sum"),
        ("rush_yards",          "sum"),
        ("rush_touchdowns",     "sum"),
        ("efficiency",          "mean"),
        ("avg_time_to_los",     "mean"),
        ("avg_rush_yards",      "mean"),
        ("rush_yards_over_expected",     "sum"),
        ("rush_yards_over_expected_per_att","mean"),
        ("rush_pct_over_expected","mean"),
        ("expected_rush_yards", "sum"),
        ("percent_attempts_gte_eight_defenders","mean"),
    ]:
        if col in ngs_s.columns:
            agg_dict[col] = (col, agg)

    df = ngs_s.groupby([name_col, team_col]).agg(**agg_dict).reset_index()
    df = df.rename(columns={name_col:"name", team_col:"team"})

    # Derived
    df["ypc"]       = df["rush_yards"] / df["rush_attempts"].clip(lower=1)
    df["td_rate"]   = df["rush_touchdowns"] / df["rush_attempts"].clip(lower=1) * 100
    df["name_last"] = df["name"].str.split().str[-1].str.upper()
    df["ryoe_att"]  = df.get("rush_yards_over_expected_per_att", pd.Series(0, index=df.index))

    # Filter volume
    df = df[df["rush_attempts"] >= min_carries].copy()

    # PFR receiving (broken tackles, also captures elusive RBs)
    pfr = load("pfr_receiving")
    pfr_s = pfr[
        (pfr["season"] == season) &
        (pfr["game_type"] == "REG")
    ].copy() if not pfr.empty else pd.DataFrame()

    if not pfr_s.empty and "rushing_broken_tackles" in pfr_s.columns:
        pfr_rb = pfr_s.groupby(["pfr_player_name","team"]).agg(
            broken_tackles = ("rushing_broken_tackles","sum"),
            drops          = ("receiving_drop",        "sum"),
        ).reset_index()
        pfr_rb["name_last"] = pfr_rb["pfr_player_name"].str.split(",").str[0].str.strip().str.upper()
        df = df.merge(pfr_rb[["name_last","broken_tackles","drops"]], on="name_last", how="left")

    # Player stats for receiving
    pstats = load("player_stats")
    if not pstats.empty:
        ps = pstats[
            (pstats["season"] == season) &
            (pstats["position"] == "RB")
        ].copy()
        if "season_type" in ps.columns:
            reg = [v for v in ps["season_type"].unique() if "reg" in str(v).lower()]
            if reg:
                ps = ps[ps["season_type"].isin(reg)]
        name_c = "player_display_name" if "player_display_name" in ps.columns else "player_name"
        team_c = "recent_team" if "recent_team" in ps.columns else "team"
        ps_agg = ps.groupby([name_c]).agg(
            rec_yards  = ("receiving_yards","sum"),
            rec_tds    = ("receiving_tds",  "sum"),
            receptions = ("receptions",     "sum"),
            rec_targets= ("targets",        "sum"),
            rush_epa   = ("rushing_epa",    "sum") if "rushing_epa" in ps.columns else ("rushing_yards","sum"),
            rec_epa    = ("receiving_epa",  "sum") if "receiving_epa" in ps.columns else ("receiving_yards","sum"),
        ).reset_index()
        ps_agg["name_last"] = ps_agg[name_c].str.split().str[-1].str.upper()
        df = df.merge(ps_agg[["name_last","rec_yards","receptions","rec_targets","rush_epa","rec_epa"]], on="name_last", how="left")
        df["total_epa"] = df.get("rush_epa", 0).fillna(0) + df.get("rec_epa", 0).fillna(0)

    # Composite
    score_map = {
        "s_ryoe":   ("ryoe_att",         False, 0.28),
        "s_eff":    ("efficiency",        False, 0.20),
        "s_ypc":    ("ypc",               False, 0.15),
        "s_poe":    ("rush_pct_over_expected", False, 0.12),
        "s_btk":    ("broken_tackles",    False, 0.12) if "broken_tackles" in df.columns else None,
        "s_recv":   ("rec_yards",         False, 0.08) if "rec_yards" in df.columns else None,
        "s_stacks": ("percent_attempts_gte_eight_defenders", False, 0.05),
    }
    score_map = {k:v for k,v in score_map.items() if v}
    total_w = sum(v[2] for v in score_map.values())

    for sname, (col, inv, w) in score_map.items():
        if col in df.columns and df[col].notna().any():
            df[sname] = norm(df[col], invert=inv)
        else:
            df[sname] = 50.0

    df["composite"] = sum(df[s] * (w/total_w) for s,(c,i,w) in score_map.items())
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    has_btk  = "broken_tackles" in df.columns and df["broken_tackles"].notna().any()
    has_recv  = "rec_yards" in df.columns and df["rec_yards"].notna().any()
    has_epa   = "total_epa" in df.columns and df["total_epa"].notna().any()

    hdr = f"  {'#':<3} {'RB':<24} {'Team':<5} {'Car':<4} {'Yds':<5} {'YPC':>5} {'RYOE/att':>9} {'Eff':>5} {'StackRt':>8}"
    if has_btk:  hdr += f" {'BTkl':>5}"
    if has_recv: hdr += f" {'RecYd':>6}"
    if has_epa:  hdr += f" {'TotEPA':>7}"
    hdr += f" {'SCORE':>5}"
    print(hdr)
    print(f"  {'-'*95}")

    for _, r in df.iterrows():
        ypc   = f"{r['ypc']:.1f}"        if pd.notna(r.get('ypc'))       else "—"
        ryoe  = f"{r['ryoe_att']:+.2f}"  if pd.notna(r.get('ryoe_att'))  else "—"
        eff   = f"{r['efficiency']:.2f}" if pd.notna(r.get('efficiency'))  else "—"
        stk   = f"{r.get('percent_attempts_gte_eight_defenders',0):.1f}%" if pd.notna(r.get('percent_attempts_gte_eight_defenders')) else "—"
        btk   = f"{int(r['broken_tackles'])}" if has_btk and pd.notna(r.get('broken_tackles')) else ""
        recv  = f"{int(r['rec_yards'])}"  if has_recv and pd.notna(r.get('rec_yards'))  else ""
        tepa  = f"{r['total_epa']:+.0f}" if has_epa  and pd.notna(r.get('total_epa'))  else ""

        line  = (f"  {int(r['rank']):<3} {r['name']:<24} {r['team']:<5} "
                 f"{int(r.get('rush_attempts',0)):<4} "
                 f"{int(r.get('rush_yards',0)):<5} "
                 f"{ypc:>5} {ryoe:>9} {eff:>5} {stk:>8}")
        if has_btk:  line += f" {btk:>5}"
        if has_recv: line += f" {recv:>6}"
        if has_epa:  line += f" {tepa:>7}"
        line += f" {r['composite']:>5.1f}"
        print(line)

    out = PROC / f"rb_rankings_{season}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",      type=int, default=2025)
    parser.add_argument("--pos",         type=str, default="ALL",
                        choices=["WR","RB","ALL"])
    parser.add_argument("--min-targets", type=int, default=50)
    parser.add_argument("--min-carries", type=int, default=75)
    args = parser.parse_args()

    if args.pos in ("WR","ALL"):
        rank_wr(args.season, args.min_targets)
    if args.pos in ("RB","ALL"):
        rank_rb(args.season, args.min_carries)
