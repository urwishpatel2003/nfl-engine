"""
ol_dl_st_rankings.py  —  OL, DL, Special Teams Rankings
========================================================
Uses:
  OL  — pfr_defense (pressure allowed, sacks allowed), snap_counts,
         PBP-derived pass block win rate proxy (sack rate by team)
  DL  — pfr_defense (sacks, pressures, hurries, tackles), snap_counts
  ST  — schedules (fg%, punt avg, return yds), player_stats (K/P)

Run:
    python ol_dl_st_rankings.py --season 2025
    python ol_dl_st_rankings.py --season 2025 --pos OL
    python ol_dl_st_rankings.py --season 2025 --pos DL
    python ol_dl_st_rankings.py --season 2025 --pos ST
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


# ──────────────────────────────────────────────────────────────────
# OL RANKINGS  (team-level — OL grades are team metrics not individual)
# ──────────────────────────────────────────────────────────────────
def rank_ol(season):
    print(f"\n{'='*85}")
    print(f"  {season} OFFENSIVE LINE RANKINGS  (team-level)")
    print(f"  Metrics: sack rate allowed, pressure rate allowed, stuffed run %, YPC")
    print(f"{'='*85}")

    # Primary source: situational_stats has pressure_rate_allowed, sack_rate_allowed
    sit = load("situational_stats")
    sit_s = sit[sit["season"] == season].copy() if not sit.empty else pd.DataFrame()

    # Secondary: PBP-derived (sack rate, stuffed run rate per team)
    pbp_file = RAW / f"pbp_{season}.parquet"
    pbp = pd.read_parquet(pbp_file) if pbp_file.exists() else pd.DataFrame()

    teams = []

    if not pbp.empty:
        pbp_off = pbp[pbp["play_type"].isin(["pass","run","qb_spike"])].copy()

        # Pass blocking: sacks allowed, pressures
        pass_plays = pbp_off[pbp_off["pass_attempt"] == 1].copy()
        sack_data  = pass_plays.groupby("posteam").agg(
            dropbacks   = ("play_id", "count"),
            sacks_allow = ("sack",    "sum"),
        ).reset_index().rename(columns={"posteam": "team"})
        sack_data["sack_rate_allowed"] = sack_data["sacks_allow"] / sack_data["dropbacks"].clip(lower=1) * 100

        # Run blocking: stuffed runs (≤0 yards), YPC
        run_plays = pbp_off[pbp_off["rush_attempt"] == 1].copy()
        run_data = run_plays.groupby("posteam").agg(
            carries     = ("play_id",      "count"),
            total_yds   = ("yards_gained", "sum"),
            stuffed     = ("yards_gained", lambda x: (x <= 0).sum()),
        ).reset_index().rename(columns={"posteam": "team"})
        run_data["ypc"]          = run_data["total_yds"] / run_data["carries"].clip(lower=1)
        run_data["stuff_rate"]   = run_data["stuffed"]   / run_data["carries"].clip(lower=1) * 100

        df = sack_data.merge(run_data[["team","ypc","stuff_rate"]], on="team", how="outer")
    else:
        df = pd.DataFrame()

    # Merge situational stats (pressure rate allowed, sack rate allowed from their data)
    if not sit_s.empty:
        sit_cols = ["team"] + [c for c in ["sack_rate_allowed","pressure_rate_allowed"] if c in sit_s.columns]
        if len(sit_cols) > 1:
            if df.empty:
                df = sit_s[sit_cols].copy()
            else:
                df = df.merge(sit_s[sit_cols].rename(columns={
                    "sack_rate_allowed":    "sack_sit",
                    "pressure_rate_allowed":"pressure_sit",
                }), on="team", how="left")
                # Prefer PBP sack rate, fall back to situational
                if "sack_sit" in df.columns and "sack_rate_allowed" not in df.columns:
                    df["sack_rate_allowed"] = df["sack_sit"]

    if df.empty:
        print("  No PBP or situational data found. Run fetch_data.py with PBP.")
        return pd.DataFrame()

    # Composite: lower sack rate + lower stuff rate + higher YPC
    score_map = {
        "s_sack":   ("sack_rate_allowed", True,  0.35),
        "s_stuff":  ("stuff_rate",        True,  0.30),
        "s_ypc":    ("ypc",               False, 0.25),
        "s_press":  ("pressure_sit",      True,  0.10) if "pressure_sit" in df.columns else None,
    }
    score_map = {k: v for k, v in score_map.items() if v}
    total_w = sum(v[2] for v in score_map.values())

    for sname, (col, inv, w) in score_map.items():
        if col in df.columns and df[col].notna().any():
            df[sname] = norm(df[col], invert=inv)
        else:
            df[sname] = 50.0

    df["composite"] = sum(df[s] * (w / total_w) for s, (c, i, w) in score_map.items())
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Print
    has_press = "pressure_sit" in df.columns and df["pressure_sit"].notna().any()
    hdr = f"  {'#':<3} {'Team':<6} {'DBs':<5} {'SackRt':>7} {'StuffRt':>8} {'YPC':>5}"
    if has_press: hdr += f" {'PressRt':>8}"
    hdr += f" {'SCORE':>6}"
    print(hdr)
    print(f"  {'-'*65}")

    for _, r in df.iterrows():
        sack  = f"{r['sack_rate_allowed']:.1f}%"    if pd.notna(r.get('sack_rate_allowed'))  else "—"
        stuff = f"{r['stuff_rate']:.1f}%"            if pd.notna(r.get('stuff_rate'))          else "—"
        ypc   = f"{r['ypc']:.2f}"                   if pd.notna(r.get('ypc'))                 else "—"
        press = f"{r['pressure_sit']:.1f}%"         if has_press and pd.notna(r.get('pressure_sit')) else ""
        dbs   = f"{int(r['dropbacks'])}"             if pd.notna(r.get('dropbacks'))            else "—"
        line  = f"  {int(r['rank']):<3} {r['team']:<6} {dbs:<5} {sack:>7} {stuff:>8} {ypc:>5}"
        if has_press: line += f" {press:>8}"
        line += f" {r['composite']:>6.1f}"
        print(line)

    out = PROC / f"ol_rankings_{season}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df


# ──────────────────────────────────────────────────────────────────
# DL RANKINGS  (individual pass rushers from pfr_defense)
# ──────────────────────────────────────────────────────────────────
def rank_dl(season, min_snaps=150):
    print(f"\n{'='*85}")
    print(f"  {season} DL / EDGE RANKINGS  |  Min ~{min_snaps} snaps")
    print(f"  Metrics: sacks, pressures, hurries, hits, tackles, missed tackle rate")
    print(f"{'='*85}")

    pfr_def = load("pfr_defense")
    if pfr_def.empty:
        print("  No pfr_defense.parquet. Run fetch_data.py.")
        return pd.DataFrame()

    pfr_s = pfr_def[
        (pfr_def["season"] == season) &
        (pfr_def["game_type"] == "REG")
    ].copy()

    if pfr_s.empty:
        print(f"  No data for {season}. Available: {sorted(pfr_def['season'].unique().tolist())}")
        return pd.DataFrame()

    # Filter to front-7 positions
    dl_positions = ["DE","DT","NT","EDGE","OLB","ILB","MLB","LB","DL","IDL"]
    if "position" in pfr_s.columns:
        pfr_s = pfr_s[pfr_s["position"].isin(dl_positions)].copy()

    agg_dict = {}
    for col, agg in [
        ("def_sacks",              "sum"),
        ("def_pressures",          "sum"),
        ("def_times_hurried",      "sum"),  # note: col name varies
        ("def_times_hitqb",        "sum"),
        ("def_tackles_combined",   "sum"),
        ("def_missed_tackles",     "sum"),
        ("def_targets",            "sum"),
        ("def_completions_allowed","sum"),
        ("def_yards_allowed",      "sum"),
        ("def_passer_rating_allowed","mean"),
        ("def_ints",               "sum"),
    ]:
        if col in pfr_s.columns:
            agg_dict[col] = (col, agg)

    name_col = "pfr_player_name" if "pfr_player_name" in pfr_s.columns else "player"
    team_col = "team"
    pos_col  = "position" if "position" in pfr_s.columns else None

    gb = [name_col, team_col] + ([pos_col] if pos_col else [])
    df = pfr_s.groupby(gb).agg(**agg_dict).reset_index()
    df = df.rename(columns={name_col: "name"})

    # Snap count proxy: use def_tackles_combined as volume filter
    vol_col = "def_tackles_combined" if "def_tackles_combined" in df.columns else None
    if vol_col:
        df = df[df[vol_col] >= 15].copy()   # at least 15 tackles = meaningful role

    # Derived
    if "def_sacks" in df.columns and "def_tackles_combined" in df.columns:
        df["sack_rate"]    = df["def_sacks"] / df["def_tackles_combined"].clip(lower=1) * 100
    if "def_missed_tackles" in df.columns and "def_tackles_combined" in df.columns:
        total_att = df["def_tackles_combined"] + df["def_missed_tackles"]
        df["missed_tkl_pct"] = df["def_missed_tackles"] / total_att.clip(lower=1) * 100
    if "def_pressures" in df.columns and "def_tackles_combined" in df.columns:
        df["pressure_rate"] = df["def_pressures"] / df["def_tackles_combined"].clip(lower=1) * 100

    # PFR name format: "Last, First"
    df["name_last"] = df["name"].str.split(",").str[0].str.strip()

    # Composite
    score_map = {
        "s_sacks":    ("def_sacks",          False, 0.30),
        "s_press":    ("def_pressures",       False, 0.25),
        "s_hits":     ("def_times_hitqb",     False, 0.15),
        "s_tackle":   ("def_tackles_combined",False, 0.15),
        "s_miss":     ("missed_tkl_pct",      True,  0.10) if "missed_tkl_pct" in df.columns else None,
        "s_int":      ("def_ints",            False, 0.05),
    }
    score_map = {k: v for k, v in score_map.items() if v}
    total_w = sum(v[2] for v in score_map.values())

    for sname, (col, inv, w) in score_map.items():
        if col in df.columns and df[col].notna().any():
            df[sname] = norm(df[col], invert=inv)
        else:
            df[sname] = 50.0

    df["composite"] = sum(df[s] * (w / total_w) for s, (c, i, w) in score_map.items())
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    has_press = "def_pressures" in df.columns and df["def_pressures"].notna().any()
    has_hits  = "def_times_hitqb" in df.columns and df["def_times_hitqb"].notna().any()
    has_miss  = "missed_tkl_pct" in df.columns and df["missed_tkl_pct"].notna().any()

    hdr = f"  {'#':<3} {'Player':<26} {'Pos':<5} {'Team':<5} {'Sacks':>6} {'Tkl':>5}"
    if has_press: hdr += f" {'Press':>6}"
    if has_hits:  hdr += f" {'QBHit':>6}"
    if has_miss:  hdr += f" {'Miss%':>6}"
    hdr += f" {'SCORE':>6}"
    print(hdr)
    print(f"  {'-'*80}")

    shown = 0
    for _, r in df.iterrows():
        if shown >= 50: break
        sacks = f"{r['def_sacks']:.1f}"          if pd.notna(r.get('def_sacks'))          else "—"
        tkl   = f"{int(r['def_tackles_combined'])}" if pd.notna(r.get('def_tackles_combined')) else "—"
        press = f"{int(r['def_pressures'])}"      if has_press and pd.notna(r.get('def_pressures')) else ""
        hits  = f"{int(r['def_times_hitqb'])}"    if has_hits  and pd.notna(r.get('def_times_hitqb'))  else ""
        miss  = f"{r['missed_tkl_pct']:.1f}%"    if has_miss  and pd.notna(r.get('missed_tkl_pct'))  else ""
        pos   = str(r.get(pos_col, ""))[:4]       if pos_col else ""
        print(f"  {int(r['rank']):<3} {r['name_last']:<26} {pos:<5} {r['team']:<5} "
              f"{sacks:>6} {tkl:>5}"
              + (f" {press:>6}" if has_press else "")
              + (f" {hits:>6}"  if has_hits  else "")
              + (f" {miss:>6}"  if has_miss  else "")
              + f" {r['composite']:>6.1f}")
        shown += 1

    out = PROC / f"dl_rankings_{season}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}  ({len(df)} qualified players)")
    return df


# ──────────────────────────────────────────────────────────────────
# SPECIAL TEAMS  (team-level from schedules + player_stats K/P)
# ──────────────────────────────────────────────────────────────────
def rank_st(season):
    print(f"\n{'='*85}")
    print(f"  {season} SPECIAL TEAMS RANKINGS  (team-level)")
    print(f"  Metrics: FG%, XP%, punt avg, kick return avg, punt return avg")
    print(f"{'='*85}")

    # Player stats for K/P
    pstats = load("player_stats")
    st_data = {}

    if not pstats.empty:
        ps_s = pstats[pstats["season"] == season].copy()
        if "season_type" in ps_s.columns:
            reg = [v for v in ps_s["season_type"].unique() if "reg" in str(v).lower()]
            if reg: ps_s = ps_s[ps_s["season_type"].isin(reg)]

        # Kickers
        k_cols = [c for c in ps_s.columns if any(x in c for x in ["fg","kick","xp","extra_point","pat"])]
        if k_cols:
            k = ps_s[ps_s["position"] == "K"].copy() if "position" in ps_s.columns else pd.DataFrame()
            if not k.empty:
                name_c = "player_display_name" if "player_display_name" in k.columns else "player_name"
                team_c = "recent_team" if "recent_team" in k.columns else "team"
                k_agg_dict = {}
                for col, agg in [
                    ("fg_made",          "sum"), ("fg_att",         "sum"),
                    ("fg_made_0_19",     "sum"), ("fg_made_20_29",  "sum"),
                    ("fg_made_30_39",    "sum"), ("fg_made_40_49",  "sum"),
                    ("fg_made_50_",      "sum"), ("fg_att_50_",     "sum"),
                    ("pat_made",         "sum"), ("pat_att",        "sum"),
                ]:
                    if col in k.columns:
                        k_agg_dict[col] = (col, agg)
                if k_agg_dict:
                    k_agg = k.groupby([name_c, team_c]).agg(**k_agg_dict).reset_index()
                    k_agg = k_agg.rename(columns={name_c: "name", team_c: "team"})
                    if "fg_made" in k_agg.columns and "fg_att" in k_agg.columns:
                        k_agg["fg_pct"] = k_agg["fg_made"] / k_agg["fg_att"].clip(lower=1) * 100
                    if "fg_made_50_" in k_agg.columns and "fg_att_50_" in k_agg.columns:
                        k_agg["fg50_pct"] = k_agg["fg_made_50_"] / k_agg["fg_att_50_"].clip(lower=1) * 100
                    if "pat_made" in k_agg.columns and "pat_att" in k_agg.columns:
                        k_agg["xp_pct"] = k_agg["pat_made"] / k_agg["pat_att"].clip(lower=1) * 100
                    st_data["kickers"] = k_agg

        # Punters
        p_cols = [c for c in ps_s.columns if "punt" in c.lower()]
        if p_cols:
            p = ps_s[ps_s["position"] == "P"].copy() if "position" in ps_s.columns else pd.DataFrame()
            if not p.empty:
                name_c = "player_display_name" if "player_display_name" in p.columns else "player_name"
                team_c = "recent_team" if "recent_team" in p.columns else "team"
                p_agg_dict = {}
                for col, agg in [
                    ("punts",            "sum"), ("punting_yards",  "sum"),
                    ("punting_net_yds",  "sum"), ("punting_inside_20","sum"),
                    ("punting_touchbacks","sum"),
                ]:
                    if col in p.columns:
                        p_agg_dict[col] = (col, agg)
                if p_agg_dict:
                    p_agg = p.groupby([name_c, team_c]).agg(**p_agg_dict).reset_index()
                    p_agg = p_agg.rename(columns={name_c: "name", team_c: "team"})
                    if "punting_yards" in p_agg.columns and "punts" in p_agg.columns:
                        p_agg["gross_avg"] = p_agg["punting_yards"] / p_agg["punts"].clip(lower=1)
                    if "punting_net_yds" in p_agg.columns and "punts" in p_agg.columns:
                        p_agg["net_avg"] = p_agg["punting_net_yds"] / p_agg["punts"].clip(lower=1)
                    if "punting_inside_20" in p_agg.columns and "punts" in p_agg.columns:
                        p_agg["inside_20_pct"] = p_agg["punting_inside_20"] / p_agg["punts"].clip(lower=1) * 100
                    st_data["punters"] = p_agg

    # ── KICKER RANKINGS ───────────────────────────────────────────────
    if "kickers" in st_data:
        k_df = st_data["kickers"]
        k_df = k_df[k_df.get("fg_att", pd.Series(0)) >= 10].copy() if "fg_att" in k_df.columns else k_df
        score_map = {
            "s_fg":    ("fg_pct",   False, 0.45),
            "s_xp":    ("xp_pct",   False, 0.25),
            "s_fg50":  ("fg50_pct", False, 0.30) if "fg50_pct" in k_df.columns else None,
        }
        score_map = {k: v for k, v in score_map.items() if v}
        total_w = sum(v[2] for v in score_map.values())
        for sname, (col, inv, w) in score_map.items():
            if col in k_df.columns and k_df[col].notna().any():
                k_df[sname] = norm(k_df[col], invert=inv)
            else:
                k_df[sname] = 50.0
        k_df["composite"] = sum(k_df[s] * (w / total_w) for s, (c, i, w) in score_map.items())
        k_df = k_df.sort_values("composite", ascending=False).reset_index(drop=True)
        k_df["rank"] = k_df.index + 1

        print(f"\n  KICKERS")
        print(f"  {'#':<3} {'Kicker':<24} {'Team':<5} {'FG':<6} {'FG%':>6} {'XP%':>6} {'FG50%':>7} {'SCORE':>6}")
        print(f"  {'-'*65}")
        for _, r in k_df.iterrows():
            fg_rec = f"{int(r.get('fg_made',0))}/{int(r.get('fg_att',0))}" if pd.notna(r.get('fg_made')) else "—"
            fg_pct = f"{r['fg_pct']:.1f}%"    if pd.notna(r.get('fg_pct'))    else "—"
            xp_pct = f"{r['xp_pct']:.1f}%"    if pd.notna(r.get('xp_pct'))    else "—"
            fg50   = f"{r['fg50_pct']:.1f}%"  if pd.notna(r.get('fg50_pct'))  else "—"
            print(f"  {int(r['rank']):<3} {r['name']:<24} {r['team']:<5} {fg_rec:<6} {fg_pct:>6} {xp_pct:>6} {fg50:>7} {r['composite']:>6.1f}")

        k_df.to_csv(PROC / f"kicker_rankings_{season}.csv", index=False)

    # ── PUNTER RANKINGS ───────────────────────────────────────────────
    if "punters" in st_data:
        p_df = st_data["punters"]
        p_df = p_df[p_df.get("punts", pd.Series(0)) >= 20].copy() if "punts" in p_df.columns else p_df
        score_map = {
            "s_net":   ("net_avg",       False, 0.40),
            "s_gross": ("gross_avg",     False, 0.25),
            "s_i20":   ("inside_20_pct", False, 0.35) if "inside_20_pct" in p_df.columns else None,
        }
        score_map = {k: v for k, v in score_map.items() if v}
        total_w = sum(v[2] for v in score_map.values())
        for sname, (col, inv, w) in score_map.items():
            if col in p_df.columns and p_df[col].notna().any():
                p_df[sname] = norm(p_df[col], invert=inv)
            else:
                p_df[sname] = 50.0
        p_df["composite"] = sum(p_df[s] * (w / total_w) for s, (c, i, w) in score_map.items())
        p_df = p_df.sort_values("composite", ascending=False).reset_index(drop=True)
        p_df["rank"] = p_df.index + 1

        print(f"\n  PUNTERS")
        has_i20 = "inside_20_pct" in p_df.columns and p_df["inside_20_pct"].notna().any()
        hdr = f"  {'#':<3} {'Punter':<24} {'Team':<5} {'Punts':<6} {'GrossAvg':>9} {'NetAvg':>7}"
        if has_i20: hdr += f" {'Inside20':>9}"
        hdr += f" {'SCORE':>6}"
        print(hdr)
        print(f"  {'-'*70}")
        for _, r in p_df.iterrows():
            punts  = f"{int(r.get('punts',0))}"       if pd.notna(r.get('punts'))      else "—"
            gross  = f"{r['gross_avg']:.1f}"           if pd.notna(r.get('gross_avg'))  else "—"
            net    = f"{r['net_avg']:.1f}"             if pd.notna(r.get('net_avg'))    else "—"
            i20    = f"{r['inside_20_pct']:.1f}%"     if has_i20 and pd.notna(r.get('inside_20_pct')) else ""
            line = f"  {int(r['rank']):<3} {r['name']:<24} {r['team']:<5} {punts:<6} {gross:>9} {net:>7}"
            if has_i20: line += f" {i20:>9}"
            line += f" {r['composite']:>6.1f}"
            print(line)

        p_df.to_csv(PROC / f"punter_rankings_{season}.csv", index=False)

    if not st_data:
        print("  No K/P data found. Check player_stats positions.")

    return st_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",      type=int, default=2025)
    parser.add_argument("--pos",         type=str, default="ALL",
                        choices=["OL","DL","ST","ALL"])
    parser.add_argument("--min-snaps",   type=int, default=150)
    args = parser.parse_args()

    if args.pos in ("OL","ALL"):
        rank_ol(args.season)
    if args.pos in ("DL","ALL"):
        rank_dl(args.season, args.min_snaps)
    if args.pos in ("ST","ALL"):
        rank_st(args.season)
