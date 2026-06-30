"""
update_styles_2025.py
=====================
Injects 2025 OL, DL, and ST rankings into team_styles.parquet.

Adds / updates these columns for season=2025:
  OL  -> ol_rank, sack_rate_allowed, stuff_rate, ypc_allowed
         leaky_under_pressure  (updated from real OL data, not situational proxy)
         elite_ol              (new: top-8 OL)
  DL  -> dl_pressure_score, top_pass_rusher, top_pass_rusher_sacks
         elite_pass_rush       (new: top-10 DL pressure composite)
         blitz_heavy_def       (updated: now blends DL sacks + team sack rate)
  ST  -> kicker_score, punter_score, kr_score
         elite_kicker          (new: top-10 kicker)
         elite_punter          (new: top-10 punter)
         elite_returner        (new: top-8 KR)

predict.py already uses:
  leaky_under_pressure  -> -3 pts penalty on off_quality
  elite_rz_offense      -> +2 pts boost
  elite_2min            -> +1.5 pts boost
  blitz_heavy_def       -> matchup clash adjustments

New columns used in predict.py after this update:
  elite_pass_rush       -> +2 pts def_quality boost (affects scoring suppression)
  elite_kicker          -> ±1.5 pts adjustment in close games (spread < 4)
  elite_punter          -> +0.5 pts field position boost to def_quality

Run:
    python update_styles_2025.py
    python update_styles_2025.py --dry-run   # preview without saving
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path(__file__).parent / "data" / "processed"
RAW  = Path(__file__).parent / "data" / "raw"


def load(path, name):
    if path.exists():
        df = pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
        print(f"  loaded {name} ({len(df)} rows)")
        return df
    print(f"  MISSING: {name} — run ol_dl_st_rankings.py first")
    return pd.DataFrame()


def update_styles(season=2025, dry_run=False):
    print(f"\n{'='*65}")
    print(f"  Updating team_styles for season {season}")
    print(f"{'='*65}\n")

    # Load base styles
    styles_path = PROC / "team_styles.parquet"
    if not styles_path.exists():
        print("ERROR: team_styles.parquet not found. Run: python run_engine.py first")
        return

    styles = pd.read_parquet(styles_path)
    print(f"Loaded team_styles: {len(styles)} rows, seasons: {sorted(styles['season'].unique().tolist())}")

    # Check if 2025 exists
    if season not in styles["season"].values:
        print(f"\nWARN: season {season} not in team_styles. Adding rows from {season-1}...")
        base = styles[styles["season"] == season - 1].copy()
        base["season"] = season
        styles = pd.concat([styles, base], ignore_index=True)

    mask = styles["season"] == season
    print(f"  {mask.sum()} rows for season {season}\n")

    # ── OL DATA ───────────────────────────────────────────────────
    ol = load(PROC / f"ol_rankings_{season}.csv", f"OL rankings {season}")
    if not ol.empty:
        print("Applying OL data...")
        ol_map = ol.set_index("team")

        # OL rank (1=best)
        styles.loc[mask, "ol_rank"] = styles.loc[mask, "team"].map(
            ol_map.get("rank", pd.Series(dtype=float)))

        # Sack rate allowed from OL (overrides situational_stats proxy)
        if "sack_rate_allowed" in ol_map.columns:
            styles.loc[mask, "sack_rate_allowed_ol"] = styles.loc[mask, "team"].map(
                ol_map["sack_rate_allowed"])

        # Stuff rate and YPC
        for col in ["stuff_rate", "ypc"]:
            if col in ol_map.columns:
                styles.loc[mask, col] = styles.loc[mask, "team"].map(ol_map[col])

        # leaky_under_pressure: bottom 8 OL by rank OR sack rate >= 9%
        def is_leaky(row):
            if row.get("ol_rank", 16) >= 25:
                return True
            if row.get("sack_rate_allowed_ol", 0) >= 9.0:
                return True
            return False

        styles.loc[mask, "leaky_under_pressure"] = styles[mask].apply(is_leaky, axis=1)

        # elite_ol: top 8 OL
        styles.loc[mask, "elite_ol"] = styles.loc[mask, "ol_rank"].fillna(32) <= 8

        # Print leaky flags
        leaky_teams = styles.loc[mask & styles["leaky_under_pressure"], "team"].tolist()
        elite_ol = styles.loc[mask & styles["elite_ol"], "team"].tolist()
        print(f"  leaky_under_pressure: {sorted(leaky_teams)}")
        print(f"  elite_ol:             {sorted(elite_ol)}")

    # ── DL DATA ───────────────────────────────────────────────────
    dl = load(PROC / f"dl_rankings_{season}.csv", f"DL rankings {season}")
    if not dl.empty:
        print("\nApplying DL data...")

        # Team-level: best pass rusher score + top sack producer
        dl_25 = dl[dl["season"] == season].copy() if "season" in dl.columns else dl.copy()

        # Get top DL player per team
        dl_top = (dl_25.sort_values("composite", ascending=False)
                       .drop_duplicates("team")
                       [["team","name_last","def_sacks","def_pressures","composite"]]
                       .rename(columns={
                           "name_last":     "top_pass_rusher",
                           "def_sacks":     "top_pass_rusher_sacks",
                           "def_pressures": "top_pass_rusher_pressures",
                           "composite":     "dl_pressure_score",
                       }))

        # Team DL composite = mean of top-3 pass rushers
        dl_team = (dl_25.sort_values("composite", ascending=False)
                        .groupby("team")
                        .head(3)
                        .groupby("team")["composite"]
                        .mean()
                        .reset_index()
                        .rename(columns={"composite": "dl_team_score"}))

        dl_map  = dl_top.set_index("team")
        dl_tmap = dl_team.set_index("team")

        for col in ["top_pass_rusher","top_pass_rusher_sacks","top_pass_rusher_pressures","dl_pressure_score"]:
            if col in dl_map.columns:
                styles.loc[mask, col] = styles.loc[mask, "team"].map(dl_map[col])

        styles.loc[mask, "dl_team_score"] = styles.loc[mask, "team"].map(dl_tmap["dl_team_score"])

        # elite_pass_rush: top 10 DL teams by team composite
        dl_team_ranked = dl_team.sort_values("dl_team_score", ascending=False).reset_index(drop=True)
        dl_team_ranked["dl_team_rank"] = dl_team_ranked.index + 1
        dl_rank_map = dl_team_ranked.set_index("team")["dl_team_rank"]
        styles.loc[mask, "dl_team_rank"] = styles.loc[mask, "team"].map(dl_rank_map)
        styles.loc[mask, "elite_pass_rush"] = styles.loc[mask, "dl_team_rank"].fillna(32) <= 10

        # Update blitz_heavy_def: fire if top pass rusher has 12+ sacks OR team sack rate >= 9%
        def is_blitz(row):
            if row.get("top_pass_rusher_sacks", 0) >= 12:
                return True
            if row.get("sack_rate_allowed_ol", 99) <= 4.5:  # very hard to sack this team
                return False
            if row.get("dl_team_score", 50) >= 60:
                return True
            return row.get("blitz_heavy_def", False)

        styles.loc[mask, "blitz_heavy_def"] = styles[mask].apply(is_blitz, axis=1)

        elite_pr = styles.loc[mask & styles["elite_pass_rush"], "team"].tolist()
        blitz = styles.loc[mask & styles["blitz_heavy_def"], "team"].tolist()
        print(f"  elite_pass_rush: {sorted(elite_pr)}")
        print(f"  blitz_heavy_def: {sorted(blitz)}")

    # ── KICKER DATA ───────────────────────────────────────────────
    kickers = load(PROC / f"kicker_rankings_{season}.csv", f"Kicker rankings {season}")
    if not kickers.empty:
        print("\nApplying kicker data...")

        # Map by team — take highest-scoring kicker per team
        k_map = (kickers.sort_values("composite", ascending=False)
                        .drop_duplicates("team")
                        .set_index("team"))

        styles.loc[mask, "kicker_score"] = styles.loc[mask, "team"].map(
            k_map["composite"] if "composite" in k_map.columns else pd.Series(dtype=float))
        styles.loc[mask, "kicker_fg_pct"] = styles.loc[mask, "team"].map(
            k_map["fg_pct"] if "fg_pct" in k_map.columns else pd.Series(dtype=float))
        styles.loc[mask, "kicker_fg50_pct"] = styles.loc[mask, "team"].map(
            k_map["fg50_pct"] if "fg50_pct" in k_map.columns else pd.Series(dtype=float))

        # elite_kicker: top 10 (score >= 70)
        styles.loc[mask, "elite_kicker"] = styles.loc[mask, "kicker_score"].fillna(50) >= 70

        elite_k = styles.loc[mask & styles["elite_kicker"], "team"].tolist()
        print(f"  elite_kicker: {sorted(elite_k)}")

        # poor_kicker: bottom 8 (score < 45) — negative flag for close games
        styles.loc[mask, "poor_kicker"] = styles.loc[mask, "kicker_score"].fillna(50) < 45
        poor_k = styles.loc[mask & styles["poor_kicker"], "team"].tolist()
        print(f"  poor_kicker:  {sorted(poor_k)}")

    # ── PUNTER DATA ───────────────────────────────────────────────
    punters = load(PROC / f"punter_rankings_{season}.csv", f"Punter rankings {season}")
    if not punters.empty:
        print("\nApplying punter data...")

        p_map = (punters.sort_values("composite", ascending=False)
                        .drop_duplicates("team")
                        .set_index("team"))

        styles.loc[mask, "punter_score"] = styles.loc[mask, "team"].map(
            p_map["composite"] if "composite" in p_map.columns else pd.Series(dtype=float))
        styles.loc[mask, "punter_net_avg"] = styles.loc[mask, "team"].map(
            p_map["net_avg"] if "net_avg" in p_map.columns else pd.Series(dtype=float))
        styles.loc[mask, "punter_inside_20_pct"] = styles.loc[mask, "team"].map(
            p_map["inside_20_pct"] if "inside_20_pct" in p_map.columns else pd.Series(dtype=float))

        # elite_punter: top 8 (score >= 65)
        styles.loc[mask, "elite_punter"] = styles.loc[mask, "punter_score"].fillna(50) >= 65

        elite_p = styles.loc[mask & styles["elite_punter"], "team"].tolist()
        print(f"  elite_punter: {sorted(elite_p)}")

    # ── KR DATA ───────────────────────────────────────────────────
    kr = load(PROC / f"kickoff_return_rankings_{season}.csv", f"KR rankings {season}")
    if not kr.empty:
        print("\nApplying KR data...")
        kr_map = (kr.sort_values("composite", ascending=False)
                    .drop_duplicates("team")
                    .set_index("team"))
        styles.loc[mask, "kr_score"] = styles.loc[mask, "team"].map(
            kr_map["composite"] if "composite" in kr_map.columns else pd.Series(dtype=float))
        styles.loc[mask, "elite_returner"] = styles.loc[mask, "kr_score"].fillna(50) >= 75
        elite_r = styles.loc[mask & styles["elite_returner"], "team"].tolist()
        print(f"  elite_returner: {sorted(elite_r)}")

    # ── SUMMARY ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  2025 TEAM FLAGS SUMMARY")
    print(f"{'='*65}")
    new_cols = ["team","ol_rank","leaky_under_pressure","elite_ol",
                "dl_team_rank","elite_pass_rush","blitz_heavy_def",
                "kicker_score","elite_kicker","poor_kicker",
                "punter_score","elite_punter"]
    avail = [c for c in new_cols if c in styles.columns]
    s25 = styles[styles["season"] == season][avail].sort_values("ol_rank").reset_index(drop=True)

    # Pretty print
    print(f"\n  {'Team':<6} {'OL#':>4} {'Leaky':>6} {'ElOL':>5} {'DL#':>4} {'ElPR':>5} {'Blitz':>6} "
          f"{'K-Scr':>6} {'ElK':>4} {'PoorK':>6} {'P-Scr':>6} {'ElP':>4}")
    print(f"  {'-'*82}")
    for _, r in s25.iterrows():
        ks = f"{r['kicker_score']:.0f}" if pd.notna(r.get('kicker_score')) else '?'
        ps = f"{r['punter_score']:.0f}" if pd.notna(r.get('punter_score')) else '?'
        print(f"  {str(r.get('team','')):<6} "
              f"{int(r['ol_rank']) if pd.notna(r.get('ol_rank')) else '?':>4} "
              f"{'Y' if r.get('leaky_under_pressure') else 'n':>6} "
              f"{'Y' if r.get('elite_ol') else 'n':>5} "
              f"{int(r['dl_team_rank']) if pd.notna(r.get('dl_team_rank')) else '?':>4} "
              f"{'Y' if r.get('elite_pass_rush') else 'n':>5} "
              f"{'Y' if r.get('blitz_heavy_def') else 'n':>6} "
              f"{ks:>6} "
              f"{'Y' if r.get('elite_kicker') else 'n':>4} "
              f"{'Y' if r.get('poor_kicker') else 'n':>6} "
              f"{ps:>6} "
              f"{'Y' if r.get('elite_punter') else 'n':>4}")

    if dry_run:
        print(f"\n  DRY RUN — no files saved")
        return styles

    # Save
    styles.to_parquet(styles_path, index=False)
    print(f"\n  Saved -> {styles_path}")

    # Also save a readable CSV for inspection
    csv_path = PROC / f"team_styles_{season}_flags.csv"
    s25.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")
    return styles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",  type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    update_styles(args.season, args.dry_run)
