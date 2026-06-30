"""
roster_update.py  —  2026 Roster Management System
====================================================
Handles all roster lifecycle events for the prediction engine.

STRUCTURE OF depth_charts.parquet:
  - Rows with season = 2021-2024: historical weekly depth charts
  - Rows with season = NaN + dt timestamp: LIVE nflverse snapshot
    The most recent dt per player = their current 2026 team

DATA SOURCES:
  - depth_charts (season=NaN rows): current team assignments (FA signings included)
  - rosters_weekly 2025 wk18: player metadata + historical quality context
  - injuries: weekly availability
  - nfl.import_seasonal_rosters([2025]): refresh of end-of-season snapshot

MODES:
  status     — show current roster state and team changes vs 2025
  offseason  — build 2026 pre-draft roster from live depth_charts
  preseason  — final 53-man roster (run after August cutdown)
  weekly     — refresh rosters + injuries for a specific week
  trade      — manually record a mid-season trade
  changes    — list all FA signings / team changes detected

Usage:
    python roster_update.py --mode status
    python roster_update.py --mode offseason
    python roster_update.py --mode weekly --season 2026 --week 1
    python roster_update.py --mode changes
    python roster_update.py --mode trade --player "Davante Adams" --from-team GB --to-team KC --season 2026 --week 8
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"

POS_GROUPS = {
    "QB":"QB","HB":"RB","FB":"RB","WR":"WR","TE":"TE",
    "LWR":"WR","RWR":"WR","SWR":"WR","SLOT":"WR",
    "LT":"OL","LG":"OL","C":"OL","RG":"OL","RT":"OL",
    "DE":"DL","DT":"DL","NT":"DL","LDE":"DL","RDE":"DL",
    "LOLB":"LB","ROLB":"LB","MLB":"LB","ILB":"LB","LB":"LB",
    "CB":"CB","FS":"S","SS":"S","DB":"S",
    "K":"K","P":"P","LS":"LS",
}


def load(name: str) -> pd.DataFrame:
    p = RAW / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def get_live_depth_charts() -> pd.DataFrame:
    """
    Extract the LIVE current roster snapshot from depth_charts.parquet.

    depth_charts has two types of rows:
      - season = 2021-2024 : historical weekly data (ignore for current roster)
      - season = NaN + dt  : live nflverse snapshot entries

    We take the most recent dt entry per player (gsis_id) from the
    NaN-season rows to get each player's current team.
    """
    dc = load("depth_charts")
    if dc.empty:
        return pd.DataFrame()

    # Live rows = season is NaN (offseason snapshot from nflverse)
    live = dc[dc["season"].isna()].copy()

    if live.empty:
        # Fallback: use most recent season's latest week
        print("  WARN: No live (season=NaN) rows in depth_charts")
        print("        Falling back to 2024 week 18 depth charts")
        latest_season = dc["season"].max()
        latest_week   = dc[dc["season"] == latest_season]["week"].max()
        live = dc[
            (dc["season"] == latest_season) &
            (dc["week"] == latest_week)
        ].copy()
        live["dt"] = str(datetime.now().date())

    # Among live rows, take the most recent dt per player
    live["dt"] = pd.to_datetime(live["dt"], errors="coerce", utc=True)
    live = (live.sort_values("dt", ascending=False)
                .drop_duplicates("gsis_id")
                .copy())

    # Standardize team column
    if "club_code" in live.columns and "team" not in live.columns:
        live = live.rename(columns={"club_code": "team"})

    # Build player_name from first + last name
    if "football_name" in live.columns:
        live["player_name"] = live["football_name"]
    elif "first_name" in live.columns and "last_name" in live.columns:
        live["player_name"] = (
            live["first_name"].fillna("") + " " +
            live["last_name"].fillna("")
        ).str.strip()

    # Map formation/position to standard group
    pos_col = next((c for c in ["pos_abb","formation","depth_position","position"]
                    if c in live.columns), None)
    if pos_col:
        live["position"] = live[pos_col].str.upper().map(POS_GROUPS).fillna(
            live[pos_col].str.upper().str[:2])

    return live[[c for c in ["gsis_id","team","player_name","position","dt",
                              "depth_team","pos_slot","pos_rank"]
                 if c in live.columns]].rename(columns={"gsis_id": "player_id"})


def get_2025_final_roster() -> pd.DataFrame:
    """Get end-of-2025 roster (week 18) as baseline for comparison."""
    rw = load("rosters_weekly")
    if rw.empty:
        return pd.DataFrame()
    wk18 = (rw[rw["season"] == 2025]
            .sort_values("week", ascending=False)
            .drop_duplicates("player_id")
            [["player_id","player_name","team","position","status"]]
            .rename(columns={"team":"team_2025"}))
    return wk18


def show_status():
    print("\n" + "="*65)
    print("  ROSTER DATA STATUS  —  " + datetime.now().strftime("%Y-%m-%d"))
    print("="*65)

    files = {
        "rosters_weekly":   ("Weekly player-team-status snapshots",
                             "season","week"),
        "rosters_seasonal": ("Season-level player records",
                             "season", None),
        "depth_charts":     ("Historical + live depth order",
                             "season", "dt"),
        "injuries":         ("Weekly injury reports",
                             "season","week"),
    }

    for name, (desc, scol, tcol) in files.items():
        df = load(name)
        if df.empty:
            print(f"\n  [{name}]  NOT FOUND"); continue

        seasons = sorted(df[scol].dropna().unique().tolist()) \
            if scol and scol in df.columns else []
        print(f"\n  [{name}]")
        print(f"    Rows:    {len(df):,}")
        print(f"    Seasons: {seasons}")
        if tcol and tcol in df.columns:
            rng = df[tcol].dropna()
            if len(rng):
                print(f"    Range:   {str(rng.min())[:10]} → {str(rng.max())[:10]}")
        print(f"    Purpose: {desc}")

    # ── Live snapshot summary ──────────────────────────────────────
    print("\n" + "="*65)
    print("  2026 SEASON READINESS")
    print("="*65)

    live = get_live_depth_charts()
    base = get_2025_final_roster()

    if live.empty:
        print("\n  ERROR: Could not extract live roster from depth_charts")
        return

    live_dt_max = load("depth_charts")["dt"].max()
    print(f"\n  Live snapshot date:  {str(live_dt_max)[:10]}")
    print(f"  Players in snapshot: {len(live):,}")
    print(f"  Players in 2025 end: {len(base):,}")

    # Team changes: merge on player_id, compare teams
    if not base.empty:
        chg = base.merge(
            live[["player_id","team","player_name"]].rename(
                columns={"team":"team_2026","player_name":"name_2026"}),
            on="player_id", how="inner"
        )
        changed = chg[chg["team_2025"] != chg["team_2026"]].copy()
        new_players = live[~live["player_id"].isin(base["player_id"])]

        print(f"  Confirmed team changes: {len(changed):,}")
        print(f"  New players (FA/rookie): {len(new_players):,}")

        # Show key position changes
        key_pos = ["QB","WR","RB","TE"]
        key_changes = changed[changed["position"].isin(key_pos)].copy()
        if not key_changes.empty:
            print(f"\n  KEY SKILL PLAYER MOVES ({len(key_changes)} total):")
            print(f"  {'Player':<26} {'Pos':<4} {'From':<5} → {'To'}")
            print(f"  {'-'*50}")
            for _, r in key_changes.sort_values("position").iterrows():
                name = str(r.get("player_name", r.get("name_2026","?")))[:24]
                print(f"  {name:<26} {str(r.get('position','?')):<4} "
                      f"{r['team_2025']:<5} → {r['team_2026']}")

    # Team-by-team QB situation
    print(f"\n  QB SITUATION BY TEAM (from live depth_charts):")
    qbs = live[live["position"] == "QB"][["player_id","team","player_name"]].copy()
    if not qbs.empty:
        for team in sorted(qbs["team"].dropna().unique()):
            team_qbs = [str(n) for n in qbs[qbs["team"] == team]["player_name"].tolist() if pd.notna(n)]
            print(f"    {team:<4}: {', '.join(team_qbs[:2]) if team_qbs else 'unknown'}")


def show_changes():
    """Detailed report of all FA signings and team changes."""
    print("\n" + "="*65)
    print("  ALL DETECTED ROSTER CHANGES  (2025 end → 2026 current)")
    print("="*65)

    live = get_live_depth_charts()
    base = get_2025_final_roster()

    if live.empty or base.empty:
        print("  Insufficient data"); return

    merged = base.merge(
        live[["player_id","team","position"]].rename(columns={"team":"team_2026"}),
        on="player_id", how="outer",
        suffixes=("_2025","_2026")
    )

    # Clean up position column
    if "position_2025" in merged.columns and "position_2026" in merged.columns:
        merged["pos"] = merged["position_2025"].fillna(merged["position_2026"])
    elif "position" in merged.columns:
        merged["pos"] = merged["position"]

    changed = merged[
        merged["team_2025"].notna() &
        merged["team_2026"].notna() &
        (merged["team_2025"] != merged["team_2026"])
    ].copy()

    for pos in ["QB","WR","RB","TE","OL","DL","LB","CB","S","K"]:
        pos_df = changed[changed["pos"] == pos]
        if pos_df.empty: continue
        print(f"\n  {pos} ({len(pos_df)}):")
        for _, r in pos_df.iterrows():
            name = str(r.get("player_name","?"))[:26]
            print(f"    {name:<28} {r['team_2025']} → {r['team_2026']}")


def build_2026_roster(refresh: bool = True):
    """
    Build synthetic 2026 week-0 roster rows and add to rosters_weekly.
    Uses live depth_charts snapshot as source of team assignments.
    """
    print("\n" + "="*65)
    print("  BUILDING 2026 PRE-SEASON ROSTER")
    print("="*65)

    if refresh:
        print("\n  Step 1: Refreshing depth_charts from nflverse...")
        try:
            import nfl_data_py as nfl
            # Fetch current + recent seasons for depth charts
            dc_new = nfl.import_depth_charts([2024, 2025])
            if not dc_new.empty:
                # Merge with existing to preserve any live snapshot rows
                dc_old = load("depth_charts")
                if not dc_old.empty:
                    combined = pd.concat([dc_old, dc_new], ignore_index=True)
                    combined = combined.drop_duplicates()
                else:
                    combined = dc_new
                combined.to_parquet(RAW / "depth_charts.parquet", index=False)
                print(f"    Saved {len(combined):,} total rows")
        except Exception as e:
            print(f"    Refresh failed: {e} — using cached data")

    print("\n  Step 2: Extracting live 2026 team assignments...")
    live = get_live_depth_charts()
    if live.empty:
        print("  ERROR: No live roster data available"); return

    print(f"    {len(live):,} players in current depth charts")

    # Merge with 2025 player metadata for names/IDs
    rw25 = load("rosters_weekly")
    if not rw25.empty:
        meta = (rw25[rw25["season"] == 2025]
                .sort_values("week", ascending=False)
                .drop_duplicates("player_id")
                [["player_id","player_name","position","birth_date",
                  "height","weight","years_exp","college"]])
        live = live.merge(meta, on="player_id", how="left",
                          suffixes=("_dc","_rw"))
        # Prefer depth_charts name/position, fall back to rosters
        for col in ["player_name","position"]:
            dc_col, rw_col = f"{col}_dc", f"{col}_rw"
            if dc_col in live.columns and rw_col in live.columns:
                live[col] = live[dc_col].fillna(live[rw_col])
                live = live.drop(columns=[dc_col, rw_col])

    # Build week-0 rows
    roster_2026 = pd.DataFrame({
        "season":      2026,
        "week":        0,
        "team":        live["team"].astype(str),
        "player_id":   live["player_id"].astype(str),
        "player_name": live.get("player_name", pd.Series("Unknown", index=live.index)).fillna("Unknown").astype(str),
        "position":    live.get("position", pd.Series("UNK", index=live.index)).fillna("UNK").astype(str),
        "status":      "ACT",
        "depth_chart_position": pd.to_numeric(
            live.get("depth_team", pd.Series(1, index=live.index)), errors="coerce"
        ).fillna(1).astype(int),
    })

    # Remove any existing 2026 week-0 rows and add fresh ones
    rw = load("rosters_weekly")
    if not rw.empty:
        rw = rw[~((rw["season"] == 2026) & (rw["week"] == 0))]
        combined = pd.concat([rw, roster_2026], ignore_index=True)
    else:
        combined = roster_2026

    # ── Sanitize all columns before writing parquet ────────────────
    # PyArrow requires uniform types per column. Mixed object columns
    # (e.g. depth_chart_position with ints AND strings) crash on write.
    for col in combined.columns:
        if combined[col].dtype == object:
            # Try numeric first; if it fails, cast everything to string
            numeric = pd.to_numeric(combined[col], errors="coerce")
            if numeric.notna().mean() > 0.8:   # mostly numeric
                combined[col] = numeric
            else:
                combined[col] = combined[col].astype(str).replace("nan", "")

    combined.to_parquet(RAW / "rosters_weekly.parquet", index=False)
    print(f"\n  Saved {len(roster_2026):,} 2026 roster rows")

    # Summary
    by_team = roster_2026.groupby("team").size()
    by_pos  = roster_2026.groupby("position").size().sort_values(ascending=False)
    print(f"\n  By position: {dict(by_pos.head(8))}")
    print(f"  Teams with data: {len(by_team)}/32")
    teams_short = by_team[by_team < 30]
    if not teams_short.empty:
        print(f"  Teams with thin rosters: {dict(teams_short)}")

    print(f"\n  Next steps:")
    print(f"    1. python run_engine.py --seasons 2025 2026 (after draft in April)")
    print(f"    2. python roster_update.py --mode offseason (weekly until August)")
    print(f"    3. python roster_update.py --mode preseason (after Aug 27 cutdowns)")

    return roster_2026


def weekly_update(season: int, week: int):
    """Refresh rosters + injuries for a specific in-season week."""
    print(f"\n{'='*65}")
    print(f"  WEEKLY ROSTER UPDATE — Season {season} Week {week}")
    print(f"  Run this every Tuesday after Monday Night Football")
    print(f"{'='*65}")

    try:
        import nfl_data_py as nfl

        # 1. Weekly rosters
        print(f"\n  [1/3] Refreshing weekly rosters...")
        new_rw = nfl.import_weekly_rosters([season])
        if not new_rw.empty:
            if "week" in new_rw.columns:
                new_wk = new_rw[new_rw["week"] == week]
            else:
                new_wk = new_rw
            rw = load("rosters_weekly")
            if not rw.empty:
                rw = rw[~((rw["season"] == season) & (rw["week"] == week))]
                combined = pd.concat([rw, new_wk], ignore_index=True)
            else:
                combined = new_wk
            combined.to_parquet(RAW / "rosters_weekly.parquet", index=False)
            print(f"    {len(new_wk):,} rows for week {week}")

        # 2. Injuries
        print(f"\n  [2/3] Refreshing injuries...")
        new_inj = nfl.import_injuries([season])
        if not new_inj.empty:
            inj_wk = new_inj[new_inj["week"] == week] \
                if "week" in new_inj.columns else new_inj
            inj = load("injuries")
            if not inj.empty:
                inj = inj[~((inj["season"] == season) & (inj["week"] == week))]
                combined_inj = pd.concat([inj, inj_wk], ignore_index=True)
            else:
                combined_inj = inj_wk
            combined_inj.to_parquet(RAW / "injuries.parquet", index=False)

            # Print key injuries
            key = inj_wk[inj_wk.get("report_status",
                         pd.Series()).isin(["Questionable","Doubtful","Out","IR"])] \
                if "report_status" in inj_wk.columns else pd.DataFrame()
            if not key.empty:
                print(f"\n  Key injury updates (week {week}):")
                for _, r in key.sort_values("position").head(20).iterrows():
                    name = str(r.get("full_name", r.get("player_name","?")))
                    print(f"    {name:<26} {str(r.get('team','?')):<4} "
                          f"{str(r.get('position','?')):<4} "
                          f"→ {r.get('report_status','?')}")

        # 3. Depth charts
        print(f"\n  [3/3] Refreshing depth charts...")
        new_dc = nfl.import_depth_charts([season])
        if not new_dc.empty:
            new_dc.to_parquet(RAW / "depth_charts.parquet", index=False)
            print(f"    {len(new_dc):,} rows")

        print(f"\n  Done. Rebuild and pick:")
        print(f"    python run_engine.py --season {season} --week {week} --skip-styles")
        print(f"    python weekly_picks.py --season {season} --week {week}")

    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Run this on your local machine with nfl_data_py installed")


def handle_trade(player: str, from_team: str, to_team: str,
                 season: int, week: int):
    """Manually record a mid-season trade."""
    print(f"\n  TRADE: {player}  {from_team} → {to_team}  from Week {week}")

    rw = load("rosters_weekly")
    if rw.empty:
        print("  rosters_weekly.parquet not found"); return

    name_mask = rw["player_name"].str.lower().str.contains(
        player.lower(), na=False)
    sea_mask  = rw["season"] == season
    matches   = rw[name_mask & sea_mask]

    if matches.empty:
        print(f"  '{player}' not found in {season} rosters")
        # Try partial match
        partial = rw[rw["player_name"].str.lower().str.contains(
            player.split()[0].lower(), na=False) & sea_mask]
        if not partial.empty:
            print("  Possible matches:")
            for nm in partial["player_name"].unique()[:5]:
                print(f"    {nm}")
        return

    pid  = matches["player_id"].iloc[0]
    name = matches["player_name"].iloc[0]

    # Update team from trade week onward
    update_mask = ((rw["player_id"] == pid) &
                   (rw["season"] == season) &
                   (rw["week"] >= week))
    count = update_mask.sum()
    rw.loc[update_mask, "team"] = to_team
    rw.to_parquet(RAW / "rosters_weekly.parquet", index=False)

    print(f"  Updated {count} rows: {name}  {from_team} → {to_team}  (wk{week}+)")
    print(f"  Rebuild composite: python run_engine.py --season {season}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["status","offseason","preseason","weekly","trade","changes"],
        default="status")
    parser.add_argument("--season",    type=int, default=2026)
    parser.add_argument("--week",      type=int, default=1)
    parser.add_argument("--player",    type=str, default="")
    parser.add_argument("--from-team", type=str, default="")
    parser.add_argument("--to-team",   type=str, default="")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()

    if args.mode == "status":
        show_status()
    elif args.mode == "changes":
        show_changes()
    elif args.mode in ("offseason", "preseason"):
        build_2026_roster(refresh=not args.no_refresh)
    elif args.mode == "weekly":
        weekly_update(args.season, args.week)
    elif args.mode == "trade":
        if not all([args.player, args.from_team, args.to_team]):
            print("Trade requires: --player NAME --from-team XX --to-team XX")
        else:
            handle_trade(args.player, args.from_team, args.to_team,
                         args.season, args.week)
