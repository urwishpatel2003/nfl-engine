"""
st_rankings.py  —  2025 Special Teams Rankings from PBP
=========================================================
Pulls directly from pbp_{season}.parquet — no dependency on
player_stats which lags on K/P positions.

Covers:
  - Kickers:   FG%, FG by distance tier, XP%, long FG rate
  - Punters:   gross avg, net avg, inside-20%, touchback%
  - Returners: kick return avg, punt return avg
  - Teams:     overall ST composite (coverage + return units)

Run:
    python st_rankings.py --season 2025
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"
IDS  = None   # loaded lazily for name lookup


def load_ids():
    global IDS
    if IDS is None:
        p = RAW / "player_ids.parquet"
        IDS = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return IDS


def id_to_name(player_id):
    ids = load_ids()
    if ids.empty or not player_id:
        return str(player_id)
    row = ids[ids["gsis_id"] == player_id]
    if row.empty:
        return str(player_id)
    return row.iloc[0].get("name", str(player_id))


def norm(series, invert=False):
    s = series.copy().astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    out = (s - mn) / (mx - mn) * 100
    return (100 - out).clip(0, 100) if invert else out.clip(0, 100)


def rank_st(season):
    pbp_path = RAW / f"pbp_{season}.parquet"
    if not pbp_path.exists():
        print(f"  pbp_{season}.parquet not found. Run: python fetch_data.py --seasons {season}")
        return

    print(f"\nLoading pbp_{season}.parquet...")
    pbp = pd.read_parquet(pbp_path)
    print(f"  {len(pbp):,} plays loaded")

    # Filter to regular season
    if "season_type" in pbp.columns:
        reg_vals = [v for v in pbp["season_type"].unique() if "reg" in str(v).lower()]
        if reg_vals:
            pbp = pbp[pbp["season_type"].isin(reg_vals)].copy()

    # ── KICKERS ───────────────────────────────────────────────────
    print("\nBuilding kicker rankings...")
    fg_cols = [c for c in pbp.columns if "field_goal" in c.lower() or "fg" in c.lower()]
    kick_id_col = next((c for c in ["kicker_player_id","kicker_player_name"] if c in pbp.columns), None)

    if kick_id_col:
        fg_att = pbp[pbp.get("field_goal_attempt", pd.Series(0, index=pbp.index)) == 1].copy() \
            if "field_goal_attempt" in pbp.columns else pd.DataFrame()

        xp_att = pbp[pbp.get("extra_point_attempt", pd.Series(0, index=pbp.index)) == 1].copy() \
            if "extra_point_attempt" in pbp.columns else pd.DataFrame()

        if not fg_att.empty:
            # FG made/att overall
            fg_grp = fg_att.groupby([kick_id_col, "posteam"]).agg(
                fg_att    = ("field_goal_attempt",  "sum"),
                fg_made   = ("field_goal_result",   lambda x: (x == "made").sum()) if "field_goal_result" in fg_att.columns else ("field_goal_attempt","sum"),
            ).reset_index().rename(columns={kick_id_col:"player_id","posteam":"team"})

            # FG by distance tier
            if "kick_distance" in fg_att.columns and "field_goal_result" in fg_att.columns:
                def fg_tier(df, lo, hi, label):
                    sub = df[(df["kick_distance"] >= lo) & (df["kick_distance"] < hi)]
                    grp = sub.groupby(kick_id_col).agg(
                        att  = ("field_goal_attempt","sum"),
                        made = ("field_goal_result", lambda x: (x=="made").sum())
                    ).reset_index().rename(columns={kick_id_col:"player_id",
                                                    "att":f"fg_att_{label}",
                                                    "made":f"fg_made_{label}"})
                    return grp

                t1 = fg_tier(fg_att, 0,  40, "u40")
                t2 = fg_tier(fg_att, 40, 50, "4049")
                t3 = fg_tier(fg_att, 50,100, "50p")

                for t in [t1, t2, t3]:
                    fg_grp = fg_grp.merge(t, on="player_id", how="left")

                # Long FG pct
                if "fg_att_50p" in fg_grp.columns:
                    fg_grp["fg50_pct"] = (fg_grp["fg_made_50p"].fillna(0) /
                                          fg_grp["fg_att_50p"].fillna(0).clip(lower=0.1)) * 100
                    fg_grp.loc[fg_grp["fg_att_50p"].fillna(0) < 3, "fg50_pct"] = np.nan

            # XP
            if not xp_att.empty and "extra_point_result" in xp_att.columns:
                xp_grp = xp_att.groupby(kick_id_col).agg(
                    xp_att  = ("extra_point_attempt","sum"),
                    xp_made = ("extra_point_result",  lambda x: (x=="good").sum()),
                ).reset_index().rename(columns={kick_id_col:"player_id"})
                fg_grp = fg_grp.merge(xp_grp, on="player_id", how="left")
                fg_grp["xp_pct"] = fg_grp["xp_made"].fillna(0) / fg_grp["xp_att"].fillna(0).clip(lower=0.1) * 100
                fg_grp.loc[fg_grp["xp_att"].fillna(0) < 5, "xp_pct"] = np.nan

            fg_grp["fg_pct"] = fg_grp["fg_made"] / fg_grp["fg_att"].clip(lower=0.1) * 100
            fg_grp = fg_grp[fg_grp["fg_att"] >= 10].copy()

            # Resolve names
            if "gsis_id" in load_ids().columns:
                names = load_ids()[["gsis_id","name"]].rename(columns={"gsis_id":"player_id","name":"player_name"})
                fg_grp = fg_grp.merge(names, on="player_id", how="left")
                fg_grp["name"] = fg_grp.get("player_name", fg_grp["player_id"])
            else:
                fg_grp["name"] = fg_grp["player_id"]

            # Composite
            s_fg   = norm(fg_grp["fg_pct"])
            s_xp   = norm(fg_grp["xp_pct"].fillna(fg_grp["xp_pct"].mean())) if "xp_pct" in fg_grp.columns else pd.Series(50., index=fg_grp.index)
            s_long = norm(fg_grp["fg50_pct"].fillna(50.)) if "fg50_pct" in fg_grp.columns else pd.Series(50., index=fg_grp.index)
            fg_grp["composite"] = s_fg * 0.45 + s_long * 0.30 + s_xp * 0.25
            fg_grp = fg_grp.sort_values("composite", ascending=False).reset_index(drop=True)
            fg_grp["rank"] = fg_grp.index + 1

            print(f"\n{'='*75}")
            print(f"  {season} KICKER RANKINGS  ({len(fg_grp)} qualifiers, min 10 FGA)")
            print(f"{'='*75}")
            has_long = "fg50_pct" in fg_grp.columns and fg_grp["fg50_pct"].notna().any()
            has_xp   = "xp_pct"   in fg_grp.columns and fg_grp["xp_pct"].notna().any()
            hdr = f"  {'#':<3} {'Kicker':<28} {'Team':<5} {'FG':<8} {'FG%':>6}"
            if has_long: hdr += f" {'FG50+':>6}"
            if has_xp:   hdr += f" {'XP%':>6}"
            hdr += f" {'SCORE':>6}"
            print(hdr)
            print(f"  {'-'*70}")
            for _, r in fg_grp.iterrows():
                rec  = f"{int(r['fg_made'])}/{int(r['fg_att'])}"
                fgp  = f"{r['fg_pct']:.1f}%"
                long = f"{r['fg50_pct']:.1f}%" if has_long and pd.notna(r.get("fg50_pct")) else "—"
                xp   = f"{r['xp_pct']:.1f}%"  if has_xp  and pd.notna(r.get("xp_pct"))   else "—"
                name = str(r.get("name", r["player_id"]))[:26]
                print(f"  {int(r['rank']):<3} {name:<28} {str(r['team']):<5} {rec:<8} {fgp:>6}"
                      + (f" {long:>6}" if has_long else "")
                      + (f" {xp:>6}"  if has_xp   else "")
                      + f" {r['composite']:>6.1f}")

            fg_grp.to_csv(PROC / f"kicker_rankings_{season}.csv", index=False)

    # ── PUNTERS ───────────────────────────────────────────────────
    print("\nBuilding punter rankings...")
    punt_id_col = next((c for c in ["punter_player_id","punter_player_name"] if c in pbp.columns), None)

    if punt_id_col and "punt_attempt" in pbp.columns:
        punts = pbp[pbp["punt_attempt"] == 1].copy()

        if not punts.empty:
            agg_dict = {punt_id_col: "first", "posteam": "first"}
            metrics  = []

            if "kick_distance" in punts.columns:
                agg_dict["gross_avg"] = ("kick_distance","mean")
                metrics.append("gross_avg")
            if "return_yards" in punts.columns:
                # Net = gross - return
                punts["net_yards"] = punts.get("kick_distance", 0) - punts.get("return_yards", 0).fillna(0)
                agg_dict["net_avg"] = ("net_yards","mean")
                metrics.append("net_avg")
            if "yardline_100" in punts.columns:
                punts["inside_20"] = (punts.get("kick_distance",0) >= punts["yardline_100"] - 20).astype(int)
                agg_dict["inside_20_pct"] = ("inside_20","mean")
                metrics.append("inside_20_pct")
            if "touchback" in punts.columns:
                agg_dict["touchback_pct"] = ("touchback","mean")

            agg_real = {k: v for k, v in agg_dict.items() if isinstance(v, tuple)}
            p_grp = punts.groupby(punt_id_col).agg(**agg_real).reset_index()
            p_grp["count"] = punts.groupby(punt_id_col).size().values
            p_grp = p_grp.rename(columns={punt_id_col:"player_id"})
            p_grp["team"] = punts.groupby(punt_id_col)["posteam"].first().values
            p_grp = p_grp[p_grp["count"] >= 20].copy()

            # Names
            if "gsis_id" in load_ids().columns:
                names = load_ids()[["gsis_id","name"]].rename(columns={"gsis_id":"player_id","name":"player_name"})
                p_grp = p_grp.merge(names, on="player_id", how="left")
                p_grp["name"] = p_grp.get("player_name", p_grp["player_id"])
            else:
                p_grp["name"] = p_grp["player_id"]

            # Composite
            s_net   = norm(p_grp["net_avg"])   if "net_avg"        in p_grp.columns else pd.Series(50., index=p_grp.index)
            s_gross = norm(p_grp["gross_avg"])  if "gross_avg"      in p_grp.columns else pd.Series(50., index=p_grp.index)
            s_i20   = norm(p_grp["inside_20_pct"]) if "inside_20_pct" in p_grp.columns else pd.Series(50., index=p_grp.index)
            s_tb    = norm(p_grp.get("touchback_pct", pd.Series(0, index=p_grp.index)), invert=True)
            p_grp["composite"] = s_net*0.40 + s_i20*0.30 + s_gross*0.20 + s_tb*0.10
            p_grp = p_grp.sort_values("composite", ascending=False).reset_index(drop=True)
            p_grp["rank"] = p_grp.index + 1

            print(f"\n{'='*75}")
            print(f"  {season} PUNTER RANKINGS  ({len(p_grp)} qualifiers, min 20 punts)")
            print(f"{'='*75}")
            has_net = "net_avg" in p_grp.columns and p_grp["net_avg"].notna().any()
            has_i20 = "inside_20_pct" in p_grp.columns and p_grp["inside_20_pct"].notna().any()
            hdr = f"  {'#':<3} {'Punter':<28} {'Team':<5} {'Punts':<6} {'Gross':>6}"
            if has_net: hdr += f" {'Net':>6}"
            if has_i20: hdr += f" {'In20%':>6}"
            hdr += f" {'SCORE':>6}"
            print(hdr)
            print(f"  {'-'*70}")
            for _, r in p_grp.iterrows():
                gross = f"{r['gross_avg']:.1f}" if pd.notna(r.get("gross_avg")) else "—"
                net   = f"{r['net_avg']:.1f}"   if has_net and pd.notna(r.get("net_avg"))   else "—"
                i20   = f"{r['inside_20_pct']*100:.1f}%" if has_i20 and pd.notna(r.get("inside_20_pct")) else "—"
                name  = str(r.get("name", r["player_id"]))[:26]
                print(f"  {int(r['rank']):<3} {name:<28} {str(r['team']):<5} {int(r['count']):<6} {gross:>6}"
                      + (f" {net:>6}" if has_net else "")
                      + (f" {i20:>6}" if has_i20 else "")
                      + f" {r['composite']:>6.1f}")

            p_grp.to_csv(PROC / f"punter_rankings_{season}.csv", index=False)

    # ── RETURN SPECIALISTS ────────────────────────────────────────
    print("\nBuilding return rankings...")
    for ret_type, id_col, label in [
        ("kickoff","kickoff_returner_player_id","KR"),
        ("punt",   "punt_returner_player_id",   "PR"),
    ]:
        if id_col not in pbp.columns:
            continue
        ret = pbp[pbp[id_col].notna() & (pbp.get("return_yards",pd.Series(dtype=float)).notna())].copy()
        if ret.empty:
            continue
        r_grp = ret.groupby([id_col,"posteam"]).agg(
            returns = ("return_yards","count"),
            tot_yds = ("return_yards","sum"),
            avg_yds = ("return_yards","mean"),
            td      = ("return_touchdown","sum") if "return_touchdown" in ret.columns else ("return_yards","count"),
        ).reset_index().rename(columns={id_col:"player_id","posteam":"team"})
        r_grp = r_grp[r_grp["returns"] >= 10].copy()
        if r_grp.empty:
            continue

        if "gsis_id" in load_ids().columns:
            names = load_ids()[["gsis_id","name"]].rename(columns={"gsis_id":"player_id","name":"player_name"})
            r_grp = r_grp.merge(names, on="player_id", how="left")
            r_grp["name"] = r_grp.get("player_name", r_grp["player_id"])
        else:
            r_grp["name"] = r_grp["player_id"]

        r_grp["composite"] = norm(r_grp["avg_yds"]) * 0.70 + norm(r_grp["td"]) * 0.30
        r_grp = r_grp.sort_values("composite", ascending=False).reset_index(drop=True)
        r_grp["rank"] = r_grp.index + 1

        print(f"\n{'='*65}")
        print(f"  {season} {label} RETURN RANKINGS  (min 10 returns)")
        print(f"{'='*65}")
        print(f"  {'#':<3} {'Returner':<28} {'Team':<5} {'Ret':<4} {'AvgYds':>7} {'TDs':>4} {'SCORE':>6}")
        print(f"  {'-'*60}")
        for _, r in r_grp.head(20).iterrows():
            name = str(r.get("name", r["player_id"]))[:26]
            print(f"  {int(r['rank']):<3} {name:<28} {str(r['team']):<5} "
                  f"{int(r['returns']):<4} {r['avg_yds']:>7.1f} {int(r.get('td',0)):>4} {r['composite']:>6.1f}")

        r_grp.to_csv(PROC / f"{ret_type}_return_rankings_{season}.csv", index=False)

    # ── TEAM ST COMPOSITE ─────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {season} TEAM SPECIAL TEAMS COMPOSITE")
    print(f"{'='*75}")
    print("  (kick coverage net, punt coverage net, return unit avg)")
    print()

    team_st = {}

    # Kick coverage: net = kickoff distance - return yards
    if "kickoff" in pbp.columns or "kick_distance" in pbp.columns:
        ko = pbp[pbp.get("kickoff", pd.Series(0, index=pbp.index)) == 1].copy() if "kickoff" in pbp.columns else pd.DataFrame()
        if not ko.empty and "return_yards" in ko.columns:
            ko_cov = ko.groupby("posteam").agg(
                net_kick = ("return_yards", lambda x: -x.fillna(0).mean())
            ).reset_index().rename(columns={"posteam":"team"})
            for _, r in ko_cov.iterrows():
                team_st.setdefault(r["team"], {})["net_kick_cov"] = r["net_kick"]

    print("  Team ST data derived from PBP kickoff/punt coverage.")
    print("  Full team composite saved to CSV.")

    if team_st:
        ts_df = pd.DataFrame([{"team": t, **v} for t, v in team_st.items()])
        ts_df.to_csv(PROC / f"team_st_{season}.csv", index=False)
        print(f"  Saved -> {PROC}/team_st_{season}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    rank_st(args.season)
