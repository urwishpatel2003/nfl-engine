"""
coaching_metrics.py  —  Build coach quality scores from PBP + schedules
=========================================================================
Computes four coaching signals that the player/EPA model misses:

  1. ADJUSTMENT SCORE  — 2H EPA vs 1H EPA delta per team
     Elite coaches (Reid, McVay, McDaniel) consistently outperform
     in the second half as they adjust to what they saw in Q1/Q2.
     Poor coaches get outschemed after halftime.

  2. DISCIPLINE SCORE  — penalty rate per team (ref-crew adjusted)
     Undisciplined teams (excess penalties) cost 3-5 pts per game.
     Removes the ref crew confound by normalizing to crew baseline.

  3. ATS COACHING EDGE  — win% as favorite vs win% as underdog
     Elite coaches punch above their talent level.
     Andy Reid covers spreads at 58%+ as a favorite. Bad coaches don't.

  4. QUARTER SCORING PROFILE  — Q1/Q2/Q3/Q4 EPA by team
     Good halftime-adjustment teams surge in Q3.
     Q3 EPA delta (vs Q1/Q2 avg) is a pure coaching signal.

Usage:
    python coaching_metrics.py --seasons 2021 2022 2023 2024 2025
    
Outputs:
    data/processed/coaching_scores.parquet   — per team per season
    data/processed/coaching_flags.csv        — human-readable summary
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"


def normalize(s: pd.Series, reverse=False) -> pd.Series:
    """Scale series to 0-100."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    out = (s - mn) / (mx - mn) * 100
    return 100 - out if reverse else out


def build_coaching_metrics(seasons: list) -> pd.DataFrame:
    print(f"\nBuilding coaching metrics for seasons {seasons}...")

    # ── Load PBP ──────────────────────────────────────────────────
    pbp_frames = []
    for s in seasons:
        p = RAW / f"pbp_{s}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["season"] = s
            pbp_frames.append(df)

    if not pbp_frames:
        print("  ERROR: No PBP files found"); return pd.DataFrame()

    pbp = pd.concat(pbp_frames, ignore_index=True)
    print(f"  Loaded {len(pbp):,} PBP rows across {len(pbp_frames)} seasons")

    # Normalize key columns
    for col in ["epa","penalty","qb_scramble","rush_attempt","pass_attempt",
                "complete_pass","success","touchdown","sack"]:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0)

    # ── 1. ADJUSTMENT SCORE: 2H EPA vs 1H EPA ─────────────────────
    print("  Computing halftime adjustment scores...")

    has_quarter = "qtr" in pbp.columns or "quarter" in pbp.columns
    quarter_col = "qtr" if "qtr" in pbp.columns else "quarter"

    if has_quarter:
        pbp["half"] = np.where(pd.to_numeric(pbp[quarter_col], errors="coerce") <= 2, 1, 2)
        half_epa = pbp.groupby(["season","posteam","half"])["epa"].mean().reset_index()
        half_epa = half_epa.pivot_table(
            index=["season","posteam"], columns="half", values="epa"
        ).reset_index()
        half_epa.columns = ["season","team","h1_epa","h2_epa"]
        # Adjustment score = how much better is 2H vs 1H
        # Positive = team improves after halftime = good coaching adjustment
        half_epa["adjustment_score"] = half_epa["h2_epa"].fillna(0) - half_epa["h1_epa"].fillna(0)
        print(f"    Built half-EPA for {len(half_epa)} team-seasons")
    else:
        half_epa = pd.DataFrame(columns=["season","team","h1_epa","h2_epa","adjustment_score"])
        print("    WARN: No quarter column in PBP — skipping adjustment score")

    # ── 2. DISCIPLINE SCORE: Penalty rate (ref-adjusted) ──────────
    print("  Computing discipline scores...")

    if "penalty" in pbp.columns and "game_id" in pbp.columns:
        # Compute ref crew avg penalty rate
        ref_rate = pbp.groupby("game_id")["penalty"].mean().reset_index()
        ref_rate.columns = ["game_id","game_penalty_rate"]

        pbp_p = pbp.merge(ref_rate, on="game_id", how="left")

        # Team penalty rate vs ref crew baseline
        team_pen = pbp_p.groupby(["season","posteam"]).agg(
            team_penalty_rate = ("penalty", "mean"),
            game_baseline      = ("game_penalty_rate", "mean"),
            total_plays        = ("epa", "count"),
        ).reset_index()
        # Adjusted penalty rate = team rate - ref crew baseline
        # Negative adjusted = disciplined (fewer penalties than crew average)
        team_pen["adjusted_penalty_rate"] = (
            team_pen["team_penalty_rate"] - team_pen["game_baseline"]
        )
        team_pen = team_pen.rename(columns={"posteam":"team"})
        print(f"    Built penalty discipline for {len(team_pen)} team-seasons")
    else:
        team_pen = pd.DataFrame(columns=["season","team","adjusted_penalty_rate"])
        print("    WARN: No penalty column — skipping discipline score")

    # ── 3. QUARTER SCORING PROFILE: Q3 surge metric ───────────────
    print("  Computing quarter-by-quarter EPA profiles...")

    if has_quarter:
        qtr_epa = pbp.groupby(["season","posteam",quarter_col])["epa"].mean().reset_index()
        qtr_epa[quarter_col] = pd.to_numeric(qtr_epa[quarter_col], errors="coerce")
        qtr_pivot = qtr_epa[qtr_epa[quarter_col].isin([1,2,3,4])].pivot_table(
            index=["season","posteam"], columns=quarter_col, values="epa"
        ).reset_index()
        qtr_pivot.columns = ["season","team","q1_epa","q2_epa","q3_epa","q4_epa"]
        # Q3 surge = Q3 EPA vs average of Q1+Q2 (pure halftime adjustment signal)
        qtr_pivot["q3_surge"] = (
            qtr_pivot["q3_epa"].fillna(0) -
            ((qtr_pivot["q1_epa"].fillna(0) + qtr_pivot["q2_epa"].fillna(0)) / 2)
        )
        # Q4 clutch = Q4 EPA vs season average (close game coaching)
        qtr_pivot["q4_clutch"] = qtr_pivot["q4_epa"].fillna(0)
        print(f"    Built quarter profiles for {len(qtr_pivot)} team-seasons")
    else:
        qtr_pivot = pd.DataFrame(columns=["season","team","q3_surge","q4_clutch"])

    # ── 4. ATS COACHING EDGE: Cover % when favored ────────────────
    print("  Computing ATS coaching edge...")

    sched = pd.read_parquet(RAW / "schedules.parquet")
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched = sched[
        sched["season"].isin(seasons) &
        (sched["game_type"].str.upper().str.strip() == "REG") &
        sched["home_score"].notna() &
        sched["spread_line"].notna()
    ].copy()

    ats_rows = []
    for _, g in sched.iterrows():
        home = g["home_team"]
        away = g["away_team"]
        spread = float(g["spread_line"])  # positive = home favored (home margin line)
        home_score = float(g["home_score"])
        away_score = float(g["away_score"])
        season = int(g["season"])
        actual_margin = home_score - away_score

        # Home team ATS — home covers if home_margin > spread_line
        home_covered = actual_margin > spread
        home_favored = spread > 0
        ats_rows.append({
            "season": season, "team": home, "spread": spread,
            "covered": home_covered, "is_favorite": home_favored,
            "margin": actual_margin, "spread_margin": actual_margin - spread
        })

        # Away team ATS — away covers if home_margin < spread_line
        away_covered = actual_margin < spread
        away_favored = spread < 0
        ats_rows.append({
            "season": season, "team": away, "spread": -spread,
            "covered": away_covered, "is_favorite": away_favored,
            "margin": -actual_margin, "spread_margin": spread - actual_margin
        })

    ats_df = pd.DataFrame(ats_rows)
    coaching_ats = ats_df.groupby(["season","team"]).agg(
        ats_cover_pct     = ("covered", "mean"),
        ats_fav_cover_pct = ("covered", lambda x: x[ats_df.loc[x.index,"is_favorite"]].mean()),
        avg_spread_margin = ("spread_margin", "mean"),  # positive = beats spread by N
        games_as_fav      = ("is_favorite", "sum"),
    ).reset_index()
    print(f"    Built ATS coaching edge for {len(coaching_ats)} team-seasons")

    # ── MERGE ALL SIGNALS ─────────────────────────────────────────
    print("\n  Merging coaching signals...")

    coaching = coaching_ats.copy()

    if not half_epa.empty:
        coaching = coaching.merge(
            half_epa[["season","team","h1_epa","h2_epa","adjustment_score"]],
            on=["season","team"], how="left"
        )

    if not team_pen.empty:
        coaching = coaching.merge(
            team_pen[["season","team","adjusted_penalty_rate","total_plays"]],
            on=["season","team"], how="left"
        )

    if not qtr_pivot.empty:
        coaching = coaching.merge(
            qtr_pivot[["season","team","q1_epa","q2_epa","q3_epa","q4_epa",
                       "q3_surge","q4_clutch"]],
            on=["season","team"], how="left"
        )

    # ── COMPOSITE COACHING SCORE (0-100) ──────────────────────────
    # Weights: ATS cover% 30%, adjustment score 30%, discipline 20%, Q3 surge 20%
    coaching["coaching_score"] = 50.0
    weight_total = 0.0

    if "ats_cover_pct" in coaching.columns and coaching["ats_cover_pct"].notna().any():
        ats_norm = normalize(coaching["ats_cover_pct"].fillna(0.5))
        coaching["coaching_score"] += (ats_norm - 50) * 0.30
        weight_total += 0.30

    if "adjustment_score" in coaching.columns and coaching["adjustment_score"].notna().any():
        adj_norm = normalize(coaching["adjustment_score"].fillna(0))
        coaching["coaching_score"] += (adj_norm - 50) * 0.30
        weight_total += 0.30

    if "adjusted_penalty_rate" in coaching.columns and coaching["adjusted_penalty_rate"].notna().any():
        # Lower penalty rate = better → reverse normalize
        disc_norm = normalize(coaching["adjusted_penalty_rate"].fillna(0), reverse=True)
        coaching["coaching_score"] += (disc_norm - 50) * 0.20
        weight_total += 0.20

    if "q3_surge" in coaching.columns and coaching["q3_surge"].notna().any():
        q3_norm = normalize(coaching["q3_surge"].fillna(0))
        coaching["coaching_score"] += (q3_norm - 50) * 0.20
        weight_total += 0.20

    if weight_total > 0:
        coaching["coaching_score"] = (coaching["coaching_score"] - 50) / weight_total + 50

    coaching["coaching_score"] = coaching["coaching_score"].clip(0, 100).round(1)

    # ── FLAGS ─────────────────────────────────────────────────────
    # elite_coaching: top quartile (score >= 65)
    # poor_coaching:  bottom quartile (score <= 40)
    # elite_adjuster: adjustment_score > 0.03 (improves significantly after half)
    # undisciplined:  adjusted_penalty_rate > 0.02
    coaching["elite_coaching"]  = coaching["coaching_score"] >= 65
    coaching["poor_coaching"]   = coaching["coaching_score"] <= 40
    coaching["elite_adjuster"]  = coaching.get("adjustment_score", pd.Series(0)) > 0.03
    coaching["undisciplined"]   = coaching.get("adjusted_penalty_rate", pd.Series(0)) > 0.02

    # Save
    out_path = PROC / "coaching_scores.parquet"
    coaching.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}  ({len(coaching)} team-seasons)")

    # Human-readable summary
    cols_to_show = ["season","team","coaching_score",
                    "ats_cover_pct","adjustment_score","adjusted_penalty_rate",
                    "q3_surge","elite_coaching","poor_coaching","elite_adjuster"]
    cols_avail = [c for c in cols_to_show if c in coaching.columns]
    csv_path = PROC / "coaching_flags.csv"
    coaching[cols_avail].sort_values(
        ["season","coaching_score"], ascending=[True,False]
    ).to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    # Print 2025 summary
    c25 = coaching[coaching["season"] == max(seasons)].sort_values(
        "coaching_score", ascending=False
    )
    if not c25.empty:
        print(f"\n  2025 COACHING SCORES (top 10 / bottom 5):")
        print(f"  {'Team':<5} {'Score':>6} {'ATS%':>6} {'Adj':>6} {'Disc':>6} {'Q3Srg':>7} {'Flags'}")
        print(f"  {'-'*60}")
        for _, r in c25.head(10).iterrows():
            flags = []
            if r.get("elite_coaching"):   flags.append("elite")
            if r.get("elite_adjuster"):   flags.append("adjuster")
            if r.get("undisciplined"):    flags.append("undiscip")
            ats  = f"{r.get('ats_cover_pct',float('nan'))*100:.0f}%" if pd.notna(r.get('ats_cover_pct')) else "n/a"
            adj  = f"{r.get('adjustment_score',float('nan')):+.3f}" if pd.notna(r.get('adjustment_score')) else "n/a"
            disc = f"{r.get('adjusted_penalty_rate',float('nan')):+.3f}" if pd.notna(r.get('adjusted_penalty_rate')) else "n/a"
            q3   = f"{r.get('q3_surge',float('nan')):+.3f}" if pd.notna(r.get('q3_surge')) else "n/a"
            print(f"  {r['team']:<5} {r['coaching_score']:>6.1f} {ats:>6} {adj:>6} {disc:>6} {q3:>7}  {', '.join(flags)}")

        print(f"\n  BOTTOM 5:")
        for _, r in c25.tail(5).iterrows():
            flags = []
            if r.get("poor_coaching"):  flags.append("poor")
            if r.get("undisciplined"): flags.append("undiscip")
            ats  = f"{r.get('ats_cover_pct',float('nan'))*100:.0f}%" if pd.notna(r.get('ats_cover_pct')) else "n/a"
            adj  = f"{r.get('adjustment_score',float('nan')):+.3f}" if pd.notna(r.get('adjustment_score')) else "n/a"
            print(f"  {r['team']:<5} {r['coaching_score']:>6.1f} {ats:>6} {adj:>6}")

    return coaching


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    args = parser.parse_args()
    build_coaching_metrics(args.seasons)
