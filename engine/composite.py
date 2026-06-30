"""
engine/composite.py
-------------------
Builds a 0-100 composite score for every active player.

Score = weighted blend of five components:
  1. Position Rank      (20%) -- where they rank 1-N in their position group by seasonal EPA
  2. Efficiency         (30%) -- EPA-based efficiency metrics, position-specific
  3. Usage             (25%) -- snap%, target share, carry share
  4. Tracking          (15%) -- NGS metrics (CPOE, separation, RYOE)
  5. Athleticism       (10%) -- combine measurables normalized by position

Each component is normalized 0-100 before blending.
Final score is then multiplied by:
  - starter_multiplier  (1.0 if starter, 0.6 if backup, 0.3 if 3rd string)
  - availability_mult   (1.0 Active, 0.75 Questionable, 0.25 Doubtful, 0.0 Out)

Output: one row per player per (season, week) with composite_score + all components.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── Component weights ──────────────────────────────────────────────
WEIGHTS = {
    "rank_score":       0.18,   # EPA-based positional rank
    "efficiency_score": 0.30,   # EPA efficiency (rolling 6-week)
    "usage_score":      0.22,   # snap%, target share, carry share
    "tracking_score":   0.22,   # NGS: CPOE / RYOE / separation + YAC-OE  (upgraded)
    "athleticism_score":0.08,   # combine measurables
}

# ── Injury multipliers ─────────────────────────────────────────────
INJURY_MULT = {
    "Active":      1.00,
    "Questionable":0.75,
    "Doubtful":    0.25,
    "Out":         0.00,
    "IR":          0.00,
    "PUP":         0.00,
}

# ── Depth chart multipliers ────────────────────────────────────────
DEPTH_MULT = {1: 1.0, 2: 0.6, 3: 0.3}


def normalize(s: pd.Series, reverse: bool = False) -> pd.Series:
    """Min-max normalize to 0-100. reverse=True for metrics where lower is better."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    norm = (s - mn) / (mx - mn) * 100
    return (100 - norm) if reverse else norm


def rolling_mean(df: pd.DataFrame, col: str, n: int = 6) -> pd.Series:
    """Per-player rolling mean over last n games (shift 1 to avoid leakage)."""
    return (
        df.groupby("player_id")[col]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )


# ══════════════════════════════════════════════════════════════════
# 1. POSITION RANK SCORE
# ══════════════════════════════════════════════════════════════════

def build_rank_score(seasonal: pd.DataFrame,
                      rosters: pd.DataFrame = None) -> pd.DataFrame:
    """
    Rank every player within their position group by season EPA.
    seasonal_stats has no position column — we join it from rosters_seasonal.
    Returns player_id, season, position, pos_rank, rank_score (0-100).
    """
    # Join position from rosters if not already present
    if "position" not in seasonal.columns:
        if rosters is None:
            rosters = pd.read_parquet(RAW / "rosters_seasonal.parquet")
        pos_map = (
            rosters[["player_id", "season", "position"]]
            .dropna(subset=["player_id", "position"])
            .drop_duplicates(["player_id", "season"])
        )
        seasonal = seasonal.merge(pos_map, on=["player_id", "season"], how="left")

    # Also join from player_stats as fallback for any still missing
    if seasonal["position"].isna().any():
        try:
            ps = pd.read_parquet(RAW / "player_stats.parquet")
            ps_pos = ps[["player_id","position"]].dropna().drop_duplicates("player_id")
            seasonal = seasonal.merge(ps_pos, on="player_id", how="left",
                                       suffixes=("","_ps"))
            if "position_ps" in seasonal.columns:
                seasonal["position"] = seasonal["position"].fillna(seasonal["position_ps"])
                seasonal = seasonal.drop(columns=["position_ps"])
        except Exception:
            pass

    pos_epa = {
        "QB": "passing_epa",
        "RB": "rushing_epa",
        "WR": "receiving_epa",
        "TE": "receiving_epa",
        "K":  "fantasy_points",
    }

    rows = []
    for pos, epa_col in pos_epa.items():
        sub = seasonal[seasonal["position"] == pos].copy()
        if sub.empty or epa_col not in sub.columns:
            continue
        sub = sub.dropna(subset=[epa_col])

        # Apply recency weight to the EPA column before ranking
        # This means a player's recent seasons dominate their rank
        if "recency_weight" in sub.columns:
            sub["weighted_epa"] = sub[epa_col] * sub["recency_weight"]
        else:
            sub["weighted_epa"] = sub[epa_col]

        # Aggregate to one row per player using weighted EPA
        # (handles multi-season case: each player gets their recency-weighted best)
        player_epa = (
            sub.groupby("player_id")
            .apply(lambda g: (g["weighted_epa"] * g.get("recency_weight", 1)).sum()
                   / g.get("recency_weight", pd.Series(1, index=g.index)).sum(),
                   include_groups=False)
            .reset_index()
            .rename(columns={0: "agg_epa"})
        )

        # Join back the most recent season for each player (for the season label)
        latest_season = sub.sort_values("season").groupby("player_id")["season"].last().reset_index()
        player_epa = player_epa.merge(latest_season, on="player_id", how="left")
        player_epa["position"] = pos

        # Rank within position across all players
        player_epa["pos_rank"] = player_epa["agg_epa"].rank(ascending=False, method="min")
        player_epa["rank_score"] = normalize(player_epa["pos_rank"], reverse=True)

        rows.append(player_epa[["player_id", "season", "position", "pos_rank", "rank_score"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# 2. EFFICIENCY SCORE
# ══════════════════════════════════════════════════════════════════

def build_efficiency_score(stats: pd.DataFrame,
                            pfr_passing: pd.DataFrame = None,
                            qbr_data: pd.DataFrame = None,
                            pfr_receiving: pd.DataFrame = None,
                            ftn_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Rolling efficiency per player per week.
    Now includes: QBR, PFR pressure metrics (QB), FTN route participation (WR/TE).
    """
    stats = stats.sort_values(["player_id", "season", "week"]).copy()
    out_parts = []

    # ── QB efficiency ──────────────────────────────────────────────
    qb = stats[stats["position"] == "QB"].copy()
    if not qb.empty:
        for col in ["passing_epa", "dakota"]:
            if col in qb.columns:
                qb[f"{col}_r6"] = rolling_mean(qb, col, 6)

        # Merge QBR if available
        if qbr_data is not None and not qbr_data.empty:
            qbr_cols = [c for c in qbr_data.columns if "qbr" in c.lower() or "pts_added" in c.lower()]
            if "game_week" in qbr_data.columns:
                qbr_data = qbr_data.rename(columns={"game_week": "week"})
            if qbr_cols and "season" in qbr_data.columns and "week" in qbr_data.columns:
                # QBR uses name_display, not player_id — match on last name + season + week
                if "name_display" in qbr_data.columns:
                    qbr_sub = qbr_data[["season","week","name_display",qbr_cols[0]]].copy()
                    qbr_sub["name_last"] = qbr_sub["name_display"].str.split().str[-1].str.upper()
                    qbr_sub = qbr_sub.rename(columns={qbr_cols[0]: "qbr_score"})
                    if "name_last" not in qb.columns:
                        qb["name_last"] = qb["player_display_name"].str.split().str[-1].str.upper()
                    qb = qb.merge(qbr_sub[["season","week","name_last","qbr_score"]],
                                  on=["name_last","season","week"], how="left")
                    if "qbr_score" in qb.columns:
                        qb["qbr_r6"] = rolling_mean(qb, "qbr_score", 6)

        # ── PFR pressure + accuracy metrics (NEW) ─────────────────
        # bad_throw_pct: best single accuracy metric (lower = better)
        # times_pressured: pocket disruption volume
        # times_hurried + times_hit: individual pressure components
        if pfr_passing is not None and not pfr_passing.empty:
            pfr_qb_cols = []
            for wanted in ["passing_bad_throw_pct", "times_pressured_pct",
                           "times_sacked", "times_hurried", "times_hit", "times_blitzed"]:
                if wanted in pfr_passing.columns:
                    pfr_qb_cols.append(wanted)

            # Try merge via pfr_player_name → player_display_name last name match
            pfr_p = pfr_passing[
                (pfr_passing["season"].isin(qb["season"].unique())) &
                (pfr_passing["game_type"] == "REG")
            ].copy() if "game_type" in pfr_passing.columns else pfr_passing.copy()

            if pfr_qb_cols and not pfr_p.empty:
                # PFR name format: "Last, First" → extract last name for matching
                pfr_p["name_last"] = pfr_p["pfr_player_name"].str.split(",").str[0].str.strip().str.upper()
                qb["name_last"]    = qb["player_display_name"].str.split().str[-1].str.upper()

                pfr_agg = pfr_p.groupby(["name_last", "season", "week"])[pfr_qb_cols].mean().reset_index()
                qb = qb.merge(pfr_agg, on=["name_last", "season", "week"], how="left")

                # Rolling 6-week for each pressure metric
                for col in pfr_qb_cols:
                    if col in qb.columns:
                        qb[f"{col}_r6"] = rolling_mean(qb, col, 6)

                # bad_throw_pct is INVERTED (lower = better accuracy)
                if "passing_bad_throw_pct_r6" in qb.columns:
                    max_btp = qb["passing_bad_throw_pct_r6"].quantile(0.95)
                    min_btp = qb["passing_bad_throw_pct_r6"].quantile(0.05)
                    qb["accuracy_score_r6"] = (
                        1.0 - (qb["passing_bad_throw_pct_r6"].clip(min_btp, max_btp) - min_btp)
                        / (max_btp - min_btp + 1e-9)
                    ) * 100

                # Pressure conversion rate (how well QB performs under pressure)
                if "times_hurried_r6" in qb.columns and "times_hit_r6" in qb.columns:
                    # More hurries + hits while maintaining EPA = elite pocket presence
                    # Use as a modifier rather than standalone score
                    qb["pressure_volume_r6"] = (
                        qb["times_hurried_r6"].fillna(0) + qb["times_hit_r6"].fillna(0)
                    )

        # ── Weighted efficiency blend ──────────────────────────────
        # Weights: passing_epa 40%, accuracy (bad_throw_pct) 30%, dakota 20%, qbr 10%
        # Only include a metric if it has actual data — redistribute weight if missing
        weight_map = {
            "passing_epa_r6":    0.40,
            "accuracy_score_r6": 0.30,   # NEW: bad throw % converted to 0-100 score
            "dakota_r6":         0.20,
            "qbr_r6":            0.10,
        }
        qb["raw_eff"] = 0.0
        total_w = 0.0
        for col, w in weight_map.items():
            if col in qb.columns and qb[col].notna().any():
                filled = qb[col].fillna(qb[col].median())
                qb["raw_eff"] += filled * w
                total_w += w

        if total_w > 0.1:
            # Normalize by actual weight used so scores stay on same scale
            qb["raw_eff"] = qb["raw_eff"] / total_w
            qb["efficiency_score"] = qb.groupby("season")["raw_eff"].transform(normalize)
        else:
            qb["efficiency_score"] = 50.0

        out_parts.append(qb[["player_id", "season", "week", "position", "efficiency_score"]])

    # ── RB efficiency ──────────────────────────────────────────────
    rb = stats[stats["position"] == "RB"].copy()
    if not rb.empty:
        rb_metrics = []
        for col in ["rushing_epa", "receiving_epa"]:
            if col in rb.columns:
                rb[f"{col}_r6"] = rolling_mean(rb, col, 6)
                rb_metrics.append(f"{col}_r6")

        # Wire in PFR rushing broken tackles (NEW)
        if pfr_receiving is not None and not pfr_receiving.empty:
            pfr_rb = pfr_receiving[
                (pfr_receiving["season"].isin(rb["season"].unique())) &
                (pfr_receiving["game_type"] == "REG")
            ].copy() if "game_type" in pfr_receiving.columns else pfr_receiving.copy()

            if "rushing_broken_tackles" in pfr_rb.columns and not pfr_rb.empty:
                pfr_rb["name_last"] = pfr_rb["pfr_player_name"].str.split(",").str[0].str.strip().str.upper()
                rb["name_last"]     = rb["player_display_name"].str.split().str[-1].str.upper()
                btk_agg = pfr_rb.groupby(["name_last","season","week"])[["rushing_broken_tackles"]].sum().reset_index()
                rb = rb.merge(btk_agg, on=["name_last","season","week"], how="left")
                rb["rushing_broken_tackles_r6"] = rolling_mean(rb, "rushing_broken_tackles", 6)
                rb_metrics.append("rushing_broken_tackles_r6")

        if rb_metrics:
            rb["raw_eff"] = rb[rb_metrics].mean(axis=1)
            rb["efficiency_score"] = rb.groupby("season")["raw_eff"].transform(normalize)
        else:
            rb["efficiency_score"] = 50.0
        out_parts.append(rb[["player_id", "season", "week", "position", "efficiency_score"]])

    # ── WR/TE efficiency ───────────────────────────────────────────
    recv = stats[stats["position"].isin(["WR", "TE"])].copy()
    if not recv.empty:
        recv_metrics = []
        for col in ["receiving_epa", "wopr", "racr"]:
            if col in recv.columns:
                recv[f"{col}_r6"] = rolling_mean(recv, col, 6)
                recv_metrics.append(f"{col}_r6")

        # Merge FTN route participation if available
        if ftn_data is not None and not ftn_data.empty:
            route_cols = [c for c in ftn_data.columns
                          if "route" in c.lower() or "target" in c.lower()]
            ftn_id = [c for c in ["player_id", "season", "week"] if c in ftn_data.columns]
            if route_cols and len(ftn_id) == 3:
                ftn_sub = ftn_data[ftn_id + route_cols[:2]].copy()
                recv = recv.merge(ftn_sub, on=["player_id", "season", "week"], how="left")
                for rc in route_cols[:1]:
                    if rc in recv.columns:
                        recv[f"{rc}_r6"] = rolling_mean(recv, rc, 6)
                        recv_metrics.append(f"{rc}_r6")

        # Merge PFR receiving (broken tackles, drop rate)
        if pfr_receiving is not None and not pfr_receiving.empty:
            pfr_rec_cols = [c for c in pfr_receiving.columns
                            if any(x in c.lower() for x in ["broken", "drop", "yac"])]
            pfr_id = [c for c in ["pfr_id", "player_id", "season", "week"] if c in pfr_receiving.columns]
            if pfr_rec_cols and len(pfr_id) >= 3:
                pfr_sub = pfr_receiving[pfr_id + pfr_rec_cols[:2]].copy()
                merge_key = "pfr_id" if "pfr_id" in pfr_sub.columns else "player_id"
                if merge_key in recv.columns:
                    recv = recv.merge(pfr_sub, on=[merge_key, "season", "week"], how="left")
                    for pc in pfr_rec_cols[:1]:
                        if pc in recv.columns:
                            recv[f"{pc}_r6"] = rolling_mean(recv, pc, 6)
                            recv_metrics.append(f"{pc}_r6")

        if recv_metrics:
            recv["raw_eff"] = recv[recv_metrics].mean(axis=1)
            recv["efficiency_score"] = recv.groupby(["season","position"])["raw_eff"].transform(normalize)
        else:
            recv["efficiency_score"] = 50.0
        out_parts.append(recv[["player_id", "season", "week", "position", "efficiency_score"]])

    if not out_parts:
        return pd.DataFrame()

    result = pd.concat(out_parts, ignore_index=True)
    result["efficiency_score"] = result["efficiency_score"].clip(0, 100)
    return result


# ══════════════════════════════════════════════════════════════════
# 3. USAGE SCORE
# ══════════════════════════════════════════════════════════════════

def build_usage_score(stats: pd.DataFrame, snaps: pd.DataFrame,
                      rosters: pd.DataFrame) -> pd.DataFrame:
    """
    Snap%, target share, carry share rolled into a usage score.
    Snap data comes from snap_counts (pfr_player_id) linked via player_ids.
    """
    stats = stats.sort_values(["player_id", "season", "week"]).copy()

    # Build snap% per player per game via player_ids crosswalk
    # snap_counts uses pfr_player_id; stats uses gsis player_id
    # We join on name+team+week as fallback if ID crosswalk fails
    snap_usage = snaps[["game_id", "season", "week", "player", "pfr_player_id",
                          "position", "team", "offense_pct", "defense_pct"]].copy()

    usage_parts = []

    # WR/TE: target_share + air_yards_share + snap%
    recv = stats[stats["position"].isin(["WR", "TE"])].copy()
    if not recv.empty:
        metrics = []
        for col in ["target_share", "air_yards_share", "wopr"]:
            if col in recv.columns:
                recv[f"{col}_r6"] = rolling_mean(recv, col, 6)
                metrics.append(f"{col}_r6")
        if metrics:
            recv["raw_usage"] = recv[metrics].mean(axis=1)
            recv["usage_score"] = recv.groupby(["season", "position"])["raw_usage"].transform(normalize)
        else:
            recv["usage_score"] = 50.0
        usage_parts.append(recv[["player_id", "season", "week", "position", "usage_score"]])

    # RB: carries + targets (total touch share)
    rb = stats[stats["position"] == "RB"].copy()
    if not rb.empty:
        rb_metrics = []
        for col in ["carries", "targets", "receptions"]:
            if col in rb.columns:
                rb[f"{col}_r6"] = rolling_mean(rb, col, 6)
                rb_metrics.append(f"{col}_r6")
        if rb_metrics:
            rb["raw_usage"] = rb[rb_metrics].mean(axis=1)
            rb["usage_score"] = rb.groupby("season")["raw_usage"].transform(normalize)
        else:
            rb["usage_score"] = 50.0
        usage_parts.append(rb[["player_id", "season", "week", "position", "usage_score"]])

    # QB: attempts (volume = opportunity)
    qb = stats[stats["position"] == "QB"].copy()
    if not qb.empty and "attempts" in qb.columns:
        qb["attempts_r6"] = rolling_mean(qb, "attempts", 6)
        qb["usage_score"] = qb.groupby("season")["attempts_r6"].transform(normalize)
        usage_parts.append(qb[["player_id", "season", "week", "position", "usage_score"]])

    if not usage_parts:
        return pd.DataFrame()

    result = pd.concat(usage_parts, ignore_index=True)
    result["usage_score"] = result["usage_score"].clip(0, 100)
    return result


# ══════════════════════════════════════════════════════════════════
# 4. TRACKING SCORE (NGS)
# ══════════════════════════════════════════════════════════════════

def build_tracking_score(ngs_pass: pd.DataFrame, ngs_rush: pd.DataFrame,
                          ngs_recv: pd.DataFrame) -> pd.DataFrame:
    """
    NGS tracking quality scores — upgraded to use multi-metric blends.

    QB:    CPOE (completion % over expectation)  — primary signal
           + aggressiveness (willingness to push ball)
    RB:    rush_yards_over_expected_per_att (RYOE)  — primary
           + efficiency (consistency)
           + stack rate (works against loaded boxes)
    WR/TE: avg_separation                           — 45% weight
           + avg_yac_above_expectation (YAC-OE)     — 35% weight
           + avg_intended_air_yards (target depth)  — 20% weight
    """
    parts = []

    # ── QB tracking ────────────────────────────────────────────────
    if not ngs_pass.empty and "completion_percentage_above_expectation" in ngs_pass.columns:
        qb = ngs_pass[ngs_pass["week"] > 0].copy()
        qb = qb.rename(columns={"player_gsis_id": "player_id"})

        # Primary: CPOE (45%), aggressiveness (20%), passer rating (20%)
        # Mobility bonus (15%): scramble rate from NGS passing
        # A QB who scrambles frequently AND efficiently is adding a dimension
        # that pure passing metrics miss entirely
        cpoe_score = normalize(qb["completion_percentage_above_expectation"])
        components = [cpoe_score * 0.45]
        total_w = 0.45

        if "aggressiveness" in qb.columns:
            components.append(normalize(qb["aggressiveness"]) * 0.20)
            total_w += 0.20
        if "passer_rating" in qb.columns:
            components.append(normalize(qb["passer_rating"]) * 0.20)
            total_w += 0.20

        # Scramble mobility: avg_time_to_throw (fast release = mobility awareness)
        # + implicit scramble threat captured through aggressiveness above
        # We use avg_time_to_throw INVERTED: shorter time = quicker release/escape
        if "avg_time_to_throw" in qb.columns and qb["avg_time_to_throw"].notna().any():
            # Invert: faster (lower) time = better pocket escapability
            mobility_score = normalize(qb["avg_time_to_throw"], reverse=True)
            components.append(mobility_score * 0.15)
            total_w += 0.15

        qb["tracking_score"] = sum(components) / total_w
        parts.append(qb[["player_id", "season", "week", "tracking_score"]].assign(position="QB"))

    # ── RB tracking ────────────────────────────────────────────────
    if not ngs_rush.empty:
        rb = ngs_rush[ngs_rush["week"] > 0].copy()
        rb = rb.rename(columns={"player_gsis_id": "player_id"})
        components = []
        weights_used = []
        # Primary: RYOE/att (50%)
        if "rush_yards_over_expected_per_att" in rb.columns:
            components.append(normalize(rb["rush_yards_over_expected_per_att"]) * 0.50)
            weights_used.append(0.50)
        # Secondary: efficiency score (30%) — consistency metric
        if "efficiency" in rb.columns:
            # efficiency is time-to-LOS based — higher = faster hitting holes
            # but it's inverse in their scale so lower value = faster = better
            components.append(normalize(rb["efficiency"], reverse=True) * 0.30)
            weights_used.append(0.30)
        # Tertiary: rush_pct_over_expected (20%)
        if "rush_pct_over_expected" in rb.columns:
            components.append(normalize(rb["rush_pct_over_expected"]) * 0.20)
            weights_used.append(0.20)
        if components:
            rb["tracking_score"] = sum(components) / sum(weights_used)
        else:
            rb["tracking_score"] = 50.0
        parts.append(rb[["player_id", "season", "week", "tracking_score"]].assign(position="RB"))

    # ── WR/TE tracking ─────────────────────────────────────────────
    if not ngs_recv.empty:
        recv = ngs_recv[ngs_recv["week"] > 0].copy()
        recv = recv.rename(columns={"player_gsis_id": "player_id"})
        components = []
        weights_used = []
        # Primary: separation (45%) — how open they get off the line
        if "avg_separation" in recv.columns:
            components.append(normalize(recv["avg_separation"]) * 0.45)
            weights_used.append(0.45)
        # Secondary: YAC above expectation (35%) — value after catch
        if "avg_yac_above_expectation" in recv.columns:
            components.append(normalize(recv["avg_yac_above_expectation"]) * 0.35)
            weights_used.append(0.35)
        # Tertiary: intended air yards (20%) — depth of route / target profile
        if "avg_intended_air_yards" in recv.columns:
            components.append(normalize(recv["avg_intended_air_yards"]) * 0.20)
            weights_used.append(0.20)
        if components:
            recv["tracking_score"] = sum(components) / sum(weights_used)
        else:
            recv["tracking_score"] = 50.0
        pos_col = recv["player_position"] if "player_position" in recv.columns else "WR"
        parts.append(recv[["player_id", "season", "week", "tracking_score"]].assign(position=pos_col))

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result["tracking_score"] = result["tracking_score"].clip(0, 100)
    return result


# ══════════════════════════════════════════════════════════════════
# 5. ATHLETICISM SCORE
# ══════════════════════════════════════════════════════════════════

COMBINE_WEIGHTS = {
    "QB":  {"forty": -0.4, "vertical": 0.3, "cone": -0.3},
    "RB":  {"forty": -0.5, "vertical": 0.2, "broad_jump": 0.3},
    "WR":  {"forty": -0.5, "vertical": 0.25, "broad_jump": 0.25},
    "TE":  {"forty": -0.3, "vertical": 0.3, "bench": 0.2, "broad_jump": 0.2},
    "OL":  {"bench": 0.5, "cone": -0.3, "shuttle": -0.2},
    "DL":  {"bench": 0.4, "forty": -0.3, "cone": -0.3},
    "LB":  {"forty": -0.4, "vertical": 0.3, "cone": -0.3},
    "CB":  {"forty": -0.5, "vertical": 0.3, "broad_jump": 0.2},
    "S":   {"forty": -0.4, "vertical": 0.3, "broad_jump": 0.3},
    "K":   {"bench": 0.5, "vertical": 0.3, "broad_jump": 0.2},
}

def build_athleticism_score(combine: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted combine score per player, normalized within position group.
    'forty' is reverse-scored (lower = faster = better).
    """
    combine = combine.copy()
    combine["pos"] = combine["pos"].str.upper().str.strip()

    results = []
    for pos, weights in COMBINE_WEIGHTS.items():
        sub = combine[combine["pos"] == pos].copy()
        if sub.empty:
            continue

        sub["raw_score"] = 0.0
        total_w = 0.0
        for col, w in weights.items():
            if col not in sub.columns:
                continue
            valid = sub[col].notna()
            if valid.sum() < 5:
                continue
            col_norm = normalize(sub[col], reverse=(w < 0))
            sub["raw_score"] += col_norm.fillna(50) * abs(w)
            total_w += abs(w)

        if total_w > 0:
            sub["raw_score"] /= total_w
        sub["athleticism_score"] = sub["raw_score"].clip(0, 100)
        results.append(sub[["player_name", "pos", "pfr_id", "athleticism_score"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# DEPTH & INJURY MULTIPLIERS
# ══════════════════════════════════════════════════════════════════

def get_depth_multipliers(depth: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Returns gsis_id -> depth_mult for a given week."""
    w = depth[(depth["season"] == season) & (depth["week"] == week)].copy()
    w["depth_mult"] = w["depth_team"].map(DEPTH_MULT).fillna(0.2)
    return w[["gsis_id", "depth_mult"]].drop_duplicates("gsis_id")


def get_injury_multipliers(injuries: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Returns gsis_id -> availability_mult for the given week's injury report."""
    w = injuries[(injuries["season"] == season) & (injuries["week"] == week)].copy()
    w["availability_mult"] = w["report_status"].map(INJURY_MULT).fillna(1.0)
    return w[["gsis_id", "availability_mult"]].drop_duplicates("gsis_id")


# ══════════════════════════════════════════════════════════════════
# MASTER: BUILD ALL COMPOSITE SCORES
# ══════════════════════════════════════════════════════════════════

def build_composite(season: int = None, week: int = None) -> pd.DataFrame:
    """
    Build composite scores for all players.
    If season/week specified, filters to that snapshot.

    Season recency weighting (Fix 2):
      Most recent season = 70% weight in rank/efficiency blending
      Prior season       = 20%
      2+ seasons ago     = 10% combined
    This means a player's 2024 EPA matters far more than their 2021 EPA.
    """
    print("Loading data...")
    stats          = pd.read_parquet(RAW / "player_stats.parquet")
    seasonal       = pd.read_parquet(RAW / "seasonal_stats.parquet")
    rosters_seas   = pd.read_parquet(RAW / "rosters_seasonal.parquet")
    snaps          = pd.read_parquet(RAW / "snap_counts.parquet")
    rosters        = pd.read_parquet(RAW / "rosters_weekly.parquet")
    depth          = pd.read_parquet(RAW / "depth_charts.parquet")

    # ── Normalize depth_charts schema ─────────────────────────────
    # Old schema (nflverse pre-2026): season, week, depth_team, club_code, gsis_id
    # New schema (nflverse 2026+):    dt, team, gsis_id, pos_slot, pos_rank (no season/week)
    if "season" not in depth.columns:
        # New schema — add synthetic season/week and map pos_slot → depth_team
        depth["season"] = season if season else 2025
        depth["week"]   = 1
        # pos_slot 1 = starter, 2 = backup, 3+ = depth
        depth["depth_team"] = depth["pos_slot"].fillna(2).clip(lower=1).astype(int)
        if "club_code" not in depth.columns and "team" in depth.columns:
            depth["club_code"] = depth["team"]
    if "gsis_id" not in depth.columns and "player_id" in depth.columns:
        depth["gsis_id"] = depth["player_id"]
    injuries       = pd.read_parquet(RAW / "injuries.parquet")
    ngs_pass       = pd.read_parquet(RAW / "ngs_passing.parquet")
    ngs_rush       = pd.read_parquet(RAW / "ngs_rushing.parquet")
    ngs_recv       = pd.read_parquet(RAW / "ngs_receiving.parquet")
    combine        = pd.read_parquet(RAW / "combine.parquet")
    player_ids     = pd.read_parquet(RAW / "player_ids.parquet")

    # New sources (graceful fallback if not yet fetched)
    def load_optional(name):
        p = RAW / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            print(f"  loaded {name}.parquet ({len(df):,} rows)")
            return df
        print(f"  SKIP {name}.parquet (not found -- run fetch_data.py)")
        return pd.DataFrame()

    ftn_data     = load_optional("ftn_data")
    pfr_passing  = load_optional("pfr_passing")
    pfr_receiving= load_optional("pfr_receiving")
    qbr_data     = load_optional("qbr")

    if season:
        # For player_stats and seasonal — fall back to most recent available season
        # if the requested season has no data (e.g. nflverse lag on 2025)
        available_stat_seasons = sorted(stats["season"].unique())
        stat_season = season if season in available_stat_seasons else (
            max(available_stat_seasons) if available_stat_seasons else season)
        if stat_season != season:
            print(f"  WARN: player_stats has no season {season}, using {stat_season} as proxy")

        stats        = stats[stats["season"] == stat_season]
        # rank_score is a PRIOR-season signal: a player's established positional skill
        # ENTERING the season. Ranking on the current season's full aggregates would leak
        # future games into every week (week 1 would already "know" the whole season).
        # So rank from completed prior seasons only; current-season form is captured
        # leak-free by efficiency_score (rolling, shift(1)).
        prior_seasonal = seasonal[seasonal["season"] < season]
        if not prior_seasonal.empty:
            seasonal = prior_seasonal          # recency-weighted across prior seasons
        else:
            # Earliest season in the dataset — no priors exist. Fall back to current
            # season (an unavoidable leak for the very first year of data only).
            available_sea_seasons = sorted(seasonal["season"].unique())
            sea_season = season if season in available_sea_seasons else (
                max(available_sea_seasons) if available_sea_seasons else season)
            seasonal = seasonal[seasonal["season"] == sea_season]
        rosters_seas = rosters_seas[rosters_seas["season"] == season] if season in rosters_seas["season"].values else rosters_seas[rosters_seas["season"] == rosters_seas["season"].max()]
        snaps        = snaps[snaps["season"] == season] if season in snaps["season"].values else snaps[snaps["season"] == snaps["season"].max()]
        rosters      = rosters[rosters["season"] == season] if season in rosters["season"].values else rosters[rosters["season"] == rosters["season"].max()]
        # Only filter depth by season if it has real season data (old schema)
        if "season" in depth.columns and depth["season"].nunique() > 1:
            depth = depth[depth["season"] == season]
        injuries     = injuries[injuries["season"] == season] if season in injuries["season"].values else injuries[injuries["season"] == injuries["season"].max()]
        ngs_pass     = ngs_pass[ngs_pass["season"] == season]
        ngs_rush     = ngs_rush[ngs_rush["season"] == season]
        ngs_recv     = ngs_recv[ngs_recv["season"] == season]
        if not ftn_data.empty and "season" in ftn_data.columns:
            ftn_data = ftn_data[ftn_data["season"] == season]
        if not pfr_passing.empty and "season" in pfr_passing.columns:
            pfr_passing = pfr_passing[pfr_passing["season"] == season]
        if not pfr_receiving.empty and "season" in pfr_receiving.columns:
            pfr_receiving = pfr_receiving[pfr_receiving["season"] == season]
        if not qbr_data.empty and "season" in qbr_data.columns:
            qbr_data = qbr_data[qbr_data["season"] == season]

    if week:
        stats    = stats[stats["week"] <= week]
        # Old schema has week column; new schema is a live snapshot (use all rows)
        depth_w  = depth[depth["week"] == week] if depth["week"].nunique() > 1 else depth
        inj_w    = injuries[injuries["week"] == week]
    else:
        # Old schema: take last week per player; new schema: already one row per player
        if depth["week"].nunique() > 1:
            depth_w = depth.sort_values("week").groupby(["season","gsis_id"]).last().reset_index()
        else:
            depth_w = depth.drop_duplicates("gsis_id")
        inj_w    = injuries.sort_values("week").groupby(["season","gsis_id"]).last().reset_index()

    print("Building component scores...")

    # ── Season recency weighting (Fix 2) ───────────────────────────
    # When building rank scores, give recent seasons more weight.
    # We do this by replicating rows weighted by season recency,
    # then normalising so the rank/EPA scores reflect current form.
    #
    # Weights:  most recent season = 0.70
    #           prior season       = 0.20
    #           2+ seasons ago     = 0.10 combined (split evenly)
    #
    # Implementation: create a recency_weight column on seasonal stats,
    # then use it as a multiplier when computing positional EPA for ranking.

    all_seasons = sorted(seasonal["season"].unique())
    n = len(all_seasons)

    if n >= 1:
        recency_map = {}
        if n == 1:
            recency_map[all_seasons[-1]] = 1.0
        elif n == 2:
            recency_map[all_seasons[-1]] = 0.70
            recency_map[all_seasons[-2]] = 0.30
        else:
            recency_map[all_seasons[-1]] = 0.70
            recency_map[all_seasons[-2]] = 0.20
            older_weight = 0.10 / (n - 2)
            for s in all_seasons[:-2]:
                recency_map[s] = older_weight

        seasonal = seasonal.copy()
        seasonal["recency_weight"] = seasonal["season"].map(recency_map).fillna(0.05)
    else:
        seasonal["recency_weight"] = 1.0

    rank_df   = build_rank_score(seasonal, rosters_seas)
    # Broadcast the prior-season rank onto the target season's weekly rows.
    # rank_df is labeled with each player's latest *prior* season; stamp the target
    # season so the player_id+season merge below lands on this season's rows.
    if season and not rank_df.empty:
        rank_df["season"] = season
    eff_df    = build_efficiency_score(
                    stats,
                    pfr_passing   = pfr_passing   if not pfr_passing.empty   else None,
                    qbr_data      = qbr_data       if not qbr_data.empty       else None,
                    pfr_receiving = pfr_receiving  if not pfr_receiving.empty  else None,
                    ftn_data      = ftn_data        if not ftn_data.empty        else None,
                )
    usage_df  = build_usage_score(stats, snaps, rosters)
    track_df  = build_tracking_score(ngs_pass, ngs_rush, ngs_recv)
    ath_df    = build_athleticism_score(combine)

    # ── Join athleticism via pfr_id crosswalk ──────────────────────
    # player_ids has gsis_id + pfr_id
    id_map = player_ids[["gsis_id", "pfr_id", "name"]].dropna(subset=["gsis_id"])
    if not ath_df.empty and "pfr_id" in ath_df.columns:
        ath_df = ath_df.merge(id_map[["gsis_id", "pfr_id"]], on="pfr_id", how="left")
        ath_df = ath_df.rename(columns={"gsis_id": "player_id"})

    # ── Base: player_stats gives us the player list ────────────────
    base_cols = ["player_id", "player_display_name", "position",
                 "recent_team", "season", "week"]
    base_cols = [c for c in base_cols if c in stats.columns]
    base = stats[base_cols].drop_duplicates().copy()

    # If we used a proxy season (e.g. 2024 stats for 2025), stamp the
    # requested season on base rows and update team assignments from
    # current rosters so predictions use the right team
    if season and "season" in base.columns and base["season"].nunique() == 1:
        proxy_yr = int(base["season"].iloc[0])
        if proxy_yr != season:
            base["season"] = season
            # Update recent_team from current season rosters
            if not rosters.empty and "player_id" in rosters.columns and "team" in rosters.columns:
                current_team = (rosters
                    .sort_values("week", ascending=False)
                    .drop_duplicates("player_id")[["player_id", "team"]]
                    .rename(columns={"team": "current_team"}))
                base = base.merge(current_team, on="player_id", how="left")
                base["recent_team"] = base["current_team"].fillna(base["recent_team"])
                base = base.drop(columns=["current_team"])

    # ── Merge rank (season-level → broadcast to all weeks) ─────────
    if not rank_df.empty:
        base = base.merge(
            rank_df[["player_id", "season", "rank_score", "pos_rank"]],
            on=["player_id", "season"], how="left"
        )
    else:
        base["rank_score"] = 50.0
        base["pos_rank"] = 999

    # ── Merge efficiency ───────────────────────────────────────────
    if not eff_df.empty:
        base = base.merge(
            eff_df[["player_id", "season", "week", "efficiency_score"]],
            on=["player_id", "season", "week"], how="left"
        )
    else:
        base["efficiency_score"] = 50.0

    # ── Merge usage ────────────────────────────────────────────────
    if not usage_df.empty:
        base = base.merge(
            usage_df[["player_id", "season", "week", "usage_score"]],
            on=["player_id", "season", "week"], how="left"
        )
    else:
        base["usage_score"] = 50.0

    # ── Merge tracking (week-level, may be sparse) ─────────────────
    if not track_df.empty:
        base = base.merge(
            track_df[["player_id", "season", "week", "tracking_score"]],
            on=["player_id", "season", "week"], how="left"
        )
    else:
        base["tracking_score"] = 50.0

    # ── Merge athleticism (career-level, no season/week) ───────────
    if not ath_df.empty and "player_id" in ath_df.columns:
        base = base.merge(
            ath_df[["player_id", "athleticism_score"]],
            on="player_id", how="left"
        )
    else:
        base["athleticism_score"] = 50.0

    # ── Fill missing component scores with position median ─────────
    for col in ["rank_score", "efficiency_score", "usage_score",
                "tracking_score", "athleticism_score"]:
        if col not in base.columns:
            base[col] = 50.0
        base[col] = base.groupby("position")[col].transform(
            lambda x: x.fillna(x.median())
        ).fillna(50.0)

    # ── Weighted composite ─────────────────────────────────────────
    base["composite_score"] = sum(
        base[comp] * w for comp, w in WEIGHTS.items()
    ).clip(0, 100)

    # ── Depth multiplier ───────────────────────────────────────────
    # Only apply depth penalty to players confirmed as backups
    # Players not in depth chart are likely starters (data gap) → default 1.0
    depth_mult = depth_w[["gsis_id", "depth_team"]].drop_duplicates("gsis_id").copy()
    depth_mult["depth_mult"] = depth_mult["depth_team"].map(DEPTH_MULT).fillna(1.0)
    base = base.merge(
        depth_mult.rename(columns={"gsis_id": "player_id"})[["player_id", "depth_mult"]],
        on="player_id", how="left"
    )
    base["depth_mult"] = base["depth_mult"].fillna(1.0)  # default: treat as starter

    # ── Injury multiplier ──────────────────────────────────────────
    inj_mult = inj_w[["gsis_id", "report_status"]].drop_duplicates("gsis_id").copy()
    inj_mult["availability_mult"] = inj_mult["report_status"].map(INJURY_MULT).fillna(1.0)
    base = base.merge(
        inj_mult.rename(columns={"gsis_id": "player_id"})[["player_id", "availability_mult"]],
        on="player_id", how="left"
    )
    base["availability_mult"] = base["availability_mult"].fillna(1.0)

    # ── Final adjusted score ───────────────────────────────────────
    base["adjusted_score"] = (
        base["composite_score"]
        * base["depth_mult"]
        * base["availability_mult"]
    ).clip(0, 100).round(1)

    # ── Tier assignment ────────────────────────────────────────────
    def assign_tier(score):
        if score >= 80: return "Elite"
        if score >= 65: return "Above Avg"
        if score >= 45: return "Average"
        if score >= 25: return "Below Avg"
        return "Poor"

    base["tier"] = base["adjusted_score"].apply(assign_tier)

    print(f"Composite scores built: {len(base):,} player-weeks")

    # Save individual season file for reference, but also return the dataframe
    # so run_engine.py can merge multiple seasons before writing composite_scores.parquet
    out_path = PROC / "composite_scores.parquet"
    base.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    return base


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--week",   type=int, default=None)
    args = parser.parse_args()
    df = build_composite(args.season, args.week)
    print("\nSample output:")
    cols = ["player_display_name", "position", "recent_team", "week",
            "composite_score", "adjusted_score", "tier", "pos_rank"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values("adjusted_score", ascending=False).head(20).to_string(index=False))
