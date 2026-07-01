"""
ml/squad.py  —  roster-talent power ratings for the upcoming season
====================================================================
Rates each team by the talent CURRENTLY on its roster + coaching, NOT by last
year's results (offseason roster/coaching turnover makes prior results stale;
2025->2026 strength persistence is only r=0.35). This is the preseason product
ranking the user asked for ("based on squad and coaching strength").

Method — map the current roster to each player's 2025 performance, z-score within
position group, aggregate the group talents with weights, add coaching:

  QB        composite (gsis; name fallback)     weight 0.34   -- dominates
  WR/TE/RB  composite (top starters)            weight 0.18
  OL        team 2025 pass-block/run-block      weight 0.12   -- team-level (continuity)
  pass rush current-roster DL 2025 pressure     weight 0.14   -- name-matched
  coverage  current-roster DB 2025 (pfr)        weight 0.14   -- name-matched
  coaching  2025 coaching_score (neutral=new)   weight 0.08

Data caveats: offense maps cleanly; defense is name-matched (noisier); OL is only
team-level; rookies/no-2025 default to replacement. Honest about being a projection.

Usage:
  python -m ml.squad                 # 2026 roster-talent ranking
  python -m ml.squad --breakdown     # show per-team component scores
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

# QB is the dominant position in the NFL, so it carries the most weight. Coaching is
# down-weighted: it's noisy, partly results-derived, and blind to 2026 staff changes.
WEIGHTS = {"qb": 0.40, "skill": 0.17, "ol": 0.10, "rush": 0.12, "cover": 0.13, "coach": 0.05}
RATING_SCALE = 9.0   # maps the blended z-score to ~points; sets the spread of the ranking


def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def _key(name):
    """Robust match key: lastname + first initial (handles Greg/Gregory, suffixes)."""
    parts = [p for p in re.sub(r"[^a-z ]", "", str(name).lower()).split() if p not in _SUFFIX]
    if not parts:
        return ""
    return parts[-1] + (parts[0][0] if parts[0] else "")


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0


# ── player-quality tables (2025) ────────────────────────────────────
def _composite_2025():
    # Use season-MEAN composite_score (pure talent) not last-week adjusted_score, which
    # multiplies in end-of-season injury/depth status (e.g. it zeroed out Jayden Daniels).
    c = pd.read_parquet(PROC / "composite_scores.parquet")
    c = c[c.season == 2025]
    talent_col = "composite_score" if "composite_score" in c.columns else "adjusted_score"
    agg = c.groupby("player_id").agg(adjusted_score=(talent_col, "mean"),
                                     name=("player_display_name", "last"),
                                     position=("position", "last")).reset_index()
    agg["nm"] = agg["name"].map(_norm)
    agg["k"] = agg["name"].map(_key)
    return agg[["player_id", "nm", "k", "position", "adjusted_score"]]


def _qb_value_table():
    """Multi-year QB value = (passing + rushing) EPA per play, recency+volume weighted.
    Far wider spread than the composite, so elite QBs actually separate from replacement."""
    ss = pd.read_parquet(RAW / "seasonal_stats.parquet")
    ss = ss[ss.season.isin([2023, 2024, 2025]) & (ss.attempts >= 100)].copy()
    ss["epa_play"] = (ss["passing_epa"].fillna(0) + ss["rushing_epa"].fillna(0)) / ss["attempts"].clip(lower=1)
    w = {2025: 0.5, 2024: 0.35, 2023: 0.15}
    ss["w"] = ss["season"].map(w) * ss["attempts"]          # weight by recency AND volume
    return ss.groupby("player_id").apply(
        lambda x: (x.epa_play * x.w).sum() / x.w.sum(), include_groups=False)


def _qb_starter():
    """Per-team QB1 (from the current depth chart) rated by multi-year EPA/play.
    Rookies / no-history default to the 10th-percentile (replacement) value."""
    dc = pd.read_parquet(RAW / "depth_2026_current.parquet")
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    qb1 = dc[(dc.pos_abb == "QB") & (dc.pos_rank == 1)].drop_duplicates("team")[["team", "gsis_id"]]
    val = _qb_value_table()
    qb1["qb"] = qb1["gsis_id"].map(val).fillna(float(val.quantile(0.10)))
    return qb1.set_index("team")["qb"]


def _team_rosters_2026():
    r = pd.read_parquet(RAW / "rosters_2026.parquet")
    r["nm"] = r["player_name"].map(_norm)
    r["nm_full"] = r["player_name"]
    return r[["team", "player_id", "nm", "nm_full", "position"]]


def _offense_group(roster, comp, positions, topn):
    """Best `topn` current-roster players at `positions`, by 2025 composite (id then name)."""
    sub = roster[roster.position.isin(positions)].copy()
    m = sub.merge(comp[["player_id", "adjusted_score"]], on="player_id", how="left")
    # name fallback for unmatched (id mismatches)
    miss = m["adjusted_score"].isna()
    if miss.any():
        byname = comp.dropna(subset=["nm"]).drop_duplicates("nm").set_index("nm")["adjusted_score"]
        m.loc[miss, "adjusted_score"] = m.loc[miss, "nm"].map(byname)
    m["adjusted_score"] = m["adjusted_score"].fillna(comp["adjusted_score"].quantile(0.15))
    return m.sort_values("adjusted_score", ascending=False).groupby("team").head(topn) \
            .groupby("team")["adjusted_score"].mean()


def _pass_rush(roster):
    dl = pd.read_csv(PROC / "dl_rankings_2025.csv")
    dl["k"] = dl["name"].map(_key)
    dl["prod"] = dl.get("def_sacks", 0).fillna(0) + 0.5 * dl.get("def_pressures", 0).fillna(0)
    bykey = dl.groupby("k")["prod"].max()
    sub = roster[roster.position.isin(["DL", "LB"])].copy()   # edge rushers often listed as LB
    sub["k"] = sub["nm_full"].map(_key)
    sub["prod"] = sub["k"].map(bykey).fillna(0.0)
    return sub.sort_values("prod", ascending=False).groupby("team").head(5).groupby("team")["prod"].sum()


def _coverage(roster):
    pf = pd.read_parquet(RAW / "pfr_defense.parquet")
    pf = pf[pf.season == 2025]
    agg = pf.groupby("pfr_player_name").agg(tgt=("def_targets", "sum"),
                                            rate=("def_passer_rating_allowed", "mean")).reset_index()
    agg = agg[agg.tgt >= 20]
    agg["k"] = agg["pfr_player_name"].map(_key)
    agg["cov"] = -agg["rate"]                      # lower passer rating allowed = better
    bykey = agg.groupby("k")["cov"].mean()
    sub = roster[roster.position.isin(["DB", "LB"])].copy()
    sub["cov"] = sub["nm_full"].map(_key).map(bykey)
    good = sub.dropna(subset=["cov"])
    return good.sort_values("cov", ascending=False).groupby("team").head(5).groupby("team")["cov"].mean()


def _ol():
    ol = pd.read_csv(PROC / "ol_rankings_2025.csv").set_index("team")
    return -_z(ol["sack_rate_allowed"]) + _z(ol.get("ypc", pd.Series(0, index=ol.index)))


def _coaching():
    # multi-year (2023-2025) mean, so a single strong season doesn't spike a team
    c = pd.read_parquet(PROC / "coaching_scores.parquet")
    c = c[c.season.isin([2023, 2024, 2025])]
    return c.groupby("team")["coaching_score"].mean()


# ── assemble ────────────────────────────────────────────────────────
def squad_ratings(breakdown: bool = False) -> pd.DataFrame:
    roster = _team_rosters_2026()
    comp = _composite_2025()
    teams = sorted(roster.team.unique())

    g = pd.DataFrame(index=teams)
    g["qb"]    = _qb_starter()                                      # depth-chart starter, EPA/play
    g["skill"] = _offense_group(roster, comp, ["WR", "RB", "TE"], 5)
    g["ol"]    = _ol()
    g["rush"]  = _pass_rush(roster)
    g["cover"] = _coverage(roster)
    g["coach"] = _coaching()
    # replacement-fill any team missing a group, then z-score each group
    for c in g.columns:
        g[c] = g[c].fillna(g[c].median())
    z = g.apply(_z)

    z["blend"] = sum(z[c] * w for c, w in WEIGHTS.items())
    z["rating"] = (z["blend"] * RATING_SCALE).round(1)
    out = z.sort_values("rating", ascending=False).reset_index().rename(columns={"index": "team"})
    out.insert(0, "rank", out.index + 1)
    return (out if breakdown else out[["rank", "team", "rating"]]), g


HFA = 2.0            # home-field points
SPREAD_SCALE = 0.9   # map roster-talent rating difference to a point spread


def predict_matchup(home: str, away: str, neutral: bool = False) -> dict:
    """Predict a matchup from the SAME roster-talent ratings as the power rankings,
    so a #1 team is favored over a #16 team (previously it used a separate SRS model)."""
    out, _ = squad_ratings()
    r = out.set_index("team")["rating"]
    if home not in r.index or away not in r.index:
        return {"error": "unknown team(s)"}
    hfa = 0.0 if neutral else HFA
    margin = float(np.clip((r[home] - r[away]) * SPREAD_SCALE + hfa, -18, 18))
    total = 44.0        # league-average total; scores split around the margin
    home_pts, away_pts = (total + margin) / 2, (total - margin) / 2
    wp = float(1 / (1 + np.exp(-margin / 13.5 * np.pi / np.sqrt(3))))
    return {
        "home": home, "away": away,
        "pred_home_score": round(home_pts, 1), "pred_away_score": round(away_pts, 1),
        "pred_margin": round(margin, 1), "pred_total": round(total, 1),
        "home_win_prob": round(wp, 3), "away_win_prob": round(1 - wp, 3),
    }


# ── team depth chart + per-player ratings (for the dashboard team page) ──
# The depth chart uses granular slots (LT, RG, LDE, RCB, ...); map them to groups.
GROUP_MAP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
             "C": "OL", "LG": "OL", "RG": "OL", "LT": "OL", "RT": "OL",
             "LDE": "DL", "RDE": "DL", "LDT": "DL", "RDT": "DL", "NT": "DL",
             "MLB": "LB", "LILB": "LB", "RILB": "LB", "SLB": "LB", "WLB": "LB",
             "LCB": "DB", "RCB": "DB", "NB": "DB", "FS": "DB", "SS": "DB",
             "PK": "K", "P": "P", "LS": "LS"}
POS_ORDER = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P", "LS"]
POS_LABEL = {"QB": "Quarterback", "RB": "Running Back", "WR": "Wide Receiver",
             "TE": "Tight End", "OL": "Offensive Line", "DL": "Defensive Line",
             "LB": "Linebacker", "DB": "Defensive Back", "K": "Kicker",
             "P": "Punter", "LS": "Long Snapper"}


_PCT_CACHE = None


def _player_pct():
    """Per-player 0-100 rating = percentile within position (2025). Comparable across
    positions: 90+=elite, ~60-80=solid starter, <40=depth. Returns lookups by id/key."""
    global _PCT_CACHE
    if _PCT_CACHE is not None:
        return _PCT_CACHE
    comp = _composite_2025()
    comp["pct"] = comp.groupby("position")["adjusted_score"].rank(pct=True) * 100
    skill = comp.set_index("player_id")["pct"]                       # WR/RB/TE (+QB fallback)
    qb = _qb_value_table(); qb_pct = qb.rank(pct=True) * 100         # QB by multi-year EPA
    dl = pd.read_csv(PROC / "dl_rankings_2025.csv"); dl["k"] = dl["name"].map(_key)
    dl["prod"] = dl.get("def_sacks", 0).fillna(0) + 0.5 * dl.get("def_pressures", 0).fillna(0)
    dl_pct = dl.groupby("k")["prod"].max().rank(pct=True) * 100      # DL by pass-rush
    pf = pd.read_parquet(RAW / "pfr_defense.parquet"); pf = pf[pf.season == 2025]
    cov = pf.groupby("pfr_player_name").agg(tgt=("def_targets", "sum"),
                                            rate=("def_passer_rating_allowed", "mean")).reset_index()
    cov = cov[cov.tgt >= 20]; cov["k"] = cov["pfr_player_name"].map(_key)
    cov_pct = (-cov.groupby("k")["rate"].mean()).rank(pct=True) * 100  # DB/LB by coverage
    _PCT_CACHE = (skill, qb_pct, dl_pct, cov_pct)
    return _PCT_CACHE


def team_depth_chart(team: str) -> list:
    dc = pd.read_parquet(RAW / "depth_2026_current.parquet").copy()
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    dc = dc[(dc.team == team) & dc.player_name.notna()].copy()
    dc["grp"] = dc["pos_abb"].map(GROUP_MAP)
    skill, qb_pct, dl_pct, cov_pct = _player_pct()

    def rate(grp, gid, name):
        k = _key(name)
        if grp == "QB":       v = qb_pct.get(gid, skill.get(gid))
        elif grp in ("WR", "RB", "TE"): v = skill.get(gid)
        elif grp == "DL":     v = dl_pct.get(k)
        elif grp in ("DB", "LB"): v = cov_pct.get(k)
        else:                 v = None
        return None if v is None or pd.isna(v) else int(round(v))

    groups = []
    for grp in POS_ORDER:
        # starters first (rank 1 of each slot), then depth; group like slots together
        sub = dc[dc.grp == grp].sort_values(["pos_rank", "pos_abb"])
        if sub.empty:
            continue
        players = [{"name": r.player_name, "slot": r.pos_abb,
                    "rank": int(r.pos_rank) if pd.notna(r.pos_rank) else None,
                    "rating": rate(grp, r.gsis_id, r.player_name)}
                   for r in sub.itertuples()]
        groups.append({"pos": grp, "label": POS_LABEL.get(grp, grp), "players": players})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breakdown", action="store_true")
    args = ap.parse_args()
    out, raw = squad_ratings(breakdown=args.breakdown)
    print(f"\n  2026 ROSTER-TALENT RANKINGS  (current squad + coaching, not 2025 results)")
    print(f"  {'-'*52}")
    for _, x in out.iterrows():
        bar = "#" * max(0, int(round(x["rating"] + 12)))
        print(f"  {x['rank']:2d}. {x['team']:<3} {x['rating']:+5.1f}  {bar}")
    print()
    if args.breakdown:
        print(raw.round(1).to_string())


if __name__ == "__main__":
    main()
