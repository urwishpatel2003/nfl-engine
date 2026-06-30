"""
engine/matchups.py
------------------
Builds the positional matchup matrix for a given game.

For each game, we match every offensive player against the
defensive player they'll face based on formation position:

  QB         vs  pass_rush (DL/LB sack rate composite)
  WR1        vs  CB1  (top corner by composite score)
  WR2        vs  CB2
  WR3/slot   vs  slot CB / nickel
  TE         vs  LB / safety in coverage
  RB (pass)  vs  LB coverage
  OL         vs  DL (run blocking matchup)
  RB (run)   vs  run defense unit

Each matchup produces:
  - off_score:       offensive player composite score
  - def_score:       defensive player composite score
  - matchup_edge:    off_score - def_score  (+= offense wins)
  - matchup_grade:   "Clear Advantage" / "Slight Edge" / "Even" /
                     "Slight Disadvantage" / "Clear Disadvantage"
  - key_matchup:     True if this matchup heavily influences game outcome
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"


MATCHUP_GRADE = {
    (12, 100):  "Clear Advantage",
    (4,   12):  "Slight Edge",
    (-4,   4):  "Even",
    (-12,  -4): "Slight Disadvantage",
    (-100,-12): "Clear Disadvantage",
}

def grade(edge: float) -> str:
    edge = float(np.clip(edge, -30, 30))   # cap before grading
    for (lo, hi), label in MATCHUP_GRADE.items():
        if lo <= edge < hi:
            return label
    return "Even"


# ── Positional matchup slot definitions ───────────────────────────
# Maps depth chart formation_position values to matchup role
OFF_SLOTS = {
    "QB":    ["QB"],
    "WR1":   ["WR", "WR1", "X"],
    "WR2":   ["WR2", "Z"],
    "SLOT":  ["SWR", "SL", "SLOT", "H"],
    "TE":    ["TE", "TE1", "TE2", "Y"],
    "RB":    ["RB", "HB", "FB"],
    "LT":    ["LT"],
    "LG":    ["LG"],
    "C":     ["C"],
    "RG":    ["RG"],
    "RT":    ["RT"],
}

DEF_SLOTS = {
    "PASS_RUSH": ["LE", "RE", "DT", "NT", "LOLB", "ROLB", "MLB"],
    "CB1":       ["LCB", "RCB", "CB"],
    "CB2":       ["LCB", "RCB", "CB"],
    "SLOT_CB":   ["SCB", "SS", "FS"],
    "LB_COV":    ["LOLB", "ROLB", "MLB", "ILB"],
    "SAFETY":    ["SS", "FS", "SAF"],
    "RUN_DEF":   ["LE", "RE", "DT", "NT", "MLB", "LOLB", "ROLB"],
}

# Which defensive slot opposes each offensive slot
MATCHUP_MAP = {
    "QB":   "PASS_RUSH",
    "WR1":  "CB1",
    "WR2":  "CB2",
    "SLOT": "SLOT_CB",
    "TE":   "LB_COV",
    "RB":   "RUN_DEF",
}

# How much each matchup slot influences the game outcome (weights)
MATCHUP_IMPORTANCE = {
    "QB":   0.35,
    "WR1":  0.22,
    "WR2":  0.13,
    "SLOT": 0.10,
    "TE":   0.08,
    "RB":   0.12,   # lower — RB vs run D is real but less decisive than passing game
}


def get_team_roster(composite: pd.DataFrame, depth: pd.DataFrame,
                    team: str, season: int, week: int,
                    side: str = "offense") -> pd.DataFrame:
    """
    Get ranked players for a team at a given week.
    side: 'offense' or 'defense'
    Returns players sorted by adjusted_score desc with depth info.
    """
    # Get composite scores for this team/week
    players = composite[
        (composite["recent_team"] == team) &
        (composite["season"] == season) &
        (composite["week"] == week)
    ].copy()

    # Get depth chart for this week — handle both old and new schema
    # Old schema: club_code, season, week columns
    # New schema (nflverse 2026+): team column only, no season/week
    team_col_depth = "club_code" if "club_code" in depth.columns else "team"
    has_season_week = "season" in depth.columns and "week" in depth.columns

    if has_season_week:
        d = depth[
            (depth[team_col_depth] == team) &
            (depth["season"] == season) &
            (depth["week"] == week)
        ].copy()
        # Fallback to any week if exact week not found
        if d.empty:
            d = depth[
                (depth[team_col_depth] == team) &
                (depth["season"] == season)
            ].sort_values("week", ascending=False).head(25).copy()
    else:
        # New schema — just filter by team
        d = depth[depth[team_col_depth] == team].copy()

    if players.empty:
        # Fall back to most recent available week
        players = composite[
            (composite["recent_team"] == team) &
            (composite["season"] == season)
        ].sort_values("week", ascending=False).drop_duplicates("player_id")

    return players.sort_values("adjusted_score", ascending=False)


def get_top_player(players: pd.DataFrame, positions: list,
                    rank: int = 1) -> pd.Series | None:
    """Get the nth-ranked player at given positions."""
    filtered = players[players["position"].isin(positions)].sort_values(
        "adjusted_score", ascending=False
    )
    if len(filtered) >= rank:
        return filtered.iloc[rank - 1]
    return None


def build_game_matchups(home_team: str, away_team: str,
                         season: int, week: int,
                         composite: pd.DataFrame = None,
                         depth: pd.DataFrame = None,
                         styles: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build the full positional matchup matrix for a single game.

    Returns DataFrame with one row per matchup slot containing:
    off_team, def_team, slot, off_player, def_player,
    off_score, def_score, matchup_edge, matchup_grade, importance, key_matchup
    """
    if composite is None:
        composite = pd.read_parquet(PROC / "composite_scores.parquet")
    if depth is None:
        depth = pd.read_parquet(RAW / "depth_charts.parquet")
    if styles is None:
        try:
            styles = pd.read_parquet(PROC / "team_styles.parquet")
        except FileNotFoundError:
            styles = None

    rows = []

    # Build matchups for both directions: home offense vs away defense, and vice versa
    for off_team, def_team in [(home_team, away_team), (away_team, home_team)]:
        off_players = get_team_roster(composite, depth, off_team, season, week)
        def_players = get_team_roster(composite, depth, def_team, season, week)

        if off_players.empty or def_players.empty:
            continue

        for slot, def_slot in MATCHUP_MAP.items():
            importance = MATCHUP_IMPORTANCE.get(slot, 0.10)

            # ── Offensive player ──────────────────────────────────
            if slot == "QB":
                off_p = get_top_player(off_players, ["QB"])
            elif slot == "WR1":
                off_p = get_top_player(off_players, ["WR"], rank=1)
            elif slot == "WR2":
                off_p = get_top_player(off_players, ["WR"], rank=2)
            elif slot == "SLOT":
                off_p = get_top_player(off_players, ["WR"], rank=3)
            elif slot == "TE":
                off_p = get_top_player(off_players, ["TE"])
            elif slot == "RB":
                off_p = get_top_player(off_players, ["RB"])
            else:
                off_p = None

            # ── Defensive player ──────────────────────────────────
            # Position data in composite comes from player_stats which
            # only has offensive positions. Use style EPA as proxy for
            # all defensive slots, scaled to a 0-100 score.
            def_p = None
            def_score_from_style = 50.0
            if styles is not None:
                def_style = styles[
                    (styles["team"] == def_team) & (styles["season"] == season)
                ]
                if def_style.empty:
                    team_hist = styles[styles["team"] == def_team]
                    if not team_hist.empty:
                        def_style = team_hist.sort_values("season", ascending=False).head(1)
                if not def_style.empty:
                    ds = def_style.iloc[0]
                    if slot == "QB":
                        # Sack rate: league avg ~6.5%, elite ~9%, poor ~4%
                        # Map to 0-100: 4%=30, 6.5%=50, 9%=75, 12%=90
                        sack_r = float(ds.get("sack_rate", 0.065) or 0.065)
                        def_score_from_style = np.clip(50 + (sack_r - 0.065) * 1500, 25, 90)
                    elif slot in ("WR1", "WR2", "SLOT"):
                        # def_epa_per_pass: league avg ~0, elite ~+0.08, poor ~-0.08
                        # Positive = defense stops passes (good for defense)
                        epa_p = float(ds.get("def_epa_per_pass", 0) or 0)
                        def_score_from_style = np.clip(50 + epa_p * 300, 25, 85)
                    elif slot == "TE":
                        epa_p = float(ds.get("def_epa_per_pass", 0) or 0)
                        def_score_from_style = np.clip(50 + epa_p * 250, 25, 80)
                    elif slot == "RB":
                        # def_epa_per_rush: league avg ~0, elite ~+0.06, poor ~-0.06
                        epa_r = float(ds.get("def_epa_per_rush", 0) or 0)
                        def_score_from_style = np.clip(50 + epa_r * 350, 25, 85)

            off_score = float(off_p["adjusted_score"]) if off_p is not None else 40.0
            off_name  = str(off_p["player_display_name"]) if off_p is not None else f"{off_team} {slot}"
            off_pos   = str(off_p["position"]) if off_p is not None else slot

            def_score = float(np.clip(def_score_from_style, 0, 100))
            def_name  = f"{def_team} {def_slot}"
            def_pos   = "DEF"
            edge = round(off_score - def_score, 1)

            rows.append({
                "game_key":       f"{away_team}@{home_team}",
                "season":         season,
                "week":           week,
                "off_team":       off_team,
                "def_team":       def_team,
                "matchup_slot":   slot,
                "def_slot":       def_slot,
                "off_player":     off_name,
                "off_position":   off_pos,
                "off_score":      round(off_score, 1),
                "def_player":     def_name,
                "def_position":   def_pos,
                "def_score":      round(def_score, 1),
                "matchup_edge":   round(float(np.clip(edge, -30, 30)), 1),
                "matchup_grade":  grade(edge),
                "importance":     importance,
                "key_matchup":    abs(edge) >= 10,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Add style clash context if available
    if styles is not None and not styles.empty:
        df = add_style_context(df, styles, season)

    return df


def add_style_context(matchups: pd.DataFrame, styles: pd.DataFrame,
                       season: int) -> pd.DataFrame:
    """Annotate matchups with relevant style clash information."""
    from engine.styles import get_style_clash_score

    style_notes = []
    for _, row in matchups.iterrows():
        off_s = styles[(styles["team"] == row["off_team"]) & (styles["season"] == season)]
        def_s = styles[(styles["team"] == row["def_team"]) & (styles["season"] == season)]

        if off_s.empty:
            team_hist = styles[styles["team"] == row["off_team"]]
            if not team_hist.empty:
                off_s = team_hist.sort_values("season", ascending=False).head(1)
        if def_s.empty:
            team_hist = styles[styles["team"] == row["def_team"]]
            if not team_hist.empty:
                def_s = team_hist.sort_values("season", ascending=False).head(1)

        if off_s.empty or def_s.empty:
            style_notes.append("")
            continue

        clashes = get_style_clash_score(off_s.iloc[0], def_s.iloc[0])
        note = " | ".join(f"{k}: {v:+d}" for k, v in clashes.items()) if clashes else ""
        style_notes.append(note)

    matchups["style_clash_notes"] = style_notes
    return matchups


def summarize_game_matchups(matchups: pd.DataFrame) -> dict:
    """
    Summarize which team has the overall matchup advantage.
    Returns dict with advantage scores and key matchups.
    """
    if matchups.empty:
        return {}

    results = {}
    for off_team in matchups["off_team"].unique():
        team_m = matchups[matchups["off_team"] == off_team]
        weighted_edge = (
            team_m["matchup_edge"] * team_m["importance"]
        ).sum() / team_m["importance"].sum()

        key_matchups = team_m[team_m["key_matchup"]].to_dict("records")

        results[off_team] = {
            "weighted_matchup_edge": round(weighted_edge, 1),
            "clear_advantages":   int((team_m["matchup_grade"] == "Clear Advantage").sum()),
            "clear_disadvantages":int((team_m["matchup_grade"] == "Clear Disadvantage").sum()),
            "key_matchups":       key_matchups,
        }

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--home",   default="KC")
    parser.add_argument("--away",   default="BUF")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--week",   type=int, default=18)
    args = parser.parse_args()

    comp_path = PROC / "composite_scores.parquet"
    if not comp_path.exists():
        print("Run composite.py first to generate composite_scores.parquet")
        exit(1)

    composite = pd.read_parquet(comp_path)
    depth     = pd.read_parquet(RAW / "depth_charts.parquet")

    matchups = build_game_matchups(
        args.home, args.away, args.season, args.week,
        composite=composite, depth=depth
    )

    print(f"\n{args.away} @ {args.home} -- Week {args.week} {args.season}")
    print("=" * 80)
    print(matchups[[
        "off_team", "matchup_slot", "off_player", "off_score",
        "def_player", "def_score", "matchup_edge", "matchup_grade"
    ]].to_string(index=False))

    print("\nGame summary:")
    summary = summarize_game_matchups(matchups)
    for team, s in summary.items():
        print(f"  {team}: weighted edge = {s['weighted_matchup_edge']:+.1f} "
              f"| advantages: {s['clear_advantages']} | disadvantages: {s['clear_disadvantages']}")
