"""
ml/matchup_context.py  —  injury-to-unit routing + scheme (play-caller) matchups
================================================================================
Two second-order effects the flat injury penalty and unit ratings miss:

1. INJURY → UNIT.  An 'Out' player weakens the specific unit he plays in, so the matchup
   engine's unit-vs-unit multiplication produces the interaction: a run-stuffer out
   downgrades def_rush, which then loses *harder* to a strong rushing offense, and barely
   matters against a pass-first team. Returns per-team deltas to the unit z-scores that
   project_game consumes (defense uses the EPA-allowed convention: + = weaker/allows more;
   offense: − = weaker).

2. SCHEME MATCHUP.  A play-caller's scheme IS his team's style. team_styles.parquet carries
   play-action / motion / blitz rates, the man/zone label, mobile-QB and QB-contain flags —
   so we can score offensive-scheme vs defensive-scheme mismatches (play-action punishes
   blitz-heavy man defenses; motion stresses man coverage; a mobile QB feasts on poor
   contain). Small, bounded points nudges + human-readable notes.

Honest scope: these make the prediction and matchup *analysis* more realistic; they do NOT
create a betting edge (the closing line already prices them).
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

_INJ = None
_STY = None


def _injuries():
    global _INJ
    if _INJ is None:
        p = RAW / "injuries.parquet"
        _INJ = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _INJ


def _styles():
    global _STY
    if _STY is None:
        p = PROC / "team_styles.parquet"
        _STY = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _STY


# ── 1. injury → unit routing (quality-weighted, starters only) ──────
# position (from the injury report) → which unit(s) it weakens.
_RUN_D = {"DT", "NT", "DL", "ILB", "MLB", "LB"}                 # interior run defenders
_PASS_D = {"CB", "DB", "S", "FS", "SS", "NB", "OLB"}           # coverage
_EDGE = {"DE", "EDGE"}                                          # edge rushers: both fronts
_OL = {"T", "G", "C", "OL", "OT", "OG", "LT", "RT", "LG", "RG", "T/G"}
_WR = {"WR", "TE"}
_RB = {"RB", "FB"}
_BASE = 0.42        # z downgrade for losing an ELITE starter; scaled down by rating
_CAP = 0.7          # max downgrade to any one unit

_RATINGS = {}       # team -> ({gsis: rec}, {norm_name: rec})


def _team_ratings(team):
    """Per-player rating + depth rank from the depth-chart model, keyed by id and name."""
    if team in _RATINGS:
        return _RATINGS[team]
    from ml.squad import team_depth_chart, _norm
    byg, bynm = {}, {}
    try:
        for grp in team_depth_chart(team):
            for p in grp["players"]:
                rec = {"rating": p.get("rating"), "rank": p.get("rank"), "group": grp["pos"]}
                if p.get("gsis"):
                    byg[p["gsis"]] = rec
                if p.get("name"):
                    bynm[_norm(p["name"])] = rec
    except Exception:
        pass
    _RATINGS[team] = (byg, bynm)
    return _RATINGS[team]


def _route(d, pos, mag):
    """Apply a downgrade of size `mag` to the unit(s) a position affects (sign-correct:
    defense +=allows more, offense -=less)."""
    if pos in _OL:
        d["z_off_pass"] -= mag; d["z_off_rush"] -= mag
    elif pos in _WR:
        d["z_off_pass"] -= mag
    elif pos in _RB:
        d["z_off_rush"] -= mag
    elif pos in _EDGE:
        d["z_def_pass"] += mag * 0.9; d["z_def_rush"] += mag * 0.5
    elif pos in _RUN_D:
        d["z_def_rush"] += mag
    elif pos in _PASS_D:
        d["z_def_pass"] += mag


def unit_injury_deltas(team: str) -> dict:
    """Unit z-score deltas from this team's Out list — only STARTERS / heavily-featured
    players count, and each is scaled by the injured player's 0-100 rating (losing an
    All-Pro >> losing a depth piece, which barely moves it since a similar player replaces him)."""
    inj = _injuries()
    d = {"z_off_pass": 0.0, "z_off_rush": 0.0, "z_def_pass": 0.0, "z_def_rush": 0.0}
    if inj.empty or "team" not in inj.columns:
        return d
    t = inj[inj["team"] == team]
    if t.empty:
        return d
    sw = t["season"].astype(int) * 100 + t["week"].astype(int)
    out = t[(sw == sw.max()) & (t["report_status"] == "Out")]
    if out.empty:
        return d
    from ml.squad import _norm
    byg, bynm = _team_ratings(team)
    for _, r in out.iterrows():
        pos = str(r.get("position") or "").upper()
        if pos == "QB":                                    # QB handled by the points penalty
            continue
        rec = byg.get(r.get("gsis_id")) or bynm.get(_norm(r.get("full_name") or ""))
        if not rec or rec.get("rating") is None:
            continue
        rating, rank = rec["rating"], rec.get("rank")
        # starter (rank 1 of the slot) OR a heavily-featured, well-rated #2
        starter = (rank == 1) or (rank == 2 and rating >= 68)
        if not starter:
            continue
        q = max(0.0, min(1.2, (rating - 45) / 45.0))       # quality above replacement
        if q <= 0:
            continue
        _route(d, pos, _BASE * q)
    d["z_off_pass"] = max(-_CAP, d["z_off_pass"]); d["z_off_rush"] = max(-_CAP, d["z_off_rush"])
    d["z_def_pass"] = min(_CAP, d["z_def_pass"]); d["z_def_rush"] = min(_CAP, d["z_def_rush"])
    return d


# ── 2. scheme (play-caller) matchups ────────────────────────────────
def _style_row(team, season=None):
    s = _styles()
    if s.empty:
        return None
    ss = s[s["team"] == team]
    if ss.empty:
        return None
    if season is not None and (ss["season"] == season).any():
        return ss[ss["season"] == season].iloc[0]
    return ss[ss["season"] == ss["season"].max()].iloc[0]


# The boolean archetype flags in team_styles are miscalibrated (poor_qb_contain is true for
# all 32, high_play_action/blitz_heavy never fire), so we use the continuous rates that have
# real variance, as league percentiles, plus the defense_label (Man / Aggressive Blitz / Zone).
_SCHEME = None
_SCHEME_COLS = ["play_action_rate", "motion_rate", "avg_blitzers", "qb_rush_rate",
                "def_qb_scramble_epa_allowed"]


def _scheme_tables(season):
    global _SCHEME
    if _SCHEME is not None and _SCHEME[0] == season:
        return _SCHEME[1]
    s = _styles()
    tab = {"label": {}}
    if not s.empty:
        s = s[s["season"] == season]
        if "defense_label" in s:
            tab["label"] = dict(zip(s["team"], s["defense_label"].fillna("")))
        for col in _SCHEME_COLS:
            if col in s:
                r = s[["team", col]].dropna()
                tab[col] = dict(zip(r["team"], r[col].rank(pct=True)))
            else:
                tab[col] = {}
    _SCHEME = (season, tab)
    return tab


def _offense_vs_defense(off_t, def_t, tab) -> tuple:
    """Points bonus for the OFFENSE from scheme mismatches + notes (percentile-based)."""
    def p(col, t):
        return tab.get(col, {}).get(t, 0.5)
    b, notes = 0.0, []
    dlabel = str(tab.get("label", {}).get(def_t, "") or "")
    # play-action punishes an aggressive/blitzing front
    if p("play_action_rate", off_t) > 0.65 and (p("avg_blitzers", def_t) > 0.6 or "Blitz" in dlabel):
        b += 0.8; notes.append("play-action vs aggressive front")
    # pre-snap motion stresses man coverage
    if p("motion_rate", off_t) > 0.65 and "Man" in dlabel:
        b += 0.6; notes.append("motion vs man coverage")
    # a run-happy QB vs a defense that leaks scramble EPA (poor contain)
    if p("qb_rush_rate", off_t) > 0.65:
        cont = p("def_qb_scramble_epa_allowed", def_t)     # high = leaky contain
        if cont > 0.65:
            b += 0.9; notes.append("mobile QB vs poor contain")
        elif cont < 0.35:
            b -= 0.6; notes.append("mobile QB vs strong contain")
    return round(b, 2), notes


def scheme_matchup(home: str, away: str, season=None) -> dict:
    """Per-team scheme-mismatch points + notes for both offensive matchups."""
    s = _styles()
    if season is None:
        season = int(s["season"].max()) if not s.empty else None
    tab = _scheme_tables(season) if season is not None else {"label": {}}
    h_b, h_notes = _offense_vs_defense(home, away, tab)   # home offense vs away defense
    a_b, a_notes = _offense_vs_defense(away, home, tab)   # away offense vs home defense
    hs, aws = _style_row(home, season), _style_row(away, season)
    return {
        "home_delta": h_b, "away_delta": a_b,
        "notes": {home: h_notes, away: a_notes},
        "labels": {
            home: {"off": str(hs.get("offense_label")) if hs is not None else None,
                   "def": str(hs.get("defense_label")) if hs is not None else None},
            away: {"off": str(aws.get("offense_label")) if aws is not None else None,
                   "def": str(aws.get("defense_label")) if aws is not None else None},
        },
    }


def clear():
    global _INJ, _STY, _SCHEME, _RATINGS
    _INJ = _STY = _SCHEME = None
    _RATINGS = {}
