"""
ml/matchup_engine.py  —  unit-vs-unit matchup engine
=====================================================
Predicts a game by matching every phase of both teams against each other:
offense (pass & rush) vs the opponent's defense (pass & rush), plus special teams
and coaching. Produces a differentiated score AND total (good offense vs bad
defense = high scoring), and the opponent adjustments that shape each player's line.

team_units()  — per-team 2025 phase ratings (league-relative), the shared basis:
    off_pass, off_rush   : offensive EPA/play passing & rushing
    def_pass, def_rush    : EPA/play ALLOWED (negative = good defense)
    pace                  : offensive plays / game
    pass_rate             : offensive pass share
    pf, pa                : points for / against per game
    st                    : special-teams net (FG% + return/coverage), points/game
    coaching              : multi-year coaching score

project_game() — combine them into each team's expected points via a
points-for/against blend refined by the pass/rush unit matchups, pace, ST,
coaching and home field.

The margin is anchored to the roster-talent rating (ml.squad) so the matchup never
contradicts the power rankings; the unit model shapes the TOTAL and player stats.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

_UNITS = None
W25_MAX = 0.40       # max weight on 2025 performance (so 2026 talent is always >= 60%)
PPG_SCALE = 4.5      # points/game per unit-talent z-score


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0


def _squad_zunits() -> pd.DataFrame:
    """Per-team 2026 roster-talent phase z-scores (from ml.squad unit components)."""
    from ml.squad import squad_ratings
    _, g = squad_ratings(breakdown=True)
    z = g.apply(_z)
    out = pd.DataFrame(index=g.index)
    out["t_off_pass"] = _z(0.60 * z.qb + 0.25 * z.skill + 0.15 * z.ol)
    out["t_off_rush"] = _z(0.50 * z.skill + 0.50 * z.ol)
    out["t_def_pass"] = _z(0.55 * z.rush + 0.45 * z.cover)     # rush = pass rush
    out["t_def_rush"] = _z(0.60 * z.rush + 0.40 * z.cover)     # front-7 proxy for run D
    out["t_off"] = _z(0.5 * out.t_off_pass + 0.5 * out.t_off_rush)
    out["t_def"] = _z(0.5 * out.t_def_pass + 0.5 * out.t_def_rush)
    out["t_coach"] = z.coach
    return out


def _continuity() -> pd.DataFrame:
    """Per-team share of 2025 production still on the 2026 roster (offense & defense)."""
    p = pd.read_parquet(RAW / "pbp_2025.parquet")
    p = p[p["week"] <= 18]
    rost = pd.read_parquet(RAW / "rosters_2026.parquet")[["player_id", "team"]]

    # offense: pass attempts + targets + carries, by player & 2025 team
    ev = []
    ev.append(p[p.pass_attempt == 1][["passer_player_id", "posteam"]].rename(columns={"passer_player_id": "pid"}))
    ev.append(p[(p.pass_attempt == 1) & p.receiver_player_id.notna()][["receiver_player_id", "posteam"]].rename(columns={"receiver_player_id": "pid"}))
    ev.append(p[p.rush_attempt == 1][["rusher_player_id", "posteam"]].rename(columns={"rusher_player_id": "pid"}))
    off = pd.concat(ev).dropna(subset=["pid"]).assign(w=1).groupby(["pid", "posteam"], as_index=False)["w"].sum()
    off = off.merge(rost, left_on="pid", right_on="player_id", how="left")
    off["ret"] = off["team"] == off["posteam"]
    cont_off = off.assign(rw=off.w * off.ret).groupby("posteam").apply(
        lambda x: x.rw.sum() / max(1.0, x.w.sum()), include_groups=False)

    # defense: tackles/coverage credited to a defender, by player & 2025 team (defteam)
    tk = []
    for c in ["solo_tackle_1_player_id", "solo_tackle_2_player_id"]:
        if c in p.columns:
            tk.append(p[[c, "defteam"]].rename(columns={c: "pid"}))
    if tk:
        dfe = pd.concat(tk).dropna(subset=["pid"]).assign(w=1).groupby(["pid", "defteam"], as_index=False)["w"].sum()
        dfe = dfe.merge(rost, left_on="pid", right_on="player_id", how="left")
        dfe["ret"] = dfe["team"] == dfe["defteam"]
        cont_def = dfe.assign(rw=dfe.w * dfe.ret).groupby("defteam").apply(
            lambda x: x.rw.sum() / max(1.0, x.w.sum()), include_groups=False)
    else:
        cont_def = cont_off * 0 + 0.6
    return pd.DataFrame({"cont_off": cont_off, "cont_def": cont_def}).fillna(0.6)


def team_units() -> pd.DataFrame:
    """Per-team 2025 phase ratings (offense pass/rush, defense pass/rush, ST, coaching)."""
    global _UNITS
    if _UNITS is not None:
        return _UNITS

    p = pd.read_parquet(RAW / "pbp_2025.parquet")
    p = p[p["week"] <= 18]
    plays = p[p["play_type"].isin(["pass", "run"]) & p["epa"].notna()].copy()
    pa, ru = plays[plays.play_type == "pass"], plays[plays.play_type == "run"]

    u = pd.DataFrame(index=sorted(set(plays.posteam.dropna())))
    # opponent-adjusted unit EPA (schedule-adjusted); falls back to raw means if unavailable
    from ml.adjust import adjusted_unit_epa
    adj = adjusted_unit_epa(2025)
    _raw = {"off_pass": pa.groupby("posteam")["epa"].mean(),
            "off_rush": ru.groupby("posteam")["epa"].mean(),
            "def_pass": pa.groupby("defteam")["epa"].mean(),      # allowed (lower=better)
            "def_rush": ru.groupby("defteam")["epa"].mean()}
    for col, rawvals in _raw.items():
        u[col] = (pd.Series({t: adj.get(t, {}).get(col) for t in u.index}).fillna(rawvals)
                  if adj else rawvals)
    gpg = plays.groupby("posteam")["game_id"].nunique()
    u["pace"] = plays.groupby("posteam").size() / gpg
    u["pass_rate"] = pa.groupby("posteam").size() / plays.groupby("posteam").size()

    # points for / against per game (from final scores)
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[(s.season == 2025) & (s.game_type.str.upper() == "REG") & s.home_score.notna()]
    h = s.rename(columns={"home_team": "t", "home_score": "pf", "away_score": "pa"})[["t", "pf", "pa"]]
    a = s.rename(columns={"away_team": "t", "away_score": "pf", "home_score": "pa"})[["t", "pf", "pa"]]
    pts = pd.concat([h, a]).groupby("t").mean()
    u["pf"] = pts["pf"]; u["pa"] = pts["pa"]

    # special teams: FG make rate + return net, expressed as points/game vs average
    fg = p[p["play_type"] == "field_goal"]
    if "field_goal_result" in fg.columns and len(fg):
        fgm = fg.assign(made=(fg["field_goal_result"] == "made").astype(int)) \
            .groupby("posteam")["made"].mean()
        u["st"] = (fgm.reindex(u.index).fillna(fgm.mean()) - fgm.mean()) * 6.0
    else:
        u["st"] = 0.0

    # coaching: multi-year mean coaching score (down-weighted noise elsewhere)
    cp = PROC / "coaching_scores.parquet"
    if cp.exists():
        c = pd.read_parquet(cp)
        u["coaching"] = c[c.season.isin([2023, 2024, 2025])].groupby("team")["coaching_score"].mean()
    u["coaching"] = u.get("coaching", pd.Series(50.0, index=u.index)).fillna(50.0)

    u = u.fillna(u.mean(numeric_only=True))
    lg_pf = float(u["pf"].mean())
    # 2025-performance z-scores (defense: higher = worse, i.e. more EPA allowed)
    for col in ["off_pass", "off_rush", "def_pass", "def_rush", "st", "coaching"]:
        u[f"z25_{col}"] = _z(u[col])

    # ── blend: majority 2026 roster talent, 2025 weighted by unit continuity ──
    tal = _squad_zunits().reindex(u.index).fillna(0.0)
    con = _continuity().reindex(u.index).fillna(0.6)
    w_off = (W25_MAX * con["cont_off"]).clip(0, W25_MAX)
    w_def = (W25_MAX * con["cont_def"]).clip(0, W25_MAX)

    u["z_off_pass"] = (1 - w_off) * tal["t_off_pass"] + w_off * u["z25_off_pass"]
    u["z_off_rush"] = (1 - w_off) * tal["t_off_rush"] + w_off * u["z25_off_rush"]
    # defense talent is "good = high"; flip to the "EPA-allowed" convention (high = worse)
    u["z_def_pass"] = (1 - w_def) * (-tal["t_def_pass"]) + w_def * u["z25_def_pass"]
    u["z_def_rush"] = (1 - w_def) * (-tal["t_def_rush"]) + w_def * u["z25_def_rush"]
    u["z_st"] = u["z25_st"]                          # ST has high continuity (kickers)
    u["z_coaching"] = tal["t_coach"]                 # coaching already multi-year in squad

    # blend points-for / points-against toward the 2026-talent expectation
    pf26 = lg_pf + tal["t_off"] * PPG_SCALE
    pa26 = lg_pf - tal["t_def"] * PPG_SCALE
    u["pf"] = (1 - w_off) * pf26 + w_off * u["pf"]
    u["pa"] = (1 - w_def) * pa26 + w_def * u["pa"]
    u["cont_off"] = con["cont_off"]; u["cont_def"] = con["cont_def"]
    _UNITS = u
    return u


# ── unit-vs-unit points model ───────────────────────────────────────
def project_game(home: str, away: str, neutral: bool = False) -> dict:
    """Expected points for each team from offense-vs-defense + ST + coaching + pace,
    with the margin anchored to the roster-talent rating (consistent with the rankings)."""
    u = team_units()
    if home not in u.index or away not in u.index:
        return {"error": "unknown team(s)"}
    from ml.squad import predict_matchup
    roster = predict_matchup(home, away, neutral)
    hfa = 0.0 if neutral else 2.0
    lg_pace = float(u["pace"].mean())

    def phase(off, deff, sign):
        # points-for/against blend (captures overall off vs def), then small unit nudges + ST + coaching
        base = 0.5 * u.loc[off, "pf"] + 0.5 * u.loc[deff, "pa"]
        # specific pass/rush mismatch nudge: good offense vs bad (high-EPA-allowed) defense
        nudge = 0.8 * ((u.loc[off, "z_off_pass"] + u.loc[deff, "z_def_pass"]) +
                       0.6 * (u.loc[off, "z_off_rush"] + u.loc[deff, "z_def_rush"]))
        pts = base + nudge + u.loc[off, "st"] + 0.4 * u.loc[off, "z_coaching"] + sign * hfa / 2
        return float(pts)

    ph, pa_ = phase(home, away, +1), phase(away, home, -1)
    # pace: more combined plays -> scale total modestly
    pace_mult = (u.loc[home, "pace"] + u.loc[away, "pace"]) / (2 * lg_pace)
    mid = (ph + pa_) / 2
    ph = mid + (ph - mid) * 1.0; pa_ = mid + (pa_ - mid) * 1.0
    total = (ph + pa_) * (0.85 + 0.15 * pace_mult)

    unit_margin = ph - pa_
    # anchor the margin to the roster-talent rating so it never contradicts the rankings
    final_margin = 0.55 * roster["pred_margin"] + 0.45 * unit_margin
    home_pts = (total + final_margin) / 2
    away_pts = (total - final_margin) / 2
    wp = float(1 / (1 + np.exp(-final_margin / 13.5 * np.pi / np.sqrt(3))))

    def edges(off, deff):
        return {"pass_off": round(float(u.loc[off, "z_off_pass"]), 2),
                "rush_off": round(float(u.loc[off, "z_off_rush"]), 2),
                "pass_def": round(float(-u.loc[deff, "z_def_pass"]), 2),   # flip so +=good D
                "rush_def": round(float(-u.loc[deff, "z_def_rush"]), 2),
                "st": round(float(u.loc[off, "z_st"]), 2),
                "coach": round(float(u.loc[off, "z_coaching"]), 2)}

    return {
        "home": home, "away": away,
        "pred_home_score": round(home_pts, 1), "pred_away_score": round(away_pts, 1),
        "pred_margin": round(final_margin, 1), "pred_total": round(total, 1),
        "home_win_prob": round(wp, 3), "away_win_prob": round(1 - wp, 3),
        "units": {home: edges(home, away), away: edges(away, home)},
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        r = project_game(sys.argv[1].upper(), sys.argv[2].upper())
        print(f"{r['away']} {r['pred_away_score']} - {r['pred_home_score']} {r['home']}  "
              f"(total {r['pred_total']}, home win {r['home_win_prob']:.0%})")
        for t, e in r["units"].items():
            print(f"  {t}: passO {e['pass_off']:+.1f} rushO {e['rush_off']:+.1f} | "
                  f"passD {e['pass_def']:+.1f} rushD {e['rush_def']:+.1f} | ST {e['st']:+.1f} coach {e['coach']:+.1f}")
        sys.exit(0)
    u = team_units()
    lg_pf = u["pf"].mean()
    print(f"league avg pts/game: {lg_pf:.1f}\n")
    print("Best offenses (off_pass EPA):")
    print(u.sort_values("off_pass", ascending=False).head(5)[["off_pass", "off_rush", "pf", "pace"]].round(3).to_string())
    print("\nBest defenses (def_pass EPA allowed, lower=better):")
    print(u.sort_values("def_pass").head(5)[["def_pass", "def_rush", "pa"]].round(3).to_string())
