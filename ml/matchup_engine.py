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

# Margin calibration. The raw roster+unit margin ranks teams well but spreads them far too wide.
# Fitted against the 76 games of the 2026 schedule that already carry Vegas lines: raw model
# margin has sd 7.66 vs the market's 4.62 (ratio 1.66) and regresses as model = 1.45*market-0.47
# — yet correlates at r=0.88. So the ORDER is right and only the SCALE is wrong; shrinking it
# cuts MAE vs the market from 3.47 to 1.85 pts and reorders nothing. (Symptom that prompted this:
# DET projected 13.9 wins and was 5-13 pts too big in all four of its priced games.)
#
# Applied to the strength differential ONLY, leaving home field at full weight — scaling the whole
# margin drags HFA down with it and leaves a -0.65 pt home bias, vs +0.20 when HFA is held out.
#
# 0.60 makes model dispersion match the market exactly on those games. Raw MAE alone would prefer
# 0.52, but weeks 1-4 lines run 0.851x a full season's spread dispersion (pooled 2021-25), so the
# full-season-equivalent factor is 0.52/0.851 = 0.61. The two routes agree; 0.60 sits between them.
# Re-fit with scratch fit_k2-style sweep if the roster model's own dispersion ever changes.
#
# IN-SEASON IT DID CHANGE, so the constant is now the FALLBACK only. Once results blend into
# the units the raw margins are far less over-dispersed (week 3 of 2026: model sd 6.4 vs
# market 5.4, ratio 1.20 — not 1.66), and a fixed 0.60 then over-shrinks: calibrated sd 3.8,
# NARROWER than the market, so a 10-pt favorite read as 5 and the underdog gained 3 points
# it had not earned (KC at MIA: units said KC by 11.6, page said 5.0 and Miami 23).
# margin_calibration() measures the ratio market sd / raw model sd over this season's priced
# regular-season games (pooled, all weeks so far incl. the upcoming one), clipped to
# [CAL_MIN, 1.0]. Same derivation as the constant — match the market's scale — just re-done
# on every refresh instead of once in August.
MARGIN_CALIBRATION = 0.60          # fallback while fewer than CAL_MIN_GAMES priced games exist
CAL_MIN, CAL_MIN_GAMES = 0.50, 12
_CAL = None


def margin_calibration() -> float:
    return calibration()["margin"]


def calibration() -> dict:
    """{margin, total, total_mean}: market sd / raw model sd for margins AND totals over this
    season's priced games, clipped to [CAL_MIN, 1]; cached until refresh. Totals are scaled
    about the model's own mean total on those games (its mean is unbiased — week 3 of 2026:
    45.6 vs market 45.1 — but its spread is 1.5x the market's, so a 51-point projection was
    a 49 once the pass/rush nudges were put on the market's scale)."""
    global _CAL
    if _CAL is not None:
        return _CAL
    fallback = {"margin": MARGIN_CALIBRATION, "total": 1.0, "total_mean": None}
    try:
        from ml.current import state
        s = pd.read_parquet(RAW / "schedules.parquet")
        st = state()
        s = s[(s["season"] == st["season"]) & (s["game_type"].fillna("REG").str.upper() == "REG")
              & s["spread_line"].notna() & s["home_team"].notna() & s["away_team"].notna()]
        # upcoming games too (their lines exist); cap at the current week + 1
        wk_cap = (int(s[s["home_score"].notna()]["week"].max()) if s["home_score"].notna().any() else 0) + 1
        s = s[s["week"] <= wk_cap]
        raws, lines, tots, vtots = [], [], [], []
        for _, g in s.iterrows():
            r = _raw_project(g["home_team"], g["away_team"])
            if r is None:
                continue
            raws.append(r["raw_margin"] - r["hfa"])          # strength differential only (HFA held out)
            lines.append(float(g["spread_line"]))
            if pd.notna(g.get("total_line")):
                tots.append(r["total"]); vtots.append(float(g["total_line"]))
        if len(raws) < CAL_MIN_GAMES:
            _CAL = fallback
        else:
            sd_raw = float(np.std(raws)) or 1e-9
            out = {"margin": float(np.clip(np.std(lines) / sd_raw, CAL_MIN, 1.0)), "total": 1.0, "total_mean": None}
            if len(tots) >= CAL_MIN_GAMES:
                sd_t = float(np.std(tots)) or 1e-9
                out["total"] = float(np.clip(np.std(vtots) / sd_t, CAL_MIN, 1.0))
                out["total_mean"] = float(np.mean(tots))
            _CAL = out
    except Exception:
        _CAL = fallback
    return _CAL


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
    # opponent-adjusted unit EPA (schedule-adjusted); falls back to raw means if unavailable.
    # IN-SEASON this is the CURRENT view (ml/current.py): this season's adjusted units shrunk
    # toward last season's by games played, so week-1 results nudge the units without
    # replacing a full season of evidence with one game.
    from ml.current import adjusted_units, weights as _cur_w, state as _cur_state
    adj = adjusted_units()
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

    # points for / against per game (from final scores) — last season, then blended with
    # this season's games by the same per-team weight
    cw = _cur_w()
    cst = _cur_state()

    def _pts(season):
        s = pd.read_parquet(RAW / "schedules.parquet")
        s = s[(s.season == season) & (s.game_type.str.upper() == "REG") & s.home_score.notna()]
        h = s.rename(columns={"home_team": "t", "home_score": "pf", "away_score": "pa"})[["t", "pf", "pa"]]
        a = s.rename(columns={"away_team": "t", "away_score": "pf", "home_score": "pa"})[["t", "pf", "pa"]]
        return pd.concat([h, a]).groupby("t").mean()
    pts = _pts(2025)
    u["pf"] = pts["pf"]; u["pa"] = pts["pa"]
    if len(cw) and cst["season"] != 2025:
        now = _pts(cst["season"]).reindex(u.index)
        w = cw.reindex(u.index).fillna(0.0)
        u["pf"] = w * now["pf"].fillna(u["pf"]) + (1 - w) * u["pf"]
        u["pa"] = w * now["pa"].fillna(u["pa"]) + (1 - w) * u["pa"]
        try:                                   # pace / pass rate from this season's plays, same blend
            pc = pd.read_parquet(RAW / f"pbp_{cst['season']}.parquet")
            pc = pc[pc["play_type"].isin(["pass", "run"]) & pc["epa"].notna() & (pc["week"] <= 18)]
            g2 = pc.groupby("posteam")["game_id"].nunique()
            pace2 = (pc.groupby("posteam").size() / g2).reindex(u.index)
            pr2 = (pc[pc.play_type == "pass"].groupby("posteam").size() / pc.groupby("posteam").size()).reindex(u.index)
            u["pace"] = w * pace2.fillna(u["pace"]) + (1 - w) * u["pace"]
            u["pass_rate"] = w * pr2.fillna(u["pass_rate"]) + (1 - w) * u["pass_rate"]
        except Exception:
            pass

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
    # IN-SEASON: the performance side now contains this season's games, so its weight vs the
    # roster projection grows with games played — never below the continuity-based preseason
    # weight, and up to 0.8 by late season (the roster prior never fully disappears).
    if len(cw):
        wp = (cw.reindex(u.index).fillna(0.0) / 1.0).clip(0, 0.8)
        w_off = pd.concat([w_off, wp], axis=1).max(axis=1)
        w_def = pd.concat([w_def, wp], axis=1).max(axis=1)

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
def _raw_project(home: str, away: str, neutral: bool = False, unit_adj: dict = None):
    """Everything up to the UNCALIBRATED margin: {raw_margin, hfa, total, u}. None if unknown."""
    u = team_units()
    if home not in u.index or away not in u.index:
        return None
    from ml.squad import predict_matchup, team_hfa
    roster = predict_matchup(home, away, neutral)
    hfa = 0.0 if neutral else team_hfa(home)
    lg_pace = float(u["pace"].mean())

    def uz(team, col):
        base = float(u.loc[team, col])
        return base + (unit_adj.get(team, {}).get(col, 0.0) if unit_adj else 0.0)

    def phase(off, deff, sign):
        base = 0.5 * u.loc[off, "pf"] + 0.5 * u.loc[deff, "pa"]
        nudge = 0.8 * ((uz(off, "z_off_pass") + uz(deff, "z_def_pass")) +
                       0.6 * (uz(off, "z_off_rush") + uz(deff, "z_def_rush")))
        return float(base + nudge + u.loc[off, "st"] + 0.4 * u.loc[off, "z_coaching"] + sign * hfa / 2)

    ph, pa_ = phase(home, away, +1), phase(away, home, -1)
    pace_mult = (u.loc[home, "pace"] + u.loc[away, "pace"]) / (2 * lg_pace)
    total = (ph + pa_) * (0.85 + 0.15 * pace_mult)
    raw_margin = 0.55 * roster["pred_margin"] + 0.45 * (ph - pa_)
    return {"raw_margin": raw_margin, "hfa": hfa, "total": total, "u": u}


def project_game(home: str, away: str, neutral: bool = False, unit_adj: dict = None) -> dict:
    """Expected points for each team from offense-vs-defense + ST + coaching + pace,
    with the margin anchored to the roster-talent rating (consistent with the rankings).
    unit_adj = {team: {z_off_pass/z_off_rush/z_def_pass/z_def_rush: delta}} injects per-game
    unit downgrades (e.g. injury-to-unit routing) so they interact through the matchup nudge."""
    u = team_units()
    if home not in u.index or away not in u.index:
        return {"error": "unknown team(s)"}
    if True:
        r = _raw_project(home, away, neutral, unit_adj)
        hfa, raw_total, raw_margin = r["hfa"], r["total"], r["raw_margin"]
        c = calibration()
        cal = c["margin"]
        final_margin = hfa + cal * (raw_margin - hfa)
        total = raw_total if c["total_mean"] is None else c["total_mean"] + c["total"] * (raw_total - c["total_mean"])
        home_pts = (total + final_margin) / 2
        away_pts = (total - final_margin) / 2
        wp = float(1 / (1 + np.exp(-final_margin / 13.5 * np.pi / np.sqrt(3))))

        def edges(off, deff):
            return {"pass_off": round(float(u.loc[off, "z_off_pass"]), 2),
                    "rush_off": round(float(u.loc[off, "z_off_rush"]), 2),
                    "pass_def": round(float(-u.loc[deff, "z_def_pass"]), 2),
                    "rush_def": round(float(-u.loc[deff, "z_def_rush"]), 2),
                    "st": round(float(u.loc[off, "z_st"]), 2),
                    "coach": round(float(u.loc[off, "z_coaching"]), 2)}

        return {
            "home": home, "away": away,
            "pred_home_score": round(home_pts, 1), "pred_away_score": round(away_pts, 1),
            "pred_margin": round(final_margin, 1), "pred_total": round(total, 1),
            "raw_margin": round(raw_margin, 1), "raw_total": round(raw_total, 1),
            "calibration": round(cal, 3), "total_calibration": round(c["total"], 3),
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
