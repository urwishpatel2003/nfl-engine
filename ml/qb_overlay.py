"""
ml/qb_overlay.py  —  QB-change adjustment for the preseason projection
=======================================================================
The regression-to-mean projection (ml/preseason.py) is blind to offseason QB
changes: it still rates a team on last year's QB. QB is the one roster factor big
enough to plausibly matter, so this module tests — and only then applies — an
adjustment for teams that changed starting QB.

Approach
  * QB quality  = a QB's mean composite adjusted_score in the PRIOR season (leak-free);
                  rookies / no-prior get a replacement level (low percentile).
  * qb_delta    = (incoming starter's prior quality) - (last year's starter's quality),
                  in composite points. 0 if the starter is unchanged.
  * The overlay adds beta * qb_delta (points/game) to the regressed rating.

Validation (the gate)
  Does qb_delta predict the RESIDUAL of the regression baseline (how much a team
  beat/missed its regressed projection)? Leave-one-season-out: does adding the QB
  term reduce out-of-sample MAE vs regression-only? If not, we DO NOT ship it.

RESULT (2022-2025, 43 QB changes): the overlay does NOT validate. qb_delta correlates
with the regression residual at -0.07 (all) / -0.16 (changed-only) — near zero and the
WRONG sign, so the fitted beta is negative and the OOS gain is a trivial 0.02 pts/game.
Regression to the mean already absorbs QB-change effects (teams change QBs after bad
years and revert anyway). So we DO NOT ship this adjustment — it would add noise. The
2026 depth chart is still shown in the dashboard as the projected starter (informational),
but it does not move the ratings. This module is kept as the documented test.

Usage:
  python -m ml.qb_overlay --validate       # run the gate (currently FAILS)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "processed"
RAWD = Path(__file__).parent.parent / "data" / "raw"
CARRYOVER_K = 0.35


# ── building blocks ─────────────────────────────────────────────────
def season_strength() -> pd.DataFrame:
    s = pd.read_parquet(RAWD / "schedules.parquet")
    s = s[s["home_score"].notna()].copy()
    h = s.rename(columns={"home_team": "team"}).assign(m=lambda d: d.home_score - d.away_score)[["season", "team", "m"]]
    a = s.rename(columns={"away_team": "team"}).assign(m=lambda d: d.away_score - d.home_score)[["season", "team", "m"]]
    return pd.concat([h, a]).groupby(["season", "team"])["m"].mean().reset_index().rename(columns={"m": "strength"})


def qb_tables():
    """Return (qb season quality, team-season primary QB, replacement level)."""
    c = pd.read_parquet(PROC / "composite_scores.parquet")
    c = c[c["position"] == "QB"].copy()
    quality = c.groupby(["player_id", "season"])["adjusted_score"].mean().reset_index()
    # primary QB per team-season = most weeks played
    wk = c.groupby(["recent_team", "season", "player_id"])["week"].nunique().reset_index(name="wks")
    prim = wk.sort_values("wks").groupby(["recent_team", "season"]).tail(1)[["recent_team", "season", "player_id"]]
    prim = prim.rename(columns={"recent_team": "team"})
    repl = float(quality["adjusted_score"].quantile(0.20))   # replacement level for rookies/no-prior
    return quality, prim, repl


def _prior_quality(quality, pid, season, repl):
    """A QB's quality ENTERING `season` = their mean composite in season-1 (else replacement)."""
    row = quality[(quality["player_id"] == pid) & (quality["season"] == season - 1)]
    return float(row["adjusted_score"].iloc[0]) if len(row) else repl


def build_transitions() -> pd.DataFrame:
    """Per team-season N: regression baseline, actual strength, and qb_delta."""
    strength = season_strength()
    quality, prim, repl = qb_tables()
    prev = strength.rename(columns={"strength": "prev"}); prev["season"] += 1
    df = strength.merge(prev, on=["season", "team"]).dropna()

    rows = []
    for _, r in df.iterrows():
        team, N = r["team"], int(r["season"])
        pN = prim[(prim.team == team) & (prim.season == N)]
        pP = prim[(prim.team == team) & (prim.season == N - 1)]
        if pN.empty or pP.empty:
            continue
        new_id, old_id = pN.player_id.iloc[0], pP.player_id.iloc[0]
        new_q = _prior_quality(quality, new_id, N, repl)          # incoming starter's prior quality
        old_q = _prior_quality(quality, old_id, N, repl)          # last year's starter's prior quality
        rows.append({"team": team, "season": N, "strength": r["strength"], "prev": r["prev"],
                     "qb_delta": (new_q - old_q) if new_id != old_id else 0.0,
                     "changed": new_id != old_id})
    return pd.DataFrame(rows)


# ── validation gate ─────────────────────────────────────────────────
def validate():
    t = build_transitions()
    t["resid"] = t["strength"] - CARRYOVER_K * t["prev"]          # regression baseline residual
    ch = t[t["changed"]]
    print(f"\n  QB-overlay validation — {len(t)} team-seasons, {len(ch)} with a QB change")
    print(f"    corr(qb_delta, regression residual): all={t.qb_delta.corr(t.resid):+.3f}  "
          f"changed-only={ch.qb_delta.corr(ch.resid):+.3f}")

    # leave-one-season-out: does adding beta*qb_delta beat regression-only OOS?
    seasons = sorted(t["season"].unique())
    base_err, qb_err = [], []
    for s in seasons:
        tr, te = t[t.season != s], t[t.season == s]
        # fit beta on train residuals: resid ~ beta*qb_delta (no intercept)
        d = tr["qb_delta"]; beta = float((d * tr["resid"]).sum() / (d ** 2).sum()) if (d ** 2).sum() > 0 else 0.0
        base = CARRYOVER_K * te["prev"]
        qb   = base + beta * te["qb_delta"]
        base_err += list((te["strength"] - base).abs())
        qb_err   += list((te["strength"] - qb).abs())
        print(f"    holdout {s}: fitted beta={beta:+.3f} pts/composite-pt")
    base_mae, qb_mae = np.mean(base_err), np.mean(qb_err)
    print(f"\n    OOS MAE  regression-only: {base_mae:.3f}   +QB overlay: {qb_mae:.3f}  "
          f"({'IMPROVES' if qb_mae < base_mae else 'no gain'} by {base_mae-qb_mae:+.3f} pts/game)")
    # global beta for application
    d = t["qb_delta"]; beta_all = float((d * t["resid"]).sum() / (d ** 2).sum())
    print(f"    global beta (for application): {beta_all:+.3f} pts/game per composite point")
    # Honest gate: the effect must be in the CORRECT direction (better QB -> better team,
    # beta>0) AND give a meaningful OOS gain. A tiny gain from a wrong-signed beta is noise.
    ok = (beta_all > 0) and ((base_mae - qb_mae) > 0.10)
    print(f"    GATE: {'PASSED' if ok else 'FAILED'} "
          f"(needs beta>0 and OOS gain>0.10; got beta={beta_all:+.3f}, gain={base_mae-qb_mae:+.3f})\n")
    return beta_all, ok


# ── 2026 application ────────────────────────────────────────────────
def qb_adjustments_2026(beta: float) -> pd.DataFrame:
    quality, prim, repl = qb_tables()
    dc = pd.read_parquet(RAWD / "depth_2026_current.parquet")
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    qb1 = dc[(dc.pos_abb == "QB") & (dc.pos_rank == 1)][["team", "player_name", "gsis_id"]]
    rows = []
    for _, r in qb1.iterrows():
        team = r["team"]
        pP = prim[(prim.team == team) & (prim.season == 2025)]
        if pP.empty:
            continue
        old_id = pP.player_id.iloc[0]
        new_q = _prior_quality(quality, r["gsis_id"], 2026, repl)   # 2026 starter's 2025 quality
        old_q = _prior_quality(quality, old_id, 2026, repl)         # 2025 starter's 2025 quality
        changed = r["gsis_id"] != old_id
        delta = (new_q - old_q) if changed else 0.0
        rows.append({"team": team, "qb_2026": r["player_name"], "qb_delta": round(delta, 1),
                     "adj": round(beta * delta, 1), "changed": changed})
    return pd.DataFrame(rows).sort_values("adj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--apply", type=int, metavar="SEASON")
    args = ap.parse_args()
    beta, ok = validate()
    if args.apply:
        adj = qb_adjustments_2026(beta)
        print(f"  2026 QB adjustments (beta={beta:+.3f}, gate {'PASSED' if ok else 'FAILED'}):")
        for _, r in adj[adj.changed].iterrows():
            print(f"    {r['team']:<3} {r['qb_2026']:<20} delta={r['qb_delta']:+6.1f}  adj={r['adj']:+.1f} pts")


if __name__ == "__main__":
    main()
