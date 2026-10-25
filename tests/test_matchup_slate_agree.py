"""
tests/test_matchup_slate_agree.py
---------------------------------
Regression guard: every page that shows a prediction for a SCHEDULED game must show the
Schedule & Picks number. Week 7 2026 (Steelers–Saints in Paris) the Matchup page predicted
the game as a Saints home game — +1.4 pts of home field the board had correctly removed —
and leaned the opposite way to the board that is graded. Both paths now run through
dashboard.server._predict_scheduled; this test fails if they ever diverge again.

Checks, for every game of the current week:
  1. /api/matchup_full margin == the slate's base prediction margin (exact, same code + data)
  2. the scores agree whenever the totals agree (the slate caches the total it was built on;
     the Matchup page re-splits on the board's total, so a moved line changes scores only)
  3. a game the schedule lists as Neutral carries context.listed_neutral and hfa == 0
  4. the Matchup page's line is the board's line (pinned contest sheet / live / nflverse)
  5. a hypothetical pairing (not on the schedule) still works and has no board row

Run:  python tests/test_matchup_slate_agree.py      (slow: builds the week's slate once)
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from dashboard import server as S
    s = S.schedules_df()
    season = int(s["season"].max())
    week = S._current_week(season)
    assert week is not None, "season complete — nothing to compare"
    slate = S._slate(season, week)
    S._overlay_locked(season, slate["games"])           # what the Schedule page shows: frozen rows after the lock
    base = {g["game_id"]: g for g in S._SCHED_PRED[(season, week)]}
    board = {g["game_id"]: g for g in slate["games"]}
    client = S.app.test_client()

    n = 0
    for gid, b in base.items():
        if b.get("pred_margin") is None:
            continue
        r = client.get(f"/api/matchup_full?home={b['home']}&away={b['away']}").get_json()
        assert "error" not in r, f"{gid}: {r.get('error')}"
        # 1. same margin
        assert abs(float(r["pred_margin"]) - float(b["pred_margin"])) < 0.05, \
            f"{gid}: matchup margin {r['pred_margin']} != slate {b['pred_margin']}"
        # 2. same scores when the totals agree
        if abs(float(r["pred_total"]) - float(b["pred_total"])) < 0.05:
            assert abs(float(r["pred_home_score"]) - float(b["pred_home"])) < 0.05, f"{gid}: home score differs"
            assert abs(float(r["pred_away_score"]) - float(b["pred_away"])) < 0.05, f"{gid}: away score differs"
        # 3. neutral venue honoured
        ctx = r.get("context") or {}
        assert bool(ctx.get("listed_neutral")) == bool(b.get("neutral")), f"{gid}: neutral flag mismatch"
        if b.get("neutral"):
            assert abs(float((r.get("components") or {}).get("hfa", 0.0))) < 1e-9, f"{gid}: neutral game carries home field"
        # 4. the line is the board's line
        bb = board.get(gid) or {}
        bt = r.get("betting") or {}
        if bb.get("vegas_spread") is not None:
            assert bt.get("has_line") and abs(float(bt["vegas_spread"]) - float(bb["vegas_spread"])) < 1e-9, \
                f"{gid}: matchup line {bt.get('vegas_spread')} != board {bb['vegas_spread']} ({bb.get('line_source')})"
            assert bt.get("line_source") == bb.get("line_source"), \
                f"{gid}: line source differs: matchup {bt.get('line_source')!r} vs board {bb.get('line_source')!r}"
        assert r.get("board") is not None and r["board"].get("pred_margin") is not None, f"{gid}: no board row"
        n += 1
    assert n, "no games compared"

    # 5. a pairing that is not on the schedule still predicts, with no board row
    teams = sorted(set(s["home_team"].dropna()))
    pair = None
    for h in teams:
        for a in teams:
            if h != a and s[(s["home_team"] == h) & (s["away_team"] == a)].empty:
                pair = (h, a)
                break
        if pair:
            break
    if pair:
        r = client.get(f"/api/matchup_full?home={pair[0]}&away={pair[1]}&neutral=1").get_json()
        assert "error" not in r and r.get("board") is None and r.get("context") is None, f"hypothetical {pair}: {r.get('error')}"
        assert abs(float((r.get("components") or {}).get("hfa", 0.0))) < 1e-9, "neutral checkbox ignored on a hypothetical"
    print(f"OK: {n} games of {season} wk{week} — Matchup page == Schedule & Picks slate")


if __name__ == "__main__":
    main()
