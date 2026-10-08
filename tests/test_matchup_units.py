"""
tests/test_matchup_units.py — the matchup payload's `units` block is each team's OWN units
=========================================================================================
Guards a display bug found 2026-10-07: project_game once reported, under each team, the
OPPONENT's pass/run defense (it was written as an "offense vs the defense it faces" view),
while the dashboard labelled the block as the team's own profile — the Bills showed the
Rams' elite pass defense. The prediction never read the block, so only the page was wrong.

Rule pinned here: units[team].pass_def == -z_def_pass[team] (good = high), and the same
for rush_def; offense/ST/coaching are the team's own z-scores. Run:
    python tests/test_matchup_units.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.matchup_engine import project_game, team_units   # noqa: E402

u = team_units()
teams = [t for t in ("BUF", "LA", "KC", "SEA") if t in u.index]
fails = 0
for home, away in ((teams[0], teams[1]), (teams[2], teams[3])):
    r = project_game(home, away)
    for t in (home, away):
        e = r["units"][t]
        want = {"pass_off": u.loc[t, "z_off_pass"], "rush_off": u.loc[t, "z_off_rush"],
                "pass_def": -u.loc[t, "z_def_pass"], "rush_def": -u.loc[t, "z_def_rush"],
                "st": u.loc[t, "z_st"], "coach": u.loc[t, "z_coaching"]}
        for k, v in want.items():
            ok = abs(e[k] - round(float(v), 2)) < 0.011
            fails += 0 if ok else 1
            print(f"  {'PASS' if ok else 'FAIL'}  {away}@{home} units[{t}].{k} = {e[k]} (own unit {round(float(v), 2)})")
print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}: the units block is each team's own units")
sys.exit(1 if fails else 0)
