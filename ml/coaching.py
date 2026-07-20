"""
ml/coaching.py  —  2026 coaching staff, schemes, and scheme-confidence
======================================================================
The scheme tendencies in team_styles are derived from 2025 play-by-play, i.e. from the
2025 play-caller. When a team changes its head coach / play-caller for 2026, those
tendencies are stale — we should trust them LESS, because we don't yet know the new
scheme. This module provides:

  • team_coaching(team) — HC / OC / DC + scheme tags from the curated data/coaching_2026.json
    (head coach is data-grounded from schedules; coordinators + tags are seeded from public
    knowledge and marked verify=true — EDIT that file to match your season).
  • scheme_confidence(team) — 1.0 if the head coach is unchanged from 2025, lower if new
    (so the scheme-matchup layer regresses a new-staff team's 2025 tendencies toward neutral).

Head-coach change is detected straight from schedules.parquet, so the confidence factor is
100% data-driven even if the coordinator table isn't filled in.
"""

import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
TABLE = Path(__file__).parent.parent / "data" / "coaching_2026.json"

NEW_HC_CONFIDENCE = 0.4     # trust a new-staff team's 2025 scheme this much

_TABLE = None
_HC_CHANGE = None


def _table() -> dict:
    global _TABLE
    if _TABLE is None:
        try:
            _TABLE = json.loads(TABLE.read_text())
        except Exception:
            _TABLE = {}
    return _TABLE


def _hc_change() -> dict:
    """{team: True if 2026 head coach differs from 2025} — straight from the schedule."""
    global _HC_CHANGE
    if _HC_CHANGE is None:
        _HC_CHANGE = {}
        p = RAW / "schedules.parquet"
        if p.exists():
            s = pd.read_parquet(p)

            def hc(yr):
                d = s[s["season"] == yr]
                m = {}
                for _, g in d.iterrows():
                    m[g["home_team"]] = g["home_coach"]
                    m[g["away_team"]] = g["away_coach"]
                return m
            h25, h26 = hc(2025), hc(2026)
            for t in h26:
                _HC_CHANGE[t] = bool(h26.get(t) != h25.get(t))
    return _HC_CHANGE


def team_coaching(team: str) -> dict:
    rec = dict(_table().get(team, {}))
    rec.setdefault("hc_new", _hc_change().get(team, False))
    rec["hc_new"] = bool(rec.get("hc_new") or _hc_change().get(team, False))
    return rec


def scheme_confidence(team: str) -> float:
    """How much to trust this team's 2025 scheme in 2026 (lower if the staff changed)."""
    return NEW_HC_CONFIDENCE if _hc_change().get(team, False) else 1.0


def clear():
    global _TABLE, _HC_CHANGE
    _TABLE = _HC_CHANGE = None
