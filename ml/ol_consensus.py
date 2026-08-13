"""
ml/ol_consensus.py  —  preseason O-line consensus anchor
========================================================
Our measured O-line grade (ml.squad._ol) is built from 2023-25 PFR pressure rates and
RB run-blocking. That is a real signal, but it is entirely BACKWARD-looking: it cannot see
free agency, the draft, trades or retirements, so every offseason it re-rates a line that
no longer exists. A team that signed two starting tackles in March still carries its old
grade; a team that lost its center and both guards still looks fine.

The published preseason consensus (data/ol_consensus_2026.json) is the opposite: it is
forward-looking and roster-aware — ten independent expert boards, already averaged, which
priced in every offseason move. That is exactly the information our data lacks, which is
why this is a blend and not a replacement: the consensus supplies "who is on the line in
2026", our data supplies measured pass-protection performance the boards only eyeball.

  ol_consensus()      — curated table → DataFrame[rank, avg, sd] indexed by team
  consensus_z()       — consensus avg rank as a z-score, HIGHER = better line
  blend(data_z)       — blend our measured z with the consensus z, per-team weight
                        scaled by how much the ten boards agree (see W_HI / W_LO)

Everything degrades quietly: a missing/corrupt file returns None and blend() hands back the
measured grade untouched, so the engine still builds with no consensus table present.

To refresh: re-paste a newer consensus table into the JSON. Nothing else changes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path(__file__).parent.parent / "data" / "ol_consensus_2026.json"

# Weight placed on the consensus, scaled by expert AGREEMENT (the table's SD column).
# The consensus is already an average of ten independent boards, so it is far better
# anchored than any single source and carries the offseason roster info our data can't see
# — hence it leads the blend. But we trust it MOST where the boards agree (DEN sd=0.3,
# CLE sd=1.4: ten-for-ten, essentially settled) and least where they are split
# (CAR sd=9.2 — ranked 3rd and 29th by different boards, i.e. nobody actually knows), and
# there our measured pressure rates deserve more of a say. Mean weight lands ~0.68.
W_HI = 0.80          # weight on consensus for the most-agreed line in the league
W_LO = 0.55          # weight on consensus for the most-disputed line in the league

_CACHE = None


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0


def ol_consensus() -> pd.DataFrame | None:
    """Curated consensus table as DataFrame[rank, avg, sd] indexed by team, or None."""
    global _CACHE
    if _CACHE is None:
        try:
            raw = json.loads(TABLE.read_text(encoding="utf-8"))
            df = pd.DataFrame(raw["teams"]).T
            _CACHE = df[["rank", "avg", "sd"]].astype(float) if len(df) else None
        except Exception:                       # missing/corrupt table → no anchor, engine still builds
            _CACHE = None
    return _CACHE


def consensus_z() -> pd.Series | None:
    """Consensus average rank as a z-score, HIGHER = better line (rank is negated).

    Uses `avg` rather than the 1-32 ordinal because the mean rank keeps the real gaps:
    DEN (1.1) is far clear of PHI (2.4), while MIN/NO/DAL/ATL/SEA (14.6-15.0) are a
    five-way tie the ordinal would falsely spread across five rank slots."""
    c = ol_consensus()
    return None if c is None else _z(-c["avg"])


def consensus_weight() -> pd.Series | None:
    """Per-team weight on the consensus, W_HI (boards agree) → W_LO (boards split).

    Scaled on the SD's rank percentile rather than its raw value so one wildly split team
    can't compress everyone else's weight toward the floor."""
    c = ol_consensus()
    if c is None:
        return None
    return W_HI - (W_HI - W_LO) * c["sd"].rank(pct=True)


def blend(data_z: pd.Series) -> pd.Series:
    """Blend our measured O-line grade with the consensus anchor. Higher = better.

    Both sides are re-z-scored first: `data_z` arrives as a weighted sum of z-scores, so its
    spread is not 1.0 and blending it raw would silently under-weight the consensus."""
    cz, w = consensus_z(), consensus_weight()
    if cz is None or w is None or data_z.empty:
        return data_z                                   # no table → measured grade, unchanged
    teams = data_z.index.union(cz.index)
    d, c = _z(data_z).reindex(teams), cz.reindex(teams)
    wt = w.reindex(teams).fillna(0.0)
    wt = wt.where(d.notna(), 1.0)                       # team we have no data for → pure consensus
    return (wt * c.fillna(0.0) + (1 - wt) * d.fillna(0.0)).rename(data_z.name)
