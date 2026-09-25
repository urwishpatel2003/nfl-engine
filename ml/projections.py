"""
ml/projections.py  —  per-player game stat projections (SportsLine-style)
==========================================================================
Projects each player's expected stat line for a specific matchup, the same core
idea as SportsLine's model (minus their proprietary Monte-Carlo sims):

    player projection = usage share x team volume x efficiency, shaped by game script

Pipeline
  1. player_profiles()  — 2025 per-game usage + efficiency + team shares, from pbp_2025
     (player_stats/seasonal_stats are stale at 2024, so we aggregate box scores from PBP).
  2. team_volume()      — league/team pace: plays, pass & rush attempts per game.
  3. project_matchup()  — combine with the matchup score/total (game script) and
     distribute team volume to the players on the current depth chart.

Honest limits: no snap-count injuries, rookies with no 2025 usage are omitted, TD
projections are noisy, opponent-defense adjustment is coarse. It's a credible
baseline, not a betting tool.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"

_PROFILE_CACHE = None
_QBDEPTH_CACHE = None

# Injury statuses that mean "won't play" → excluded from projections. Questionable
# players are assumed active (they suit up ~75% of the time).
OUT_STATUSES = ("Out", "Doubtful")

# nflverse roster `status` / `status_description_abbr` → short reserve-list label. Anyone
# on one of these lists cannot suit up regardless of the injury report (which usually
# doesn't even list them), so they're unavailable to every projection layer.
RESERVE_STATUS = {"RES": "RES", "EXE": "EXE", "RET": "RET", "CUT": "CUT"}
RESERVE_DESC = {"R01": "IR", "R48": "IR",          # Reserve/Injured (+ designated-to-return)
                "R04": "PUP", "R05": "NFI", "R27": "NFI",
                "E02": "EXE"}                      # Commissioner exempt list
_STATUS_PRIORITY = {"ACT": 0, "DEV": 1}            # a player cut by A and signed by B keeps B


def current_reports(inj: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each team's most recent injury report WITHIN THE CURRENT SEASON — the newest season
    present in injuries.parquet. It deliberately never falls back to a prior season: before a
    team's first report of the year it just has no rows. (The old "latest week per team"
    rule resurrected last January's Out list on week-1 Wednesday for every team whose
    report hadn't published yet, quietly deleting healthy starters from the projection.)"""
    if inj is None:
        p = RAW / "injuries.parquet"
        inj = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if inj.empty or not {"season", "week", "team"} <= set(inj.columns):
        return pd.DataFrame()
    cur = inj[inj["season"] == inj["season"].max()].copy()
    cur["week"] = cur["week"].astype(int)
    return cur[cur["week"] == cur.groupby("team")["week"].transform("max")]


def reserve_ids(team: str | None = None) -> dict:
    """{gsis_id: label} for players on a reserve/exempt list in the current roster release
    (IR, PUP, NFI, exempt, retired). Optional team filter."""
    p = RAW / "rosters_2026.parquet"
    if not p.exists():
        return {}
    r = pd.read_parquet(p, columns=["team", "player_id", "status", "status_description_abbr"])
    r = r.dropna(subset=["player_id"])
    r["_pri"] = r["status"].map(_STATUS_PRIORITY).fillna(9)
    r = r.sort_values("_pri").drop_duplicates("player_id")        # ACT row wins across teams
    r = r[r["status"].isin(RESERVE_STATUS)]
    if team is not None:
        r = r[r["team"] == team]
    return {str(g): RESERVE_DESC.get(str(d), RESERVE_STATUS.get(str(s), "RES"))
            for g, s, d in zip(r["player_id"], r["status"], r["status_description_abbr"])}


_REST_WORDS = ("rest", "not injury", "personal", "coach", "veteran")
_DNP = "Did Not Participate In Practice"


def _injury_dnp(row) -> bool:
    """A DNP with an actual injury listed and NO game designation yet. Wednesday–Thursday
    reports carry practice status only; the Out/Questionable call comes Friday. A player
    who is not practising because of an injury (Daniels, elbow) is treated as unavailable
    until that call arrives — a rest-day DNP ('not injury related - resting player') is not."""
    st = row.get("report_status")
    if not (st is None or (isinstance(st, float) and pd.isna(st))):
        return False
    if str(row.get("practice_status") or "") != _DNP:
        return False
    why = str(row.get("practice_primary_injury") or "").lower()
    return bool(why) and not any(w in why for w in _REST_WORDS)


def unavailable_map(statuses=OUT_STATUSES) -> dict:
    """{gsis_id: label} of players who won't play: ruled Out/Doubtful in each team's
    current-season report (same report the dashboard's injury panel shows), injury DNPs that
    have no designation yet (label 'DNP'), PLUS everyone on a reserve/exempt list in the
    roster release. The label is the reason so a projection can say WHO it left out and why."""
    ids = dict(reserve_ids())
    inj = current_reports()
    if not inj.empty and {"gsis_id", "report_status"} <= set(inj.columns):
        inj = inj.dropna(subset=["gsis_id"])
        for _, r in inj.iterrows():
            g, st = str(r["gsis_id"]), r.get("report_status")
            if st in statuses:
                ids[g] = str(st)                       # a game designation outranks a reserve tag
            elif _injury_dnp(r):
                ids[g] = "DNP"
    return ids


def unavailable_ids(statuses=OUT_STATUSES) -> set:
    """The keys of unavailable_map(): every gsis_id excluded from projections."""
    return set(unavailable_map(statuses))


def _depth_qbs() -> dict:
    """{team: [(gsis_id, name), …]} ordered by depth-chart rank."""
    global _QBDEPTH_CACHE
    if _QBDEPTH_CACHE is None:
        dc = pd.read_parquet(RAW / "depth_2026_current.parquet").copy()
        dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
        q = dc[dc.pos_abb == "QB"].sort_values(["team", "pos_rank"])
        _QBDEPTH_CACHE = {t: [(r.gsis_id, r.player_name) for r in g.itertuples()]
                          for t, g in q.groupby("team")}
    return _QBDEPTH_CACHE


_RANK_CACHE = None


def _depth_rank() -> dict:
    """{gsis_id: best pos_rank on the current depth chart} (1 = starter)."""
    global _RANK_CACHE
    if _RANK_CACHE is None:
        try:
            dc = pd.read_parquet(RAW / "depth_2026_current.parquet", columns=["gsis_id", "pos_rank"])
            dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
            _RANK_CACHE = dc.dropna().groupby("gsis_id")["pos_rank"].min().astype(int).to_dict()
        except Exception:
            _RANK_CACHE = {}
    return _RANK_CACHE


def _available_qb(team: str, unavail: set):
    """First depth-chart QB who isn't ruled out (falls back to QB1 if all are)."""
    qs = _depth_qbs().get(team, [])
    for gid, name in qs:
        if gid not in unavail:
            return gid, name
    return qs[0] if qs else (None, None)


# ── 1. player box-score aggregation from PBP ────────────────────────
_BOX_CACHE: dict = {}


def _player_box(season: int):
    """Aggregate one season's regular-season box-score totals per player from play-by-play.
    Returns (box, team_pa, team_ra); an empty triple if the season has no PBP yet."""
    if season in _BOX_CACHE:
        return _BOX_CACHE[season]
    path = RAW / f"pbp_{season}.parquet"
    if not path.exists():
        empty = pd.DataFrame(columns=["player_id", "games"])
        return empty, pd.DataFrame(columns=["posteam", "team_pa", "team_g"]), pd.DataFrame(columns=["posteam", "team_ra"])
    p = pd.read_parquet(path)
    p = p[(p["week"] <= 18) & p["play_type"].isin(["pass", "run"])].copy()
    p["ret_td"] = p.get("return_touchdown", 0)
    p["is_td"] = (p["touchdown"] == 1) & (p["ret_td"] != 1)

    # passing (by passer)
    pa = p[p["pass_attempt"] == 1].copy()
    pa["cmp_yds"] = pa["yards_gained"] * pa["complete_pass"]
    pa["ptd"] = (pa["is_td"] & (pa["complete_pass"] == 1)).astype(int)
    passing = pa.groupby("passer_player_id").agg(
        g_pass=("game_id", "nunique"), pass_att=("pass_attempt", "sum"),
        cmp=("complete_pass", "sum"), pass_yds=("cmp_yds", "sum"),
        pass_td=("ptd", "sum"), interc=("interception", "sum")).reset_index() \
        .rename(columns={"passer_player_id": "player_id"})

    # rushing (by rusher)
    ru = p[p["rush_attempt"] == 1].copy()
    ru["rtd"] = ru["is_td"].astype(int)
    rushing = ru.groupby("rusher_player_id").agg(
        g_rush=("game_id", "nunique"), carries=("rush_attempt", "sum"),
        rush_yds=("yards_gained", "sum"), rush_td=("rtd", "sum")).reset_index() \
        .rename(columns={"rusher_player_id": "player_id"})

    # receiving (by receiver)
    rc = p[(p["pass_attempt"] == 1) & p["receiver_player_id"].notna()].copy()
    rc["rec_yds"] = rc["yards_gained"] * rc["complete_pass"]
    rc["rectd"] = (rc["is_td"] & (rc["complete_pass"] == 1)).astype(int)
    receiving = rc.groupby("receiver_player_id").agg(
        g_rec=("game_id", "nunique"), targets=("pass_attempt", "sum"),
        rec=("complete_pass", "sum"), rec_yds=("rec_yds", "sum"),
        rec_td=("rectd", "sum"), air=("air_yards", "sum")).reset_index() \
        .rename(columns={"receiver_player_id": "player_id"})

    # team pass/rush attempts per game (for shares)
    team_pa = pa.groupby("posteam").agg(team_pa=("pass_attempt", "sum"),
                                        team_g=("game_id", "nunique")).reset_index()
    team_ra = ru.groupby("posteam")["rush_attempt"].sum().reset_index(name="team_ra")

    box = passing.merge(rushing, on="player_id", how="outer").merge(receiving, on="player_id", how="outer")
    box["games"] = box[["g_pass", "g_rush", "g_rec"]].max(axis=1)
    _BOX_CACHE[season] = (box.fillna(0), team_pa, team_ra)
    return _BOX_CACHE[season]


_VOL = ["pass_att", "carries", "targets"]
_TOT = ["pass_att", "cmp", "pass_yds", "pass_td", "interc", "carries", "rush_yds", "rush_td",
        "targets", "rec", "rec_yds", "rec_td", "air"]


def _combined_box():
    """Last season + this season's box totals per player.

    EFFICIENCY counts (completions, yards, TDs…) are simply summed across both seasons —
    more attempts is more evidence, and the shrinkage in player_profiles handles the size.
    VOLUME per game is blended toward this season's per-game rate by the player's own games
    played, w = g/(g+4) (the same rule ml/current.py uses for teams): a role changes with a
    new team or a new depth chart, and two starts already say more about a player's usage
    than last year's mop-up snaps. Returns (box, games_prior, games_cur)."""
    from ml.current import state, K_GAMES
    st = state()
    cur, prior = st["season"], st["prior"] or (st["season"] - 1)
    b1, _, _ = _player_box(prior)
    b2, _, _ = _player_box(cur) if st["in_progress"] else (pd.DataFrame(columns=["player_id", "games"]), None, None)
    b1 = b1.set_index("player_id"); b2 = b2.set_index("player_id") if len(b2) else b2
    ids = b1.index.union(b2.index) if len(b2) else b1.index
    out = pd.DataFrame(index=ids)
    for c in _TOT:
        out[c] = b1[c].reindex(ids).fillna(0) + (b2[c].reindex(ids).fillna(0) if len(b2) and c in b2.columns else 0)
    g1 = b1["games"].reindex(ids).fillna(0)
    # THIS season's per-game usage is over the TEAM's games, not the player's: a player who
    # has not appeared while his team played two games is being used zero times a game, and
    # that is evidence (last season stays per player-game so an injury year is not a zero).
    try:
        team_of = pd.read_parquet(RAW / "rosters_2026.parquet", columns=["player_id", "team"]) \
            .dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")["team"]
        tg = team_of.reindex(ids).map(st["games"]).fillna(0).astype(float)
    except Exception:
        tg = pd.Series(0.0, index=ids)
    g2 = tg if st["in_progress"] else pd.Series(0.0, index=ids)
    # ROLE PRIOR: what a player in this depth-chart slot typically gets per game — last
    # season's league distribution at his position, by rank (WR1 ≈ 70th percentile of WR
    # targets/game, WR2 ≈ 45th, WR3 ≈ 25th, deeper ≈ 10th). Every season's per-game rate is
    # shrunk toward it by games played, so one 7-target game does not make a newcomer the
    # team's WR1, and a 17-game veteran barely moves.
    try:
        rost = pd.read_parquet(RAW / "rosters_2026.parquet", columns=["player_id", "position"]) \
            .dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")["position"]
    except Exception:
        rost = pd.Series(dtype=object)
    rank = pd.Series(_depth_rank()).reindex(ids).fillna(4).clip(1, 4).astype(int)
    pos = rost.reindex(ids).fillna("")
    # base = everyone with 4+ games (an 8-game floor kept only real contributors and made the
    # WR2/WR3 priors read like starters); depth-chart slots repeat rank 1 (LWR/RWR/SWR), so
    # rank 2 is already a backup and gets a backup's prior
    base = b1[b1["games"] >= 4]
    base_pos = rost.reindex(base.index).fillna("")
    q_of_rank = {1: 0.65, 2: 0.30, 3: 0.15, 4: 0.05}
    w1, w2 = g1 / (g1 + K_GAMES), g2 / (g2 + K_GAMES)
    for c in _VOL:
        prior = pd.Series(0.0, index=ids)
        bpg = base[c] / base["games"].clip(lower=1)
        for p_ in ("QB", "RB", "WR", "TE", "FB"):
            sub = bpg[base_pos == p_]
            if not len(sub):
                continue
            for r_, q_ in q_of_rank.items():
                m = (pos == p_) & (rank == r_)
                if m.any():
                    prior[m] = float(sub.quantile(q_))
        pg1 = (b1[c].reindex(ids).fillna(0) / g1.clip(lower=1)).where(g1 > 0, 0.0)
        pg2 = (b2[c].reindex(ids).fillna(0) / g2.clip(lower=1)).where(g2 > 0, 0.0) if len(b2) and c in b2.columns else pd.Series(0.0, index=ids)
        last = w1 * pg1 + (1 - w1) * prior            # last season, shrunk toward the role prior
        out[c + "_pg"] = (w2 * pg2 + (1 - w2) * last).values
    out["games"] = g1 + (b2["games"].reindex(ids).fillna(0) if len(b2) else 0)
    out["games_cur"] = b2["games"].reindex(ids).fillna(0) if len(b2) else 0
    return out.reset_index().rename(columns={"index": "player_id"})


# ── 2. per-game profiles + usage shares ─────────────────────────────
def player_profiles() -> pd.DataFrame:
    """Per-player per-game usage + efficiency, with current team & position from the roster."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    box = _combined_box()

    # attach current team + position + name from the 2026 roster / depth chart
    rost = pd.read_parquet(RAW / "rosters_2026.parquet")[["player_id", "player_name", "team", "position"]]
    df = box.merge(rost, on="player_id", how="inner")
    df = df[df["games"] > 0].copy()

    # per-game volume: blended last-season / this-season rates from _combined_box
    df["att_pg"] = df["pass_att_pg"]
    # Per-attempt EFFICIENCY is shrunk toward a replacement-level prior by sample size:
    #     rate = (n·observed + K·prior) / (n + K)
    # Without this a backup's mop-up line became a starter's projection — Malik Willis, 38
    # attempts in 2025 at 11.1 yds/att and 79% completion, projected for 340 yards against
    # Kansas City once handed 32 attempts. Volume (per-game counts) is NOT shrunk; it is
    # re-derived from team volume and depth-chart role at projection time anyway.
    # K is "how many attempts before the observed rate outweighs the prior": 150 for a QB
    # (about a month as a starter), 60 carries, 40 targets. The QB prior is the same
    # replacement line a rookie/unknown starter gets (ROOKIE_QB); the skill priors are
    # league-typical rates. Priors, not fitted constants.
    def shrink(num, den, prior, k):
        return (num + k * prior) / (den + k)
    pa_, ca_, tg_ = df["pass_att"], df["carries"], df["targets"]
    df["cmp_pct"] = shrink(df["cmp"], pa_, ROOKIE_QB["cmp_pct"], K_PASS_ATT)
    df["ypa"] = shrink(df["pass_yds"], pa_, ROOKIE_QB["ypa"], K_PASS_ATT)
    df["ptd_pa"] = shrink(df["pass_td"], pa_, ROOKIE_QB["ptd_pa"], K_PASS_ATT)
    df["int_pa"] = shrink(df["interc"], pa_, ROOKIE_QB["int_pa"], K_PASS_ATT)
    df["carry_pg"] = df["carries_pg"]
    df["ypc"] = shrink(df["rush_yds"], ca_, SKILL_PRIOR["ypc"], K_CARRIES)
    df["rtd_carry"] = shrink(df["rush_td"], ca_, SKILL_PRIOR["rtd_carry"], K_CARRIES)
    df["tgt_pg"] = df["targets_pg"]
    df["catch_pct"] = shrink(df["rec"], tg_, SKILL_PRIOR["catch_pct"], K_TARGETS)
    df["ypt"] = shrink(df["rec_yds"], tg_, SKILL_PRIOR["ypt"], K_TARGETS)
    df["rectd_tgt"] = shrink(df["rec_td"], tg_, SKILL_PRIOR["rectd_tgt"], K_TARGETS)

    # team shares: the player's blended per-game volume over his CURRENT team's blended
    # per-game volume (a share of last year's old team means nothing for this year's plan)
    tv = team_volume()
    df["team_pa_pg"] = df["team"].map(lambda t: tv["by_team"].get(t, {}).get("pa_pg", tv["lg_pa"]))
    df["team_ra_pg"] = df["team"].map(lambda t: tv["by_team"].get(t, {}).get("ra_pg", tv["lg_ra"]))
    df["target_share"] = np.where(df["team_pa_pg"] > 0, df["tgt_pg"] / df["team_pa_pg"], 0)
    df["carry_share"] = np.where(df["team_ra_pg"] > 0, df["carry_pg"] / df["team_ra_pg"], 0)
    num = df.select_dtypes("number").columns          # float32 -> float64 for clean rounding
    df[num] = df[num].astype("float64")
    _PROFILE_CACHE = df
    return df


# ── 3. team volume (pace) ───────────────────────────────────────────
_TV_CACHE = None


def team_volume() -> dict:
    """League-average and per-team pass/rush attempts per game: last season blended toward
    this season by each team's games played (ml/current.py weights)."""
    global _TV_CACHE
    if _TV_CACHE is not None:
        return _TV_CACHE
    from ml.current import state, weights
    st = state()
    prior = st["prior"] or (st["season"] - 1)

    def _vol(season):
        _, team_pa, team_ra = _player_box(season)
        t = team_pa.merge(team_ra, on="posteam", how="outer").fillna(0)
        t["pa_pg"] = t["team_pa"] / t["team_g"].clip(lower=1)
        t["ra_pg"] = t["team_ra"] / t["team_g"].clip(lower=1)
        return t.set_index("posteam")[["pa_pg", "ra_pg"]]
    t = _vol(prior)
    w = weights()
    if st["in_progress"] and len(w):
        now = _vol(st["season"]).reindex(t.index)
        wv = w.reindex(t.index).fillna(0.0)
        for c in ("pa_pg", "ra_pg"):
            t[c] = wv * now[c].fillna(t[c]) + (1 - wv) * t[c]
    _TV_CACHE = {"by_team": t.to_dict("index"),
                 "lg_pa": float(t["pa_pg"].mean()), "lg_ra": float(t["ra_pg"].mean())}
    return _TV_CACHE


# QB starters (depth chart) and a rookie/replacement prior
_QB_START = None


def _qb_starters() -> dict:
    """{team: (gsis_id, name)} for the current depth-chart QB1."""
    global _QB_START
    if _QB_START is None:
        dc = pd.read_parquet(RAW / "depth_2026_current.parquet")
        dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
        qb1 = dc[(dc.pos_abb == "QB") & (dc.pos_rank == 1)].drop_duplicates("team")
        _QB_START = {r.team: (r.gsis_id, r.player_name) for r in qb1.itertuples()}
    return _QB_START


# league-average starter line (used for rookies / no-2025-usage starters)
ROOKIE_QB = {"cmp_pct": 0.62, "ypa": 6.4, "ptd_pa": 0.036, "int_pa": 0.028,
             "carry_pg": 3.0, "ypc": 4.0, "rtd_carry": 0.03}
# league-typical per-touch rates: the prior a small-sample RB/WR/TE is shrunk toward
SKILL_PRIOR = {"ypc": 4.2, "rtd_carry": 0.03, "catch_pct": 0.65, "ypt": 7.5, "rectd_tgt": 0.045}
K_PASS_ATT, K_CARRIES, K_TARGETS = 150, 60, 40      # sample size at which observed = prior weight


# ── 4. distribute team volume to players for a matchup ──────────────
def _distribute(team: str, team_pa: float, team_ra: float, off_tds: float, prof: pd.DataFrame,
                pass_factor: float = 1.0, rush_factor: float = 1.0, unavail: set = frozenset()) -> dict:
    """Allocate a team's projected pass/rush volume to its AVAILABLE players (injured players
    are dropped so their carries/targets redistribute), with the opponent-defense adjustment
    (pass_factor for the air game, rush_factor for the ground)."""
    roster = prof[(prof.team == team) & (~prof.player_id.isin(unavail))]
    pass_tds = off_tds * 0.62 * pass_factor
    rush_tds = off_tds - off_tds * 0.62          # ground TDs unaffected by pass factor

    # QB: first depth-chart QB who isn't ruled out (rookie/no-2025 -> replacement prior).
    # QB rushing is the QB's OWN 2025 rate (scrambles + designed runs both count as rush
    # attempts in PBP), so a dual threat (Lamar ~10 car/g) projects very differently from a
    # pocket passer (~2) — carries, matchup-scaled yards, AND a share of the rushing TDs.
    sid, sname = _available_qb(team, unavail)
    qrow = roster[roster.player_id == sid]
    q = qrow.iloc[0] if not qrow.empty else None
    rk = ROOKIE_QB
    qb_car = float(q.carry_pg) if q is not None else rk["carry_pg"]
    qb_ypc = float(q.ypc) if q is not None else rk["ypc"]
    qb_tdw = qb_car * (float(q.rtd_carry) if q is not None else rk["rtd_carry"])
    if q is None:
        # rookie / unknown starter — use the depth-chart name with a replacement line
        qb_line = {"name": sname or "Starter", "pos": "QB", "rookie": True,
                   "pass_att": round(team_pa), "cmp": round(team_pa * rk["cmp_pct"]),
                   "pass_yds": round(team_pa * rk["ypa"] * pass_factor), "pass_td": round(pass_tds, 1),
                   "int": round(team_pa * rk["int_pa"], 1)}
    else:
        qb_line = {"name": q.player_name, "pos": "QB", "rookie": False,
                   "pass_att": round(team_pa), "cmp": round(team_pa * q.cmp_pct),
                   "pass_yds": round(team_pa * q.ypa * pass_factor), "pass_td": round(pass_tds, 1),
                   "int": round(team_pa * q.int_pa, 1)}
    qb_line["carries"] = round(qb_car)
    qb_line["rush_yds"] = round(qb_car * qb_ypc * rush_factor)

    # Rushers: QB carries come off the top of team volume, the backfield (top 4) splits the
    # rest; rushing TDs are shared across QB + RBs by each rusher's own TD rate — so Hurts'
    # and Lamar's sneak/read-option TDs stop being handed to their running backs.
    rbs = roster[(roster.position == "RB") & (roster.carry_pg > 1)].sort_values(
        "carry_pg", ascending=False).head(4).copy()
    rb_ra = max(team_ra - qb_car, team_ra * 0.5)
    tdw_total = max(1e-6, qb_tdw + ((rbs.carry_pg * rbs.rtd_carry).sum() if not rbs.empty else 0.0))
    qb_line["rush_td"] = round(rush_tds * qb_tdw / tdw_total, 1)
    # Receivers: concentrate targets on the actual pass-catchers (top 6), scaled by pass matchup,
    # then RECONCILED to the QB line — receptions sum to his completions, yards to his passing
    # yards, TDs to his passing TDs. Receiver and passer rates are estimated separately, so
    # without this the receiving column ran ~10% above the passing column on the same page.
    recs = roster[roster.position.isin(["WR", "TE", "RB"]) & (roster.tgt_pg > 0.5)].sort_values(
        "tgt_pg", ascending=False).head(6).copy()
    rec_lines = []
    if not recs.empty:
        denom = recs["tgt_pg"].sum(); tdw = max(1e-6, (recs.tgt_pg * recs.rectd_tgt).sum())
        raw = []
        for _, r in recs.iterrows():
            tg = team_pa * 0.95 * (r.tgt_pg / denom)
            raw.append({"name": r.player_name, "pos": r.position, "targets": tg, "rec": tg * r.catch_pct,
                        "rec_yds": tg * r.ypt * pass_factor, "rec_td": pass_tds * (r.tgt_pg * r.rectd_tgt) / tdw})
        s_rec = sum(x["rec"] for x in raw) or 1.0
        s_yds = sum(x["rec_yds"] for x in raw) or 1.0
        s_td = sum(x["rec_td"] for x in raw) or 1.0
        k_rec, k_yds, k_td = qb_line["cmp"] / s_rec, qb_line["pass_yds"] / s_yds, qb_line["pass_td"] / s_td
        for x in raw:
            rec_lines.append({"name": x["name"], "pos": x["pos"], "targets": round(x["targets"]),
                              "rec": round(x["rec"] * k_rec), "rec_yds": round(x["rec_yds"] * k_yds),
                              "rec_td": round(x["rec_td"] * k_td, 1)})
    rec_by_name = {x["name"]: x for x in rec_lines}

    rush_lines = []
    if not rbs.empty:
        denom = rbs["carry_pg"].sum()
        for _, r in rbs.iterrows():
            car = rb_ra * (r.carry_pg / denom)
            rl = rec_by_name.get(r.player_name)                 # the RB's receiving is the reconciled line
            rush_lines.append({"name": r.player_name, "pos": "RB",
                               "carries": round(car), "rush_yds": round(car * r.ypc * rush_factor),
                               "rush_td": round(rush_tds * (r.carry_pg * r.rtd_carry) / tdw_total, 1),
                               "targets": rl["targets"] if rl else 0, "rec": rl["rec"] if rl else 0,
                               "rec_yds": rl["rec_yds"] if rl else 0, "rec_td": rl["rec_td"] if rl else 0})
    return {"qb": qb_line, "rush": rush_lines, "rec": rec_lines}


def injury_impact(team: str, unavail: set = None) -> dict:
    """Points penalty for a team's ruled-out contributors (drives the spread). QB dominates;
    skill players give a smaller, capped hit since a replacement recovers most of the usage."""
    if unavail is None:
        unavail = unavailable_ids()
    prof = player_profiles()
    r = prof[prof.team == team]
    pen, who = 0.0, []
    # QB1 out → penalty scaled by the gap to the best available backup
    from ml.squad import _qb_value_table
    qbp = _qb_value_table().rank(pct=True) * 100
    qs = _depth_qbs().get(team, [])
    if qs and qs[0][0] in unavail:
        s = float(qbp.get(qs[0][0], 60.0))
        b = next((float(qbp.get(g, 40.0)) for g, _ in qs[1:] if g not in unavail), 35.0)
        d = max(0.0, (s - b) / 100.0 * 7.0)                # elite→replacement ≈ up to 7 pts
        if d > 0.1:
            pen += d; who.append(f"{qs[0][1]} (QB)")
    # skill starters out → net loss after a ~65% replacement recovers most of the share.
    # Only the top two on the CURRENT depth chart count: usage is last season's and travels
    # with the player, so a back who carried 9/g elsewhere and now sits 4th on this chart
    # (Pacheco, IR, on DET's chart at RB4) was charging DET for a share it never planned on.
    rank = _depth_rank()
    for _, p in r[r.player_id.isin(unavail)].iterrows():
        if p.position in ("RB", "WR", "TE") and (rank.get(p.player_id) or 9) <= 2:
            share = float(p.get("target_share", 0) or 0) + float(p.get("carry_share", 0) or 0)
            loss = min(2.0, share * 6.0 * 0.35)
            if loss > 0.2:
                pen += loss; who.append(f"{p.player_name} ({p.position})")
    return {"pts": round(min(pen, 10.0), 1), "players": who}


def project_matchup(home: str, away: str, neutral: bool = False) -> dict:
    """Full matchup: unit-vs-unit score + opponent-adjusted player stat lines, using only
    AVAILABLE players (injured players are excluded and their usage redistributed)."""
    from ml.matchup_engine import project_game, team_units
    pred = project_game(home, away, neutral=neutral)
    u = team_units()
    tv = team_volume()
    prof = player_profiles()
    umap = unavailable_map()
    unavail = set(umap)
    points = {home: pred["pred_home_score"], away: pred["pred_away_score"]}
    margin = {home: pred["pred_margin"], away: -pred["pred_margin"]}

    def excluded(team) -> list:
        """Who the projection left out and why — skill players with real 2025 usage only,
        so the note names the absences that actually moved the box score."""
        ex = prof[(prof.team == team) & prof.player_id.isin(unavail)
                  & ((prof.position == "QB") | (prof.tgt_pg > 0.5) | (prof.carry_pg > 1))].copy()
        ex["use"] = ex["tgt_pg"].fillna(0) + ex["carry_pg"].fillna(0)
        return [{"name": r.player_name, "pos": r.position, "status": umap.get(str(r.player_id), "Out")}
                for r in ex.sort_values("use", ascending=False).itertuples()
                if umap.get(str(r.player_id)) not in ("CUT", "RET")]   # not on the team ≠ injured

    teams = {}
    for team, opp in [(home, away), (away, home)]:
        vol = tv["by_team"].get(team, {"pa_pg": tv["lg_pa"], "ra_pg": tv["lg_ra"]})
        # game script: trailing team throws more (each point of deficit ~0.35 plays pass-ward)
        shift = float(np.clip(-margin[team] * 0.35, -6, 6))
        team_pa = vol["pa_pg"] + shift
        team_ra = max(12.0, vol["ra_pg"] - shift)
        off_tds = max(0.0, (points[team] - 1.2) / 7.0)   # approx offensive TDs
        # opponent-defense adjustment: bad D (positive z_def EPA allowed) -> more player yards
        pass_factor = float(np.clip(1 + 0.14 * u.loc[opp, "z_def_pass"], 0.75, 1.30)) if opp in u.index else 1.0
        rush_factor = float(np.clip(1 + 0.14 * u.loc[opp, "z_def_rush"], 0.75, 1.30)) if opp in u.index else 1.0
        teams[team] = _distribute(team, team_pa, team_ra, off_tds, prof, pass_factor, rush_factor, unavail)
        teams[team]["excluded"] = excluded(team)

    return {"home": home, "away": away, "pred": pred, "teams": teams}


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        import json
        r = project_matchup(sys.argv[1].upper(), sys.argv[2].upper())
        print(f"{r['away']} @ {r['home']}: {r['pred']['pred_away_score']}-{r['pred']['pred_home_score']}")
        for tm in (r["home"], r["away"]):
            t = r["teams"][tm]
            print(f"\n{tm}:")
            if t["qb"]:
                q = t["qb"]; print(f"  QB {q['name']}: {q['cmp']}/{q['pass_att']}, {q['pass_yds']} yds, {q['pass_td']} TD, {q['int']} INT")
            for x in t["rush"][:3]:
                print(f"  RB {x['name']}: {x['carries']} car, {x['rush_yds']} yds, {x['rush_td']} TD | {x['rec']}-{x['rec_yds']} rec")
            for x in t["rec"][:4]:
                print(f"  {x['pos']} {x['name']}: {x['rec']}/{x['targets']}, {x['rec_yds']} yds, {x['rec_td']} TD")
        sys.exit(0)
    prof = player_profiles()
    print(f"players with 2025 usage: {len(prof)}")
    for pos in ["QB", "RB", "WR"]:
        top = prof[prof.position == pos].nlargest(3, "att_pg" if pos == "QB" else "tgt_pg" if pos != "RB" else "carry_pg")
        print(f"\nTop {pos}:")
        cols = (["player_name", "team", "att_pg", "ypa", "cmp_pct", "ptd_pa"] if pos == "QB"
                else ["player_name", "team", "carry_pg", "ypc", "tgt_pg", "carry_share"] if pos == "RB"
                else ["player_name", "team", "tgt_pg", "ypt", "catch_pct", "target_share"])
        print(top[cols].round(2).to_string(index=False))
