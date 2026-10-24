"""
social/content.py  —  the week's post, built from the frozen picks board
=======================================================================
week_content(season, week) -> {
    season, week, picks: [ {rank, tier, team, team_name, opp, opp_name, home, away, at_home,
                            line: "NE -2.5", cover_prob, edge, kickoff, logo, color} ... ],
    record: {w, l, p, pct, units, n}            # top-5 ATS to date this season (ml.ledger.grade)
    captions: {youtube_title, youtube_desc, instagram, x}
}
Picks come from the same path the Schedule & Picks page uses (dashboard.server._slate +
_overlay_locked), so a post always shows the board that is on the record — after the Saturday
lock that is the frozen one. Imports from the dashboard are done inside the function so this
module can be imported by the server without a cycle.
"""

import os
from datetime import datetime, timezone

SITE = os.environ.get("PUBLIC_BASE_URL", "https://www.halfbloodbettingprince.com").rstrip("/")   # the brand domain (www is canonical; the apex redirects at Cloudflare)
# the brand: the accounts are "Half Blood Betting Prince", wizard theme (owner, 2026-10-22).
# Generic wizardry only (spells, wands, crystal ball) — no quotes or imagery from the books.
BRAND = os.environ.get("BRAND_NAME", "Half Blood Betting Prince")
HANDLE = os.environ.get("BRAND_HANDLE", "@HalfBloodBettingPrince")
BRAND_TAG = "#" + "".join(ch for ch in BRAND if ch.isalnum())
DISCLAIMER = "Model output for tracking. Not betting advice."


def _line_txt(g: dict) -> str:
    """The pick as a bettor reads it: team + its number (home perspective spread → team's)."""
    sp = g.get("vegas_spread")
    team = g["ats_pick"]
    if sp is None:
        return team
    num = -float(sp) if team == g["home"] else float(sp)     # home line is -spread for the home team
    if abs(num) < 0.05:
        return f"{team} PK"
    return f"{team} {num:+.1f}".replace(".0", "")


def week_content(season: int, week: int) -> dict:
    from dashboard import server as S
    from ml.ledger import grade
    slate = S._slate(season, week)
    games = slate["games"]
    S._overlay_locked(season, games)
    picks = []
    for g in sorted([g for g in games if g.get("pick_rank")], key=lambda x: x["pick_rank"]):
        team = g["ats_pick"]
        at_home = team == g["home"]
        opp = g["away"] if at_home else g["home"]
        # a pick whose game is already final (the Thursday game on a Saturday post) carries its result
        result, score = None, None
        if g.get("final") and g.get("home_score") is not None and g.get("away_score") is not None and g.get("vegas_spread") is not None:
            hs, as_ = float(g["home_score"]), float(g["away_score"])
            margin = (hs - as_) if at_home else (as_ - hs)             # the pick's own margin
            line = -float(g["vegas_spread"]) if at_home else float(g["vegas_spread"])   # the pick's own number
            result = "P" if abs(margin + line) < 1e-9 else ("W" if margin + line > 0 else "L")
            score = f"{int(as_ if not at_home else hs)}-{int(hs if not at_home else as_)}"   # pick's points first
        picks.append({
            "result": result, "score": score,
            "rank": int(g["pick_rank"]), "tier": g.get("pick_tier"),
            "team": team, "team_name": g["home_name"] if at_home else g["away_name"],
            "opp": opp, "opp_name": g["away_name"] if at_home else g["home_name"],
            "home": g["home"], "away": g["away"], "at_home": at_home,
            "line": _line_txt(g), "cover_prob": g.get("cover_prob"), "edge": g.get("edge"),
            "kickoff": f"{g.get('gameday', '')} {g.get('gametime', '')}".strip(),
            "logo": g.get("home_logo") if at_home else g.get("away_logo"),
            "opp_logo": g.get("away_logo") if at_home else g.get("home_logo"),
            "color": g.get("home_color") if at_home else g.get("away_color"),
            "final": bool(g.get("final")),
        })
    rec = {}
    try:
        r = grade(season)
        t = (r.get("summary") or {}).get("ats_top") or {}
        rec = {"w": t.get("w", 0), "l": t.get("l", 0), "p": t.get("p", 0), "n": t.get("n", 0),
               "pct": t.get("pct"), "units": t.get("units")}
    except Exception:
        rec = {}
    content = {"season": season, "week": week, "picks": picks, "record": rec,
               "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    content["captions"] = captions(content)
    return content


def _rec_txt(rec: dict) -> str:
    if not rec or not rec.get("n"):
        return ""
    s = f"{rec['w']}-{rec['l']}" + (f"-{rec['p']}" if rec.get("p") else "")
    pct = f" ({round(rec['pct'] * 100)}%)" if rec.get("pct") is not None else ""
    return f"Top-5 ATS this season: {s}{pct}"


def captions(c: dict) -> dict:
    from social.script import hashtags
    wk, season = c["week"], c["season"]
    lines = [f"{p['rank']}. {p['line']} vs {p['opp']} · {round((p['cover_prob'] or 0) * 100)}% cover"
             for p in c["picks"]]
    rec = _rec_txt(c["record"])
    best = c["picks"][0] if c["picks"] else None
    hook = f"The number one spell is {round((best['cover_prob'] or 0) * 100)}% to cover." if best else ""
    # the dashboard URL is deliberately NOT in any post (owner, 2026-10-22: "hide the application url")
    body = f"Week {wk} — five spells against the spread, from the {BRAND}\n\n" + "\n".join(lines) \
        + (f"\n\n{rec}" if rec else "") + f"\n\nFollow {HANDLE} for the graded record every week.\n\n{DISCLAIMER}"
    yt_tags = hashtags(c, "youtube")
    ig_tags = hashtags(c, "instagram")
    x_tags = hashtags(c, "x")
    # X: 280 characters. Picks first, record if it fits, tags last.
    x = f"Week {wk} spells — the Prince's top 5 ATS\n" + "\n".join(
        f"{p['rank']}. {p['line']} ({round((p['cover_prob'] or 0) * 100)}%)" for p in c["picks"])
    for extra in ((f"\n{rec}" if rec else ""), "\n" + " ".join(x_tags)):
        if len(x) + len(extra) <= 275:
            x += extra
    return {
        # YouTube ranks Shorts on the first seconds + the title; hashtags in the description count
        "youtube_title": f"Week {wk} NFL picks: 5 spells against the spread (#1 is {round((best['cover_prob'] or 0) * 100)}% to cover) #Shorts"[:100] if best
                         else f"Week {wk} NFL picks: five spells against the spread #Shorts",
        "youtube_desc": (hook + "\n\n" if hook else "") + body + "\n\n" + " ".join(yt_tags),
        "youtube_tags": [t.lstrip("#") for t in yt_tags] + ["NFL picks", "sports betting", "against the spread"],
        "instagram": (body + "\n.\n.\n.\n" + " ".join(ig_tags))[:2190],
        "x": x,
    }
