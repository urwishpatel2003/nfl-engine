"""
social/script.py  —  the spoken script and the hashtags
=======================================================
build_script(content) -> ordered segments [{key, text, pick}]: a hook, the five picks as a
COUNTDOWN (5 → 1, the strongest cover last — the reason to keep watching), and a close with
the season record and the follow ask. Every number is read the way a bettor says it ("plus five
and a half", "pick 'em"), and the line is short enough to sit on screen as a caption.

hashtags(content, platform) -> list: a platform-tuned base set plus the five teams' nicknames
(#Falcons, #49ers …) which is what the recommendation feeds key on in week-of-game searches.
"""

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen", "twenty"]


def _n(x: float) -> str:
    x = abs(float(x))
    whole, half = int(x), abs(x - int(x)) >= 0.25
    w = _ONES[whole] if whole <= 20 else str(whole)
    return f"{w} and a half" if half else w


def line_words(line: str) -> str:
    """'ATL +5.5' → 'plus five and a half'; 'CAR -2.5' → 'minus two and a half'; 'NE PK' → \"pick 'em\"."""
    parts = line.split()
    if len(parts) < 2 or parts[1] == "PK":
        return "pick 'em"
    v = float(parts[1])
    return ("plus " if v > 0 else "minus ") + _n(v)


def nick(name: str) -> str:
    return (name or "").split()[-1] if name else ""


def _pct(p) -> int:
    return round((p or 0) * 100)


def build_script(c: dict) -> list:
    picks = sorted(c["picks"][:5], key=lambda p: p["rank"])
    best = picks[0] if picks else None
    wk = c["week"]
    segs = []
    # Every word costs a quarter second of a 45-second budget: "percent to cover" is said once
    # in the hook and once on the number one play; the picks in between are team, number,
    # opponent, percent.
    hook = f"Five NFL spreads the model loves in week {wk}."
    if best:
        hook += f" Number one is {_pct(best['cover_prob'])} percent to cover. Stay for it."
    segs.append({"key": "intro", "text": hook, "pick": None})
    for p in reversed(picks):                                  # countdown: 5 → 1
        where = "against" if p["at_home"] else "at"
        if p["rank"] == 1:
            t = f"And number one. {nick(p['team_name'])} {line_words(p['line'])} {where} the {nick(p['opp_name'])}. " \
                f"{_pct(p['cover_prob'])} percent to cover."
            e = abs(float(p["edge"])) if p.get("edge") is not None else 0
            if e >= 3:
                t += f" {round(e)} points off the line."
        else:
            t = f"{_ONES[p['rank']].capitalize()}. {nick(p['team_name'])} {line_words(p['line'])} {where} the " \
                f"{nick(p['opp_name'])}. {_pct(p['cover_prob'])} percent."
        segs.append({"key": f"pick{p['rank']}", "text": t, "pick": p})
    rec = c.get("record") or {}
    close = "That's the board."
    if rec.get("n"):
        close += f" {rec['w']} and {rec['l']} on the top five this season."
    close += " Follow for the graded record. Not betting advice."
    segs.append({"key": "outro", "text": close, "pick": None})
    return segs


_BASE = {
    "youtube": ["#NFL", "#NFLPicks", "#NFLBetting", "#SportsBetting", "#ATS", "#Shorts"],
    "instagram": ["#nfl", "#nflpicks", "#nflbets", "#sportsbetting", "#bettingpicks", "#freepicks", "#ats",
                  "#againstthespread", "#football", "#nflsunday", "#bettingtips"],
    "x": ["#NFL", "#NFLPicks"],
}


def hashtags(c: dict, platform: str) -> list:
    wk = c["week"]
    base = list(_BASE.get(platform, _BASE["youtube"]))
    base.append(f"#NFLWeek{wk}" if platform != "instagram" else f"#nflweek{wk}")
    teams = []
    for p in c["picks"][:5]:
        for name in (p.get("team_name"), p.get("opp_name")):
            t = "#" + "".join(ch for ch in nick(name) if ch.isalnum())
            if platform == "instagram":
                t = t.lower()
            if t not in teams and len(t) > 1:
                teams.append(t)
    limit = {"youtube": 14, "instagram": 24, "x": 3}[platform]
    return (base + teams)[:limit]
