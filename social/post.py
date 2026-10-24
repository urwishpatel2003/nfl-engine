"""
social/post.py  —  render the week's picks and publish them, once per platform per week
=======================================================================================
post_week(season, week, dry_run=False, platforms=None, force=False)
    1. content  (social.content.week_content — the frozen board + season record + captions)
    2. render   data/processed/social/{season}_wk{NN}/picks.png + picks.mp4 + content.json
    3. publish  each platform whose credentials are set and that has not already posted this
                week (data/processed/social_posts.jsonl is the ledger — one line per attempt;
                a success is never repeated unless force=True). dry_run stops after step 2.

post_if_due(season)
    The automatic trigger, called at the end of every refresh when SOCIAL_AUTO=1: posts the
    current week once the Saturday lock has passed (the board is frozen, so what is posted is
    what gets graded), while games are still to be played, and only if nothing has been posted
    for that week yet. The light refresh runs every 4 h from 09:00 UTC, so the 13:00 UTC run
    lands right after the 13:00 UTC lock.

CLI
    python -m social.post --season 2026 --week 7 --dry-run        # render only, print captions
    python -m social.post --season 2026 --week 7 --platforms x    # publish to X only
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROC = Path(__file__).parent.parent / "data" / "processed"
DIR = PROC / "social"
LOG = PROC / "social_posts.jsonl"
PLATFORMS = ("youtube", "instagram", "x")
_LAST = {"running": False, "result": None, "started": None}


def _log_rows() -> list:
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def posted(season: int, week: int, platform: str) -> dict | None:
    for r in reversed(_log_rows()):
        if r.get("season") == season and r.get("week") == week and r.get("platform") == platform and r.get("ok"):
            return r
    return None


def _append(row: dict) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def render_week(season: int, week: int) -> dict:
    from social.content import week_content
    from social.render import render_image, render_video
    from social.script import build_script
    from social.audio import tts_segments, mix, SR
    c = week_content(season, week)
    if not c["picks"]:
        raise RuntimeError(f"no picks on the board for {season} wk{week}")
    out = DIR / f"{season}_wk{week:02d}"
    out.mkdir(parents=True, exist_ok=True)
    # the video: a spoken countdown script, each card held for its line (+ a beat), music under
    segs = build_script(c)
    voices = tts_segments(segs, out)
    min_hold = {"intro": 2.6, "outro": 3.0}
    holds = [max(min_hold.get(s["key"], 2.2), (len(v) / SR if v is not None else 0) + 0.4) for s, v in zip(segs, voices)]
    audio = mix(voices, holds, str(out / "audio.wav"))
    c["script"] = [s["text"] for s in segs]
    c["voiced"] = sum(v is not None for v in voices)
    c["duration_s"] = round(sum(holds), 1)
    c["files"] = {"image": str(render_image(c, str(out / "picks.png"))),
                  "video": str(render_video(c, str(out / "picks.mp4"), segments=segs, holds=holds, audio=audio))}
    c["public"] = {k: f"{os.environ.get('PUBLIC_BASE_URL', 'https://www.halfbloodbettingprince.com').rstrip('/')}"
                      f"/social/{season}_wk{week:02d}/{Path(v).name}" for k, v in c["files"].items()}
    (out / "content.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    return c


def post_week(season: int, week: int, dry_run: bool = False, platforms=None, force: bool = False) -> dict:
    from social import publish as P
    platforms = [p for p in (platforms or PLATFORMS) if p in PLATFORMS]
    c = render_week(season, week)
    res = {"season": season, "week": week, "files": c["files"], "captions": c["captions"], "results": {}}
    if dry_run:
        res["dry_run"] = True
        return res
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for pf in platforms:
        if not force and posted(season, week, pf):
            res["results"][pf] = {"ok": True, "skipped": "already posted"}
            continue
        have = {"youtube": P.have_youtube, "instagram": P.have_instagram, "x": P.have_x}[pf]()
        if not have:
            res["results"][pf] = {"ok": False, "error": "no credentials"}
            continue
        if pf == "youtube":
            r = P.publish_youtube(c["files"]["video"], c["captions"]["youtube_title"], c["captions"]["youtube_desc"],
                                  tags=c["captions"].get("youtube_tags"))
        elif pf == "instagram":
            r = P.publish_instagram_image(c["public"]["image"], c["captions"]["instagram"])
            if r["ok"] and os.environ.get("IG_REEL") == "1":
                r["reel"] = P.publish_instagram_reel(c["public"]["video"], c["captions"]["instagram"], c["public"]["image"])
        else:
            r = P.publish_x(c["captions"]["x"], c["files"]["image"])
        res["results"][pf] = r
        _append({"ts": now, "season": season, "week": week, "platform": pf, **{k: r.get(k) for k in ("ok", "url", "error")}})
    return res


def post_if_due(season: int) -> str:
    """Called after every refresh. Returns a one-line status for the refresh log."""
    if os.environ.get("SOCIAL_AUTO") != "1":
        return "off (SOCIAL_AUTO != 1)"
    try:
        from dashboard import server as S
        from ml.ledger import week_lock_time
        week = S._current_week(season)
        if week is None:
            return "season complete"
        games = S._slate(season, week)["games"]
        lock = week_lock_time(games)
        now = datetime.now(timezone.utc)
        if lock is None or now < lock:
            return f"wk{week}: before the lock"
        if all(g.get("final") for g in games):
            return f"wk{week}: all games final"
        if any(posted(season, week, p) for p in PLATFORMS):
            return f"wk{week}: already posted"
        r = post_week(season, week)
        return f"wk{week}: " + ", ".join(f"{p}={'ok' if v.get('ok') else v.get('error')}" for p, v in r["results"].items())
    except Exception as e:
        return f"error: {str(e)[:160]}"


def status() -> dict:
    from social import publish as P
    return {"running": _LAST["running"], "started": _LAST["started"], "result": _LAST["result"],
            "log": _log_rows()[-20:], "auto": os.environ.get("SOCIAL_AUTO") == "1",
            "credentials": {"youtube": P.have_youtube(), "instagram": P.have_instagram(), "x": P.have_x()}}


def run_background(season: int, week: int, dry_run: bool, platforms, force: bool) -> None:
    """For the API: render + publish in a thread; status() reports it."""
    _LAST.update(running=True, started=datetime.now(timezone.utc).isoformat(timespec="seconds"), result=None)
    try:
        _LAST["result"] = post_week(season, week, dry_run=dry_run, platforms=platforms, force=force)
    except Exception as e:
        _LAST["result"] = {"error": str(e)[:200]}
    finally:
        _LAST["running"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="render the image and video, print the captions, publish nothing")
    ap.add_argument("--platforms", nargs="*", default=None, choices=PLATFORMS)
    ap.add_argument("--force", action="store_true", help="post again even if this week is already on the log")
    a = ap.parse_args()
    r = post_week(a.season, a.week, dry_run=a.dry_run, platforms=a.platforms, force=a.force)
    print(json.dumps({k: v for k, v in r.items() if k != "captions"}, indent=1))
    print("\n--- captions ---")
    for k, v in r["captions"].items():
        print(f"\n[{k}]\n{v}")


if __name__ == "__main__":
    main()
