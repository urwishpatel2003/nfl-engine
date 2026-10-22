"""
social/publish.py  —  the three publishers (official APIs, credentials from the environment)
============================================================================================
YouTube Shorts   YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN  (+ YT_PRIVACY: public|unlisted|private)
                 Data API v3: refresh the access token, resumable upload of the vertical video with
                 "#Shorts" in the title — YouTube files it as a Short from the aspect ratio + length.
Instagram        IG_USER_ID, IG_ACCESS_TOKEN  (a Business/Creator account linked to a Facebook Page;
                 long-lived token) + the media must be reachable by URL → the dashboard serves the
                 rendered files at {PUBLIC_BASE_URL}/social/…  Image feed post by default; IG_REEL=1
                 also publishes the video as a Reel (container → poll FINISHED → publish).
X                X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET  (OAuth 1.0a user context;
                 the free tier allows posting). v1.1 media upload, then v2 POST /tweets.

Every function returns {"ok": bool, "url": str|None, "error": str|None}. Nothing here retries
blindly: a failure is logged by social/post.py and the week stays un-posted on that platform,
so the next refresh (or a manual POST /api/social/post) tries again.
"""

import os
import time

import requests

_T = 60   # seconds; uploads are small (≤ ~10 MB)


def have_youtube() -> bool:
    return all(os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))


def have_instagram() -> bool:
    return all(os.environ.get(k) for k in ("IG_USER_ID", "IG_ACCESS_TOKEN"))


def have_x() -> bool:
    return all(os.environ.get(k) for k in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"))


def _res(ok, url=None, error=None):
    return {"ok": bool(ok), "url": url, "error": None if ok else (error or "unknown")[:300]}


# ── YouTube ───────────────────────────────────────────────────────────
def _yt_token() -> str:
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": os.environ["YT_CLIENT_ID"], "client_secret": os.environ["YT_CLIENT_SECRET"],
        "refresh_token": os.environ["YT_REFRESH_TOKEN"], "grant_type": "refresh_token"}, timeout=_T)
    r.raise_for_status()
    return r.json()["access_token"]


def publish_youtube(video_path: str, title: str, description: str, tags=None) -> dict:
    try:
        tok = _yt_token()
        meta = {"snippet": {"title": title[:100], "description": description[:5000],
                            "tags": tags or ["NFL", "NFL picks", "ATS", "sports betting", "Shorts"],
                            "categoryId": "17"},                           # Sports
                "status": {"privacyStatus": os.environ.get("YT_PRIVACY", "public"),
                           "selfDeclaredMadeForKids": False}}
        size = os.path.getsize(video_path)
        r = requests.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                          headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json; charset=UTF-8",
                                   "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": "video/mp4"},
                          json=meta, timeout=_T)
        if r.status_code >= 400:
            return _res(False, error=f"YouTube init {r.status_code}: {r.text[:200]}")
        loc = r.headers.get("Location")
        with open(video_path, "rb") as f:
            u = requests.put(loc, headers={"Authorization": f"Bearer {tok}", "Content-Type": "video/mp4",
                                           "Content-Length": str(size)}, data=f, timeout=300)
        if u.status_code >= 400:
            return _res(False, error=f"YouTube upload {u.status_code}: {u.text[:200]}")
        vid = u.json().get("id")
        return _res(True, url=f"https://youtube.com/shorts/{vid}")
    except Exception as e:
        return _res(False, error=f"YouTube: {e}")


# ── Instagram ─────────────────────────────────────────────────────────
_G = "https://graph.facebook.com/v21.0"


def _ig_publish(creation_id: str) -> dict:
    ig, tok = os.environ["IG_USER_ID"], os.environ["IG_ACCESS_TOKEN"]
    p = requests.post(f"{_G}/{ig}/media_publish", data={"creation_id": creation_id, "access_token": tok}, timeout=_T)
    if p.status_code >= 400:
        return _res(False, error=f"Instagram publish {p.status_code}: {p.text[:200]}")
    mid = p.json().get("id")
    try:
        link = requests.get(f"{_G}/{mid}", params={"fields": "permalink", "access_token": tok}, timeout=_T).json().get("permalink")
    except Exception:
        link = None
    return _res(True, url=link or f"instagram media {mid}")


def publish_instagram_image(image_url: str, caption: str) -> dict:
    try:
        ig, tok = os.environ["IG_USER_ID"], os.environ["IG_ACCESS_TOKEN"]
        r = requests.post(f"{_G}/{ig}/media", data={"image_url": image_url, "caption": caption[:2200], "access_token": tok}, timeout=_T)
        if r.status_code >= 400:
            return _res(False, error=f"Instagram container {r.status_code}: {r.text[:200]}")
        return _ig_publish(r.json()["id"])
    except Exception as e:
        return _res(False, error=f"Instagram: {e}")


def publish_instagram_reel(video_url: str, caption: str, cover_url: str | None = None) -> dict:
    try:
        ig, tok = os.environ["IG_USER_ID"], os.environ["IG_ACCESS_TOKEN"]
        data = {"media_type": "REELS", "video_url": video_url, "caption": caption[:2200],
                "share_to_feed": "true", "access_token": tok}
        if cover_url:
            data["cover_url"] = cover_url
        r = requests.post(f"{_G}/{ig}/media", data=data, timeout=_T)
        if r.status_code >= 400:
            return _res(False, error=f"Instagram reel container {r.status_code}: {r.text[:200]}")
        cid = r.json()["id"]
        for _ in range(60):                                        # Instagram transcodes first (≤ ~5 min)
            st = requests.get(f"{_G}/{cid}", params={"fields": "status_code,status", "access_token": tok}, timeout=_T).json()
            code = st.get("status_code")
            if code == "FINISHED":
                return _ig_publish(cid)
            if code in ("ERROR", "EXPIRED"):
                return _res(False, error=f"Instagram reel {code}: {st.get('status')}")
            time.sleep(5)
        return _res(False, error="Instagram reel: still processing after 5 minutes")
    except Exception as e:
        return _res(False, error=f"Instagram reel: {e}")


# ── X ─────────────────────────────────────────────────────────────────
def publish_x(text: str, image_path: str | None = None) -> dict:
    try:
        from requests_oauthlib import OAuth1
        auth = OAuth1(os.environ["X_API_KEY"], os.environ["X_API_SECRET"],
                      os.environ["X_ACCESS_TOKEN"], os.environ["X_ACCESS_SECRET"])
        media_ids = []
        if image_path:
            with open(image_path, "rb") as f:
                m = requests.post("https://upload.twitter.com/1.1/media/upload.json", auth=auth,
                                  files={"media": f}, timeout=_T)
            if m.status_code >= 400:
                return _res(False, error=f"X media {m.status_code}: {m.text[:200]}")
            media_ids = [m.json()["media_id_string"]]
        body = {"text": text[:280]}
        if media_ids:
            body["media"] = {"media_ids": media_ids}
        t = requests.post("https://api.x.com/2/tweets", auth=auth, json=body, timeout=_T)
        if t.status_code >= 400:
            return _res(False, error=f"X tweet {t.status_code}: {t.text[:200]}")
        tid = t.json().get("data", {}).get("id")
        return _res(True, url=f"https://x.com/i/web/status/{tid}")
    except Exception as e:
        return _res(False, error=f"X: {e}")
