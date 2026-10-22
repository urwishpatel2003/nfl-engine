"""
social/  —  automated weekly posts of the model's top-5 ATS picks
================================================================
content.py  builds the week's post (picks from the FROZEN board, the season record, captions)
render.py   draws the image (1080×1350) and the vertical video (1080×1920, ~20 s) with PIL +
            the ffmpeg binary that ships in the imageio-ffmpeg wheel (no system ffmpeg needed)
publish.py  the three publishers — YouTube Shorts (Data API v3, resumable upload), Instagram
            (Graph API container → publish; Reels for the video), X (v1.1 media + v2 tweet)
post.py     orchestrator: render → publish where credentials exist → log; dry-run renders only.
            CLI: python -m social.post --season 2026 --week 7 --dry-run
yt_auth.py  one-time local helper that prints the YouTube refresh token to the terminal

Credentials are Railway service variables only (see docs/SOCIAL_SETUP.md) — never in the
repo, never through chat. Without them the pipeline still renders and logs "no credentials".
"""
