# Social posts — setup

The server renders the week's top-5 ATS picks into an image and a vertical video and posts
them automatically after the Saturday lock (13:00 UTC) when `SOCIAL_AUTO=1`. Each platform
posts once per week; attempts are logged on the volume (`data/processed/social_posts.jsonl`).
Everything works without credentials in dry-run mode (renders only).

Try it locally first:

```bash
python -m social.post --season 2026 --week 7 --dry-run
```

Files land in `data/processed/social/2026_wk07/` (`picks.png`, `picks.mp4`, `content.json`),
and the captions print to the terminal.

## Railway variables (all optional; a platform without its variables is skipped)

| Variable | What |
|---|---|
| `SOCIAL_AUTO` | `1` to post automatically after the Saturday lock |
| `BRAND_NAME`, `BRAND_HANDLE` | default `Half Blood Betting Prince` / `@HalfBloodBettingPrince` — on the image, in the script, captions and the brand hashtag |
| `PUBLIC_BASE_URL` | the site URL Instagram fetches media from (default `https://nfl-engine-production.up.railway.app`) |
| `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` | YouTube Shorts |
| `YT_PRIVACY` | `public` (default), `unlisted` or `private` |
| `IG_USER_ID`, `IG_ACCESS_TOKEN` | Instagram image post |
| `IG_REEL` | `1` to also publish the video as a Reel |
| `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_SECRET` | X post with the image |

Set them in Railway → service → Variables. Never put them in the repo or in chat.

## YouTube Shorts

1. Google Cloud Console → create a project → APIs & Services → enable **YouTube Data API v3**.
2. OAuth consent screen: External, add your Google account as a test user (or publish the app).
3. Credentials → Create credentials → OAuth client ID → type **Desktop app**. Copy the client id and secret.
4. On the laptop, in PowerShell:
   ```powershell
   $env:YT_CLIENT_ID = "<client id>"; $env:YT_CLIENT_SECRET = "<client secret>"
   python -m social.yt_auth
   ```
   Sign in with the channel's account and allow the upload scope. The terminal prints `YT_REFRESH_TOKEN`.
5. Set the three `YT_*` variables on Railway. Quota: an upload costs 1,600 of the 10,000 daily units, one post a week is nothing.

The video is 1080×1920, about 20 s, title ends in `#Shorts`, so YouTube files it as a Short.

## Instagram

Requires a **Business or Creator** Instagram account linked to a **Facebook Page**.

1. developers.facebook.com → create an app (type Business) → add the **Instagram Graph API** product.
2. Graph API Explorer: select the app, generate a user token with `instagram_basic`,
   `instagram_content_publish`, `pages_show_list`, `pages_read_engagement`.
3. On the laptop, in PowerShell inside the repo (App ID and App Secret are under the app's Settings → Basic):
   ```powershell
   $env:IG_APP_ID = "<app id>"; $env:IG_APP_SECRET = "<app secret>"; $env:IG_SHORT_TOKEN = "<token from the explorer>"
   python -m social.ig_auth
   ```
   It exchanges the token for a long-lived one (60 days), finds the Instagram account linked to
   your Page, and prints `IG_USER_ID` and `IG_ACCESS_TOKEN`.
4. Set those two on Railway. Before the 60 days are up, run the helper again with the current
   long-lived token as `IG_SHORT_TOKEN` and update the variable.

Instagram downloads the media from a public URL, which is why the dashboard serves
`/social/<season>_wk<NN>/picks.png` and `picks.mp4`. Content publishing is limited to 100 posts per 24 h.

## X

1. developer.x.com → sign up for the Free tier → create a project and an app.
2. App settings → User authentication settings → OAuth 1.0a, permissions **Read and write**, type Web/Bot, any callback URL.
3. Keys and tokens → generate the **API key and secret** and the **Access token and secret** (after the permission change, regenerate the access token so it carries write).
4. Set the four `X_*` variables on Railway. Free tier allows 1,500 posts a month.

## The video: script, voice, music, captions

`social/script.py` writes the spoken script from the board: a hook ("Five NFL spread picks the
model loves in week 7. The number one play is 70% to cover. Stay for it."), the five picks as a
countdown from 5 to 1 so the strongest cover is the payoff, and a close with the season record
and the follow ask. Each line is spoken by a neural voice (`edge-tts`, no key) and shown as a
caption on the card, since most Shorts are watched muted. A procedural beat plays under the
voice, ducked while it speaks. Hashtags are platform-tuned and include the ten teams' nicknames.

| Variable | What |
|---|---|
| `TTS_VOICE` | default `en-US-ChristopherNeural`; try `en-US-AndrewMultilingualNeural`, `en-US-AriaNeural`, `en-GB-RyanNeural` |
| `TTS_RATE` | default `+8%` |
| `MUSIC_FILE` | path to a licensed track on the volume to use instead of the generated beat (e.g. `/app/data/processed/social/music.mp3`) |
| `MUSIC_DB` | music level under the voice, default `-16` |

If the voice service is unreachable the video still renders with captions and music, and the
refresh log says which segments failed.

## Manual trigger and status

```bash
# render only (dry run), then look at the files on the volume
curl -X POST "https://<site>/api/social/post?week=7&dry=1" -H "X-Refresh-Token: <REFRESH_TOKEN>"
# publish now to X only
curl -X POST "https://<site>/api/social/post?week=7&platforms=x" -H "X-Refresh-Token: <REFRESH_TOKEN>"
curl "https://<site>/api/social/status"
```

`force=1` re-posts a week that is already on the log (a second tweet, a second video).
