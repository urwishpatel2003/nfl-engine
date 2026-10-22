"""
social/ig_auth.py  —  one-time, LOCAL: turn a short-lived Instagram token into what Railway needs
=================================================================================================
After creating the Meta app and generating a user token in the Graph API Explorer (scopes
instagram_basic, instagram_content_publish, pages_show_list, pages_read_engagement), run on
the laptop:

    $env:IG_APP_ID = "<app id>"; $env:IG_APP_SECRET = "<app secret>"; $env:IG_SHORT_TOKEN = "<token from the explorer>"
    python -m social.ig_auth

It exchanges the token for a LONG-LIVED one (60 days), lists the Facebook Pages the token can see
and the Instagram Business/Creator account linked to each, and prints the two values to set on
the Railway service: IG_USER_ID and IG_ACCESS_TOKEN. Nothing is written to disk and nothing is
sent anywhere but graph.facebook.com. Run it again before the 60 days are up (with the current
long-lived token as IG_SHORT_TOKEN) to get a fresh one.
"""

import os
import sys

import requests

G = "https://graph.facebook.com/v21.0"


def main():
    app_id, secret, short = (os.environ.get(k) for k in ("IG_APP_ID", "IG_APP_SECRET", "IG_SHORT_TOKEN"))
    if not (app_id and secret and short):
        sys.exit("set IG_APP_ID, IG_APP_SECRET and IG_SHORT_TOKEN in this shell first")
    r = requests.get(f"{G}/oauth/access_token", params={"grant_type": "fb_exchange_token", "client_id": app_id,
                                                        "client_secret": secret, "fb_exchange_token": short}, timeout=30)
    if r.status_code >= 400:
        sys.exit(f"token exchange failed: {r.text[:300]}")
    long_tok = r.json()["access_token"]
    pages = requests.get(f"{G}/me/accounts", params={"access_token": long_tok, "fields": "id,name,instagram_business_account"}, timeout=30).json()
    found = []
    for p in pages.get("data", []):
        ig = (p.get("instagram_business_account") or {}).get("id")
        if ig:
            info = requests.get(f"{G}/{ig}", params={"access_token": long_tok, "fields": "username,name"}, timeout=30).json()
            found.append((p["name"], ig, info.get("username")))
    if not found:
        print("No Instagram Business/Creator account is linked to a Page this token can see.\n"
              "Link the Instagram account to a Facebook Page you admin (Instagram → Settings → Sharing to other apps → Facebook),\n"
              "then generate the token again.")
        sys.exit(1)
    print("\nInstagram accounts reachable with this token:")
    for name, ig, user in found:
        print(f"  Page '{name}' → Instagram @{user}  (IG_USER_ID = {ig})")
    print("\nSet on the Railway service:")
    print(f"  IG_USER_ID      = {found[0][1]}")
    print(f"  IG_ACCESS_TOKEN = {long_tok}")
    print("\n(long-lived token: 60 days — run this again with it as IG_SHORT_TOKEN before then)")


if __name__ == "__main__":
    main()
