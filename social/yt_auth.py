"""
social/yt_auth.py  —  one-time, LOCAL: mint the YouTube refresh token
=====================================================================
Run on the laptop, never on the server:

    $env:YT_CLIENT_ID = "<client id>"; $env:YT_CLIENT_SECRET = "<client secret>"
    python -m social.yt_auth

It opens the Google consent page in the browser (scope youtube.upload), catches the redirect
on http://localhost:8765, exchanges the code and PRINTS the refresh token to the terminal.
Set that as YT_REFRESH_TOKEN on the Railway service. Nothing is written to disk and nothing
is sent anywhere but Google. The OAuth client must be of type "Desktop app" (Google Cloud
Console → APIs & Services → Credentials), with the YouTube Data API v3 enabled.
"""

import os
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

SCOPE = "https://www.googleapis.com/auth/youtube.upload"
REDIRECT = "http://localhost:8765"


def main():
    cid, sec = os.environ.get("YT_CLIENT_ID"), os.environ.get("YT_CLIENT_SECRET")
    if not cid or not sec:
        sys.exit("set YT_CLIENT_ID and YT_CLIENT_SECRET in this shell first")
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": REDIRECT, "response_type": "code", "scope": SCOPE,
        "access_type": "offline", "prompt": "consent"})
    code = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            code["v"] = q.get("code", [None])[0]
            self.send_response(200); self.end_headers()
            self.wfile.write(b"Done. You can close this tab and return to the terminal.")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("localhost", 8765), H)
    print("Opening the consent page… (if it does not open, paste this URL into a browser)\n" + url)
    webbrowser.open(url)
    while not code.get("v"):
        srv.handle_request()
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "code": code["v"], "client_id": cid, "client_secret": sec, "redirect_uri": REDIRECT,
        "grant_type": "authorization_code"}, timeout=30)
    r.raise_for_status()
    tok = r.json()
    print("\nYT_REFRESH_TOKEN =", tok.get("refresh_token"))
    print("(set it on the Railway service as a variable; it does not expire while the app stays in use)")


if __name__ == "__main__":
    main()
