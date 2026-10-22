"""
social/render.py  —  the picks image and the vertical video
===========================================================
render_image(content, path)   1080×1350 (Instagram 4:5; X shows it as is)
render_video(content, path)   1080×1920 vertical, 30 fps, ~20 s: intro card, one card per pick,
                              outro with the record — YouTube Shorts / Instagram Reels

Drawing is PIL only. Fonts: PIL ≥ 10.1 ships a scalable default (ImageFont.load_default(size)),
so nothing has to be downloaded or vendored. Team logos are the ESPN PNGs nflverse links to,
fetched once and cached under data/processed/social/logos/; when one cannot be fetched a disc in
the team colour with the abbreviation stands in. Video frames are piped raw to the ffmpeg binary
that ships in the imageio-ffmpeg wheel (libx264, yuv420p, faststart) — no system ffmpeg.
"""

import io
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROC = Path(__file__).parent.parent / "data" / "processed"
LOGOS = PROC / "social" / "logos"

# Half Blood Betting Prince — wizard theme: deep plum night, gold accents, emerald for a strong cover
BG = (16, 9, 34)
PANEL = (30, 19, 60)
LINE = (74, 54, 120)
TXT = (242, 236, 255)
MUTED = (172, 160, 205)
ACCENT = (224, 184, 74)        # gold
GOOD = (72, 212, 138)          # emerald
AMBER = (240, 170, 80)


def _star(d: ImageDraw.ImageDraw, cx, cy, r, fill, inner=0.38):
    """A four-point sparkle."""
    pts = []
    for k in range(8):
        a = k * math.pi / 4 - math.pi / 2
        rr = r if k % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=fill)


def _sparkles(im: Image.Image, n: int = 28, seed: int = 11, avoid=None) -> None:
    """Scatter small gold sparkles over the background (deterministic), keeping clear of `avoid`
    boxes so they never sit on text."""
    import random
    rnd = random.Random(seed)
    d = ImageDraw.Draw(im)
    W, H = im.size
    placed = 0
    for _ in range(n * 6):
        if placed >= n:
            break
        x, y = rnd.randint(30, W - 30), rnd.randint(30, H - 30)
        if any(bx0 <= x <= bx1 and by0 <= y <= by1 for bx0, by0, bx1, by1 in (avoid or [])):
            continue
        r = rnd.choice([5, 6, 8, 10, 13])
        tone = rnd.choice([(224, 184, 74), (200, 160, 60), (160, 130, 220)])   # gold, dim gold, lilac
        _star(d, x, y, r, tone)
        placed += 1


def _brand(d: ImageDraw.ImageDraw, cx, y, size=30):
    from social.content import BRAND
    f = _font(size)
    t = BRAND.upper()
    w = d.textlength(t, font=f)
    d.text((cx, y), t, font=f, fill=ACCENT, anchor="mm")
    _star(d, cx - w / 2 - size * 0.9, y, size * 0.42, ACCENT)
    _star(d, cx + w / 2 + size * 0.9, y, size * 0.42, ACCENT)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # PIL < 10.1
        for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()


def _hex(c: str, fallback=(51, 65, 85)):
    try:
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return fallback


def _logo(url: str, team: str, color: str, size: int) -> Image.Image:
    """The team logo at `size` px (cached), else a coloured disc with the abbreviation."""
    LOGOS.mkdir(parents=True, exist_ok=True)
    p = LOGOS / f"{team}.png"
    img = None
    if p.exists():
        try:
            img = Image.open(p).convert("RGBA")
        except Exception:
            img = None
    if img is None and url:
        try:
            import requests
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            img.save(p)
        except Exception:
            img = None
    if img is None:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([0, 0, size - 1, size - 1], fill=_hex(color))
        f = _font(int(size * 0.34))
        d.text((size / 2, size / 2), team, font=f, fill=TXT, anchor="mm")
        return img
    img.thumbnail((size, size))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2), img)
    return canvas


def _pct(p) -> str:
    return f"{round((p or 0) * 100)}%"


def _rec_line(rec: dict) -> str:
    if not rec or not rec.get("n"):
        return "Season record builds from this week"
    s = f"{rec['w']}-{rec['l']}" + (f"-{rec['p']}" if rec.get("p") else "")
    pct = f" · {round(rec['pct'] * 100)}%" if rec.get("pct") is not None else ""
    u = rec.get("units")
    units = f" · {u:+.1f}u" if isinstance(u, (int, float)) else ""
    return f"Top-5 ATS this season: {s}{pct}{units}"


def _rounded(d: ImageDraw.ImageDraw, box, r, fill, outline=None):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=2 if outline else 0)


# ── the image ─────────────────────────────────────────────────────────
def render_image(c: dict, path: str, size=(1080, 1350)) -> str:
    W, H = size
    im = Image.new("RGB", (W, H), BG)
    # sparkles only in the top band around the brand line and the bottom-right corner, never on text
    _sparkles(im, n=18, seed=c["week"], avoid=[(40, 86, W - 40, H - 30), (250, 20, W - 250, 86)])
    d = ImageDraw.Draw(im)
    _brand(d, W / 2, 58, 30)
    d.text((60, 96), f"WEEK {c['week']}  ·  FIVE SPELLS AGAINST THE SPREAD", font=_font(42), fill=TXT)
    d.text((60, 154), "the five strongest covers in the crystal ball · cast before kickoff, graded after", font=_font(25), fill=MUTED)
    top, row_h, gap = 216, 168, 14           # 5 rows end at 1126, leaving room for the record + site + disclaimer
    for i, p in enumerate(c["picks"][:5]):
        y = top + i * (row_h + gap)
        _rounded(d, [60, y, W - 60, y + row_h], 22, PANEL, LINE)
        # rank badge
        _rounded(d, [84, y + 24, 84 + 64, y + 24 + 64], 16, ACCENT if p.get("tier") == "edge" else (60, 72, 100))
        d.text((84 + 32, y + 24 + 32), str(p["rank"]), font=_font(36), fill=TXT, anchor="mm")
        # logo
        im.paste(_logo(p.get("logo"), p["team"], p.get("color"), 120), (172, y + 31), _logo(p.get("logo"), p["team"], p.get("color"), 120))
        # pick + opponent
        d.text((320, y + 34), p["line"], font=_font(62), fill=TXT)
        where = "vs" if p["at_home"] else "at"
        if p.get("result"):                                     # already played: the result, not the kickoff
            tail = {"W": f"covered  {p.get('score') or ''}", "L": f"missed  {p.get('score') or ''}", "P": f"push  {p.get('score') or ''}"}[p["result"]]
            d.text((320, y + 112), f"{where} {p['opp_name']}  ·  ", font=_font(27), fill=MUTED)
            d.text((320 + d.textlength(f"{where} {p['opp_name']}  ·  ", font=_font(27)), y + 112), tail, font=_font(27),
                   fill=GOOD if p["result"] == "W" else (AMBER if p["result"] == "P" else (235, 95, 110)))
        else:
            d.text((320, y + 112), f"{where} {p['opp_name']}  ·  {p['kickoff'][5:] if p.get('kickoff') else ''}", font=_font(27), fill=MUTED)
        # cover probability
        d.text((W - 96, y + 54), _pct(p.get("cover_prob")), font=_font(60), fill=GOOD if (p.get("cover_prob") or 0) >= 0.55 else AMBER, anchor="ra")
        d.text((W - 96, y + 126), "to cover", font=_font(24), fill=MUTED, anchor="ra")
    y = top + 5 * (row_h + gap) + 14
    d.text((60, y), _rec_line(c.get("record")), font=_font(32), fill=TXT)
    from social.content import DISCLAIMER, HANDLE
    d.text((60, y + 50), f"follow {HANDLE} for the graded record", font=_font(26), fill=ACCENT)
    d.text((60, H - 64), DISCLAIMER, font=_font(22), fill=MUTED)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)
    return path


# ── the video ─────────────────────────────────────────────────────────
def _card_intro(c, W, H):
    im = Image.new("RGB", (W, H), BG)
    _sparkles(im, n=60, seed=c["week"] + 100, avoid=[(60, H * 0.28, W - 60, H * 0.66), (60, H - 360, W - 60, H - 120)])
    d = ImageDraw.Draw(im)
    _brand(d, W / 2, H * 0.33, 46)
    d.text((W / 2, H * 0.44), f"WEEK {c['week']}", font=_font(150), fill=TXT, anchor="mm")
    d.text((W / 2, H * 0.53), "FIVE SPELLS AGAINST THE SPREAD", font=_font(58), fill=TXT, anchor="mm")
    d.text((W / 2, H * 0.60), "cast before kickoff · graded after", font=_font(36), fill=MUTED, anchor="mm")
    return im


def _card_pick(c, p, W, H):
    im = Image.new("RGB", (W, H), BG)
    _sparkles(im, n=34, seed=c["week"] * 10 + p["rank"], avoid=[(80, 100, W - 80, H - 80)])
    d = ImageDraw.Draw(im)
    d.text((W / 2, 150), f"SPELL {p['rank']} OF 5  ·  WEEK {c['week']}", font=_font(40), fill=ACCENT, anchor="mm")
    lg = _logo(p.get("logo"), p["team"], p.get("color"), 420)
    im.paste(lg, ((W - 420) // 2, 260), lg)
    d.text((W / 2, 780), p["line"], font=_font(170), fill=TXT, anchor="mm")
    where = "vs" if p["at_home"] else "at"
    d.text((W / 2, 900), f"{where} {p['opp_name']}", font=_font(56), fill=MUTED, anchor="mm")
    ol = _logo(p.get("opp_logo"), p["opp"], None, 140)
    im.paste(ol, ((W - 140) // 2, 960), ol)
    cp = p.get("cover_prob") or 0
    d.text((W / 2, 1230), _pct(cp), font=_font(190), fill=GOOD if cp >= 0.55 else AMBER, anchor="mm")
    d.text((W / 2, 1360), "chance to cover", font=_font(44), fill=MUTED, anchor="mm")
    if p.get("edge") is not None:
        d.text((W / 2, 1440), f"{abs(float(p['edge'])):.1f} pts from the line", font=_font(38), fill=MUTED, anchor="mm")
    if p.get("result"):                                         # already played: the result, not the kickoff
        tail = {"W": "COVERED", "L": "MISSED", "P": "PUSH"}[p["result"]] + (f"  ·  {p['score']}" if p.get("score") else "")
        d.text((W / 2, 1560), tail, font=_font(40), fill=GOOD if p["result"] == "W" else (AMBER if p["result"] == "P" else (235, 95, 110)), anchor="mm")
    elif p.get("kickoff"):
        d.text((W / 2, 1560), p["kickoff"], font=_font(36), fill=MUTED, anchor="mm")
    # progress bar across the five picks (the video counts DOWN, 5 → 1, so pick 5 is step one)
    done = 6 - p["rank"]
    for k in range(5):
        x0 = 140 + k * ((W - 280) // 5)
        d.rounded_rectangle([x0, H - 60, x0 + (W - 280) // 5 - 14, H - 46], radius=7,
                            fill=ACCENT if k < done else LINE)
    return im


def _card_outro(c, W, H):
    im = Image.new("RGB", (W, H), BG)
    _sparkles(im, n=40, seed=c["week"] + 200, avoid=[(60, H * 0.26, W - 60, H - 100)])
    d = ImageDraw.Draw(im)
    d.text((W / 2, H * 0.30), "THE SPELLBOOK", font=_font(54), fill=ACCENT, anchor="mm")
    y = int(H * 0.37)
    for p in c["picks"][:5]:
        d.text((160, y), f"{p['rank']}", font=_font(44), fill=MUTED)
        d.text((240, y), p["line"], font=_font(54), fill=TXT)
        d.text((W - 160, y + 6), _pct(p.get("cover_prob")), font=_font(48), fill=GOOD if (p.get("cover_prob") or 0) >= 0.55 else AMBER, anchor="ra")
        y += 96
    d.text((W / 2, y + 60), _rec_line(c.get("record")), font=_font(40), fill=TXT, anchor="mm")
    from social.content import DISCLAIMER, HANDLE
    d.text((W / 2, y + 140), f"follow {HANDLE}", font=_font(44), fill=ACCENT, anchor="mm")
    d.text((W / 2, y + 200), "the graded record, every week", font=_font(34), fill=MUTED, anchor="mm")
    d.text((W / 2, H - 140), DISCLAIMER, font=_font(30), fill=MUTED, anchor="mm")
    return im


def _wrap(d: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def _caption(im: Image.Image, text: str, y_top: int) -> None:
    """The spoken line as an on-screen caption (most Shorts are watched muted)."""
    d = ImageDraw.Draw(im)
    f = _font(37)
    lines = _wrap(d, text, f, im.width - 160)[:4]
    y = y_top
    for ln in lines:
        w = d.textlength(ln, font=f)
        d.rounded_rectangle([im.width / 2 - w / 2 - 16, y - 5, im.width / 2 + w / 2 + 16, y + 44], radius=11, fill=(0, 0, 0))
        d.text((im.width / 2, y + 19), ln, font=f, fill=TXT, anchor="mm")
        y += 53


def render_video(c: dict, path: str, size=(1080, 1920), fps=30, segments=None, holds=None, audio=None) -> str:
    """One card per script segment (intro, the picks as a countdown, outro), each held `holds[i]`
    seconds with a 12-frame cross-fade, the spoken line captioned on the card, and the mixed
    audio muxed in. Without a script: the five picks in rank order, silent, 3 s each."""
    import imageio_ffmpeg
    W, H = size
    if segments is None:
        segments = [{"key": "intro", "text": "", "pick": None}] + \
                   [{"key": f"pick{p['rank']}", "text": "", "pick": p} for p in c["picks"][:5]] + \
                   [{"key": "outro", "text": "", "pick": None}]
        holds = [2.5] + [3.0] * (len(segments) - 2) + [3.5]
    cards = []
    for s, h in zip(segments, holds):
        if s["key"] == "intro":
            im = _card_intro(c, W, H)
        elif s["key"] == "outro":
            im = _card_outro(c, W, H)
        else:
            im = _card_pick(c, s["pick"], W, H)
        if s.get("text"):
            _caption(im, s["text"], H - 330 if s["key"] in ("intro", "outro") else 1630)
        cards.append((im, h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
            "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fade = 12
    prev = None
    for im, secs in cards:
        if prev is not None:
            for k in range(1, fade + 1):
                proc.stdin.write(Image.blend(prev, im, k / fade).tobytes())
        frames = max(1, int(secs * fps) - (fade if prev is not None else 0))
        raw = im.tobytes()
        for _ in range(frames):
            proc.stdin.write(raw)
        prev = im
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the video")
    return path
