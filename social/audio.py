"""
social/audio.py  —  voiceover + music for the picks video
=========================================================
tts_segments(segments, out_dir)  one neural voice clip per script segment (edge-tts: Microsoft's
                                 Edge "Read aloud" voices, no key; TTS_VOICE / TTS_RATE env), decoded
                                 to 24 kHz mono float PCM through the bundled ffmpeg. A segment whose
                                 synthesis fails comes back as None (silence) so the video still ships.
music(seconds)                   a procedural beat (kick / snare / hats / bass / pad at 92 bpm) made with
                                 numpy, so there is nothing to license. Put a licensed track at the path
                                 in MUSIC_FILE to use that instead (decoded + looped).
mix(voices, holds, path)         lays each clip at its card's start, loops the music under it at
                                 MUSIC_DB (default −16 dB), soft-limits and writes a 16-bit WAV.
"""

import asyncio
import math
import os
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 24000


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(path: str) -> np.ndarray:
    """Any audio file → mono float32 PCM at SR."""
    out = subprocess.run([_ffmpeg(), "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


async def _say(text: str, path: str, voice: str, rate: str):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate).save(path)


def tts_segments(segments: list, out_dir) -> list:
    voice = os.environ.get("TTS_VOICE", "en-US-ChristopherNeural")   # energetic male; try AndrewMultilingual / AriaNeural
    rate = os.environ.get("TTS_RATE", "+18%")    # brisk: the whole countdown has to land inside ~45 s
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcms = []
    for s in segments:
        p = out_dir / f"voice_{s['key']}.mp3"
        try:
            asyncio.run(_say(s["text"], str(p), voice, rate))
            pcm = decode(str(p))
            # trim leading/trailing silence so the caption and the voice line up
            idx = np.where(np.abs(pcm) > 0.01)[0]
            pcm = pcm[max(0, idx[0] - SR // 20): idx[-1] + SR // 10] if len(idx) else pcm
            pcms.append(pcm)
        except Exception as e:
            print(f"[social] tts failed for {s['key']}: {str(e)[:120]}", flush=True)
            pcms.append(None)
    return pcms


# ── procedural music ──────────────────────────────────────────────────
def _env(n, a=0.002, d=0.25):
    t = np.arange(n) / SR
    return np.exp(-t / d) * np.minimum(1.0, t / max(a, 1e-4))


def _kick(n=int(SR * 0.35)):
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 18) + 45
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.12) * 0.9


def _snare(n=int(SR * 0.22)):
    rng = np.random.default_rng(3)
    t = np.arange(n) / SR
    return (rng.standard_normal(n) * 0.35 + np.sin(2 * np.pi * 190 * t) * 0.5) * _env(n, 0.001, 0.07)


def _hat(n=int(SR * 0.06), open_=False):
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0)                     # brighten
    return x * _env(n, 0.0005, 0.08 if open_ else 0.025) * 0.18


def _bass(freq, n):
    t = np.arange(n) / SR
    saw = 2 * (t * freq - np.floor(0.5 + t * freq))
    return (0.6 * saw + 0.4 * np.sin(2 * np.pi * freq * t)) * _env(n, 0.004, 0.5) * 0.35


def _pad(freqs, n):
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t + k) for k, f in enumerate(freqs)) / len(freqs)
    e = np.minimum(1.0, t / 0.4) * np.minimum(1.0, (n / SR - t) / 0.4)
    return x * e * 0.12


def music(seconds: float, bpm: int = 92) -> np.ndarray:
    """A four-bar loop (kick / snare / hats / bass / pad) tiled to `seconds`."""
    beat = 60.0 / bpm
    bar = 4 * beat
    loop_n = int(4 * bar * SR)
    out = np.zeros(loop_n, dtype=np.float32)

    def put(x, at):
        i = int(at * SR)
        j = min(loop_n, i + len(x))
        out[i:j] += x[: j - i]
    roots = [55.0, 55.0, 65.41, 49.0]            # A1 A1 C2 G1
    for b in range(4):
        base = b * bar
        put(_bass(roots[b], int(bar * SR)), base)
        put(_pad([roots[b] * 4, roots[b] * 6, roots[b] * 8 * 1.5], int(bar * SR)), base)
        for k in range(4):
            t = base + k * beat
            if k in (0, 2):
                put(_kick(), t)
            if k == 2 and b % 2 == 1:
                put(_kick(), t + beat * 0.5)      # syncopation on bars 2 and 4
            if k in (1, 3):
                put(_snare(), t)
            for h in range(2):
                put(_hat(open_=(k == 3 and h == 1)), t + h * beat / 2)
    reps = int(math.ceil(seconds * SR / loop_n)) + 1
    full = np.tile(out, reps)[: int(seconds * SR)]
    return full / (np.max(np.abs(full)) + 1e-6)


def mix(voices: list, holds: list, path: str, music_db: float | None = None) -> str:
    total = int(sum(holds) * SR)
    voice = np.zeros(total, dtype=np.float32)
    t = 0
    for v, h in zip(voices, holds):
        if v is not None:
            n = min(len(v), total - t)
            voice[t: t + n] += v[:n]
        t += int(h * SR)
    mf = os.environ.get("MUSIC_FILE")
    if mf and Path(mf).exists():
        m = decode(mf)
        m = np.tile(m, int(math.ceil(total / max(1, len(m)))))[:total]
        m = m / (np.max(np.abs(m)) + 1e-6)
    else:
        m = music(total / SR)
    db = float(os.environ.get("MUSIC_DB", -16)) if music_db is None else music_db
    gain = 10 ** (db / 20)
    fade = int(1.5 * SR)
    ramp_in, ramp_out = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
    m[:fade] *= ramp_in
    m[-fade:] *= ramp_out
    # duck the music a little while the voice speaks
    speaking = np.convolve(np.abs(voice) > 0.02, np.ones(int(0.25 * SR)) / int(0.25 * SR), mode="same") > 0.05
    m = m * np.where(speaking, 0.55, 1.0)
    out = voice * 0.95 + m * gain
    out = np.tanh(out * 1.2) / 1.2                                   # soft limiter
    pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
