# SPDX-License-Identifier: Apache-2.0
"""Device gates for a running text-to-speech server: one PASS or FAIL line per gate, exit 1 if any failed.

    python3 tts_gates.py --url http://127.0.0.1:8000 --model <served name> [--streams 4] [--min-speed 1.0]
                         [--max-first-audio 0.5] [--whisper <openai/whisper-large-v3 or a local path>]

Gates (every timing is per call, from this client, against one server on one card):
  health        GET /health is 200
  single        one new text streamed as pcm: speed vs real time >= --min-speed, first audio <= --max-first-audio
  sanity        that clip: plausible length for the text, not silent, not clipped, energy in the speech band
  concurrent    --streams requests at once: each at least real time (speed >= 1.0)
  formats       wav (whole), sse, mp3, flac and opus decode to audio of the expected length
  hangup        a listener that hangs up after the first chunk; then health is still 200 and a request works
  whisper       (with --whisper) a Whisper transcript of the clips matches the text: WER <= --max-wer
"""

import argparse
import base64
import concurrent.futures as cf
import io
import json
import re
import sys
import time
import urllib.request

import numpy as np

TEXT = (
    "The quick brown fox jumps over the lazy dog. A gentle breeze carried the smell of rain across the "
    "valley, and the old lighthouse keeper smiled as he climbed the stairs one more time."
)
SHORT = "Every gate prints one line, and this sentence is spoken to check the formats."
RESULTS: list[bool] = []


def gate(name: str, ok: bool, detail: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def post(url: str, body: dict, timeout: float = 600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_pcm(base: str, body: dict, sample_rate: int) -> dict:
    """Stream raw pcm; returns the samples, time to first audio and speed vs real time."""
    t0 = time.monotonic()
    first = None
    buf = bytearray()
    with post(base + "/v1/audio/speech", {**body, "response_format": "pcm"}) as r:
        while chunk := r.read1(65536):
            if first is None:
                first = time.monotonic() - t0
            buf += chunk
    wall = time.monotonic() - t0
    x = np.frombuffer(bytes(buf[: len(buf) // 2 * 2]), dtype="<i2").astype(np.float32) / 32768.0
    secs = len(x) / sample_rate
    return {"audio": x, "secs": secs, "wall": wall, "first": first or wall, "speed": secs / wall if wall else 0.0}


def sanity(x: np.ndarray, sr: int, text: str) -> tuple[bool, str]:
    secs = len(x) / sr
    words = len(text.split())
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    clipped = float(np.mean(np.abs(x) > 0.999)) if len(x) else 1.0
    spec = np.abs(np.fft.rfft(x[: sr * 30])) ** 2
    freqs = np.fft.rfftfreq(min(len(x), sr * 30), 1.0 / sr)
    band = float(spec[(freqs >= 80) & (freqs <= 4000)].sum() / max(spec.sum(), 1e-12))
    # Read speech runs at about 2 to 4 words per second; allow 1 to 6.
    ok = words / 6 <= secs <= words / 1 and rms > 0.01 and clipped < 0.001 and band > 0.6
    return ok, f"{secs:.2f}s for {words} words, rms {rms:.3f}, clipped {clipped:.4f}, 80-4000 Hz energy {band:.2f}"


def decode(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(data), dtype="float32")
    return (x if x.ndim == 1 else x.mean(axis=1)), sr


def wer(ref: str, hyp: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()  # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def transcribe(model: str, clips: list[tuple[np.ndarray, int]]) -> list[str]:
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model, device="cpu", dtype=torch.float32)
    out = []
    for x, sr in clips:
        if sr != 16000:  # linear resampling is enough for a transcript check
            n = int(len(x) * 16000 / sr)
            x = np.interp(np.arange(n) * sr / 16000, np.arange(len(x)), x).astype(np.float32)
        out.append(asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "en"})["text"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request (model-specific fields)")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--min-speed", type=float, default=1.0)
    ap.add_argument("--max-first-audio", type=float, default=0.5)
    ap.add_argument("--whisper", default=None)
    ap.add_argument("--max-wer", type=float, default=0.1)
    a = ap.parse_args()
    base, sr = a.url.rstrip("/"), a.sample_rate
    body = {"model": a.model, **json.loads(a.extra)}
    if a.voice:
        body["voice"] = a.voice

    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            gate("health", r.status == 200, r.read().decode()[:200])
    except Exception as e:  # noqa: BLE001
        gate("health", False, str(e))
        return 1

    stream_pcm(base, {**body, "input": "Warm up."}, sr)  # one request so lazy first-call setup is not timed
    one = stream_pcm(base, {**body, "input": TEXT}, sr)
    gate(
        "single",
        one["speed"] >= a.min_speed and one["first"] <= a.max_first_audio,
        f"{one['secs']:.2f}s of audio in {one['wall']:.2f}s = {one['speed']:.2f}x real time, "
        f"first audio {one['first'] * 1000:.0f} ms (per call, one card)",
    )
    ok, detail = sanity(one["audio"], sr, TEXT)
    gate("sanity", ok, detail)

    names = [
        "Alpha",
        "Bravo",
        "Charlie",
        "Delta",
        "Echo",
        "Foxtrot",
        "Golf",
        "Hotel",
        "India",
        "Juliet",
        "Kilo",
        "Lima",
        "Mike",
        "November",
        "Oscar",
        "Papa",
    ]
    texts = [f"{names[i % len(names)]} speaking. {TEXT}" for i in range(a.streams)]
    with cf.ThreadPoolExecutor(a.streams) as ex:
        runs = list(ex.map(lambda t: stream_pcm(base, {**body, "input": t}, sr), texts))
    speeds = [r["speed"] for r in runs]
    firsts = [r["first"] for r in runs]
    gate(
        "concurrent",
        min(speeds) >= 1.0 and all(sanity(r["audio"], sr, t)[0] for r, t in zip(runs, texts, strict=False)),
        f"{a.streams} streams at {min(speeds):.2f}x to {max(speeds):.2f}x real time each, first audio "
        f"{min(firsts) * 1000:.0f} to {max(firsts) * 1000:.0f} ms (per call, one card)",
    )

    lengths, bad = {}, []
    for fmt in ("wav", "mp3", "flac", "opus"):
        try:
            with post(base + "/v1/audio/speech", {**body, "input": SHORT, "response_format": fmt}) as r:
                x, rate = decode(r.read())
            lengths[fmt] = len(x) / rate
        except Exception as e:  # noqa: BLE001
            bad.append(f"{fmt}: {e}")
    try:
        pcm = bytearray()
        done = False
        with post(base + "/v1/audio/speech", {**body, "input": SHORT, "response_format": "pcm", "stream": True}) as r:
            for line in r:
                if line.startswith(b"data: "):
                    ev = json.loads(line[6:])
                    if ev["type"] == "speech.audio.delta":
                        pcm += base64.b64decode(ev["audio"])
                    done = done or ev["type"] == "speech.audio.done"
        lengths["sse"] = len(pcm) / 2 / sr
        if not done:
            bad.append("sse: no speech.audio.done event")
    except Exception as e:  # noqa: BLE001
        bad.append(f"sse: {e}")
    plausible = all(len(SHORT.split()) / 6 <= s <= len(SHORT.split()) for s in lengths.values())
    gate("formats", not bad and plausible, ", ".join(f"{k} {v:.2f}s" for k, v in lengths.items()) + "; ".join(bad))

    try:
        with post(base + "/v1/audio/speech", {**body, "input": TEXT * 3, "response_format": "pcm"}) as r:
            r.read1(4096)  # first audio, then hang up mid-stream
        time.sleep(2)
        after = stream_pcm(base, {**body, "input": SHORT}, sr)
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            alive = r.status == 200
        gate("hangup", alive and after["secs"] > 1.0, f"health {alive}, next request {after['secs']:.2f}s of audio")
    except Exception as e:  # noqa: BLE001
        gate("hangup", False, str(e))

    if a.whisper:
        clips = [(one["audio"], sr)] + [(r["audio"], sr) for r in runs[:2]]
        refs = [TEXT] + texts[:2]
        hyps = transcribe(a.whisper, clips)
        rates = [wer(r, h) for r, h in zip(refs, hyps, strict=False)]
        gate("whisper", max(rates) <= a.max_wer, f"WER {', '.join(f'{x:.3f}' for x in rates)}; first: {hyps[0]!r}")

    print(f"{sum(RESULTS)}/{len(RESULTS)} gates passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
