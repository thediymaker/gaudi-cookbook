# SPDX-License-Identifier: Apache-2.0
"""API and throughput gates against a running EmbeddingGemma 2 server: one PASS or FAIL line per gate.

    python3 tests/hpu/api_gates.py [--url http://127.0.0.1:8000] [--min-text-tok-s 50000]

  forms        an image as data URI, bare base64, short form and `messages` gives the same vector; mixed inputs,
               `dimensions` with base64 encoding, `prompt_name` and token ids work
  url_guard    URLs that resolve to loopback or link-local addresses are refused with a 400
  text_speed   1,024 passages of about 240 tokens, 16 concurrent requests of 16 inputs: tokens per second, and
               the single-query latency (p50 of 40 sequential requests)
  media_speed  images (640x480 JPEG), audio (6 s WAV) and video (5 frames): items per second, 16 concurrent
All rates are per call against one server on one card.
"""

import argparse
import base64
import concurrent.futures as cf
import io
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np

RESULTS: list[bool] = []


def gate(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def vec(r, i=0):
    return np.array(r["data"][i]["embedding"], dtype=np.float64)


def jpeg(w, h, seed):
    from PIL import Image

    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8).repeat(16, 0).repeat(16, 1)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def wav(seconds, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    x = (0.2 * np.sin(2 * np.pi * (150 + 40 * np.sin(2 * np.pi * 0.5 * t)) * t)).astype(np.float32)
    pcm = (x * 32767).astype("<i2").tobytes()
    head = b"RIFF" + (36 + len(pcm)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    head += (1).to_bytes(2, "little") * 2 + sr.to_bytes(4, "little") + (2 * sr).to_bytes(4, "little")
    head += (2).to_bytes(2, "little") + (16).to_bytes(2, "little") + b"data" + len(pcm).to_bytes(4, "little")
    return head + pcm


def mp4(frames=5):
    import av

    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        s = c.add_stream("mpeg4", rate=1)
        s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
        for i in range(frames):
            a = np.full((240, 320, 3), 40 * i, dtype=np.uint8)
            a[60:180, 80 + 20 * i : 160 + 20 * i] = 255
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def burst(url, body, n, workers=16):
    t = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        outs = list(ex.map(lambda _: post(url, body), range(n)))
    dt = time.time() - t
    bad = [o for s, o in outs if s != 200]
    if bad:
        raise RuntimeError(bad[0])
    return dt, sum(o["usage"]["prompt_tokens"] for _, o in outs), sum(len(o["data"]) for _, o in outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--min-text-tok-s", type=float, default=50000)
    a = ap.parse_args()
    url = a.url.rstrip("/") + "/v1/embeddings"
    img = jpeg(640, 480, 0)
    img_b64 = base64.b64encode(img).decode()
    img_uri = "data:image/jpeg;base64," + img_b64
    wav_b64 = base64.b64encode(wav(6.0)).decode()
    vid_uri = "data:video/mp4;base64," + base64.b64encode(mp4()).decode()

    forms = [
        {"input": [[{"type": "image_url", "image_url": {"url": img_uri}}]]},
        {"input": {"image": img_b64}},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_uri}}]}]},
    ]
    res = [post(url, b) for b in forms]
    same = all(s == 200 for s, _ in res) and min(float(vec(res[0][1]) @ vec(r)) for _, r in res) > 0.9999
    s1, r1 = post(
        url,
        {
            "prompt_name": "query",
            "input": [
                "a red square on a grey background",
                [{"type": "image_url", "image_url": {"url": img_uri}}],
                [{"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}],
                [{"type": "text", "text": "a short clip"}, {"type": "video_url", "video_url": {"url": vid_uri}}],
            ],
        },
    )
    s2, r2 = post(url, {"input": "dimensions check", "dimensions": 256, "encoding_format": "base64"})
    n256 = len(np.frombuffer(base64.b64decode(r2["data"][0]["embedding"]), dtype=np.float32)) if s2 == 200 else 0
    s3, r3 = post(url, {"input": [2, 1596, 3]})
    detail = f"image forms agree {same}, mixed request {s1}, dimensions 256 -> {n256} floats, token ids {s3}"
    if s1 == 200:
        detail += f", image usage {r1['usage']['prompt_tokens']} tokens for 4 inputs"
    gate("forms", same and s1 == 200 and n256 == 256 and s3 == 200, detail)

    codes = [
        post(url, {"input": {"image": u}})[0]
        for u in ("http://127.0.0.1:8000/health", "http://169.254.169.254/latest/meta-data")
    ]
    gate("url_guard", codes == [400, 400], f"loopback {codes[0]}, link-local {codes[1]}")

    text = " ".join(["Each card holds its own copy of the model and serves requests in fixed-shape batches."] * 15)
    passages = [f"{i} {text}" for i in range(1024)]
    for i in range(5):
        post(url, {"input": [f"warm up {i}"]})
    lat = []
    for i in range(40):
        t = time.time()
        post(url, {"input": [f"how do I request a node for job {i}"]})
        lat.append(time.time() - t)
    chunks = [passages[i : i + 16] for i in range(0, len(passages), 16)]
    t = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda c: post(url, {"input": c}), chunks))
    dt = time.time() - t
    tok = sum(o["usage"]["prompt_tokens"] for _, o in outs)
    tok_s = tok / dt
    gate(
        "text_speed",
        tok_s >= a.min_text_tok_s,
        f"{tok_s:,.0f} tok/s, {len(passages) / dt:.0f} passages/s ({tok / len(passages):.0f} tokens each), "
        f"single query p50 {np.median(lat) * 1000:.1f} ms (per call, one card)",
    )

    try:
        di, _, ni = burst(url, {"input": [[{"type": "image_url", "image_url": {"url": img_uri}}]] * 4}, 32)
        da, _, na = burst(url, {"input": [[{"type": "input_audio", "input_audio": {"data": wav_b64}}]] * 4}, 32)
        dv, _, nv = burst(url, {"input": [[{"type": "video_url", "video_url": {"url": vid_uri}}]]}, 16, 8)
        gate(
            "media_speed",
            True,
            f"{ni / di:.1f} images/s, {na / da:.1f} audio clips/s (6 s), {nv / dv:.1f} videos/s "
            "(5 frames) (per call, one card)",
        )
    except Exception as e:  # noqa: BLE001
        gate("media_speed", False, str(e)[:300])

    print(f"{sum(RESULTS)}/{len(RESULTS)} gates passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
