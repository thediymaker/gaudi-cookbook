# SPDX-License-Identifier: Apache-2.0
"""Fetch and decode request media: images (PIL), audio (16 kHz mono float32), video (sampled frames).

Sources: `data:<mime>;base64,...` URIs, bare base64, or http(s) URLs. URL fetches go
only to public addresses (private, loopback and link-local addresses are refused), follow at most 3
redirects (each re-checked), and stop at MAX_MEDIA_BYTES.
"""

import base64
import binascii
import io
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

MAX_MEDIA_BYTES = 64 * 1024 * 1024
FETCH_TIMEOUT = 30.0
ALLOW_URLS = os.environ.get("EG2_ALLOW_URL_FETCH", "1") == "1"  # 0: only inline media (data: URIs, base64)
MAX_IMAGE_PIXELS = 64_000_000
SAMPLE_RATE = 16000


class MediaError(ValueError):
    """Bad or unreadable media; reported to the caller as a 400."""


def _check_host(host):
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise MediaError(f"cannot resolve host {host!r}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise MediaError(f"URL host {host!r} resolves to a non-public address; send the media inline instead")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _fetch(url):
    if not ALLOW_URLS:
        raise MediaError("URL fetching is disabled; send media as a data: URI or base64")
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(4):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise MediaError("media URLs must be http(s) or data: URIs")
        _check_host(parts.hostname)
        req = urllib.request.Request(url, headers={"User-Agent": "embeddinggemma-2-server"})
        try:
            resp = opener.open(req, timeout=FETCH_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise MediaError(f"fetching media failed: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise MediaError(f"fetching media failed: {e}") from None
        with resp:
            data = resp.read(MAX_MEDIA_BYTES + 1)
        if len(data) > MAX_MEDIA_BYTES:
            raise MediaError(f"media is larger than {MAX_MEDIA_BYTES // (1024 * 1024)} MB")
        return data
    raise MediaError("too many redirects")


def load_bytes(src):
    """`src` is a data: URI, http(s) URL, or bare base64."""
    if not isinstance(src, str) or not src:
        raise MediaError("media source must be a non-empty string")
    if src.startswith("data:"):
        head, _, payload = src.partition(",")
        if ";base64" not in head:
            raise MediaError("data: URIs must be base64-encoded")
        src = payload
    elif src.startswith(("http://", "https://")):
        return _fetch(src)
    try:
        data = base64.b64decode(src, validate=True)
    except (binascii.Error, ValueError):
        raise MediaError("media is not a URL, data: URI, or valid base64") from None
    if len(data) > MAX_MEDIA_BYTES:
        raise MediaError(f"media is larger than {MAX_MEDIA_BYTES // (1024 * 1024)} MB")
    return data


def decode_image(data):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # PIL raises many types
        raise MediaError(f"cannot decode image: {e}") from None
    return img.convert("RGB")


def decode_audio(data, max_seconds):
    """Any container/codec PyAV can read -> float32 mono at 16 kHz."""
    import av

    try:
        with av.open(io.BytesIO(data)) as c:
            if not c.streams.audio:
                raise MediaError("no audio stream found")
            res = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
            chunks, n = [], 0
            limit = int(max_seconds * SAMPLE_RATE)
            for frame in c.decode(audio=0):
                for f in res.resample(frame):
                    a = f.to_ndarray().reshape(-1)
                    chunks.append(a)
                    n += len(a)
                if n > limit:
                    raise MediaError(f"audio is longer than {max_seconds:.0f} s")
            for f in res.resample(None):
                chunks.append(f.to_ndarray().reshape(-1))
    except MediaError:
        raise
    except Exception as e:
        raise MediaError(f"cannot decode audio: {e}") from None
    if not chunks:
        raise MediaError("audio stream is empty")
    wave = np.concatenate(chunks).astype(np.float32)
    if len(wave) > limit:
        raise MediaError(f"audio is longer than {max_seconds:.0f} s")
    return wave


def decode_video(data, pick_indices, max_seconds):
    """Decode only the frames the model samples. `pick_indices(total, fps, duration)` returns
    native frame indices (may repeat). Returns uint8 [k, H, W, 3]."""
    import av

    try:
        with av.open(io.BytesIO(data)) as c:
            if not c.streams.video:
                raise MediaError("no video stream found")
            s = c.streams.video[0]
            fps = float(s.average_rate or s.guessed_rate or 0)
            if fps <= 0:
                raise MediaError("cannot read the video frame rate")
            if s.duration is not None and s.time_base is not None:
                duration = float(s.duration * s.time_base)
            elif c.duration is not None:
                duration = c.duration / 1_000_000
            else:
                duration = 0.0
            total = s.frames or int(round(duration * fps))
            if total <= 0 or duration <= 0:
                # Unknown length (some streams): count frames first.
                total = sum(1 for _ in c.decode(video=0))
                duration = total / fps
                c.seek(0)
            if duration > max_seconds:
                raise MediaError(f"video is longer than {max_seconds:.0f} s")
            want = pick_indices(total, fps, duration)
            needed = set(want)
            last = max(want)
            got = {}
            prev = None
            for i, frame in enumerate(c.decode(video=0)):
                if i in needed:
                    got[i] = frame.to_ndarray(format="rgb24")
                prev = frame
                if i >= last:
                    break
            if prev is None:
                raise MediaError("video has no decodable frames")
            fallback = got[max(got)] if got else prev.to_ndarray(format="rgb24")
    except MediaError:
        raise
    except Exception as e:
        raise MediaError(f"cannot decode video: {e}") from None
    # A container that over-reports its length: reuse the last decoded frame.
    return np.stack([got.get(i, fallback) for i in want])
