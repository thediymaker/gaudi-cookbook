# SPDX-License-Identifier: Apache-2.0
"""The OpenAI-compatible POST /v1/audio/speech surface shared by the text-to-speech servers.

  response_format pcm                     raw 16-bit mono PCM, streamed as it is made
  response_format wav                     one whole file; streamed (unknown-length header) with stream_format "audio"
  stream: true / stream_format "sse"      OpenAI speech.audio.delta / speech.audio.done events (pcm or wav only)
  response_format mp3 | opus | flac | aac encoded once the whole clip is made

A server hands `speech_response` an async iterator of PCM16 bytes. The response waits for the first chunk before
it answers, so a request that fails before making any audio gets a real error status instead of a cut-off stream.
Closing the response (a listener who hangs up) closes the iterator, which is how servers free the request's slot.
"""

import asyncio
import base64
import io
import json
import logging
import re
import subprocess
import time

import numpy as np
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram

STREAM_TYPES = {"pcm": "audio/pcm", "wav": "audio/wav"}
WHOLE_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac"}
# libsndfile writes these in-process; aac goes through ffmpeg.
_SOUNDFILE = {
    "wav": ("WAV", "PCM_16"),
    "mp3": ("MP3", "MPEG_LAYER_III"),
    "opus": ("OGG", "OPUS"),
    "flac": ("FLAC", "PCM_16"),
}

log = logging.getLogger("speech-api")


class SpeechError(ValueError):
    """A request the server refuses, with the field it is about."""

    def __init__(self, message: str, param: str | None = None, status: int = 400):
        super().__init__(message)
        self.param = param
        self.status = status


def error_response(status: int, message: str, param: str | None = None) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": kind, "param": param, "code": status}}, status)


def output_mode(fmt: str, stream: bool | None, stream_format: str | None) -> str:
    """'sse', 'stream' or 'whole' for a request; raises SpeechError for combinations that are not served."""
    if fmt not in STREAM_TYPES and fmt not in WHOLE_TYPES:
        raise SpeechError(
            f"`response_format` `{fmt}` is not supported. Use pcm, wav, mp3, opus, flac or aac.", "response_format"
        )
    if stream_format not in (None, "audio", "sse"):
        raise SpeechError("`stream_format` must be `audio` or `sse`.", "stream_format")
    if stream_format == "sse" or (stream_format is None and stream is True):
        if fmt not in STREAM_TYPES:
            raise SpeechError("Streaming needs `response_format` pcm or wav.", "response_format")
        return "sse"
    if fmt == "pcm" or (fmt == "wav" and stream_format == "audio"):
        return "stream"
    return "whole"


def wav_header(sample_rate: int, data_bytes: int | None = None) -> bytes:
    """A 16-bit mono WAV header. Without a length (a stream of unknown size) the sizes are 0xFFFFFFFF, which
    players read as "until the data ends"."""
    size = 0xFFFFFFFF if data_bytes is None else data_bytes
    riff = 0xFFFFFFFF if data_bytes is None else 36 + data_bytes
    return b"".join(
        [
            b"RIFF",
            riff.to_bytes(4, "little"),
            b"WAVEfmt ",
            (16).to_bytes(4, "little"),
            (1).to_bytes(2, "little"),
            (1).to_bytes(2, "little"),
            sample_rate.to_bytes(4, "little"),
            (sample_rate * 2).to_bytes(4, "little"),
            (2).to_bytes(2, "little"),
            (16).to_bytes(2, "little"),
            b"data",
            size.to_bytes(4, "little"),
        ]
    )


def to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def encode(pcm16: bytes, fmt: str, sample_rate: int) -> bytes:
    """A whole clip in `fmt`."""
    if fmt == "wav":
        return wav_header(sample_rate, len(pcm16)) + pcm16
    samples = np.frombuffer(pcm16, dtype="<i2")
    if fmt in _SOUNDFILE:
        import soundfile as sf

        container, subtype = _SOUNDFILE[fmt]
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format=container, subtype=subtype)
        return buf.getvalue()
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-f",
        "adts",
        "-codec:a",
        "aac",
        "pipe:1",
    ]
    done = subprocess.run(cmd, input=pcm16, capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode(errors="replace").strip() or "aac encoding failed")
    return done.stdout


# ---------------------------------------------------------------------- text
# End punctuation (then any closing quotes or brackets) before a space, or CJK end punctuation anywhere.
_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*(?=\s)|[。！？；]")
_CLAUSE_END = re.compile(r"(?<=[,;:\u2014\u2013，、；：])\s*")


def _pack(pieces: list[str], limit: int, sep: str = " ") -> list[str]:
    out: list[str] = []
    for piece in pieces:
        if out and len(out[-1]) + len(sep) + len(piece) <= limit:
            out[-1] = f"{out[-1]}{sep}{piece}"
        else:
            out.append(piece)
    return out


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    for clause in (c.strip() for c in _CLAUSE_END.split(sentence)):
        if not clause:
            continue
        if len(clause) <= limit:
            pieces.append(clause)
        elif " " in clause:
            pieces.extend(_pack(clause.split(" "), limit))
        else:  # no spaces (CJK, or one enormous token): cut at the limit
            pieces.extend(clause[i : i + limit] for i in range(0, len(clause), limit))
    return _pack(pieces, limit)


def split_text(text: str, limit: int) -> list[tuple[str, bool]]:
    """Cut text into chunks of at most `limit` characters at natural breaks: (chunk, ends_paragraph).

    Sentences are kept whole where they fit and packed together up to the limit; a longer sentence breaks at
    clause punctuation, then between words. Models degrade on long inputs, so servers synthesize chunk by chunk
    with a short pause between them.
    """
    chunks: list[tuple[str, bool]] = []
    for para in (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        sentences, start = [], 0
        for m in _SENTENCE_END.finditer(para):
            sentences.append(para[start : m.end()].strip())
            start = m.end()
        sentences.append(para[start:].strip())
        pieces = [piece for s in sentences if s for piece in _split_long(s, limit)]
        packed = _pack(pieces, limit)
        chunks.extend((c, i == len(packed) - 1) for i, c in enumerate(packed))
    return chunks


# ------------------------------------------------------------------- metrics
class SpeechMetrics:
    def __init__(self, registry=None):
        kw = {"registry": registry} if registry is not None else {}
        self.requests = Counter("tts_requests_total", "Speech requests by outcome.", ["outcome"], **kw)
        self.first_audio = Histogram(
            "tts_time_to_first_audio_seconds",
            "Request start to first audio sent.",
            buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30),
            **kw,
        )
        self.speed = Histogram(
            "tts_speed_vs_realtime",
            "Seconds of audio made per second of wall time.",
            buckets=(0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16),
            **kw,
        )
        self.audio_seconds = Counter("tts_audio_seconds_total", "Seconds of audio produced.", **kw)
        self.active = Gauge("tts_active_requests", "Requests being synthesized or waiting for a slot.", **kw)
        self.fatal = Gauge("tts_fatal", "The replica hit an unrecoverable HPU fault and waits to be restarted.", **kw)


# ------------------------------------------------------------------ response
async def speech_response(
    audio,
    *,
    mode: str,
    fmt: str,
    sample_rate: int,
    headers: dict,
    usage: dict,
    metrics: SpeechMetrics,
    error_status=lambda exc: 500,
    describe: str = "",
):
    """Turn an async iterator of PCM16 bytes into the response `mode` asks for.

    `usage` (input_tokens, output_tokens) is read when the stream ends, so the producer may fill it as it goes.
    `error_status(exc)` maps a failure to 500 or 503 (and is where a server latches a device fatality).
    """
    start = time.monotonic()
    st = {"bytes": 0, "first": None}
    metrics.active.inc()

    def finish(outcome: str) -> None:
        wall = time.monotonic() - start
        secs = st["bytes"] / 2 / sample_rate
        metrics.active.dec()
        metrics.requests.labels(outcome).inc()
        metrics.audio_seconds.inc(secs)
        if outcome == "ok" and wall > 0 and secs > 0:
            metrics.speed.observe(secs / wall)
        first = f"{st['first']:.3f}s" if st["first"] is not None else "-"
        log.info(
            "%s %s audio=%.2fs wall=%.2fs first_audio=%s speed=%.2fx",
            outcome,
            describe,
            secs,
            wall,
            first,
            secs / wall if wall else 0.0,
        )

    def record(pcm: bytes) -> bytes:
        if st["first"] is None:
            st["first"] = time.monotonic() - start
            metrics.first_audio.observe(st["first"])
        st["bytes"] += len(pcm)
        return pcm

    try:
        first = await audio.__anext__()
    except StopAsyncIteration:
        first = None
    except Exception as exc:
        await audio.aclose()
        finish("error")
        return error_response(error_status(exc), f"Speech generation failed: {exc}")
    if first is None:
        finish("error")
        return error_response(500, "No audio was produced.")

    async def pcm_chunks():
        outcome = "error"
        try:
            yield record(first)
            async for pcm in audio:
                yield record(pcm)
            outcome = "ok"
        except (GeneratorExit, asyncio.CancelledError):
            outcome = "cancelled"
            raise
        except Exception as exc:
            error_status(exc)
            raise
        finally:
            await audio.aclose()
            finish(outcome)

    if mode == "sse":

        async def events():
            try:
                async for pcm in pcm_chunks():
                    delta = {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()}
                    yield f"data: {json.dumps(delta)}\n\n"
            except Exception as exc:  # headers are gone; report it in-band
                failed = {"type": "speech.audio.error", "error": {"message": str(exc), "type": "server_error"}}
                yield f"data: {json.dumps(failed)}\n\n"
                return
            total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            yield f"data: {json.dumps({'type': 'speech.audio.done', 'usage': {**usage, 'total_tokens': total}})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    if mode == "stream":

        async def raw():
            if fmt == "wav":
                yield wav_header(sample_rate)
            try:
                async for pcm in pcm_chunks():
                    yield pcm
            except Exception:  # the status is already sent; all that is left is to end the stream
                return

        return StreamingResponse(raw(), media_type=STREAM_TYPES[fmt], headers=headers)

    pcm = bytearray()
    try:
        async for part in pcm_chunks():
            pcm += part
    except Exception as exc:
        return error_response(error_status(exc), f"Speech generation failed: {exc}")
    try:
        body = await asyncio.to_thread(encode, bytes(pcm), fmt, sample_rate)
    except Exception as exc:
        return error_response(500, f"{fmt} encoding failed: {exc}")
    return Response(body, media_type=WHOLE_TYPES[fmt], headers={**headers, "X-Response-Format": fmt})
