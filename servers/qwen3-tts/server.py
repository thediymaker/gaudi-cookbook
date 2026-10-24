# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible, streaming POST /v1/audio/speech for Qwen3-TTS-12Hz CustomVoice on one Gaudi card.

Generation runs in engine.FastQwen3TTS; the response formats are in speech_api. Also GET /v1/audio/voices,
/v1/models, /health and /metrics. Run with servers/common on PYTHONPATH (the Dockerfile sets it).
"""

import sigpipe_guard  # noqa: F401  (first: blocks SIGPIPE before torch loads, see the module)

import asyncio
import contextlib
import logging
import os
import re
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

import engine as fe
from hpu_service import Health, ProbeNoiseFilter, epipe_selfcheck
from speech_api import SpeechError, SpeechMetrics, error_response, output_mode, speech_response, split_text, to_pcm16

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("qwen3-tts")

MODEL_ID = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "qwen3-tts")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "vivian")
PORT = int(os.getenv("PORT", "8000"))
MAX_SLOTS = int(os.getenv("MAX_SLOTS", "16"))  # streams generated together in one frame step
MAX_PENDING = int(os.getenv("MAX_PENDING", "64"))  # requests waiting for a slot; past this, 429
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
MAX_INSTRUCTION_CHARS = 1000
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "400"))  # text per generation segment
CHUNK_PAUSE_S, PARAGRAPH_PAUSE_S = 0.12, 0.35
SYNC_EVERY = int(os.getenv("SYNC_EVERY", "4"))  # frames per device round between host syncs
SAMPLE_RATE = fe.SAMPLE_RATE
# OpenAI voice names speak with the first speaker (in sorted order); X-Voice-Mapped-From says so.
OPENAI_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
}

metrics = SpeechMetrics()
health = Health(deadline_s=float(os.getenv("TTS_WATCHDOG_SECONDS", "300")), on_fatal=lambda _: metrics.fatal.set(1))
engine: fe.FastQwen3TTS | None = None
voices: dict[str, str] = {}  # lowercase -> model speaker key
languages: list[str] = []


def load() -> None:
    global engine, voices, languages
    try:
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        t0 = time.time()
        path = MODEL_ID if os.path.isdir(MODEL_ID) else snapshot_download(MODEL_ID)
        tts = Qwen3TTSModel.from_pretrained(path, dtype=torch.bfloat16)
        tts.model = tts.model.to("hpu", dtype=torch.bfloat16).eval()
        tts.device = torch.device("hpu")
        stk = tts.model.speech_tokenizer
        stk.model = stk.model.to("hpu", dtype=torch.bfloat16).eval()
        stk.device = torch.device("hpu")
        fe.hpu_sync()
        log.info("weights on the HPU in %.1fs", time.time() - t0)
        eng = fe.FastQwen3TTS(tts, max_slots=MAX_SLOTS, sync_every=SYNC_EVERY, step_guard=health.device_step)
        eng.on_fatal = lambda exc: health.note_fatal(exc, force=True)
        epipe_selfcheck(lambda: float((torch.ones(8, device="hpu") * 2).sum().item()) == 16.0)
        eng.warmup()
        eng.start()
        seg = eng.build_segment("Ready.", "vivian", "English", None)
        seg["max_frames"] = 60
        t = time.time()
        n = sum(len(a) for a in eng.synthesize([seg]))  # proves the captured graphs replay
        log.info("smoke test: %.2fs of audio in %.2fs", n / SAMPLE_RATE, time.time() - t)
        voices = {str(k).lower(): str(k) for k in tts.model.get_supported_speakers()}
        languages = sorted(str(x).lower() for x in tts.model.get_supported_languages())
        engine = eng
        health.ready()
        log.info("ready in %.1fs: voices=%s languages=%s", time.time() - t0, sorted(voices), languages)
    except Exception as exc:
        log.exception("startup failed")
        health.failed_startup(exc)


@contextlib.asynccontextmanager
async def lifespan(_app):
    health.start_watchdog()
    threading.Thread(target=load, name="loader", daemon=True).start()
    yield


app = FastAPI(title="Qwen3-TTS on Gaudi", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | dict[str, str] | None = None
    language: str | None = None
    instructions: str | None = None  # plain-language style, e.g. "speak slowly and warmly"
    response_format: str = "mp3"
    speed: float = 1.0
    stream: bool | None = None
    stream_format: str | None = None
    max_new_tokens: int = Field(default=2048, ge=1, le=4096)  # frame cap per segment


def resolve_voice(voice) -> tuple[str, str | None]:
    """(model speaker key, the OpenAI name it was mapped from)."""
    if isinstance(voice, dict):
        voice = voice.get("id") or voice.get("name")
    name = (voice or DEFAULT_VOICE).strip().lower()
    if name in voices:
        return voices[name], None
    if name in OPENAI_VOICES:
        return voices[sorted(voices)[0]], name
    raise SpeechError(f"Unknown voice `{voice}`. Voices: {', '.join(sorted(voices))}.", "voice")


_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def frame_cap(text: str, requested: int) -> int:
    """A ceiling that stops a segment that never emits end of speech, without clipping slow speech: about one
    frame (80 ms) per Latin character and three per CJK character is normal; allow twice that plus 3 s."""
    cjk = len(_CJK.findall(text))
    return max(16, min(requested, 2 * (len(text) - cjk) + 6 * cjk + 40))


def plan_segments(text: str, voice: str, language: str, instructions: str | None):
    """Segments that fit the largest prefill bucket with the instructions in front: long instructions or
    token-dense scripts get smaller segments."""
    largest = engine.prefill_buckets[-1]
    limit = CHUNK_CHARS
    while True:
        chunks = split_text(text, limit)
        segs = [engine.build_segment(c, voice, language, instructions) for c, _ in chunks]
        worst = max(len(s["tid"]) for s in segs)
        if worst <= largest:
            break
        if limit <= 50:
            raise SpeechError(
                f"`instructions` plus the shortest piece of `input` is {worst} tokens; the limit is {largest}.",
                "instructions",
            )
        limit //= 2
    pauses = [PARAGRAPH_PAUSE_S if para else CHUNK_PAUSE_S for _, para in chunks[:-1]]
    return chunks, segs, pauses


async def audio_of(segments, pauses, usage):
    """PCM16 bytes as the engine makes them. Closing this generator (a listener who hung up) frees the slot."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    req = fe.Request(segments, pauses, push=lambda item: loop.call_soon_threadsafe(q.put_nowait, item))
    engine.submit(req)
    try:
        while (item := await q.get()) is not None:
            yield to_pcm16(item)
        if req.error:
            raise RuntimeError(req.error)
        usage["output_tokens"] = req.frames
    finally:
        req.cancel()


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if (why := health.unavailable()) is not None:
        return error_response(503, why)
    try:
        if req.model not in (None, SERVED_MODEL_NAME, MODEL_ID):
            raise SpeechError(f"Only `{SERVED_MODEL_NAME}` is served here.", "model")
        text = req.input.strip()
        if not text or len(text) > MAX_INPUT_CHARS:
            raise SpeechError(f"`input` must be 1 to {MAX_INPUT_CHARS} characters.", "input")
        if abs(req.speed - 1.0) > 1e-6:
            raise SpeechError("This model speaks at one speed; send `speed` 1.0 or leave it out.", "speed")
        fmt = req.response_format.lower()
        mode = output_mode(fmt, req.stream, req.stream_format)
        voice, mapped_from = resolve_voice(req.voice)
        language = (req.language or "auto").strip().lower()
        if language not in languages:
            raise SpeechError(f"Unknown language `{req.language}`. Languages: {', '.join(languages)}.", "language")
        instructions = (req.instructions or "").strip() or None
        if instructions and len(instructions) > MAX_INSTRUCTION_CHARS:
            raise SpeechError(f"`instructions` is longer than {MAX_INSTRUCTION_CHARS} characters.", "instructions")
        if len(engine.pending) >= MAX_PENDING:
            raise SpeechError("Too many requests are waiting; retry shortly.", status=429)
        chunks, segments, pauses = plan_segments(text, voice, language, instructions)
    except SpeechError as e:
        return error_response(e.status, str(e), e.param)
    for (chunk, _), seg in zip(chunks, segments, strict=True):
        seg["max_frames"] = frame_cap(chunk, req.max_new_tokens)
    usage = {"input_tokens": sum(s["text_tokens"] for s in segments), "output_tokens": 0}
    headers = {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Voice": voice,
    }
    if mapped_from:
        headers["X-Voice-Mapped-From"] = mapped_from

    def error_status(exc) -> int:
        return 503 if health.note_fatal(exc) or health.fatal else 500

    return await speech_response(
        audio_of(segments, pauses, usage),
        mode=mode,
        fmt=fmt,
        sample_rate=SAMPLE_RATE,
        headers=headers,
        usage=usage,
        metrics=metrics,
        error_status=error_status,
        describe=f"voice={voice} chars={len(text)} segments={len(segments)}",
    )


@app.get("/v1/audio/voices")
def list_voices():
    return {"voices": sorted(voices), "languages": languages}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model", "root": MODEL_ID}]}


@app.get("/health")
def health_check():
    status, body = health.status()
    if engine is not None:
        body.update(busy_slots=sum(1 for s in engine.slots if s), pending=len(engine.pending), **engine.stats)
    return JSONResponse(body, status_code=status)


@app.get("/metrics")
def prometheus():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    logging.getLogger("uvicorn.access").addFilter(ProbeNoiseFilter())
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
