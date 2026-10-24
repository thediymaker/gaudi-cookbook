# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible, streaming POST /v1/audio/speech for Chatterbox and Chatterbox Turbo on one Gaudi card.

Both models load on the same card and are named by `model`: `chatterbox` (the original, the default) or
`chatterbox-turbo`. The response formats are in speech_api. Also GET /v1/audio/voices, /v1/models, /health and
/metrics. Run with servers/common on PYTHONPATH (the Dockerfile sets it).

One thread (DeviceWorker) owns the card and loops: admit waiting requests into free T3 slots (up to MAX_CONCURRENT
per model; more wait, and past MAX_QUEUE waiting they get 429), advance every decoding slot of each model by one
batched block of 8 tokens, then run one batched flow pass over the requests that have a chunk ready (the listener
with the least audio buffered first) and vocode each chunk. Requests share each device step instead of taking
turns. Long input is split into sentence chunks of at most 300 characters, spoken back to back with a short pause.

Voices: `default` (the checkpoint's built-in voice) and every audio file in VOICES_DIR (5 to 10 s of clean speech),
enrolled at startup under its file name for both models. OpenAI voice names speak with the default voice.
"""

import sigpipe_guard  # noqa: F401  (first: blocks SIGPIPE before torch loads, see the module)

import asyncio
import contextlib
import itertools
import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from hpu_service import Health, ProbeNoiseFilter, epipe_selfcheck
from speech_api import SpeechError, SpeechMetrics, error_response, output_mode, speech_response, split_text, to_pcm16

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("chatterbox")

VARIANTS = [v.strip() for v in os.getenv("CHATTERBOX_VARIANTS", "turbo,original").split(",") if v.strip()]
MODEL_NAMES = {"turbo": "chatterbox-turbo", "original": "chatterbox"}
DEFAULT_MODEL = os.getenv("CHATTERBOX_DEFAULT_MODEL", "chatterbox")
PORT = int(os.getenv("PORT", "8000"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4096"))
CHUNK_CHARS = 300
CHUNK_PAUSE_S, PARAGRAPH_PAUSE_S = 0.2, 0.35
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))  # requests per model decoding together, each in a T3 slot
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "32"))
WATERMARK = os.getenv("WATERMARK", "1") == "1"
VOICES_DIR = os.getenv("VOICES_DIR", "")
# Under load, a ready flow chunk waits (so more requests' chunks share one batched pass) while its listener still has
# more than this much audio buffered.
FLOW_DEFER_LEAD_S = 1.0
SLOW_STEP_S = 1.0  # device steps slower than this are logged with the graph compiles they hit
SAMPLE_RATE = 24000
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


def _compiles() -> int:
    """HPU graph compilations so far in this process."""
    try:
        from habana_frameworks.torch.hpu.metrics import metric_global

        return int(dict(metric_global("graph_compilation").stats()).get("TotalNumber", 0))
    except Exception:  # noqa: BLE001
        return 0


_DONE = object()
_SEQ = itertools.count()


class Job:
    """One speech request: its text chunks, a T3 slot while it runs, and an output for float32 audio, then _DONE
    or an exception."""

    def __init__(self, variant: str, chunks, voice, params: dict, push=None):
        self.variant = variant
        self.chunks = chunks
        self.voice = voice
        self.params = params
        self.push = push or (lambda item: None)  # self-test jobs drop their audio
        self.cancelled = False
        self.slot: int | None = None
        self.idx = -1  # current text chunk
        self.utt = None  # model.Utterance for chunks[idx]
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.first_t: float | None = None  # when the first audio went out
        self.sent_s = 0.0  # seconds of audio sent
        self.seq = next(_SEQ)

    def put(self, item):
        if isinstance(item, np.ndarray):
            if self.first_t is None:
                self.first_t = time.perf_counter()
            self.sent_s += len(item) / SAMPLE_RATE
        self.push(item)

    def lead(self, now: float) -> float:
        """Seconds of audio the listener has buffered (most urgent = smallest). Requests still waiting for their
        first audio come first, oldest first."""
        if self.first_t is None:
            return -1e9 + self.seq
        return self.sent_s - (now - self.first_t)


class DeviceWorker(threading.Thread):
    """Owns the card. Each round: admit waiting requests into free T3 slots (prefill), advance every decoding slot
    of each model by one batched block of tokens, then give every request with enough tokens one flow and vocoder
    chunk."""

    def __init__(self):
        super().__init__(name="hpu-worker", daemon=True)
        self.inbox: queue.Queue = queue.Queue()
        self.models: dict = {}
        self.waiting: list[Job] = []
        self.active: list[Job] = []

    def submit(self, job: Job):
        self.inbox.put(job)

    def _load(self):
        import torch

        import model

        torch.set_num_threads(int(os.getenv("TORCH_THREADS", "8")))
        epipe_selfcheck(lambda: float((torch.ones(8, device="hpu") * 2).sum().item()) == 16.0)
        for v in VARIANTS:
            t0 = time.time()
            m = model.FastChatterbox(v, watermark=WATERMARK, slots=MAX_CONCURRENT)
            self.models[v] = m
            if VOICES_DIR and Path(VOICES_DIR).is_dir():
                for f in sorted(Path(VOICES_DIR).iterdir()):
                    if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
                        try:
                            m.enroll(f.stem.lower(), str(f))
                            log.info("%s: enrolled voice %s from %s", v, f.stem.lower(), f.name)
                        except Exception as e:  # noqa: BLE001
                            log.warning("%s: could not enroll %s: %s", v, f.name, e)
            m.warmup()
            log.info("%s ready in %.1fs (voices: %s)", v, time.time() - t0, ", ".join(m.voices))

    def _device(self, fn, *a, **kw):
        """One unit of device work under the watchdog; slow ones are logged with the compiles they hit."""
        t = time.perf_counter()
        c0 = _compiles()
        try:
            with health.device_step():
                return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t
            if dt > SLOW_STEP_S:
                log.warning(
                    "slow device step %s: %.0f ms (graph compiles +%d)",
                    getattr(fn, "__qualname__", fn),
                    dt * 1000,
                    _compiles() - c0,
                )

    def _next_chunk(self, job: Job) -> bool:
        """Start the job's next text chunk in its slot. False when there is none."""
        if job.utt is not None:
            job.usage["output_tokens"] += len(job.utt.toks)
        job.idx += 1
        if job.idx >= len(job.chunks):
            return False
        m = self.models[job.variant]
        job.utt = m.utterance(job.chunks[job.idx][0], job.voice, **job.params)
        job.usage["input_tokens"] += job.utt.n_input
        self._device(m.t3e.start, job.slot, job.utt.embeds, **job.utt.t3_kw)
        return True

    def _finish(self, job: Job, item=_DONE):
        if job.slot is not None:
            self.models[job.variant].t3e.release(job.slot)
            job.slot = None
        if job in self.active:
            self.active.remove(job)
        job.put(item)

    def _fail(self, job: Job, exc: Exception):
        log.error("synthesis failed: %s: %s", type(exc).__name__, exc)
        health.note_fatal(exc)
        self._finish(job, exc)

    def run(self):
        try:
            self._load()
            self._selftest()
            health.ready()
        except Exception as exc:  # noqa: BLE001
            log.exception("model load failed")
            health.failed_startup(exc)
        while True:
            if not self.active and not self.waiting:
                self.waiting.append(self.inbox.get())  # idle: block until work arrives
            while True:
                try:
                    self.waiting.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            self._iterate()

    def _selftest(self):
        """1, 2 and MAX_CONCURRENT requests per model through this very loop before reporting ready: it settles the
        lazy-mode graphs of the serving path (whose composition differs from the model-level warm-up), so the first
        real requests pay no compile."""
        t0 = time.time()
        text = "Warming up the serving path. This sentence is spoken and thrown away."
        for v, m in self.models.items():
            for n in sorted({1, min(2, MAX_CONCURRENT), MAX_CONCURRENT}):
                self.waiting += [
                    Job(v, split_text(text, CHUNK_CHARS), m.voices["default"], {"seed": 7} if i else {})
                    for i in range(n)
                ]
                while self.waiting or self.active:
                    self._iterate()
        log.info("serving-path self-test done in %.1fs", time.time() - t0)

    def _iterate(self):
        """One scheduling round over waiting and active requests."""
        still = []
        for job in self.waiting:  # admit, oldest first, into the lowest free slot
            if job.cancelled:
                job.put(_DONE)
                continue
            if (why := health.unavailable()) is not None and not health.loading:
                job.put(RuntimeError(why))
                continue
            slot = self.models[job.variant].free_slot()
            if slot is None:
                still.append(job)
                continue
            job.slot = slot
            self.active.append(job)
            try:
                self._next_chunk(job)
            except Exception as exc:  # noqa: BLE001
                self._fail(job, exc)
        self.waiting = still
        for v, m in self.models.items():  # one batched T3 block per model
            if not any(m.t3e.decoding):
                continue
            try:
                res = self._device(m.t3e.run_block)
            except Exception as exc:  # noqa: BLE001
                for job in [j for j in self.active if j.variant == v]:
                    self._fail(job, exc)
                continue
            for job in self.active:
                if job.variant == v and job.slot in res and job.utt is not None:
                    job.utt.feed(*res[job.slot])
        for job in [j for j in self.active if j.cancelled]:
            self._finish(job)
        now = time.perf_counter()
        for v, m in self.models.items():  # flow + vocoder: ready requests of a model in one batched pass
            ready = sorted((j for j in self.active if j.variant == v and j.utt.ready()), key=lambda j: j.lead(now))
            ready = ready[: m.flow.batch_sizes[-1]]
            if not ready:
                continue
            # Defer while every ready listener is comfortably ahead and other requests are still decoding: a later
            # pass then batches more requests' chunks.
            others = sum(1 for j in self.active if j.variant == v) - len(ready)
            if ready[0].lead(now) > FLOW_DEFER_LEAD_S and others > 0 and any(m.t3e.decoding):
                continue
            try:
                plans = [(j, j.utt.plan()) for j in ready]
                reqs = [p for _, p in plans if p is not None]
                mels = iter(self._device(m.flow.run_many, reqs) if reqs else [])
            except Exception as exc:  # noqa: BLE001
                for job in ready:
                    self._fail(job, exc)
                continue
            for job, p in plans:
                try:
                    pcm = self._device(job.utt.finish, next(mels) if p is not None else None)
                    if pcm.size:
                        job.put(pcm)
                    if job.utt.finished:
                        if self._next_chunk(job):
                            para = job.chunks[job.idx - 1][1]
                            job.put(
                                np.zeros(int(SAMPLE_RATE * (PARAGRAPH_PAUSE_S if para else CHUNK_PAUSE_S)), np.float32)
                            )
                        else:
                            self._finish(job)
                except Exception as exc:  # noqa: BLE001
                    self._fail(job, exc)


worker = DeviceWorker()


@contextlib.asynccontextmanager
async def lifespan(_app):
    health.start_watchdog()
    worker.start()
    yield


app = FastAPI(title="Chatterbox on Gaudi", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | dict[str, str] | None = None
    response_format: str = "mp3"
    speed: float = 1.0
    stream: bool | None = None
    stream_format: str | None = None
    # Chatterbox's own controls. Turbo ignores exaggeration and cfg_weight (it has no emotion input and was
    # distilled without CFG) but reads paralinguistic tags in the text, such as [laugh] or [sigh].
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    seed: int | None = None


def resolve_model(name: str | None) -> str:
    name = name or DEFAULT_MODEL
    for v in VARIANTS:
        if name in (MODEL_NAMES[v], f"ResembleAI/{MODEL_NAMES[v]}"):
            return v
    raise SpeechError(f"Unknown model `{name}`. Served: {', '.join(MODEL_NAMES[v] for v in VARIANTS)}.", "model")


def fit_chunks(m, voice, chunks):
    """Split further any chunk whose T3 prompt would not fit the prefill limit (unusual characters can tokenize to
    several ids each)."""
    out = []
    for text, para in chunks:
        todo, pieces = [text], []
        while todo:
            t = todo.pop(0)
            if len(t) < 24 or m.prefill_len(voice, len(m.text_ids(t))) <= m.t3e.max_prefill:
                pieces.append(t)
                continue
            cut = t.rfind(" ", 0, len(t) // 2 + 12)
            cut = cut if cut > 0 else len(t) // 2
            todo[:0] = [t[:cut].strip(), t[cut:].strip()]
        out += [(pc, para and i == len(pieces) - 1) for i, pc in enumerate(pieces) if pc]
    return out


async def audio_of(job: Job):
    """PCM16 bytes as the worker makes them. Closing this generator (a listener who hung up) frees the slot."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    job.push = lambda item: loop.call_soon_threadsafe(q.put_nowait, item)
    worker.submit(job)
    try:
        while (item := await q.get()) is not _DONE:
            if isinstance(item, Exception):
                raise item
            yield to_pcm16(item)
    finally:
        job.cancelled = True


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if (why := health.unavailable()) is not None:
        return error_response(503, why)
    try:
        variant = resolve_model(req.model)
        text = req.input.strip()
        if not text or len(text) > MAX_INPUT_CHARS:
            raise SpeechError(f"`input` must be 1 to {MAX_INPUT_CHARS} characters.", "input")
        if abs(req.speed - 1.0) > 1e-6:
            raise SpeechError("This model speaks at one speed; send `speed` 1.0 or leave it out.", "speed")
        fmt = req.response_format.lower()
        mode = output_mode(fmt, req.stream, req.stream_format)
        m = worker.models[variant]
        vname = req.voice.get("id") or req.voice.get("name") if isinstance(req.voice, dict) else req.voice
        vname = (vname or "default").strip().lower()
        mapped_from = None
        if vname not in m.voices:
            if vname not in OPENAI_VOICES:
                raise SpeechError(f"Unknown voice `{vname}`. Voices: {', '.join(m.voices)}.", "voice")
            mapped_from, vname = vname, "default"
        if len(worker.waiting) + worker.inbox.qsize() >= MAX_QUEUE:
            raise SpeechError("Too many requests are waiting; retry shortly.", status=429)
    except SpeechError as e:
        return error_response(e.status, str(e), e.param)
    voice = m.voices[vname]
    params = {
        "exaggeration": req.exaggeration,
        "cfg_weight": req.cfg_weight,
        "temperature": req.temperature,
        "seed": req.seed,
    }
    job = Job(variant, fit_chunks(m, voice, split_text(text, CHUNK_CHARS)), voice, params)
    headers = {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Model-Id": MODEL_NAMES[variant],
    }
    if mapped_from:
        headers["X-Voice-Mapped-From"] = mapped_from

    def error_status(exc) -> int:
        return 503 if health.note_fatal(exc) or health.fatal else 500

    return await speech_response(
        audio_of(job),
        mode=mode,
        fmt=fmt,
        sample_rate=SAMPLE_RATE,
        headers=headers,
        usage=job.usage,
        metrics=metrics,
        error_status=error_status,
        describe=f"model={MODEL_NAMES[variant]} voice={vname} chars={len(text)} chunks={len(job.chunks)}",
    )


@app.get("/v1/audio/voices")
def voices(model: str | None = None):
    try:
        m = worker.models.get(resolve_model(model))
    except SpeechError as e:
        return error_response(400, str(e), "model")
    return {"voices": list(m.voices) if m is not None else ["default"]}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAMES[v], "object": "model", "owned_by": "resemble-ai", "root": f"ResembleAI/{MODEL_NAMES[v]}"}
            for v in VARIANTS
        ],
    }


@app.get("/health")
def health_check():
    status, body = health.status()
    body.update(active=len(worker.active), waiting=len(worker.waiting))
    return JSONResponse(body, status_code=status)


@app.get("/metrics")
def prometheus():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    logging.getLogger("uvicorn.access").addFilter(ProbeNoiseFilter())
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
