# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible, streaming POST /v1/audio/speech for Orpheus 3B: a CPU sidecar in front of vLLM.

vLLM (serve_llm.sh, on the card) serves the Orpheus LLM, which writes speech as SNAC audio codes. This server cuts
the input into sentence-sized chunks, prompts vLLM's /v1/completions for each with token ids, decodes the streamed
codes to 24 kHz PCM with SNAC on the CPU as they arrive, and sends audio while the rest is still being generated.
The response formats are in speech_api. Also GET /v1/audio/voices, /v1/models, /health (this server and vLLM) and
/metrics (vLLM's metrics followed by this server's). Run with servers/common on PYTHONPATH.
"""

import asyncio
import contextlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, generate_latest
from pydantic import BaseModel

from speech_api import SpeechError, SpeechMetrics, error_response, output_mode, speech_response, split_text, to_pcm16
from speech_stream import END_OF_SPEECH, CodeParser, FrameDecoder, prompt_ids, snac_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("orpheus")

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
LLM_MODEL = os.getenv("ORPHEUS_MODEL", "canopylabs/orpheus-3b-0.1-ft")  # the tokenizer; the same files vLLM serves
LLM_NAME = os.getenv("ORPHEUS_LLM_NAME", "orpheus-3b-llm")  # vLLM's --served-model-name
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "orpheus-3b")
SNAC_MODEL = os.getenv("SNAC_MODEL", "hubertsiuzdak/snac_24khz")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "tara")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
# = vLLM's --max-num-seqs. A ninth listener waits for a slot instead of slowing every stream below real time.
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))
# SNAC on the CPU costs about 20 ms of two-thread time per 85 ms frame: 8 streams keep about 3 workers busy.
DECODE_WORKERS = int(os.getenv("DECODE_WORKERS", "6"))
CHUNK_CHARS = 250
TEMPERATURE, TOP_P = 0.6, 0.8
REPETITION_PENALTY = 1.1  # Orpheus needs at least 1.1 to end its clips; without it about a quarter ran past the text
# Speech runs about 85 codes per second at about 15 characters per second. Twice that per character, with a floor
# for very short chunks, bounds a chunk that never reaches end of speech without clipping a slow speaker.
TOKENS_PER_CHAR, MIN_TOKENS = 12, 600
CHUNK_PAUSE_S, PARAGRAPH_PAUSE_S = 0.12, 0.35
SAMPLE_RATE = 24000

VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]  # in the model card's order of naturalness
OPENAI_VOICES = {
    "alloy": "tara",
    "ash": "dan",
    "ballad": "leo",
    "coral": "leah",
    "echo": "zac",
    "fable": "leo",
    "nova": "mia",
    "onyx": "dan",
    "sage": "zoe",
    "shimmer": "jess",
    "verse": "zac",
}

registry = CollectorRegistry()  # its own registry, appended to vLLM's metrics without name clashes
metrics = SpeechMetrics(registry)
tokenizer = None
snac = None
client: httpx.AsyncClient | None = None
decode_pool = ThreadPoolExecutor(max_workers=DECODE_WORKERS, thread_name_prefix="snac")
slots = asyncio.Semaphore(MAX_CONCURRENT)


def decode_frames(frames: list[list[int]]):
    codes = [torch.tensor([layer], dtype=torch.int32) for layer in snac_layers(frames)]
    with torch.inference_mode():
        return snac.decode(codes)[0, 0].float().numpy()


@contextlib.asynccontextmanager
async def lifespan(_app):
    global tokenizer, snac, client
    from snac import SNAC
    from transformers import AutoTokenizer

    torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    snac = SNAC.from_pretrained(SNAC_MODEL).eval()
    decode_frames([[0] * 7] * 4)  # the first call pays a one-time CPU setup
    client = httpx.AsyncClient(base_url=VLLM_URL, timeout=httpx.Timeout(None, connect=5.0))
    log.info("ready: model=%s voices=%s", SERVED_MODEL_NAME, VOICES)
    yield
    await client.aclose()


app = FastAPI(title="Orpheus TTS", lifespan=lifespan)


class SpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = None
    response_format: str = "mp3"
    speed: float = 1.0
    instructions: str | None = None  # accepted and ignored: the model has no style prompt (use voices and tags)
    stream: bool | None = None
    stream_format: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None


def resolve_voice(voice: str | None) -> tuple[str, str | None]:
    name = (voice or DEFAULT_VOICE).strip().lower()
    if name in VOICES:
        return name, None
    if name in OPENAI_VOICES:
        return OPENAI_VOICES[name], name
    raise SpeechError(f"Unknown voice `{voice}`. Voices: {', '.join(VOICES)}.", "voice")


async def chunk_audio(text: str, voice: str, sampling: dict, usage: dict):
    """PCM16 for one chunk as SNAC frames arrive. Closing it closes the vLLM request too."""
    ids = prompt_ids(tokenizer(f"{voice}: {text}").input_ids)
    usage["input_tokens"] += len(ids)
    payload = {
        "model": LLM_NAME,
        "prompt": ids,
        "max_tokens": max(MIN_TOKENS, TOKENS_PER_CHAR * len(text)),
        "stop_token_ids": [END_OF_SPEECH],
        "skip_special_tokens": False,
        "stream": True,
        **sampling,
    }
    parser = CodeParser()
    decoder = FrameDecoder(decode_frames)
    loop = asyncio.get_running_loop()
    async with client.stream("POST", "/v1/completions", json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode(errors="replace")[:500]
            raise RuntimeError(f"vLLM answered {resp.status_code}: {body}")
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for frame in parser.feed(json.loads(line[6:])["choices"][0].get("text", "")):
                samples = await loop.run_in_executor(decode_pool, decoder.push, frame)
                if samples is not None:
                    yield to_pcm16(samples)
    tail = await loop.run_in_executor(decode_pool, decoder.flush)
    usage["output_tokens"] += parser.tokens
    if decoder.frames == 0:
        log.warning("no audio for chunk %r", text[:80])
    if tail is not None:
        yield to_pcm16(tail)


async def synthesize(text: str, voice: str, sampling: dict, usage: dict):
    """PCM16 for the whole input, chunk after chunk with short pauses, holding a slot for the whole stream."""
    async with slots:
        chunks = split_text(text, CHUNK_CHARS)
        for i, (chunk, ends_paragraph) in enumerate(chunks):
            async with contextlib.aclosing(chunk_audio(chunk, voice, sampling, usage)) as stream:
                async for pcm in stream:
                    yield pcm
            if i < len(chunks) - 1:
                yield bytes(int(SAMPLE_RATE * (PARAGRAPH_PAUSE_S if ends_paragraph else CHUNK_PAUSE_S)) * 2)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if client is None:
        return error_response(503, "The model is still loading.")
    try:
        if req.model not in (None, SERVED_MODEL_NAME):
            raise SpeechError(f"Only `{SERVED_MODEL_NAME}` is served here.", "model")
        text = req.input.strip()
        if not text or len(text) > MAX_INPUT_CHARS:
            raise SpeechError(f"`input` must be 1 to {MAX_INPUT_CHARS} characters.", "input")
        if abs(req.speed - 1.0) > 1e-6:
            raise SpeechError("This model speaks at one speed; send `speed` 1.0 or leave it out.", "speed")
        fmt = req.response_format.lower()
        mode = output_mode(fmt, req.stream, req.stream_format)
        voice, mapped_from = resolve_voice(req.voice)
    except SpeechError as e:
        return error_response(e.status, str(e), e.param)
    sampling = {
        "temperature": TEMPERATURE if req.temperature is None else req.temperature,
        "top_p": TOP_P if req.top_p is None else req.top_p,
        "repetition_penalty": REPETITION_PENALTY if req.repetition_penalty is None else req.repetition_penalty,
    }
    usage = {"input_tokens": 0, "output_tokens": 0}
    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Sample-Rate": str(SAMPLE_RATE)}
    if mapped_from:
        headers["X-Voice-Mapped-From"] = mapped_from
    return await speech_response(
        synthesize(text, voice, sampling, usage),
        mode=mode,
        fmt=fmt,
        sample_rate=SAMPLE_RATE,
        headers=headers,
        usage=usage,
        metrics=metrics,
        describe=f"voice={voice} chars={len(text)}",
    )


@app.get("/v1/audio/voices")
async def voices():
    return {"voices": VOICES}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "canopylabs"}]}


@app.get("/health")
async def health():
    """Ready only when vLLM is: an orchestrator then takes the pod out of service while vLLM restarts. Probe
    /v1/models for liveness, so a vLLM restart does not restart this server too."""
    if client is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    try:
        r = await client.get("/health", timeout=5.0)
    except httpx.HTTPError as e:
        return JSONResponse({"status": "llm unreachable", "detail": str(e)}, status_code=503)
    if r.status_code != 200:
        return JSONResponse({"status": f"llm unhealthy ({r.status_code})"}, status_code=503)
    return {"status": "ok"}


@app.get("/metrics")
async def prometheus():
    upstream = ""  # vLLM's own metrics first, so dashboards built on them see this model too
    if client is not None:
        try:
            r = await client.get("/metrics", timeout=5.0)
            if r.status_code == 200:
                upstream = r.text
        except httpx.HTTPError:
            pass
    return Response(upstream + generate_latest(registry).decode(), media_type="text/plain; version=0.0.4")
