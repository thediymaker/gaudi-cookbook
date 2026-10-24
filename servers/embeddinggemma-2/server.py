# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible POST /v1/embeddings for EmbeddingGemma 2 (text, image, audio, video) on one Gaudi card.

Request fields follow vLLM's embeddings API, so clients can switch between this and vLLM-served embedding models:
  input                   a string, a list of strings, token ids (a list of ints or lists of ints), or multimodal
                          items (below)
  messages                vLLM-style chat messages instead of `input`: every content part of every message, in
                          order, becomes ONE embedding
  dimensions              Matryoshka width, 1 to 768 (trained at 128, 256, 512 and 768)
  encoding_format         float | base64
  truncate_prompt_tokens  -1 = the model maximum (text and token-id inputs only)
Extra:
  prompt_name             adds the model's task prefix to inputs that contain text (query, document, CodeRetrieval,
                          ...; see /v1/prompts). Media-only inputs never get a prefix, as the model card says.

A multimodal item is a list of content parts (or a single part), embedded as one vector in the order given:
  {"type": "text", "text": "..."}
  {"type": "image_url", "image_url": {"url": "<https URL | data: URI>"}}
  {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
  {"type": "audio_url", "audio_url": {"url": "..."}}
  {"type": "video_url", "video_url": {"url": "..."}}
Short forms {"text": ...}, {"image": src}, {"audio": src}, {"video": src} also work. An image costs up to 280
tokens, audio 25 tokens per second, video 140 tokens per frame at 1 frame per second (at most 32 frames). The whole
item must fit in 8192 tokens. Run with servers/common on PYTHONPATH (the Dockerfile sets it).
"""

import sigpipe_guard  # noqa: F401  (first: blocks SIGPIPE before torch loads, see the module)

import asyncio
import base64
import contextlib
import logging
import os
import threading
import time
import uuid
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import media
from engine import MAX_LEN, MODALITIES, MODEL, NATIVE_DIM, TOKENS_PER_BATCH, Engine, Prepared
from hpu_service import Health, ProbeNoiseFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("embeddinggemma-2")

SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", "embeddinggemma-2")
PORT = int(os.environ.get("PORT", "8000"))
MAX_INPUTS = 2048
MAX_MEDIA_PER_INPUT = 32
MAX_AUDIO_SECONDS = 320.0
MAX_VIDEO_SECONDS = 1800.0
MAX_BODY_BYTES = 64 * 1024 * 1024
# Task prefixes from the checkpoint's config_sentence_transformers.json.
PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "SearchQuery": "task: search result | query: ",
    "Document": "title: none | text: ",
    "QuestionAnswering": "task: question answering | query: ",
    "FactChecking": "task: fact checking | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
}
PLACEHOLDER = {"image": "<|image|>", "audio": "<|audio|>", "video": "<|video|>"}

health = Health(deadline_s=600.0)
engine: Engine | None = None
batcher = None
stats = {"requests": 0, "inputs": 0, "tokens": 0, "errors": 0, "images": 0, "audio_clips": 0, "videos": 0}
in_flight = 0  # requests handed to the batcher and not answered yet (event loop only)


def load():
    global engine, batcher
    try:
        t = time.time()
        engine = Engine()
        log.info("loaded %s (%s) in %.1fs", MODEL, ", ".join(sorted(engine.modalities)), time.time() - t)
        t = time.time()
        engine.warmup(log=log.info)
        log.info("warmup done in %.1fs", time.time() - t)
        batcher = Batcher()
        health.ready()
    except Exception as exc:
        log.exception("startup failed")
        health.failed_startup(exc)


@contextlib.asynccontextmanager
async def lifespan(_app):
    health.start_watchdog()
    threading.Thread(target=load, name="loader", daemon=True).start()
    yield


app = FastAPI(title="EmbeddingGemma 2 on Gaudi", lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    input: Any = None
    messages: list[dict] | None = None
    model: str | None = None
    dimensions: int | None = None
    encoding_format: str = "float"
    truncate_prompt_tokens: int | None = None
    prompt_name: str | None = None
    user: str | None = None


class BadRequest(ValueError):
    pass


def error(status, message, kind="invalid_request_error"):
    stats["errors"] += 1
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind, "code": status}})


def _part(p):
    """One content part -> ("text", str) | (kind, source)."""
    if isinstance(p, str):
        return ("text", p)
    if not isinstance(p, dict):
        raise BadRequest("content parts must be strings or objects")
    t = p.get("type")
    if t is None:  # short forms
        keys = [k for k in ("text", "image", "audio", "video") if k in p]
        if len(keys) != 1:
            raise BadRequest("a content part needs a `type`, or exactly one of text/image/audio/video")
        return (keys[0], p[keys[0]])
    if t == "text":
        return ("text", p.get("text", ""))
    if t in ("image_url", "audio_url", "video_url"):
        v = p.get(t)
        return (t[:-4], v.get("url") if isinstance(v, dict) else v)
    if t == "input_audio":
        return ("audio", (p.get("input_audio") or {}).get("data"))
    if t in ("image", "audio", "video"):
        return (t, p.get(t) or p.get("data") or p.get("url"))
    raise BadRequest(f"unsupported content part type {t!r}")


def parse_items(req):
    """-> list of ("ids", [int]) | ("text", str) | ("parts", [(kind, value), ...])."""
    if req.messages is not None:
        if req.input is not None:
            raise BadRequest("send either `input` or `messages`, not both")
        parts = []
        for m in req.messages:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(("text", c))
            elif isinstance(c, list):
                parts.extend(_part(p) for p in c)
            elif c is not None:
                raise BadRequest("message content must be a string or a list of parts")
        if not parts:
            raise BadRequest("messages contain no content")
        return [("parts", parts)]
    raw = req.input
    if raw is None:
        raise BadRequest("`input` is required")
    if isinstance(raw, str):
        return [("text", raw)]
    if isinstance(raw, dict):
        return [("parts", [_part(raw)])]
    if not isinstance(raw, list) or not raw:
        raise BadRequest("input must not be empty")
    if all(isinstance(x, int) for x in raw):
        return [("ids", raw)]
    items = []
    for x in raw:
        if isinstance(x, str):
            items.append(("text", x))
        elif isinstance(x, dict):
            items.append(("parts", [_part(x)]))
        elif isinstance(x, list) and x and all(isinstance(t, int) for t in x):
            items.append(("ids", x))
        elif isinstance(x, list) and x:
            items.append(("parts", [_part(p) for p in x]))
        else:
            raise BadRequest("each input must be a string, token ids, or a list of content parts")
    return items


def prepare_parts(parts, prefix):
    """Decode media and run the processor. CPU work; runs in a worker thread."""
    n_media = sum(1 for k, _ in parts if k != "text")
    if n_media > MAX_MEDIA_PER_INPUT:
        raise BadRequest(f"at most {MAX_MEDIA_PER_INPUT} media items per input")
    if not n_media:
        return Prepared(engine.tokenize([prefix + " ".join(v for _, v in parts)], truncate=False)[0]), {}
    texts, images, audios, videos = [], [], [], []
    for kind, v in parts:
        if kind == "text":
            if not isinstance(v, str):
                raise BadRequest("text parts must be strings")
            texts.append(v)
            continue
        if kind not in PLACEHOLDER:
            raise BadRequest(f"unsupported media type {kind!r}")
        if kind not in engine.modalities:
            raise BadRequest(f"{kind} input is not enabled on this server")
        data = media.load_bytes(v)
        if kind == "image":
            images.append(media.decode_image(data))
        elif kind == "audio":
            audios.append(media.decode_audio(data, MAX_AUDIO_SECONDS))
        else:
            videos.append(media.decode_video(data, engine.sample_video_indices, MAX_VIDEO_SECONDS))
        texts.append(PLACEHOLDER[kind])
    has_text = any(k == "text" and v.strip() for k, v in parts)
    p = engine.prepare((prefix if has_text else "") + " ".join(texts), images=images, audios=audios, videos=videos)
    return p, {"images": len(images), "audio_clips": len(audios), "videos": len(videos)}


class Batcher:
    """Collects prepared inputs from concurrent requests and embeds them together on the card."""

    def __init__(self):
        self.pending = []  # (items, future, loop)
        self.cv = threading.Condition()
        threading.Thread(target=self._run, name="batcher", daemon=True).start()

    def submit(self, items):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self.cv:
            self.pending.append((items, fut, loop))
            self.cv.notify()
        return fut

    def _run(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
                # Whole requests, until roughly 8 forwards' worth of tokens.
                batch, budget = [], TOKENS_PER_BATCH * 8
                while self.pending and (not batch or budget > 0):
                    batch.append(self.pending.pop(0))
                    budget -= sum(len(p.ids) for p in batch[-1][0])
            flat = [p for items, _, _ in batch for p in items]
            try:
                with health.device_step():
                    out = engine.embed_items(flat)
                k = 0
                for items, fut, loop in batch:
                    loop.call_soon_threadsafe(_resolve, fut, out[k : k + len(items)], None)
                    k += len(items)
            except Exception as exc:  # report to every waiting request
                health.note_fatal(exc)
                for _, fut, loop in batch:
                    loop.call_soon_threadsafe(_resolve, fut, None, exc)


def _resolve(fut, value, exc):
    if fut.done():
        return
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(value)


@app.get("/health")
def health_check():
    status, body = health.status()
    return JSONResponse(body, status_code=status)


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "google",
                "root": MODEL,
                "max_model_len": MAX_LEN,
                "modalities": list(MODALITIES),
            }
        ],
    }


@app.get("/v1/prompts")
def prompts():
    return PROMPTS


@app.get("/metrics")
def metrics():
    lines = [f"eg2_{k}_total {v}" for k, v in stats.items()]
    # vLLM-named series too, so autoscalers and dashboards built for vLLM read this server the same way.
    label = f'{{model_name="{SERVED_NAME}"}}'
    queued = len(batcher.pending) if batcher else 0
    lines += [
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{label} {in_flight - queued}",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{label} {queued}",
        "# TYPE vllm:request_success_total counter",
        f"vllm:request_success_total{label} {stats['requests']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    if (why := health.unavailable()) is not None:
        return error(503, why, "service_unavailable")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return error(413, f"request body is larger than {MAX_BODY_BYTES // (1024 * 1024)} MB")
    try:
        req = EmbeddingRequest.model_validate_json(body)
    except Exception as e:
        return error(400, f"invalid request: {str(e)[:300]}")
    if req.model and req.model not in (SERVED_NAME, MODEL):
        return error(404, f"model '{req.model}' not found; this server serves '{SERVED_NAME}'", "not_found_error")
    if req.encoding_format not in ("float", "base64"):
        return error(400, "encoding_format must be 'float' or 'base64'")
    dim = req.dimensions or NATIVE_DIM
    if not 1 <= dim <= NATIVE_DIM:
        return error(400, f"dimensions must be between 1 and {NATIVE_DIM} (trained: 128, 256, 512, 768)")
    prefix = ""
    if req.prompt_name is not None:
        if req.prompt_name not in PROMPTS:
            return error(400, f"unknown prompt_name; one of {sorted(PROMPTS)}")
        prefix = PROMPTS[req.prompt_name]
    limit = req.truncate_prompt_tokens
    if limit is not None:
        limit = MAX_LEN if limit == -1 else limit
        if not 1 <= limit <= MAX_LEN:
            return error(400, f"truncate_prompt_tokens must be -1 or 1..{MAX_LEN}")

    try:
        items = parse_items(req)
        if len(items) > MAX_INPUTS:
            raise BadRequest(f"at most {MAX_INPUTS} inputs per request")
        texts = [v for k, v in items if k == "text"]
        tokenized = iter(engine.tokenize([prefix + t for t in texts], truncate=False)) if texts else iter(())
        prepared, counts = [], {}
        for kind, v in items:
            if kind == "text":
                prepared.append(Prepared(next(tokenized)))
            elif kind == "ids":
                if prefix:
                    raise BadRequest("prompt_name needs text input, not token ids")
                prepared.append(Prepared(list(v)))
            else:
                p, c = await asyncio.to_thread(prepare_parts, v, prefix)
                prepared.append(p)
                for key, n in c.items():
                    counts[key] = counts.get(key, 0) + n
    except (BadRequest, media.MediaError) as e:
        return error(400, str(e))
    except Exception as e:  # processor errors on odd media
        return error(400, f"could not process input: {str(e)[:300]}")

    eos = engine.tok.eos_token_id
    for i, p in enumerate(prepared):
        if len(p.ids) > MAX_LEN and (limit is None or p.has_media):
            what = "media input" if p.has_media else "input"
            hint = "Send fewer or shorter media items." if p.has_media else "Shorten it or set truncate_prompt_tokens."
            return error(400, f"{what} {i} has {len(p.ids)} tokens; the maximum is {MAX_LEN}. {hint}")
        if limit is not None and not p.has_media and len(p.ids) > limit:
            p.ids = p.ids[: limit - 1] + [eos]  # keep the trailing EOS, as sentence-transformers does

    global in_flight
    in_flight += 1
    try:
        vecs = await batcher.submit(prepared)
    except Exception as e:
        return error(500, f"embedding failed: {str(e)[:300]}", "server_error")
    finally:
        in_flight -= 1
    if dim != NATIVE_DIM:
        vecs = torch.nn.functional.normalize(vecs[:, :dim], dim=-1)
    n_tok = sum(len(p.ids) for p in prepared)
    stats["requests"] += 1
    stats["inputs"] += len(prepared)
    stats["tokens"] += n_tok
    for key, n in counts.items():
        stats[key] += n
    data = []
    for i, v in enumerate(vecs):
        emb = (
            base64.b64encode(v.numpy().astype(np.float32).tobytes()).decode()
            if req.encoding_format == "base64"
            else v.tolist()
        )
        data.append({"object": "embedding", "index": i, "embedding": emb})
    return {
        "id": f"embd-{uuid.uuid4().hex}",
        "object": "list",
        "created": int(time.time()),
        "model": SERVED_NAME,
        "data": data,
        "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok},
    }


if __name__ == "__main__":
    logging.getLogger("uvicorn.access").addFilter(ProbeNoiseFilter())
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
