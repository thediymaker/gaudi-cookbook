# EmbeddingGemma 2 server

Text, image, audio and video embeddings from [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
(Apache-2.0, not gated) on one Gaudi 2 card, behind a vLLM-compatible `POST /v1/embeddings`. vllm-gaudi v0.30.0
cannot serve the model (its HPU prompt attention is causal only, and upstream support is newer than the release), so
this server runs transformers' own model classes in lazy mode with HPU graphs. Design and measurements:
[EmbeddingGemma 2 without vLLM](../../docs/results/embeddinggemma-2.md).

| | |
| --- | --- |
| Base image | `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` (no vLLM) |
| Packages | `transformers==5.20.2` (the model classes first shipped in 5.19), PyAV from the base image; all pinned in `requirements.txt` |
| Output | 768-d, L2-normalized; Matryoshka widths 128, 256 and 512 through `dimensions` |
| Inputs | up to 8192 tokens each: an image costs up to 280 tokens, audio 25 per second, video 140 per frame at 1 frame per second (at most 32 frames) |
| Startup | about 2.5 minutes: load, then every graph compiled (`warmup done in 135.7s`) |

## Files

| File | What it is |
| --- | --- |
| `engine.py` | the HPU path: language, vision and audio forwards wrapped as HPU graphs, every shape from fixed buckets, masks built so a captured graph always applies them, the audio `unfold` replaced by an index gather, batching by length bucket |
| `server.py` | the HTTP side: request parsing (strings, token ids, content parts, `messages`), task prefixes, truncation, a batcher that embeds concurrent requests together |
| `media.py` | fetch and decode: `data:` URIs, base64, or http(s) URLs that resolve to public addresses only; images through PIL, audio and video through PyAV |
| `install.sh`, `requirements.txt`, `Dockerfile` | the image |
| `tests/test_embeddinggemma2_cpu.py` | CPU tests: the gather against `Tensor.unfold`, media decoding and the URL guard, request parsing |
| `tests/hpu/parity.py` | device parity against the stock model in fp32 on the CPU (below) |
| `tests/hpu/api_gates.py` | API forms, the URL guard and throughput against a running server |
| `tests/hpu/run_gates.sh` | runs both |

## Build and run

```bash
docker build -f servers/embeddinggemma-2/Dockerfile -t <registry>/embeddinggemma-2-gaudi:<tag> .
docker run -it --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all --cap-add=sys_nice --ipc=host -p 8000:8000 \
  -e HF_HOME=/hf -v ~/.cache/huggingface:/hf <registry>/embeddinggemma-2-gaudi:<tag>
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `EG2_MODEL` | `google/embeddinggemma-2` | Hugging Face id or a local directory |
| `SERVED_MODEL_NAME` | `embeddinggemma-2` | the `model` name requests use |
| `EG2_MODALITIES` | `text,image,audio,video` | inputs to load and compile; leaving out image and video (or audio) skips that tower |
| `EG2_ALLOW_URL_FETCH` | `1` | `0` accepts only inline media (data URIs and base64) |
| `PT_HPU_RECIPE_CACHE_CONFIG` | unset | node-local recipe cache to shorten restarts |

## API

```bash
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "embeddinggemma-2", "prompt_name": "query", "dimensions": 256,
  "input": ["how do I request a node",
            [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}],
            [{"type": "text", "text": "a recording of"}, {"type": "input_audio", "input_audio": {"data": "<base64 wav>"}}]]}'
```

Fields as vLLM's embeddings API: `input` (a string, strings, token ids, or content parts), `messages` instead of
`input` (every part of every message becomes one vector), `dimensions`, `encoding_format` (`float` or `base64`),
`truncate_prompt_tokens` (text only; the trailing end-of-sequence token is kept). `prompt_name` adds the
checkpoint's task prefix to inputs that contain text (`GET /v1/prompts`); media-only inputs never get one. Content
parts: `text`, `image_url`, `input_audio`, `audio_url`, `video_url`, or the short forms `{"image": src}`,
`{"audio": src}`, `{"video": src}`. Audio in any format PyAV decodes is downmixed and resampled to 16 kHz mono.
`/metrics` also exports `vllm:num_requests_running`, `vllm:num_requests_waiting` and `vllm:request_success_total`,
so autoscalers and dashboards built for vLLM read this server the same way.

## Verify

```bash
tests/hpu/run_gates.sh
```

Measured on one Gaudi 2 card, Gaudi software 1.24.1, in a pod started from the stock base image with this
directory's `install.sh`; the client ran in the same pod (per call, one card):

```text
PASS capture_order: cosine 0.999708, 0.999873, 0.999867 (min 0.9995); stock bf16 on CPU: 0.999624, 0.999859, 0.999911
PASS text: cosine 0.999979, 0.999959, 0.999968 (min 0.9995); stock bf16 on CPU: 0.999975, 0.999957, 0.999941
PASS image: cosine 0.999959, 0.999966, 0.999952 (min 0.9995); stock bf16 on CPU: 0.999971, 0.999947, 0.999933
PASS audio: cosine 0.999949, 0.999566, 0.999939 (min 0.9995); stock bf16 on CPU: 0.999978, 0.999517, 0.999972
PASS video: cosine 0.999970 (min 0.9995); stock bf16 on CPU: 0.999974
PASS mixed_batch: min cosine 0.999955 over 10 inputs
PASS startup: healthy after 151s
PASS forms: image forms agree True, mixed request 200, dimensions 256 -> 256 floats, token ids 200, image usage 1113 tokens for 4 inputs
PASS url_guard: loopback 400, link-local 400
PASS text_speed: 95,627 tok/s, 348 passages/s (275 tokens each), single query p50 4.9 ms (per call, one card)
PASS media_speed: 29.6 images/s, 17.1 audio clips/s (6 s), 13.7 videos/s (5 frames) (per call, one card)
```

`parity.py` compares the HPU engine with transformers' own `EmbeddingGemma2Model` in fp32 on the CPU (one input at a
time, no padding, the stock `unfold`), on synthetic inputs: filler text up to 3,009 tokens, generated images, tone
glides as audio, five generated video frames. Each line also prints the stock model in bf16 on the CPU against the
same reference: the HPU path sits at that bf16 floor, which on these inputs is as low as 0.9995 (a 35 s tone clip),
so the default bound is 0.9995. The known failure modes are far below it: a language graph captured on an unpadded
batch without the mask fix replays at about 0.962 on padded rows, and HPU SDPA on padded rows gives about 0.6.
`capture_order` captures each graph on an unpadded batch first, which is the order that exposes the first one.

Throughput varies between runs and with the client: the write-up records 145k to 235k tok/s across runs with a
client in another pod; the 95,627 tok/s above had the client in the server's own pod.

## Limits

- The vision encoder runs at about 29 ms per image whatever the batch size, and limits image and video throughput.
- Recompressing an image changes its vector (a JPEG re-saved at PIL's default quality scored 0.917 against the
  original), so send the original bytes when vectors must be compared across clients.
- The URL guard resolves the host before fetching; run the server where it cannot reach private services anyway if
  untrusted clients can send URLs, or set `EG2_ALLOW_URL_FETCH=0`.
