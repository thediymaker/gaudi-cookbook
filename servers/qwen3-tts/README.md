# Qwen3-TTS server

Streaming text-to-speech for [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
(Apache-2.0) on one Gaudi 2 card, behind an OpenAI-compatible `POST /v1/audio/speech`. The weights load through the
`qwen-tts` package; generation runs in this server's own loop. Design and measurements:
[Fast text-to-speech](../../docs/results/tts-on-gaudi.md#qwen3-tts).

| | |
| --- | --- |
| Base image | `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` (no vLLM) |
| Packages | `qwen-tts==0.1.1`, `transformers==4.57.6`, `torchaudio==2.11.0` (CPU wheel, no dependencies); all pinned in `requirements*.txt` |
| Voices | the checkpoint's 9 speakers (`GET /v1/audio/voices`); ryan and aiden are native English. OpenAI voice names speak with the first speaker. |
| Languages | `auto` and the checkpoint's 10 languages; `instructions` takes a plain-language style |
| Audio | 24 kHz mono, 16-bit |
| Concurrency | 16 streams advance together in one graph step (`MAX_SLOTS`); up to 64 more wait, then 429 |
| Startup | about 4 minutes (healthy after 216 s and 245 s in two fresh pods): load, graph capture (about 2.2 minutes), a smoke synthesis |

## Files

| File | What it is |
| --- | --- |
| `engine.py` | `FastQwen3TTS`: static KV caches, one captured HPU graph per frame step (talker step, all 15 code-predictor steps, sampling, state update) per (slot bucket 4/16, KV bucket 512 to 4096), prefill graphs per prompt bucket (64 to 1024 tokens), Gumbel-max top-k sampling on the device, a host sync every 4 frames, and the causal codec decoder run in windows with 100 frames of left context as frames arrive |
| `server.py` | the HTTP side: request checks, sentence chunking (at most 400 characters per segment, a short pause between), the per-segment frame cap, health and metrics |
| `install.sh`, `requirements.txt`, `requirements-nodeps.txt`, `Dockerfile` | the image |
| `tests/test_cpu.py` | CPU tests: the Gumbel-max draw against the exact distribution, buckets, the frame cap |
| `tests/hpu/run_gates.sh` | device gates (below) |

## Build and run

```bash
docker build -f servers/qwen3-tts/Dockerfile -t <registry>/qwen3-tts-gaudi:<tag> .
docker run -it --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all --cap-add=sys_nice --ipc=host -p 8000:8000 \
  -e HF_HOME=/hf -v ~/.cache/huggingface:/hf <registry>/qwen3-tts-gaudi:<tag>
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `QWEN_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Hugging Face id or a local directory |
| `SERVED_MODEL_NAME` | `qwen3-tts` | the `model` name requests use |
| `DEFAULT_VOICE` | `vivian` | voice for requests without one |
| `MAX_SLOTS` | `16` | streams generated together; a frame step costs about 10 ms at 1, 4 or 16 |
| `SYNC_EVERY` | `4` | frames per device round between host syncs |
| `TTS_WATCHDOG_SECONDS` | `300` | a device round running longer than this fails `/health` |
| `PT_HPU_RECIPE_CACHE_CONFIG` | unset | for example `/cache/qwen3-tts,True,32768` on node-local disk to shorten restarts |

Set `HF_HUB_OFFLINE=1` once the weights are in the cache.

## API

```bash
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-tts", "input": "Hello from Gaudi.", "voice": "ryan", "response_format": "pcm"}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

`response_format` `pcm` streams; `wav` is one whole file unless `stream_format` is `audio`; `stream: true` sends
server-sent events; mp3, opus, flac and aac come back whole (see [servers/README.md](../README.md#patterns-these-servers-share)).
Extra fields: `language`, `instructions` (at most 1000 characters), `max_new_tokens` (a frame cap per segment,
default 2048). `speed` other than 1.0, an unknown voice or an unknown language is a 400.

## Verify

```bash
tests/hpu/run_gates.sh --whisper openai/whisper-large-v3
```

Measured on one Gaudi 2 card, Gaudi software 1.24.1, in a pod started from the stock base image with this
directory's `install.sh` (per call, one card, warm):

```text
PASS startup: healthy after 245s
PASS single: 13.92s of audio in 1.89s = 7.35x real time, first audio 50 ms (per call, one card)
PASS sanity: 13.92s for 34 words, rms 0.064, clipped 0.0000, 80-4000 Hz energy 0.95
PASS concurrent: 16 streams at 4.60x to 5.12x real time each, first audio 52 to 214 ms (per call, one card)
PASS formats: wav 7.60s, mp3 6.32s, flac 5.28s, opus 6.88s, sse 6.32s
PASS hangup: health True, next request 4.80s of audio
PASS whisper: WER 0.000, 0.000, 0.000
```

The gate thresholds: at least 4x real time and at most 300 ms to first audio for one stream; every one of 16
concurrent streams at least real time; Whisper large-v3 word error rate at most 0.1 per clip.

## Limits

- CustomVoice only: voice design and voice cloning (the other Qwen3-TTS checkpoints) are not wired up.
- Sampling uses the checkpoint's defaults (temperature 0.9, top-k 50, repetition penalty 1.05); requests cannot
  change them. Top-p below 1 in a checkpoint's generation config is ignored, with a warning.
