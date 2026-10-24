# Orpheus 3B server

Streaming text-to-speech for Canopy Labs' [Orpheus 3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) on one
Gaudi 2 card. Orpheus is a plain Llama 3.2 3B that writes speech as [SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz)
codec tokens (7 per 85 ms frame), so vLLM runs the language model and a CPU sidecar turns its tokens into audio as
they arrive, behind an OpenAI-compatible `POST /v1/audio/speech`. Design and measurements:
[Fast text-to-speech](../../docs/results/tts-on-gaudi.md#orpheus-3b).

> **Licence and access.** The checkpoint is gated. Accept its terms on Hugging Face with the account whose token you
> use, then download it. The model card says Apache-2.0, but Orpheus is fine-tuned from Llama 3.2 3B Instruct, so the
> [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt)
> applies as well. This directory contains code only. SNAC is MIT.

| | |
| --- | --- |
| Image | one image for both containers, built FROM a stock vllm-gaudi `v0.26.0` image (as in [Getting started](../../docs/getting-started.md)); `install.sh` applies the `v0.26.0` fix profile plus [penalty-shapes](../../fixes/penalty-shapes/README.md) and adds `snac==1.2.1` and `soundfile==0.13.1` |
| vLLM | `serve_llm.sh`: bf16, 8 sequences, 4,096-token context, sync scheduling, every bucket warmed at start (warmup took 130 s) |
| Sidecar | `server.py` on the CPU (6 to 12 cores), port 8000; it gets no card |
| Voices | tara, leah, jess, leo, dan, mia, zac, zoe; OpenAI voice names map onto them (`X-Voice-Mapped-From`) |
| Tags | `<laugh>` `<chuckle>` `<sigh>` `<cough>` `<sniffle>` `<groan>` `<yawn>` `<gasp>` inline in the text |
| Audio | 24 kHz mono, 16-bit |
| Concurrency | 8 streams (`MAX_CONCURRENT` = `--max-num-seqs`); a ninth waits for a slot rather than slowing every stream below real time |

## Files

| File | What it is |
| --- | --- |
| `serve_llm.sh` | the vLLM command and bucket settings, with the reason for each non-default flag |
| `server.py` | the sidecar: sentence chunks (at most 250 characters), one streaming `/v1/completions` call per chunk with token ids, SNAC decoding in a thread pool, the speech formats |
| `speech_stream.py` | the pure parts: the Orpheus prompt, the parser that turns streamed `<custom_token_N>` text into frames, and the frame decoder that sends audio in strides of 2, 2, 4, 4, then 8 frames, each decoded with 2 frames of context per side |
| `install.sh`, `requirements.txt`, `Dockerfile` | the image |
| `tests/test_speech_stream.py` | CPU tests of the parser and the frame decoder (every frame heard exactly once, in order) |
| `tests/hpu/run_gates.sh` | device gates against a running pod (below) |

## Get the checkpoint

The repository also holds about 41 GB of optimizer and FSDP training state. Download only the serving files:

```bash
hf download canopylabs/orpheus-3b-0.1-ft --local-dir /models/orpheus-3b \
  --include "*.json" "model-*.safetensors"
```

## Build and run

```bash
docker build -f servers/orpheus/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> \
  -t <registry>/orpheus-gaudi:<tag> .
```

Run two containers from that image in one pod (or on one host), sharing the network namespace:

| Container | Command | Resources |
| --- | --- | --- |
| `vllm` | `./serve_llm.sh /models/orpheus-3b` | one `habana.ai/gaudi`, hugepages, 8 to 16 cores |
| `tts` | `ORPHEUS_MODEL=/models/orpheus-3b python3 -m uvicorn server:app --host 0.0.0.0 --port 8000` | CPU only, 6 to 12 cores |

Point the service at port 8000. Use the sidecar's `/health` for readiness (it is 503 until vLLM is healthy) and
`/v1/models` for its liveness, so a vLLM restart does not restart the sidecar too.

| Sidecar variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_URL` | `http://127.0.0.1:8001` | the vLLM container |
| `ORPHEUS_MODEL` | `canopylabs/orpheus-3b-0.1-ft` | the tokenizer: the same files vLLM serves |
| `SNAC_MODEL` | `hubertsiuzdak/snac_24khz` | the codec |
| `SERVED_MODEL_NAME` | `orpheus-3b` | the `model` name requests use |
| `MAX_CONCURRENT` | `8` | streams at once; keep it equal to `--max-num-seqs` |
| `DECODE_WORKERS` | `6` | SNAC decode threads (about 20 ms of CPU per 85 ms frame) |

## API

```bash
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "orpheus-3b", "voice": "tara", "input": "Well <sigh> that took longer than I hoped.", "response_format": "pcm"}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

Formats as in [servers/README.md](../README.md#patterns-these-servers-share). Sampling defaults are temperature 0.6,
top-p 0.8 and repetition penalty 1.1; a request may set `temperature`, `top_p` and `repetition_penalty`. Orpheus
needs a repetition penalty of at least 1.1: without one, about a quarter of clips ran past their text. `speed` other
than 1.0 is a 400; `instructions` is ignored (the model has no style prompt). Each chunk is capped at 12 tokens per
character (at least 600), which stops a chunk that never emits end of speech without clipping slow speech.

## Why these vLLM settings

- `--no-async-scheduling`: with vllm-gaudi v0.26.0's async scheduler (the default), sampled ids never reach a
  request's output history, so repetition, presence and frequency penalties do nothing.
- penalty-shapes: with sync scheduling the penalty tensors change shape every decode step and the HPU recompiles (4
  streams ran at 0.08x real time before the fix); the fix pads them to 256-token buckets.
- `--max-num-seqs 8`: with the penalty on, raw decode measured 143 tok/s at 1 stream, 106 tok/s per stream at 8 and
  40 tok/s per stream at 16, where speech needs about 82 (per call, one card); the cause of the 16-wide slowdown is
  not found.
- `VLLM_SKIP_WARMUP=false`: a first-touch graph compile mid-sentence would stall a live stream.

## Verify

```bash
tests/hpu/run_gates.sh --whisper openai/whisper-large-v3
```

Measured on one Gaudi 2 card, Gaudi software 1.24.1, with the vLLM container from a stock vllm-gaudi `v0.26.0` image
after this directory's `install.sh`, and the sidecar and the gate client in a second, card-less container of the same
pod (per call, one card; Whisper large-v3 in fp32 on the CPU):

```text
PASS health: {"status":"ok"}
PASS single: 9.90s of audio in 5.76s = 1.72x real time, first audio 251 ms (per call, one card)
PASS sanity: 9.90s for 34 words, rms 0.031, clipped 0.0000, 80-4000 Hz energy 0.96
PASS concurrent: 8 streams at 1.07x to 1.18x real time each, first audio 758 to 837 ms (per call, one card)
PASS formats: wav 5.29s, mp3 5.80s, flac 5.29s, opus 5.12s, sse 5.89s
PASS hangup: health True, next request 5.72s of audio
PASS whisper: WER 0.000, 0.000, 0.000
7/7 gates passed
```

The gates need at least 1.5x real time for one stream and 1.0x for each of 8 streams. One stream runs well above real
time; 8 streams leave little headroom, which is why `MAX_CONCURRENT` stops at 8. The write-up's 1.91x for one stream
came from a different run; expect some spread between runs.
