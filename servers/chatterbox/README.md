# Chatterbox server

Streaming text-to-speech for Resemble AI's [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) (0.5B Llama T3
plus a 10-step flow decoder with classifier-free guidance) and
[Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (350M GPT-2 T3 plus a distilled decoder, run
at 2 steps), both MIT, served from one Gaudi 2 card behind an OpenAI-compatible `POST /v1/audio/speech`. The
checkpoints load through the `chatterbox-tts` package; T3, the flow and the HiFT vocoder run in this server's own
loops on the HPU. Design and measurements: [Fast text-to-speech](../../docs/results/tts-on-gaudi.md#chatterbox-and-chatterbox-turbo).

| | |
| --- | --- |
| Base image | `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` (no vLLM) |
| Packages | `chatterbox-tts==0.1.7` and its model packages without their dependencies (they pin torch 2.6), `transformers==5.2.0`; all pinned in `requirements*.txt` |
| Models | `chatterbox` (the default when `model` is absent) and `chatterbox-turbo` |
| Voices | `default`, plus every clip in `VOICES_DIR`; OpenAI voice names speak with `default` |
| Audio | 24 kHz mono, 16-bit, with Resemble's Perth watermark as upstream applies it |
| Concurrency | 4 requests per model decode together (`MAX_CONCURRENT`); up to 32 more wait, then 429 |
| Startup | about 7 minutes for both models with the weights on a network share (healthy after 428 s), including a self-test of 1, 2 and 4 requests per model through the serving loop |

## Files

| File | What it is |
| --- | --- |
| `t3.py` | `T3Engine`: batched T3 decoding on a static KV cache, one HPU graph per step for 1, 2 or 4 active slots, prefill per 64-position bucket, sampling on the device without boolean scatters (min-p, top-k and top-p as thresholds, the draw as an exponential race with per-request noise, so `seed` works), tokens read back every 8 steps |
| `s3gen.py` | `FlowRunner` (the flow per 25-token bucket and batch size, the encoder and one ODE step captured, chunks that continue the audio already sent), `HiftVocoder` (HiFT on the HPU with the STFTs as convolutions), `HiftStreamer` (left context, held-back frames, crossfade) |
| `model.py` | `FastChatterbox`: loads a checkpoint, voices, warm-up of every graph a request can reach, and `Utterance`, one text chunk from tokens to audio |
| `server.py` | the device worker that owns the card (admit, one batched T3 block per model, one batched flow pass, most urgent listener first) and the HTTP side |
| `install.sh`, `requirements.txt`, `requirements-nodeps.txt`, `Dockerfile` | the image |
| `tests/test_cpu.py` | CPU tests: the convolution STFT against `torch.stft`/`torch.istft`, top-p and top-k thresholds against their definitions, the exponential race as a draw, the streamer emitting every sample once |
| `tests/hpu/run_gates.sh` | device gates for both models (below) |

## Build and run

```bash
docker build -f servers/chatterbox/Dockerfile -t <registry>/chatterbox-gaudi:<tag> .
docker run -it --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all --cap-add=sys_nice --ipc=host -p 8000:8000 \
  -e HF_HOME=/hf -v ~/.cache/huggingface:/hf <registry>/chatterbox-gaudi:<tag>
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHATTERBOX_VARIANTS` | `turbo,original` | which models to load |
| `CHATTERBOX_DEFAULT_MODEL` | `chatterbox` | the model for requests without `model` |
| `MAX_CONCURRENT` | `4` | requests per model decoding together (each takes a T3 slot) |
| `MAX_QUEUE` | `32` | waiting requests before 429 |
| `MAX_INPUT_CHARS` | `4096` | longest `input` |
| `VOICES_DIR` | unset | reference clips (wav, mp3, flac or ogg, 5 to 10 s of clean speech) enrolled at startup under their file names; each new prompt length adds flow graphs to the warm-up |
| `WATERMARK` | `1` | `0` turns the Perth watermark off |
| `TORCH_THREADS` | `8` | CPU threads (text processing, voice enrollment, the watermark) |
| `TTS_WATCHDOG_SECONDS` | `300` | a device step running longer than this fails `/health` |
| `PT_HPU_RECIPE_CACHE_CONFIG` | unset | node-local recipe cache to shorten restarts |

## API

```bash
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "chatterbox-turbo", "input": "Hello from Gaudi. [chuckle]", "response_format": "pcm"}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

Formats as in [servers/README.md](../README.md#patterns-these-servers-share). Chatterbox's own fields:
`exaggeration` (0 to 2, emotion intensity), `cfg_weight` (0 to 1), `temperature`, `seed`. Turbo ignores
`exaggeration` and `cfg_weight` (it has no emotion input and was distilled without guidance) but reads
paralinguistic tags in the text, such as `[laugh]`, `[chuckle]`, `[sigh]` or `[whispering]` (the full list is in the
checkpoint's `added_tokens.json`). Long input is split into sentence chunks of at most 300 characters, spoken back to
back with a 0.2 s pause (0.35 s at paragraph breaks).

## Verify

```bash
tests/hpu/run_gates.sh --whisper openai/whisper-large-v3
```

Measured on one Gaudi 2 card, Gaudi software 1.24.1, in a pod started from the stock base image with this
directory's `install.sh` (per call, one card, warm):

```text
PASS startup: healthy after 428s
== chatterbox-turbo
PASS single: 9.52s of audio in 0.80s = 11.91x real time, first audio 104 ms (per call, one card)
PASS sanity: 9.52s for 34 words, rms 0.038, clipped 0.0000, 80-4000 Hz energy 0.99
PASS concurrent: 4 streams at 5.75x to 6.11x real time each, first audio 195 to 300 ms (per call, one card)
PASS formats: wav 4.32s, mp3 4.48s, flac 4.52s, opus 3.96s, sse 4.40s
PASS hangup: health True, next request 4.20s of audio
PASS whisper: WER 0.000, 0.000, 0.000
== chatterbox
PASS single: 8.48s of audio in 1.38s = 6.13x real time, first audio 197 ms (per call, one card)
PASS sanity: 8.48s for 34 words, rms 0.126, clipped 0.0000, 80-4000 Hz energy 0.98
PASS concurrent: 4 streams at 2.56x to 2.77x real time each, first audio 338 to 601 ms (per call, one card)
PASS formats: wav 3.92s, mp3 3.84s, flac 3.76s, opus 3.72s, sse 3.88s
PASS hangup: health True, next request 4.00s of audio
PASS whisper: WER 0.000, 0.000, 0.000
```

The gate thresholds: one stream at least 5x real time and at most 300 ms to first audio for Turbo (3x and 500 ms
for the original); every one of 4 concurrent streams at least real time; Whisper large-v3 word error rate at most
0.1 per clip. The Perth watermark detector was not part of these gates.

Device steps slower than one second are logged as `slow device step ... (graph compiles +N)`; after the self-test
there should be none.

## Limits

- The original model's flow is compute-bound (a batch of 4 costs about 3.9 times a batch of 1), so it tops out near
  11x real time in total per card.
- English only: the multilingual checkpoint is not wired up.
