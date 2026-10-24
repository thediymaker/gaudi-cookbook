# Servers

Model servers that do not run on vllm-gaudi: hand-written HPU loops for models whose stock code is a Python
`generate` loop or an encoder vLLM cannot serve, and one vLLM sidecar. Each one is a small FastAPI service with an
OpenAI-compatible API, its own image built from a stock base, and device gates. The write-ups with the design and
the measurements are [Fast text-to-speech](../docs/results/tts-on-gaudi.md) and
[EmbeddingGemma 2 without vLLM](../docs/results/embeddinggemma-2.md).

| Server | Model | API | Base image | Card |
| --- | --- | --- | --- | --- |
| [`qwen3-tts/`](qwen3-tts/README.md) | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) (Apache-2.0) | `/v1/audio/speech`, streaming | Gaudi 1.24.1 `pytorch-installer-2.11.0` | 1 |
| [`chatterbox/`](chatterbox/README.md) | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) and [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (MIT) | `/v1/audio/speech`, streaming | Gaudi 1.24.1 `pytorch-installer-2.11.0` | 1 (both models) |
| [`orpheus/`](orpheus/README.md) | [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) (gated; Apache-2.0 card, Llama 3.2 terms) | `/v1/audio/speech`, streaming | stock vllm-gaudi `v0.26.0` | 1 (vLLM) + a CPU sidecar |
| [`embeddinggemma-2/`](embeddinggemma-2/README.md) | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (Apache-2.0) | `/v1/embeddings`, text, image, audio, video | Gaudi 1.24.1 `pytorch-installer-2.11.0` | 1 |

The base image is `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0`; the
vllm-gaudi image is built as in [Getting started](../docs/getting-started.md). Everything was run on Gaudi 2,
Gaudi software 1.24.1, lazy mode.

## Layout

```text
servers/
  common/                  shared by the servers below
    sigpipe_guard.py       blocks SIGPIPE before torch loads (see "SIGPIPE" below)
    hpu_service.py         fatality latch, device watchdog, probe log filter, EPIPE self-check
    speech_api.py          the /v1/audio/speech formats, text splitting and metrics for the TTS servers
    tts_gates.py           device gates for any of the TTS servers (speed, first audio, sanity, Whisper)
    install_lib.sh         base-image checks used by every install.sh
    gate_lib.sh            starts a server for its gates
  <server>/
    README.md              what it serves, build, run, API, verify
    install.sh             the only build step; runs inside the base image and fails closed
    Dockerfile             FROM the base image; copies servers/ to /opt/gaudi-cookbook and runs install.sh
    requirements*.txt      every Python dependency, pinned
    server.py, ...         the server
    tests/                 CPU tests (pytest; skip cleanly without torch or the web stack)
    tests/hpu/run_gates.sh device gates: one PASS or FAIL line per gate, exit 0 only if all pass
```

## Build and run

From the repository root, for example:

```bash
docker build -f servers/qwen3-tts/Dockerfile -t <registry>/qwen3-tts-gaudi:<tag> .
docker run -it --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all --cap-add=sys_nice --ipc=host -p 8000:8000 \
  -e HF_HOME=/hf -v ~/.cache/huggingface:/hf <registry>/qwen3-tts-gaudi:<tag>
```

On Kubernetes, use the pod settings in [Pod spec](../docs/kubernetes/pod-spec.md) (`runtimeClassName: habana`,
`hostIPC: true`, hugepages, one `habana.ai/gaudi`) with `/health` as the startup, readiness and liveness probe. Every
server captures its graphs before `/health` turns 200, so give the startup probe the budget its README states.
Requests that arrive during startup get a 503.

Without Docker, inside the base image: run the server's `install.sh`, then
`PYTHONPATH=servers/common:$PYTHONPATH python3 servers/<server>/server.py` (Orpheus: see its README).

## Verify

```bash
python3 -m pytest servers                                    # CPU tests
servers/<server>/tests/hpu/run_gates.sh [--whisper openai/whisper-large-v3]   # on one card, in the image
```

`run_gates.sh` starts the server on the card (or uses `--url` for one already running) and prints one line per
gate. Every timing it prints is per call, from a client next to the server, on one card. The Whisper gate
transcribes the clips on the CPU and needs the Whisper checkpoint.

## Patterns these servers share

**SIGPIPE.** The Habana runtime installs a native SIGPIPE handler that treats the signal as a device fatality. A
listener that hangs up mid-stream, or an encoder subprocess that exits early, would end the HPU context. Every
server that drives the card imports `sigpipe_guard` first, which blocks the signal in the main thread before torch
loads, so every thread inherits the mask and the write fails with `EPIPE` instead. At startup the servers provoke an
`EPIPE` and check that the card still computes (`EPIPE self-check passed` in the log). For a service you do not
want to edit, the same two lines in a `sitecustomize.py` on `PYTHONPATH` do the job.

**Fail health on a dead card.** A Synapse fatality is terminal for the process, and a wedged card raises nothing.
`hpu_service.Health` latches the first fatality (narrow markers, so a bad request never takes a replica out), and a
watchdog latches when one unit of device work runs past a deadline (`TTS_WATCHDOG_SECONDS`, default 300 s; one
step takes well under a second once warm). `/health` then returns 503 so the orchestrator restarts the process.

**No compile on the request path.** Every shape a request can produce is drawn from a fixed set of buckets and
compiled (captured as an HPU graph) at startup. The TTS servers also run a few real requests through the serving
loop before reporting healthy.

**Streaming.** `pcm` streams as it is made, `wav` streams with `stream_format: "audio"` (sizes `0xFFFFFFFF`),
`stream: true` or `stream_format: "sse"` sends OpenAI `speech.audio.delta` and `speech.audio.done` events; mp3, opus,
flac and aac come back whole. The response waits for the first audio before it sends a status, so a request that
fails early gets a real error. A listener who hangs up frees its slot at the next scheduling round.

## Licences

The code here is Apache-2.0. The models are not covered by it: accept each model's licence on Hugging Face. Orpheus
is gated and fine-tuned from Llama 3.2 3B Instruct, so the Llama 3.2 Community License applies too. The servers
install the upstream model packages (`qwen-tts`, Apache-2.0; `chatterbox-tts` and `resemble-perth`, MIT; `snac`,
MIT; transformers, Apache-2.0) from PyPI; no upstream code is copied into this directory. Where a module
re-implements an upstream forward pass or loop for the HPU, its docstring names the source and its licence.
