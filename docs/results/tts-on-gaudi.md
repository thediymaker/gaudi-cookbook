# Fast text-to-speech on Gaudi 2

Three open text-to-speech models stream faster than real time from one Gaudi 2 card. vLLM cannot serve
Qwen3-TTS or Chatterbox, so their Hugging Face style generate loops were rewritten for the HPU. Orpheus 3B is a
plain Llama that emits audio tokens, so it runs on vLLM with a CPU sidecar that decodes audio as tokens
arrive. The patterns apply to any model that ships as a Python `generate` loop.

!!! summary "At a glance"
    - **Checkpoints:** [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
      (Apache 2.0); [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) and
      [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (MIT);
      [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) (Apache 2.0 on the
      card, gated, fine-tuned from Llama 3.2 3B Instruct, so the Llama 3.2 Community License also applies),
      decoded with [SNAC 24 kHz](https://huggingface.co/hubertsiuzdak/snac_24khz) (MIT).
    - **Hardware:** one Gaudi 2 card per model, Gaudi software 1.24.1, lazy mode. Orpheus runs on vllm-gaudi
      v0.26.0.
    - **Status:** all three serve production traffic.
    - **Headline:** Qwen3-TTS 19.8 s of audio in 2.6 s, first audio about 50 ms (per-call, one card, warm).
      Chatterbox Turbo 10x to 13x real time, first audio about 125 ms (per-call, one card, warm).

## The problem

In lazy mode the stock `generate` paths are slow for reasons unrelated to the math. The KV cache is dynamic
or sized from the prompt, so every new text length compiles new graphs (minutes per first request). Qwen3-TTS
runs a nested `generate` for its code predictor on every frame. Sampling ran on the CPU or used boolean
`scatter_` warpers that sync the host every token. Nothing streamed.

For context only (different conditions: stock packages, different texts, not an A/B against the results
below), warm stock Qwen3-TTS took 77 s for 9.5 s of audio, and the earlier Chatterbox server ran at a
real-time factor of 1.3 to 1.8.

## Reusable patterns for generate loops on HPU

1. **Own the loop.** Keep the layers and weights, drop `generate`. Allocate static KV caches in a few fixed
   cache buckets (Qwen3-TTS: 512 to 4096) and pad prompts to fixed buckets (64 to 1024 tokens).
2. **One graph per step, covering everything.** For Qwen3-TTS one captured graph holds the talker step, all
   15 code-predictor steps, sampling and the state update. Capture one per (batch-slot bucket x cache bucket)
   at startup, so no request compiles.
3. **Sample on the device.** Gumbel-max (argmax of log p plus Gumbel noise, equivalently argmax of p / E with
   E drawn from Exp(1)) is an exact categorical draw with no `multinomial`. Express min-p, top-k and top-p as
   thresholds instead of boolean scatters. Per-request noise keeps `seed` meaningful.
4. **Sync rarely.** Read tokens back every 4 steps (Qwen3-TTS) or 8 (Chatterbox). End of speech is seen a
   few steps late and the extra steps are dropped.
5. **Batch requests into slots** that advance in the same graph step. A Qwen3-TTS frame costs about 10 ms at
   1, 4 or 16 slots (per-call, one card).
6. **Lazy-mode hygiene.** Read whole static buffers to the CPU and slice there. Leave no lazy op pending
   between components. Replace host-syncing helpers (the stock Chatterbox chunk-mask helper synced about 14
   times per flow estimator call) and dynamic shapes such as boolean indexing.
7. **Warm up in serving order.** A static KV cache takes its memory layout from the last graph that wrote it,
   so a decode graph replayed after a new prefill bucket recompiles once. Warm up each prefill bucket followed
   by a decode at each slot count.
8. **Split graphs that are too big.** All 10 Chatterbox flow steps as one graph compiled for more than 15
   minutes; one graph per step, replayed 10 times, does not.

## Qwen3-TTS

A talker LM writes the first codebook of each 12.5 Hz frame, a code predictor fills the other 15, and a causal
decoder makes 24 kHz audio. The server applies patterns 1 to 5: up to 16 slots, one graph per frame,
Gumbel-max top-k at the model's defaults (temperature 0.9, top-k 50), a sync every 4 frames. Graph capture
at startup takes about 2.2 minutes.

Audio is decoded in fixed windows as frames arrive. Upstream's 25 frames of left context leave audible clicks
at chunk edges; with 100 frames the streamed audio measures 42 dB SNR against a whole-clip decode. Chunks
grow from 4 to 48 frames.

## Chatterbox and Chatterbox Turbo

The original is a 0.5B Llama "T3" token model plus a 10-step flow decoder with classifier-free guidance. Turbo
pairs a 350M GPT-2 T3 with a distilled decoder, run at 2 steps. Both share the HiFT vocoder and are served from
one card.

- **T3:** own loop with a static KV cache, device sampling, one graph per step, 4 slots per model (graphs for
  1, 2 or 4 active). A decode step costs 3.3 ms (original, with CFG) or 2.3 ms (Turbo) (per-call, one card).
- **Flow:** each chunk runs on the reference prompt, the last 25 tokens spoken, the new tokens and 3
  lookahead tokens, with the mel already sent folded into the prompt so audio continues. Inputs pad to
  25-token buckets; chunks from several requests share a pass.
- **HiFT on the HPU:** its STFT and inverse STFT (n_fft 16, hop 4) become fixed convolutions matching
  `torch.stft` and `torch.istft` to 1e-6. It streams with 16 mel frames of left context, 8 held back and a
  40 ms crossfade. With only 4 held back, seams measured 23 dB SNR against whole-clip vocoding versus 33 dB
  elsewhere; with 8 they match.

The upstream Perth watermark is kept.

## Orpheus 3B

Orpheus writes speech as SNAC codec tokens (7 per 85 ms frame), so vLLM runs the autoregressive part. The
sidecar streams sentence-sized chunks from `/v1/completions` and decodes frames with SNAC on the CPU, each
decode covering the frames sent plus 2 frames of context per side, in strides of 2, 2, 4, 4, then 8. The
reference decoder (a 4-frame window per frame) cost 68 ms of CPU per 85 ms frame and capped one stream at
1.26x real time (per-call, CPU).

Two vllm-gaudi v0.26.0 issues mattered. With async scheduling (the default) sampled ids are not appended to
the output history, so repetition, presence and frequency penalties do nothing; Orpheus needs a repetition
penalty of about 1.1, and without it about a quarter of clips ran past their text. The deployment uses
`--no-async-scheduling`. With sync scheduling the penalty tensors change shape every step and the HPU
recompiles; the [penalty-shapes](../fixes/penalty-shapes.md) fix pads them to buckets.

## Measured results

One Gaudi 2 card, warm, all graphs captured.

| Model | Load | Result | Measured as |
| --- | --- | --- | --- |
| Qwen3-TTS | 1 stream, new text | 19.8 s of audio in 2.6 s (about 8x real time) | per-call, one card |
| Qwen3-TTS | 1 stream | first audio about 0.05 s | per-call, one card |
| Qwen3-TTS | 16 streams | 4.6x to 5.3x real time each | per-call, one card |
| Chatterbox | 1 stream | 5x to 7x real time, first audio about 210 ms | per-call, one card |
| Chatterbox | 4 streams | 2.8x to 3.0x real time each, first audio at most 0.6 s | per-call, one card |
| Chatterbox Turbo | 1 stream | 10x to 13x real time, first audio about 125 ms | per-call, one card |
| Chatterbox Turbo | 4 streams | 5.9x to 7.1x real time each, first audio at most 0.3 s | per-call, one card |
| Orpheus 3B | 1 stream | 1.91x real time, first audio 0.24 s | per-call, one card |
| Orpheus 3B | 4 streams | 1.37x to 1.39x real time, first audio 0.59 s | per-call, one card |
| Orpheus 3B | 8 streams | 1.13x to 1.15x real time, longest gap 0.12 s with a 0.3 s buffer | per-call, one card |

| Model | Quality gate | Result |
| --- | --- | --- |
| Qwen3-TTS | Whisper round trip | 36 of 36 clips word-perfect |
| Chatterbox, Turbo | Whisper large-v3, 12 sentences x 2 seeds | WER 1.5% and 1.4% (misses are number formatting) |
| Chatterbox, Turbo | Perth detector on streamed audio | 1.000 (unwatermarked control 0.006 to 0.03) |
| Orpheus 3B | Whisper large-v3, 8 voices x 4 passages x 2, with emotion tags | median WER 0.00, no clip ran past its text |

## Findings about the platform

**`torch.multinomial` on the HPU samples a distorted distribution.** Against the target distribution its draws
had a total variation distance of about 0.10, where sampling noise alone gives 0.012. Device Gumbel-max
(pattern 3) is exact and also removes the host sync.

**The Habana runtime's SIGPIPE handler turns a client hang-up into a device fatality.** When a listener
disconnects mid-stream, the next socket write raises SIGPIPE, and the handler the runtime installs takes the
device down instead of letting the write fail with `EPIPE`. Any Python service that drives the HPU directly
and writes to sockets or pipes (including a pipe to an encoder subprocess) is exposed. Block the signal before
torch is imported, so every thread inherits the mask:

```python
import signal
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGPIPE})
import torch  # only after the mask is set
```

As a `sitecustomize.py` on `PYTHONPATH`, the two lines cover services you do not want to edit. This was
found in hand-written HPU servers; vLLM's own processes were not tested.

See also [platform findings](platform-findings.md).

## Limits

- Qwen3-TTS serves the CustomVoice checkpoint (9 speakers); voice design and cloning are not wired up.
- Startup is long because every graph is captured up front: about 6 minutes (Qwen3-TTS) and 10 (Chatterbox).
- Orpheus with a repetition penalty falls below real time at 16 streams (40 tok/s per stream against 92
  without, per-call, one card); the cause is not found, so it serves 8 streams per card.

## Availability

The servers are published in [`servers/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/servers) with their images, CPU tests
and device gates:

- [`servers/qwen3-tts/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/servers/qwen3-tts): the Qwen3-TTS engine and server, built on the stock Gaudi
  1.24.1 PyTorch image with the `qwen-tts` package.
- [`servers/chatterbox/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/servers/chatterbox): Chatterbox and Chatterbox Turbo from one card, built on
  the same image with the `chatterbox-tts` package.
- [`servers/orpheus/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/servers/orpheus): the vLLM command for Orpheus and the SNAC sidecar, built on a
  stock vllm-gaudi v0.26.0 image with the v0.26.0 fix profile and [penalty-shapes](../fixes/penalty-shapes.md).
  The checkpoint is gated: accept its licence on Hugging Face first.
- [`servers/common/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/servers/common): the SIGPIPE guard, the fatality latch and watchdog, the shared
  `/v1/audio/speech` formats and the gate script every TTS server uses.

Each server's `tests/hpu/run_gates.sh` measures real-time factor and first audio, checks the audio's length and
spectrum, and with `--whisper` transcribes the clips.
