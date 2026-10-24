# EmbeddingGemma 2 on Gaudi 2 without vLLM

EmbeddingGemma 2 is a 740M multimodal embedding model: text, images, audio and video, alone or mixed, into
one 768-dimension vector. vllm-gaudi v0.30.0 cannot serve it, because the model uses bidirectional attention
and the HPU attention path in that release is causal only. A small custom server built on transformers runs it
on one Gaudi 2 card instead, with HPU graphs in lazy mode. It matches a CPU fp32 reference to a cosine of at
least 0.9995 for every modality, as close as the stock model in bf16 on the CPU gets, and embeds text at roughly
95k to 235k tokens per second per card, depending on where the client runs.

!!! summary "At a glance"
    - **Checkpoint:** [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), Apache 2.0,
      not gated. 740M parameters in BF16 (text 271M, vision 170M, audio 300M), 768-d output with Matryoshka
      sizes 128, 256 and 512, 8192 tokens per input.
    - **Hardware:** one Gaudi 2 card per replica, Gaudi software 1.24.1, lazy mode, Habana's PyTorch 2.11
      container with transformers 5.20.2 and PyAV 12.1.0.
    - **Status:** serving production traffic.
    - **Headline:** cosine at least 0.9995 against a CPU fp32 reference, every modality, the same floor as the
      stock model in bf16 on the CPU. Text about 95k to 235k tok/s (per-call, one card). Flickr30k 1k caption to image R@1 86.5.

## Why vLLM could not serve it

- **Model support.** Upstream vLLM added the model after the vLLM commit that vllm-gaudi v0.30.0 tracks
  ([vllm#60254](https://github.com/vllm-project/vllm/pull/60254)), and that support needs transformers 5.19
  or newer.
- **Attention.** EmbeddingGemma 2 is bidirectional, with a 513-token two-sided sliding window on the text
  model. vllm-gaudi's HPU attention backend accepts `ENCODER_ONLY` layers, but its prompt path calls
  `prompt_attention(..., is_causal=True)` regardless. A vLLM port therefore needs an HPU attention change, not
  only a model file. Without that change every token would attend only to its left, and the vectors would
  be wrong.

## What the custom server does

The server exposes the same `/v1/embeddings` request and response shape as vLLM's embedding models (`input`,
`dimensions`, `encoding_format`, `truncate_prompt_tokens`), plus content parts for images, audio and video and
vLLM-style `messages`. Each request is padded into a small fixed set of shapes, and every shape is compiled at
startup (about 145 s cold), so no request triggers a graph compile.

| Path | Input shape | Buckets |
| --- | --- | --- |
| Language model | token ids `[B, L]`, media soft tokens `[B, L, 512]`, mask | L 64 to 8192, at most 8192 tokens per forward |
| Vision (image) | patches `[n, 2520, 768]` | n 1 to 16 |
| Vision (video frame) | patches `[n, 1260, 768]` | n 1 to 32 |
| Audio | mel `[n, T, 128]`, mask | T 10 s to 328 s |

An image costs up to 280 tokens depending on aspect ratio, audio 25 tokens per second, and video 140 tokens per
frame at 1 frame per second (up to 32 frames, sampled by the processor's own frame sampler). Media tokens are
assembled on the CPU and passed to the language model as soft tokens.

### Three numeric traps

**HPU SDPA is wrong on padded rows: use eager attention.** With `attn_implementation="sdpa"`, padded rows came
back with a cosine of about 0.6 against the reference, and the 4k and 8k buckets produced NaN with the text
model's sliding-window mask. Eager attention over the same padded batches is correct. Loading the model through
sentence-transformers on HPU inherits the same problem.

**The mask-skip trap.** By default, transformers drops the attention mask when `attention_mask.all()` is
true, so a batch with no padding runs with no mask at all. That is a Python branch on tensor contents. In lazy
mode with HPU graphs, a graph captured on an unpadded batch has no mask in it, and replaying it later on a
padded batch silently ignores the padding (cosine 0.962 against the reference). The server builds every mask
with `allow_is_bidirectional_skip=False`, so the captured graph always applies the mask. The startup warm-up
order happened to hide this for text; a test that captures the unpadded graph first reproduces it.

**Keep the trailing EOS on truncation**, as sentence-transformers does, so truncated inputs match the
reference pipeline.

### Multimodal support

Two parts of the stock multimodal code do not suit HPU graphs:

- The vision model strips padding with a boolean index, which produces a dynamic shape and a new graph for
  every image size. The server keeps the padded `[n, 280]` output and its mask instead.
- The audio encoder's attention uses `Tensor.unfold`, which has no HPU kernel and falls back to the CPU on
  every call. The server replaces it with an index gather that builds the same tensor.

Media is fetched from `data:` URIs, base64, or public http(s) URLs (private, loopback and link-local addresses
are refused). Audio in any format PyAV decodes is downmixed to mono and resampled to 16 kHz.

## Measured results

### Correctness

| Gate | Result |
| --- | --- |
| Cosine vs CPU fp32 reference (processor plus stock model, itself identical to sentence-transformers): text up to 8192 tokens, images of two aspect ratios, two images plus text, audio of 6 s and 35 s, text plus audio, video | at least 0.9995 for every case (lowest 0.99957, a 35 s audio clip; the stock model in bf16 on the CPU scores 0.99952 on the same clip) |
| Same inputs in mixed batches vs one at a time | same vectors |
| Flickr30k 1k test, caption to image | R@1 86.5, R@5 97.5 |
| Flickr30k 1k test, image to caption | R@1 95.8, R@5 99.8 |
| LibriSpeech speech to transcript, 73 clips (`hf-internal-testing/librispeech_asr_dummy`), both directions | R@1 100 (a sanity check, not a benchmark) |

### Throughput, one Gaudi 2 card

| Load | Throughput | Single request | Measured as |
| --- | --- | --- | --- |
| Text, 240-token passages, client in another pod | about 145k to 235k tok/s across runs | 5 ms per query | per-call, one card |
| Text, 275-token passages, client in the server's pod (the published gate) | 95,627 tok/s, 348 passages/s | 4.9 ms p50 per query | per-call, one card |
| Text, 240-token passages, one benchmark run | 148,900 tok/s, 613 passages/s | 5.1 ms p50 per query | per-call, one card |
| Images, 640x480 JPEG | about 35 to 50 images/s | 52 ms | per-call, one card |
| Audio, 5.9 s clips | about 25 clips/s | not measured | per-call, one card |
| Video, 5 frames | about 20 videos/s | not measured | per-call, one card |

Media throughput is limited by the encoders, not by preprocessing. The vision encoder takes about 29 ms per
image whatever the batch size, roughly 10% of what its FLOPs allow; switching the vision tower to SDPA does not
make it faster. CPU preprocessing is about 20 ms per image and 7 ms per audio clip.

## Limits

- The vision encoder is the bottleneck for image and video load and has not been profiled yet.
- Recompressing an image changes its vector: a JPEG re-saved at PIL's default quality scored 0.917 against the
  original. Send the original bytes when vectors must be compared across clients.
- Serving it on vLLM would need upstream model support in a vllm-gaudi release and a bidirectional (and
  sliding-window) prompt attention path in the HPU backend.

## Availability

The server is published in [`servers/embeddinggemma-2/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/servers/embeddinggemma-2): the engine, the
API server, media handling, an image built on the stock Gaudi 1.24.1 PyTorch image with transformers 5.20.2, CPU
tests, and device gates (`tests/hpu/run_gates.sh`) that check cosine against the stock model in fp32 on the CPU and
measure throughput. It depends on transformers' EmbeddingGemma 2 model code, not on vllm-gaudi.
