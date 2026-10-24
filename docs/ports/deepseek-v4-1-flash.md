# DeepSeek-V4.1-Flash on Gaudi 2

DeepSeek-V4.1-Flash is a 40-layer MoE model (384 routed experts, top-6) with an unusual stack: compressed sparse
attention with a learned indexer, an encoder-decoder split in which layers 21 to 39 all read one compressed KV cache
written by layer 20, hyper-connections (mHC) normalised by a Sinkhorn loop, and two hashed Engram embedding tables of
about 200 GB. Neither vllm-gaudi nor upstream vLLM has an HPU path for it, so this port is a full vLLM model plugin
for Gaudi 2. It serves production traffic at 128K context, one 8-card node per replica.

!!! summary "At a glance"
    - **Checkpoint:** [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), MIT;
      production serves the weights-only update
      [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT, as a
      lossless re-layout made by the port's conversion script (the original release converts the same way).
    - **Hardware:** 8 Gaudi 2 cards per replica (TP 8 attention, EP 8 experts), 61 GiB of weights per card,
      131,072-token context. Gaudi software 1.24.1, lazy mode, vllm-gaudi `releases/v0.30.0` with vLLM v0.30.0.
    - **Status:** serving production traffic. The DSpark draft model is built and gated, not in production.
    - **Headline:**
        - Decode 1.5 to 15.1 tok/s from per-layer HPU graphs (same-pod A/B), and 23.6 to 35.0 tok/s from decode-step
          work (same-node A/B on one test pod).
        - Prefill 1.35x to 1.55x faster for 16K to 120K prompts (same-node A/B).
        - DSpark: 1.9x to 2.8x per stream at one stream on code, agent, multi-turn and long-context prompts (same
          server, drafts suppressed as the control).

## Why it needed a port

- Upstream vLLM v0.30.0 ships `deepseek_v41` with NVIDIA and AMD paths only: every compute kernel is Triton,
  TileLang, DeepGEMM, FlashMLA or a CUDA op.
- vllm-gaudi `releases/v0.30.0` has no DeepSeek-V4 or V4.1 model, and its runner allocates only full-attention, MLA
  and Mamba cache specs; the sliding-window and circular-buffer specs upstream declares raise "Unknown KV cache spec
  type".
- On HPU the checkpoint's quantization config silently resolves to plain `fp8`. vllm-gaudi's MXFP4 method routes
  non-gpt-oss models with softmax, while V4.1 uses `sqrtsoftplus` with a correction bias.
- V4.1's experts compute `silu(min(g, 10)) * clamp(u, -10, 10)`. The native `silu` MXFP4 kernel has no clamp (57% to
  65% wrong on saturated input in a one-card probe).

## What was built

**Lossless checkpoint re-layout.** DeepSeek's routed experts are already OCP MXFP4 (E2M1, one E8M0 scale per 32
values); AMD's Quark MXFP4 release decodes to the same values, but re-quantizes the shared experts lossily. So the
port starts from DeepSeek's own weights: expert bytes go into vLLM's `gpt_oss_mxfp4` fused layout with gate and up
rows interleaved (-0 canonicalised to +0); FP8 block-scaled linears are dequantized to bf16 offline, which is exact;
the Engram tables keep their bytes, split into 24 prime-sized hash buckets of about 4.1 GB because one 100 GB tensor
exhausts the loader's host memory. Result: 517.68 GB, 0 mismatches over 12 independent validators. The clamp runs
exactly on the stock `bias_mxfp4_fused_weights` op by folding constants (scales +7, alpha 2^-7, limit 10 x 2^7, up
bias -1, 2^-14 in the routing weights).

**The model on HPU.** Every cache (sliding window, 2:1 and 1:1 compressed, index keys) is paged in one KV group with
no per-request state slots. Each row carries a tag (request and block allocation epoch, position), so a read of a row
another request wrote is masked; a count mode counts such reads, and that counter is the isolation gate. Fake FP8/FP4
cache quantization is reproduced bit for bit; Gaudi's fp32 division is not correctly rounded, so FP4 rounding is
decided by exact midpoint comparison. Block tables are sized by the batch's real context: the indexer had been scoring
131,072 entries per query at 128K whatever the prompt length. A whole-model HPU graph never finished compiling
(compile time grows faster than linearly with graph size), so decode replays one graph per 10 layers.

**Decoder replay.** Every decoder layer reads layer 20's 1:1 compressed cache, which depends only on layer 20's input.
A prompt token needs the decoder only for what decode reads later: the last position and the window rows of the last
128 tokens. So layers 0 to 19 and layer 20's compressor run on every prompt token, and the rest runs on the last 140
tokens of each chunk with decoder windows floored there. This is how DeepSeek's own engine,
[DeepInfer](https://github.com/deepseek-ai/DeepInfer), serves the model. Prefix-cache hits stop at least 140 tokens
before the end; `prompt_logprobs` requests take the full stack, which gives a same-server quality A/B.

**MXFP4 decode kernel.** The [MXFP4 MoE kernels](../kernels/mxfp4-moe.md) gained a variant for interleaved rows with
V4.1's exact clamp in the epilogue, for decode calls of up to 16 tokens. One card, real layer weights, one token:
89 µs vs 369 µs for the stock op, relative L2 error vs fp32 1.9e-3 vs 5.9e-3 (per-call, one card).

**Decode step.** A rank-0 step was 12,435 device nodes, with compute busy 18 of 32 ms in a one-card emulation. The
largest item was the mHC Sinkhorn loop (20 iterations on 4x4 matrices, about 42 nodes per call, 80 calls per step),
now one bit-identical TPC kernel. The fp32 mix GEMM moved from the MME (54 µs) to a TPC multiply and sum (8 µs);
power-of-two helpers became bit masks; addressing was hoisted and RoPE fused. The step is now 6,513 nodes.

**Prefill.** The routed MoE is only about 18% of a short 4096-token chunk; the per-query key gather and, from 32K, the
indexer's full stable sort dominate. Three changes: banded window attention plus the compressed part under one
softmax; a sort-free threshold top-k returning the same sets; and the index dot as a bf16 GEMM, bit-identical because
both operands are fake-FP4.

**DSpark.** The checkpoint's own drafter (three stages reading the target's hidden states at layers 37 to 39, up to
5 tokens per call) runs as one HPU graph per batch bucket. The verify sampler keeps the target's distribution, and an
acceptance tracker stops drafting where it cannot pay for the verify step.

## Correctness

The reference is a real-weight fp32 CPU oracle that streams layers.

| Gate | Result |
| --- | --- |
| CPU port vs DeepSeek's reference code, teacher-forced | at most 7e-7 (fp32) |
| Layer parity, one HPU vs the oracle | relative error 1.7e-3 to 5.5e-3 |
| Perplexity vs oracle (1536 tokens) | 1.618 vs 1.6239 (1017 weights, final configuration) |
| Decode vs prefill, teacher-forced (mean \|Δlogprob\| < 0.05, at most 3% off argmax) | pass, 0.017 to 0.019 |
| Needles | 18/18 at 2K to 121K x 3 depths; hit at 19K and 65K in every later configuration |
| Isolation under preemption | own secret 48/48, 0 foreign hits; 0 violating rows in 13.2e9 cache reads |
| DSpark, decode vs prefill argmax | 1.13% of tokens off vs 1.40% without drafts |

## Performance

One stream unless stated. The rows are separate measurements, not a chain.

| Change | Before | After | Measured as |
| --- | --- | --- | --- |
| Per-layer decode HPU graphs vs eager | 1.5 tok/s | 15.1 tok/s | same-pod A/B |
| MXFP4 decode kernel | 16.7 tok/s | 23.3 tok/s | same-node A/B, flag off and on |
| Both custom kernels (`DSV41_MXK` off and on), the published recipe on a cookbook-built image | 20.7 tok/s | 37.6 tok/s | same-pod A/B; all decode-step and prefill changes on in both arms |
| Decode-step work | 23.6 tok/s | 35.0 tok/s | same-node A/B on one test pod; the "after" image also has the prefill changes, which decode never takes |
| Decode-step work, 4 streams aggregate | 67.2 tok/s | 93.7 tok/s | same pod, both with tag counters on |
| Prefill TTFT, 16K / 33K / 64K / 120K | 9.93 / 21.59 / 48.70 / 117.61 s | 7.35 / 14.97 / 32.56 / 75.89 s | same-node A/B, warm shapes, fresh prompts; the "after" image also has the decode changes |
| Decoder replay, ms per 4096-token chunk at context 0 / 64K | 2407 / 4681 | 1357 / 1929 | per-call, one card (TP rank 0 emulated, no collectives) |
| Final configuration, cold TTFT at 16K / 64K / 120K | 9.7 / 48 / 115 s | 4.9 / 21.6 / 43.5 s | different conditions: previous image on production pods vs a canary with replay, prefill and decode changes |

Decoder replay quality on one server, 12 prompts of 2.4K to 100K tokens: mean |Δlogprob| 0.021 against the full
path's own 0.019, and 1.43% vs 1.52% of tokens off the argmax.

DSpark, per-stream decode ratio, greedy, 400 tokens per stream. Measured as: same server, drafts suppressed for the
control.

| Prompts | 1 stream | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| Code edit | 2.76x | 1.95x | 1.71x | 1.30x |
| Agent | 2.10x | 1.66x | 1.39x | 1.21x |
| Multi-turn | 1.87x | 1.98x | 1.54x | 1.22x |
| Long context (56K) | 2.15x | 1.76x | 1.41x | 1.19x |
| Prose | 1.38x | 1.14x | 1.04x | 0.96x |

The control arm runs 3% to 13% below a server without speculation at 1 to 2 streams. Against that production-identical
server (same node, restarted without speculation), essays run 1.28x / 1.17x / 1.02x / 0.97x at 1 / 2 / 4 / 8 streams.

## What it taught about Gaudi

- **Many-small-op decode is launch-bound.** Count device nodes before optimising compute: a 4x4 Sinkhorn loop cost
  about 3,400 nodes per step, and a tiny TPC kernel removed them bit-exactly.
- **HPU graph caches key on the input's view state.** Passing views where warm-up captured fresh tensors re-captured
  every warmed graph in serving, and memory grew until a long request failed to allocate.
- **vllm-gaudi's `all_gather` calls `mark_step`;** inside a captured HPU graph that gave stale values for batches
  above 1. An all-reduce replaced it.
- Two runner bugs that affect every model with prefix caching or preemption were found here and are now in the fixes
  pack:
  [resumed-prefill-logits](../fixes/resumed-prefill-logits.md) and [prompt-order](../fixes/prompt-order.md). More in
  [platform findings](../results/platform-findings.md).

## Availability

The port is published as working code in [`ports/deepseek-v4-1-flash/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/deepseek-v4-1-flash),
with the served configuration as the recipe [DeepSeek-V4.1-Flash-1017](../recipes/deepseek-v4-1-flash-1017.md):

- the model plugin (`dsv41_hpu`), with every speed-up behind a switch that is off by default;
- the conversion script and an independent validator: re-run as published on one MoE layer and on all dense
  tensors, it reproduced the served checkpoint's sha256 for every tensor (1,266 of 1,266);
- one source patch (the kernel-library hook in `vllm_gaudi/__init__.py`) and the Sinkhorn TPC kernel, built into the
  [MXFP4 kernel library](../kernels/mxfp4-moe.md) as variant `dsv41`;
- the startup warm-up, a fail-closed checkpoint check, CPU tests and the device gates.

The generic runner bugs it depends on are fixes from the [fixes pack](../fixes/index.md), applied with the v0.30.0
profile. The DSpark draft model is not included; it is not served.
