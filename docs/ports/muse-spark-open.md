# Muse Spark Open on Gaudi 2

Muse Spark Open is an 816B-parameter mixture-of-experts model (42B active) with a latent MoE in every layer, a
norm on each expert's output, and interleaved sliding-window and global attention. vllm-gaudi has no path for it,
and the stock fused MoE op cannot apply a norm before the expert sum. This port runs the published MXFP4
checkpoint, unconverted, on one Gaudi 2 node and serves production traffic at 256K tokens per request (text only).

**Built with Muse Spark Open.**

!!! summary "At a glance"
    - **Checkpoint:** [amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4),
      Muse Model License (Muse Spark Open version), which asks every product using the model to show
      "Built with Muse Spark Open". Loaded as published, no repack.
    - **Hardware:** 8 Gaudi 2 cards, one node. Attention TP8, MoE EP2 x TP4. 256K tokens per request, 4 sequences;
      needles verified to 1,040,011 tokens.
    - **Stack:** Gaudi software 1.24.1, vllm-gaudi `releases/v0.30.0` (vLLM v0.30.0), lazy mode.
    - **Status:** serving production traffic.
    - **Headline:**
        - Custom MXFP4 decode kernels: 22.3 to 33.5 tok/s at 1 stream, 10.9 to 23.3 tok/s per stream at 4
          (different conditions: same deployment and clients, only image and kernel flag changed; the arms are not
          recorded as running on the same node).
        - Grouped decode launches: 34.5 to 39.7 tok/s at 1 stream (same-pod A/B).
        - Perplexity within 0.4% of an fp32 CPU oracle.

## Why it needed a port

- Upstream vLLM support is a draft pull request, [vllm#61879](https://github.com/vllm-project/vllm/pull/61879),
  with NVIDIA and ROCm paths only. Its platform-neutral files (config, processor, parsers) are reused; the HPU
  model is new.
- `post_expert_norm` applies to each expert's output before the weighted sum. Habana's `mxfp4_fused_weights` op
  returns only the sum.
- vllm-gaudi v0.30.0 gives every layer full-length KV, so the 46 sliding layers (window 2048) store context they
  never read: 31,744 B per token per card.
- vllm-gaudi rejects the AMD Quark format on HPU. A small plugin claims the quantisation method and reads the
  packed E2M1 and E8M0 bytes directly.

## What was built

62 layers, hidden 8192, 128 query and 16 KV heads of size 64, 46 sliding layers with RoPE and 16 global NoPE
layers. Attention is stock vLLM `Attention` (16 q / 2 kv heads per card) plus a runner patch: the window gate for
this model, an exact 2048-key decode window, and a prefill off-by-one fix. The fp32 router (256 experts, top 8),
residual stream, norms and logit softcap follow the reference numerics. Experts run
W4A16, not the W4A4 the checkpoint declares; the CPU oracle uses the same rule. Weights and runtime take 66.0 GiB of
the 94.62 GiB visible per card.

### The latent MoE: A′ with a per-expert norm

Each (token, route) pair becomes its own row: T tokens give T x 8 rows, a `[T*8, 1]` routing table and unit
weights. One call of the native MXFP4 op with `top_k = 1` then returns one row per route instead of the weighted
sum. The layer all-reduces those rows in fp32, applies `post_expert_norm` per row, multiplies by the router
weights and sums in fp32, then unpools 4096 to 8192. The op costs per touched expert slice, not per row, so the
expansion costs about as much as a fused top-8 call up to 8 tokens.

### EP and TP layout

Two expert groups of four cards (EP2 x TP4): each group owns 128 experts, and each card holds a quarter of every
local expert's intermediate dimension. One fp32 all-reduce over all 8 cards sums the partial per-route rows before
the norm (a norm of a partial sum is wrong); non-local rows are exact zeros. Pure EP8 keeps the norm local but lost
33% per stream at batch 1, because each all-reduce waits for the card whose 32 experts drew the most routes.

### Windowed sliding-window KV

The sliding layers get vLLM's `SlidingWindowSpec` through the hybrid KV cache manager, plus a runner patch
(13 runner and 2 worker hooks). Core forms 4 KV groups (sliding 16/15/15, global 16) over one backing allocation,
so a block id costs 1 MiB per card: 8,192 B per token per card, 3.875x more tokens. Core still frees
out-of-window blocks and owns the sliding-window prefix-hit rule and preemption, so prefix caching stays exact.
The model hands each layer its group's slot mapping and window list in its own layer loop.
[MiMo-V2.6-Flash](mimo-v2-6-flash.md) uses the same design.

### Three runtime bugs and their final fixes

1. **FusedSDPA explicit-bias workspace (generic).** A prefill chunk with cached context runs the global layers
   through FusedSDPA with an explicit bias. The call keeps about 4 B x heads x query rows x keys of HBM for the
   life of the process, also when queries are tiled. Around 64K tokens that was 15.5 GiB per card with 13.1 GiB
   free, and logits came back NaN. Fix: past 64K keys, keep a compact key-validity row instead of the dense bias,
   pick the query tile so the workspace stays under 2 GiB (128 rows at 128K, 16 at 1M), build each tile's bias
   rows bit-exactly with `where`, and bound the tiles in flight.
2. **`prompt_logprobs` over a whole chunk.** vllm-gaudi computes fp32 log-softmax over `[N, 202048]` in one
   graph, several 3.3 GB tensors at N = 4096. Fix: 256 rows at a time (max |Δ| 1.9e-6). See
   [prompt-logprobs](../fixes/prompt-logprobs.md).
3. **Host run-ahead in lazy mode (generic).** The host launches a layer's prefill graph in about 7 ms; the device
   needs 20 to 80 ms. So the host queues all 62 layers, each holding its intermediates until it runs, and HBM
   filled while decodes shared the step. Fix: record an HPU event after each layer launch in prompt forwards and
   wait on the event one layer back. A 4096-token chunk then peaked 7.2 GiB below the limit; on one card the
   bounded loop took the same total time as the unbounded one.

## Correctness

Real model at TP8, against a streaming fp32 CPU oracle built from the transformers reference.

| Gate | Result |
| --- | --- |
| Perplexity vs oracle, 5 gated passages (within 1.5%, top-1 at least 97%, KL at most 0.02) | worst +0.38%, top-1 97.56%, KL 0.0013 |
| Decode vs prefill, mean abs Δlogprob, 8 x 512 greedy tokens past the window (gate 0.05) | 0.0019; 0.0010 to 0.0030 on later configurations |
| Needles 2K to 120K x 3 depths, plain-fact wording (a "secret code" wording draws refusals); no-needle control | 15/15; 0/2 |
| Needles at 256K / 512K / 1,040,011 tokens, windowed KV | 3/3 / 3/3 / 1/1 |
| Isolation: 768 outputs in two runs, prompts to 100K, 879 and 68 preemptions | 0 foreign tokens; controls fail with the fixes removed |
| Forced tool choice: required / named / streaming / none / round trip | 10/10, 10/10, 5/5, 5/5, 3/3 |
| Structured outputs, validated by schema, regex or choice | 54/54 |
| MXFP4 decode kernels on | perplexity identical, decode vs prefill 0.0013 |

## Performance

Decode milestones, each measured on its own. Clients differ between rows (greedy 256-token runs or sampled
2K-prompt load), so compare within a row only.

| Step | Before | After | Measured as |
| --- | --- | --- | --- |
| Eager bring-up, 1 stream | | 2.75 to 2.86 tok/s | two eager servers, different nodes |
| Whole-model decode HPU graphs, 1 stream | 2.73 to 2.78 | 14.55 | chained A/Bs on one node, a restart per change: graphs, layout, run-ahead bound with the serving shape, async |
| MoE layout EP8 to EP2 x TP4, 1 stream | 14.55 | 21.1 | same chain |
| Async scheduling, 1 stream | 21.0 | 22.8 | same chain |
| MXFP4 decode kernels, sampled 2K prompts, per stream at 1 / 4 streams | 22.3 / 10.9 | 33.5 / 23.3 | different conditions: same deployment, manifest and clients, image and flag swapped, node not recorded as the same |
| Decode layer grouping, greedy, 1 stream / 4 streams aggregate | 34.5 / 99.8 | 39.7 / 99.7 | same-pod A/B |
| Kernel revision v7 to v8, greedy, 1 stream / 4 streams aggregate | 38.5 / 99.6 | 38.7 / 114.2 | same-node A/B, two test pods back to back |

Long context, one server each:

| Measurement | Result |
| --- | --- |
| 256K needle prompt, end to end (serving shape) | 346 to 461 s |
| 512K needle prompt, end to end | 25 to 31 min |
| 1M needle prompt, end to end (batch use only) | 6,435 s; the prefill alone takes about 105 minutes |
| Decode at 262K / 520K context, 1 stream | 16.7 / 17.0 tok/s |

## What it taught about Gaudi

- **FusedSDPA with an explicit bias retains workspace** proportional to heads x query rows x keys for the life of
  the process, so any chunked long prefill pays it. See [platform findings](../results/platform-findings.md).
- **Lazy-mode host run-ahead** turns a per-layer transient into a whole-model one. Bounding in-flight layers with
  HPU events costs nothing when the device is the bottleneck.
- **Each captured decode HPU graph keeps about 0.5 MiB per block of its bucket** until the process exits; long
  decode buckets need a cap or a smaller KV pool.
- **The stock MXFP4 op unpacks every touched expert slice to bf16**, about 57 µs per slice per card, 83% to 88% of
  busy decode time at batch 4 to 16. See the [MXFP4 MoE kernels](../kernels/mxfp4-moe.md).
- **Host launches bound batch-1 decode once the kernels are fast**: 313 graph launches, 28.7 ms of host time, for
  14 ms of device work. vllm-gaudi cuts a graph after every class named `*DecoderLayer`; skipping that hook in
  decode graphs gave the 34.5 to 39.7 step.
- **vllm-gaudi v0.30.0 sliding-window prefill admits W+1 keys** once a chunk has cached context; this port patches
  it for this model only.
- The duplicated-token defect behind [async-inputs](../fixes/async-inputs.md) was found here, and the
  [long-prefill share](../fixes/vllm-long-prefill-share.md) keeps other users at 74% to 75% of their decode speed
  while a 250K prompt reads in.

## Availability

The integration code (plugin, runner patches, loader) is not published in this repository yet. It depends on model
code outside vllm-gaudi: the platform-neutral Muse Spark files of the upstream vLLM draft pull request. Generic
pieces are published as [fixes](../fixes/index.md) and [MXFP4 MoE kernels](../kernels/mxfp4-moe.md).
