# Mistral Large 4 on Gaudi 2

Mistral Large 4 is a 1T-parameter mixture-of-experts model (52B active) with MLA attention, a DSA-style sparse key
indexer, cross-layer KV sharing and a speculative draft head (DSpark) in the checkpoint. vllm-gaudi has no path for
it, and its experts (I 6144) are twice as wide as the MXFP4 kernels allow. This port runs AMD's published MXFP4
checkpoint, unconverted, on one Gaudi 2 node at 128K tokens per request with DSpark drafting (text only).

!!! summary "At a glance"
    - **Checkpoint:** [amd/Mistral-Large-4-1T-A52B-Quark-MXFP4](https://huggingface.co/amd/Mistral-Large-4-1T-A52B-Quark-MXFP4),
      Mistral Ecosystem Open-Weight License. Loaded as published, no repack.
    - **Hardware:** 8 Gaudi 2 cards, one node. Attention TP 8, experts EP 8 as 256 virtual experts. 128K tokens per
      request, 8 sequences, a 243,200-token int8 KV pool.
    - **Stack:** Gaudi software 1.24.1, vllm-gaudi `releases/v0.30.0` (vLLM v0.30.0), lazy mode.
    - **Status:** serving production traffic.
    - **Headline:**
        - Decode 1.7 to 21.9 tok/s at one stream with decode HPU graphs (same server), then 39.6 with DSpark,
          expert placement and the grouped-route kernel (10 prompts, one at a time).
        - Decode-step TPC kernels: 57.1 to 38.8 ms per plain decode step at 4.3K context (chained same-server A/Bs,
          tokens identical in every arm).
        - Perplexity against an fp32 CPU reference of the same checkpoint: ratio 0.998, top-1 97.4%, KL 0.019;
          MMLU-Pro 84.5% on 1,500 questions with AMD's protocol, against AMD's 85.0%.

## Why it needed a port

- Upstream vLLM support is [vllm#64534](https://github.com/vllm-project/vllm/pull/64534). Off NVIDIA and AMD GPUs
  it raises `NotImplementedError`: the sparse MLA, the indexer and its speculative decoding are CUDA and ROCm kernels.
  Its platform-neutral frontend files (params.json adapter, tokenizer, renderer, parser) are reused.
- vllm-gaudi v0.30.0 has no model class, no MLA KV format the indexer can use, no Quark claim, and no DSpark path
  (DSpark needs vLLM's model runner V2).
- The image's mistral-common 1.11.7 has no `low` and `medium` reasoning efforts; the model card asks for 1.13.0.
- The cookbook's MXFP4 kernels cap the expert width at I 3072 with K at most 6144. Mistral's experts are I 6144,
  K 6144.
- Gaudi 2's FP8 e4m3 tops out at 240, not 448, so the base's FP8 non-expert linears are converted to BF16 at load.

## What was built

72 layers, hidden 6144, 64 heads, vocabulary 204,800. Attention is MLA (KV LoRA 512, nope 64 plus rope 64, no query
LoRA) with a sparse indexer (32 heads of 128, top-2048 keys): 44 layers have their own indexer and 28 reuse the
selection of the layer before. Nine layers read layer 35's latent cache, so 63 layers own a cache. Every fourth
layer has no RoPE. Queries are scaled by a learned length term (SSMax), RoPE frequencies are learned (LeRoPE) and
logits are soft-capped at 30. The MoE has 128 experts (top-4, softmax) and one shared FP8 expert per layer; layer 0
is dense. The plugin's model, loader and kernels were checked op by op against a pure-torch reference (`ref/` in the
port) before any end-to-end run.

### Virtual experts and placement

Each I 6144 expert is re-sliced at load into two experts of I 3072: 256 virtual experts, top-8. The halves of one
expert are summed by the MoE like any two routes, so the result is the same function. No second copy is stored;
the loader reads each half straight from the published shards. Placement B puts the two halves of every expert on
different ranks, which spreads each token's work across more cards: the 1-stream verify step went from 79 to 67.6 ms
and the plain step from 46 to 42 ms (same-server runs).

### The `ml4` kernel build

The `ml4` variant of the [MXFP4 MoE kernels](../kernels/mxfp4-moe.md) is the v8 kernel set (decode up to 16 tokens
and the prefill unpack) plus grouped-route kernels for 2 to 64 tokens, which unpack each local expert once per unit
of up to 2 routes instead of once per route. Units stop at 2 routes because a third does not fit Gaudi 2's 16 KB of
vector local memory. The grouped path gives the v8 output bit for bit (parity in this repository: 10/10 cases from
1 to 64 tokens on the checkpoint's layer 5). On one card it ran 1.16x to 1.29x faster than v8 at 4 to 64 tokens;
in the server, verify went from 67.6 to 64.4 ms and the 8-stream plain aggregate from 87.8 to 94.7 tok/s (same-server
runs). Against the stock MXFP4 op the kernels are 4x to 6x faster per call at 2 to 16 tokens with real weights
(one card), and closer to fp32 (rel_l2 1.6e-3 against 3.0e-3 to 4.0e-3).

### Decode graphs and decode-step kernels

Eager decode was host-bound at about 640 ms per step. Capturing decode as 9 HPU graphs of 8 layers took one stream
from 1.69 to 21.9 tok/s (same server; graph output equal to eager). After that the step was mostly graph nodes:
each dependent node costs about 1 µs, and about 12 ms of a 57 ms step was gaps between nodes. The port's model-only
TPC kernels (`kernels/k1/`, 21 kernels) remove nodes without changing a bit:

| Arm | What it replaces | Effect | Measured as |
| --- | --- | --- | --- |
| K1, fused decode indexer | paged gather, scores, ReLU weights and radix top-k (7 kernels) | 47 µs per layer at 4K up to 212 µs at 128K; 252/252 index sets exact | one card, A/A |
| K2, sparse attention gather | int8 latent dequant and gather in registers | bitwise; 15 µs per layer at 1 row, 55 at 4 | one card |
| K1 + K2 | | plain step -6.3% at 2.3K to -14.4% at 118K | same server, 2 runs each |
| Indexer numerics rewrite and paged-row addressing | integer and bit helpers in fewer nodes | -2.4 to -2.6 ms per step, exact | chained same-server A/Bs |
| Node-cut kernels (RMS tail, RoPE, fake-FP8, SSMax, latent pack, router) | 1 to 10 native nodes each | -5.3 ms, then -4.4 ms once RoPE and the latent pack copied the compiler's fma form, then -1.8 ms for the router | chained same-server A/Bs, median of 3, token md5 identical 24/24 |
| All of them | | plain step 57.10 to 38.77 ms (17.5 to 25.8 tok/s at one stream, 4.3K context); forced-draft verify 82.1 to 59.7 ms | the same chain |

### DSpark speculative decoding

The checkpoint's draft head (3 MLA layers with a 2048-token window, a 5-layer tap and a rank-512 recurrent head)
proposes 3 tokens per step. The port drives it from the n-gram slot of vLLM's speculative config, runs verify rows
inside the decode graphs, and uses an exact rejection sampler. Verify runs as `[request, 4]` rows, so the indexer
scores the cache once per request instead of once per draft: at a 128K table the plain step went from 86 to 68 ms
and verify from 149 to 97 ms (same server, greedy identical 8/8). An acceptance tracker stops drafting when it does
not pay. Greedy, 10 prompts one at a time: 0.49 acceptance (2.48 tokens per step), 39.6 tok/s against 23.8 plain
(24.9 on prose, 53.2 on a math chat). Grammar requests (tools, JSON) also draft: drafts are cut at the first token
the grammar rejects.

### KV: int8 latent, prefix caching, offload

| Piece | What it does | Evidence |
| --- | --- | --- |
| Int8 latent codec `i8-t128-f32` | int8 in four tiles of 128 with fp32 absmax scales, 656 B per token per layer, plus byte indexer keys (lossless) | KL 0.0191 against the fp32 reference (bf16 KV 0.0154, bar 0.02); needles 30/30 to 129K; MMLU-Pro 300 paired questions 82.3% against 81.0% and 81.7% with bf16 KV (McNemar p 0.52); decode -6% at 2K |
| Pool | 1,900 blocks of 128 = 243,200 tokens | two device-memory fixes made room: no value cache (vllm-gaudi allocated one per layer, 6.4 GiB per card) and prompt graphs writing into caller buffers |
| Prefix caching | cache rows tagged by block allocation, so a hit is valid and stale rows are masked | turn 2 of a 32K conversation 37 s to 1.5 to 1.8 s first token; 110K about 300 s to 5 to 16 s |
| Offload (vLLM `OffloadingConnector`) | evicted blocks kept in 160 GB of host memory | a 128K conversation evicted from HBM reloads in 5 to 9 s instead of a 157 s prefill, follow-up output identical; decode 44.0 / 21.6 / 13.1 against 44.3 / 21.3 / 13.3 tok/s at 1 / 4 / 8 streams without it (two servers at once, noise) |

### Prefill

Short prompts run in per-layer HPU graphs for 64 to 768-row chunks, with the MoE kernel inside the 64-row graph:
50-token time to first token from 736 to 279 ms, and the longest decode stall when a request arrives from 1.0 to
0.5 s (same server). Long prompts split the indexer's mask rows over the TP ranks (exact), use 3/4-power-of-two key
widths, and run prompt attention through bf16 FusedSDPA: cold time to first token 32K 37.4 to 24 s, 110K 373 to
195 s, 128K about 395 to 153 s (same server, cumulative arms, each run twice; decode unchanged). A host sync after
each prompt chunk cut host memory per 128K prompt from +162.7 GiB to +3.3 GiB at a 1% cost in time to first token.

A start-up warm-up module sends every prompt width, decode shape and follow-up turn once before the server reports
ready: on a fresh server a 20K prompt's first token took 69 s cold and 17 s after the warm-up.

## Correctness

The reference is the port's own fp32 CPU implementation of the model on the same checkpoint, exact against the
Hugging Face reference on a tiny model (1e-7 to 1e-5). A bf16 reference was not usable: two bf16 runs differed by
up to 21% in perplexity on prose (router near-ties drift through 72 layers), while fp32 runs agree to 5e-5.

| Gate | Result |
| --- | --- |
| Perplexity, 4 passages x 4,096 tokens (story, code, a technical document, Flatland), about 1,023 scored rows each, against the fp32 reference | served configuration (int8 KV): ratio 0.9977, top-1 97.39%, KL 0.0191. Bars from the reference's own spread: ratio within 7.2%, top-1 at least 95.9%, KL at most 0.067 |
| Perplexity with FusedSDPA prompt attention (bf16 reference for ratio and top-1, fp32 for KL) | 1.033 / 97.0% / 0.037 against 1.006 / 97.6% / 0.021 for the fp32 attention path on the same server; code and the technical document stay at KL 0.002 or less |
| Controls | stock routing (no virtual split) gives perplexity x3,300 to x7,600; turning KV sharing off gives needles 0/15 |
| MMLU-Pro, AMD's protocol (5-shot chat, effort medium, temperature 1.0, top-p 0.95, 16K tokens), 1,500 stratified questions, bf16 KV | 84.47% ± 0.93, against AMD's 85.04% on all 12,032 (AMD's FP8 base: 85.40%) |
| Needles | 30/30 to 129K with int8 KV; neutral wording 20/20 to 64K (an earlier wording read as an injection and drew refusals) |
| Decode against prefill, teacher-forced | mean abs logprob difference 0.034 (bar 0.05) |
| Determinism | decode graphs equal eager; two servers 8/8 identical; same configuration md5 24/24 across restarts; every shipped kernel keeps tokens identical |
| Isolation under forced preemption | 12/12 requests got their own answers, 0 foreign, 0 errors |

## Performance

Decode milestones at one stream. Each row is its own measurement; compare within a row.

| Step | Before | After | Measured as |
| --- | --- | --- | --- |
| Decode HPU graphs (9 x 8 layers) | 1.69 tok/s | 21.9 tok/s | same server |
| DSpark | 21.9 | 33.9 (mean of 10 prompts) | same server |
| Placement B and the grouped-route kernel | 33.9 | 39.6 | same server, 10 prompts |
| MoE and attention decode changes (router padding, decode attention as batched matmuls, MoE decode chain) | 39.6 | 45.4 DSpark chat; 25.6 plain | benchmark, 512-token chat prompts |
| Decode-step kernels, plain step at 4.3K | 57.1 ms | 38.8 ms | chained same-server A/Bs |

Served configuration, DSpark chat, per stream at 1 / 4 / 8 streams: 44.9 / 22.0 / 13.6 tok/s (aggregate 45 / 95 /
115), measured through the production service before the decode-step kernels. After them, a 600-token probe at one
stream gave 53.3 tok/s on the production deployment; 4 and 8 streams were not re-measured there. The run of this
repository's recipe is on the [recipe page](../recipes/mistral-large-4.md#performance).

Plain decode slows with context because the indexer turns on past 2048 tokens: 44.6 ms per step at 1.1K, 56.4 at
2.3K and 73.7 at 118K (one stream, before the decode-step kernels, same server).

## What it taught about Gaudi

- **Graph nodes, not math, bound a one-stream decode step.** Each dependent node costs about 1.0 to 1.3 µs; 12 ms
  of a 57 ms step were gaps. Exact TPC kernels that remove nodes were the main lever once the MoE was fast.
- **The graph compiler rewrites arithmetic.** It fuses RoPE rotations into fma below 8192 pairs, drops bf16 round
  trips that feed fp32 consumers, and divides as `x * reciprocal`. A kernel that must be bit-exact has to copy those
  forms, and must be re-gated after a compiler change.
- **Gaudi 2's e4m3 range is 240.** FP8 latent KV and FP8 or int8 non-expert linears failed the perplexity bar; int8
  with an absmax scale per 128 values passed at the same byte size.
- **A `.view()` to another dtype is a copy on HPU, not an alias.** vllm-gaudi's KV offload views every cache as
  int8, so for this model's bf16-declared caches it stored a snapshot and loaded into a copy the model never read.
  The port keeps same-dtype views and moves bits with `index_select` and `index_put_`.
- **The MoE all-reduce waits for the busiest expert rank**, 4 to 11 ms per step. Placement of expert halves helped;
  hot-expert replication did not pay for its per-layer choice.
- **A lazy-mode out-of-memory error reads "tensor_data is empty".** The allocator's live peak always reads near its
  limit, so judge headroom with a hold test (3 GiB held per card), not the peak counter.
- **The host can queue a whole long prompt of lazy launches ahead of the device** and hold +160 GiB of host memory
  for it; a sync per prompt chunk bounds it.
- **vllm-gaudi allocates a value cache per layer unless the model is marked MLA**: 6.4 GiB per card wasted here.
- **`reset_prefix_cache` does not clear the CPU offload tier**, so a "cold" rerun can still hit host cache.
- **A bf16 reference is not a reference for a 72-layer MoE.** Use fp32.

More are in [platform findings](../results/platform-findings.md).

## What did not work

| Idea | Result |
| --- | --- |
| FP8 (e4m3) latent KV | KL 0.0231, over the 0.02 bar. Replaced by the int8 codec |
| FP8 or int8 non-expert linears (for 256K memory) | KL 0.0221 and 0.0244, decode -6% to -11% |
| Fitted expert order (placement F) | no speed gain, prose KL 0.027 |
| Hot-expert replication (copies of the busiest experts on extra ranks) | +2 to +4 GiB per rank; the per-layer choice costs 1.3 to 1.6 ms, plain decode a wash, 4 copies failed KL (0.040) |
| Four-way split (I 1536) | 25% more kernel work; slower |
| A native I 6144 kernel | not worth it over the virtual split |
| Async scheduling with DSpark | -4.1 to -4.4 ms per verify step at one stream and exact there, but not bit-equal to synchronous output at 4 and 8 streams; not shipped |
| 256K context | works with context parallelism over the 8 cards (pool 589,824 tokens, 256K first token 1,284 s, needles 15/15 at 128K, 192K and 259K), but short-context decode lost 17% to 22% and short requests waited about 33 minutes behind a 236K prefill. Without it, 160K ran out of device memory |

## Availability

The port is published in [`ports/mistral-large-4`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/mistral-large-4):
the plugin, the reference model, the decode-step kernels, a Dockerfile and `install.sh` that build it from the stock
`releases/v0.30.0` image, CPU tests and the device gates. The served configuration is the recipe
[Mistral Large 4 (MXFP4)](../recipes/mistral-large-4.md). The frontend files of
[vllm#64534](https://github.com/vllm-project/vllm/pull/64534) are vendored as the stock v0.30.0 files plus the
pull request's hunks, under their Apache-2.0 headers. The MXFP4 MoE kernels are built from
[`kernels/mxfp4-moe`](../kernels/mxfp4-moe.md) (`--variant ml4`). Generic pieces are separate
[fixes](../fixes/index.md).

The vision tower (built and parity-tested at bf16 cosine 0.946 against an fp32 tower, not served), the 256K
context-parallel layout and the CPU reference runs that produced the perplexity bars are not published.
