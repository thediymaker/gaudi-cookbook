# Beam-1.0-501B-A23B on Gaudi 2

Beam-1.0 is a 501B-parameter MoE model (23B active) with 52 layers: 3 dense and 49 MoE layers of 256 experts, top-8
with group-limited routing, and grouped-query attention in which 43 layers use a 512-token sliding window with RoPE
and 9 use full attention without positional encoding (NoPE) plus a length-dependent query scale. The checkpoint ships
compressed-tensors FP8 (per-channel weights, dynamic per-token activations), a scheme vllm-gaudi already serves, but
there is no model code for the architecture on HPU. This port adds it as a vLLM plugin, with a faster loader. It serves
production traffic at 256K context on one 8-card node per replica.

!!! summary "At a glance"
    - **Checkpoint:** [reflection/Beam-1.0-501B-A23B-FP8](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8),
      Apache-2.0, served as published (no repack).
    - **Hardware:** 8 Gaudi 2 cards per replica: TP 8 attention, EP 8 experts (32 per card), 262,144-token context,
      16 sequences. Gaudi software 1.24.1, lazy mode, vllm-gaudi `releases/v0.30.0` with vLLM v0.30.0.
    - **Status:** serving production traffic.
    - **Headline:**
        - Weight load 2177 s to 356 s with a bit-identical model state (same-node A/B, real checkpoint, TP 8).
        - Decode 20.6 to 30.5 tok/s at batch 1 (chained A/Bs on one pod, steps below).
        - Copy and needle tests pass at 250K to 255K tokens.

## Why it needed a port

- Upstream vLLM added Beam in [vllm#62074](https://github.com/vllm-project/vllm/pull/62074) with an NVIDIA path only
  (Triton, CuTe DSL and a CUDA all-reduce). vllm-gaudi `releases/v0.30.0` has no Beam model.
- The Beam reasoning and tool parsers ([vllm#62052](https://github.com/vllm-project/vllm/pull/62052)) sit on a
  response-template framework and response grammar that are not in vLLM v0.30.0.
- The stock loader took 25 to 51 minutes per start at TP 8.

## What was built

**Model code.** Attention uses vllm-gaudi's stock HPU attention with per-layer sliding windows; the Beam pieces
(per-head affine QK norm, RoPE or the NoPE query scale, a sigmoid output gate) run in torch around it. The MoE is the
stock fused MoE with sigmoid grouped top-k, correction bias and a 2.5 routed scale, on the stock compressed-tensors FP8
per-channel method. Two numerical details matter:

- **The router runs in fp32.** The correction bias is close to a constant of about 28 across experts, so bf16 loses
  the differences that pick the experts.
- **bf16 rounding points.** The reference rounds to bf16 at many steps (sandwich norms, depth-scaled residuals,
  gate). The HPU graph compiler removes bf16 round trips it sees as redundant, so an exact mode rounds with integer
  ops. Production runs with it off: it makes every recipe bigger, and the gates pass without it (see Correctness).

**Frontend.** The parsers and the always-on response grammar are backported to vLLM v0.30.0. Tool-call arguments
stream as fragments instead of arriving when the call closes, which had left long tool calls silent for minutes. The
grammar bitmask is applied on the device: vllm-gaudi copies the logits to the host to mask them, a sync that ended
every grammar step. The device mask is bit-exact against xgrammar's CPU kernel.

**The `beam` load format.** The stock path was CPU- and dispatch-bound, not storage-bound. Per rank it read all 503 GB,
ran vllm-gaudi's Gaudi 2 FP8 rescale (`x 0.5`) on every full CPU tensor before slicing, and copied 76,267 tensors one
by one with a device sync after each. The opt-in `--load-format beam`:

- reads only the expert rows a rank keeps (under EP, only its own experts);
- applies the identical rescale to the kept region, in 16 reader threads, after checking it against the stock wrapper
  on all 256 FP8 bit patterns;
- assembles each MoE layer's weight and scale stacks on the host and copies each to the device once: 196 synced
  copies per rank instead of 75,264.

It needs no converted copy and no extra disk. The load is now bound by reads.

**EP 8 and decode HPU graphs.** Eager decode ran at 0.68 tok/s, host-bound in lazy `mark_step` calls; decode HPU
graphs are required. Under graphs the step is still host-bound on graph launches: lazy collectives cut each layer into
7 cached graphs, 367 launches per step at about 105 µs of host time each, about 40 ms, while the device is busy 11 ms at
batch 1 and 19 ms at batch 4. Expert parallelism wins here because the per-card MoE op takes 32 experts (128 tensor
inputs) instead of 256 sliced experts (1,024 inputs), a much cheaper launch, and group-limited routing (4 of 8 groups,
one group per card) keeps the device imbalance small. Because a step costs about the same at batch 1 and 4, 16
sequences are nearly free per step.

**256K.** Only 9 of 52 layers keep full-length KV, so the pool holds 776,448 tokens per replica, about 3 times a
256K request.

## Correctness

The reference is an fp32 CPU oracle that streams real layers, plus a CPU W8A8 variant that models per-token FP8
activations. The "floor" is the W8A8 variant scored against the fp32 oracle.

| Gate | Result |
| --- | --- |
| CPU reference vs Hugging Face `modeling_beam.py`, tiny config | bit-exact |
| Perplexity vs W8A8 oracle, 3 passages (257 to 1699 tokens) | within -0.35% to +0.69% (TP), -1.25% to +0.16% (EP); top-1 agreement and KL at the floor |
| Production numerics (EP 8, exact rounding off) | pass; one sub-gate at its edge: chat passage top-1 94.53% vs a 94.48% gate |
| Decode vs prefill, mean \|Δlogprob\| (gate 0.05) | 0.030 (TP), 0.039 (EP), 0.035 (production) |
| Needles | 24/24 at 2K to 126K x 3 depths; 3/3 at 250K; no-needle control 0/2 |
| Copy test (64 tokens reproduced exactly) | 64/64 at 32K, 65K, 130K and 255K |
| Isolation under preemption | 0 foreign content in 84 outputs with 60 preemptions; own code 84/84 |
| [beam-verifier](https://github.com/reflection-oss/beam-verifier) API suite | 133/133 (defaults and temperature 0) |
| Weight loader | model state hash identical on 51,516 tensors x 8 ranks; prompt logprobs identical |

Outputs are not bit-identical across batch compositions or cache hits: per-token FP8 activations turn kernel-shape
differences into flips at near-ties (top-1/top-2 gaps of 0.05 to 0.58 nats). No foreign content appeared in any run.

**EP vs TP quality.** An early comparison suggested EP 8 cost quality on long generations. A controlled study found
no real gap: run the same day on the same internal benchmark, production's EP 8 configuration and the same
configuration without EP scored within run-to-run noise of each other. The one numerical difference is inside the
fused FP8 MoE op, which quantizes each expert's down-projection input per token over the whole 2560-wide intermediate
under EP (as a CUDA W8A8 reference does) and over a 320-wide slice under TP. Per layer that leaves EP outputs a median
5% further from the oracle, but against the fp32 oracle on long generations EP and TP agree equally under exact
rounding. A flag that splits each expert into slices to match TP's numerics was built; it changed neither oracle
agreement nor benchmark scores and cost 41% to 64% of 16-stream throughput, so production stays on EP 8.

## Performance

| Step | 1 stream | Measured as |
| --- | --- | --- |
| Eager decode vs decode HPU graphs, TP experts, 4 sequences | 0.68 to 22.7 tok/s | same pod, two server starts |
| Baseline: TP experts, 4 sequences | 20.6 tok/s | chained A/Bs on one pod, one change per step |
| + grammar mask on the device | 20.8 tok/s | |
| + EP 8 experts | 25.6 tok/s | |
| + 16 sequences, bounded bucket file | 26.0 tok/s | |
| + exact rounding off (production) | 30.5 tok/s | |

On the same chain, chat with 2K-token prompts gives 68.5 tok/s aggregate at 4 streams for the baseline and 100.7 for
the final step, and the final step reaches 286.4 tok/s aggregate at 16 streams. TTFT for a 2K prompt went from 1.69 s
to 0.52 s.

| Load path | Loading weights | Measured as |
| --- | --- | --- |
| Stock (`auto`) | 2177 s | same-node A/B, real checkpoint, TP 8; both read about 500 GB over NFS |
| `--load-format beam` | 356 s | |

At 256K (same image, production configuration): a 255K-token copy test takes about 108 s end to end; 15 decodes next to
a 250K prompt peak at 86.0 of 96 GiB per card.

## What it taught about Gaudi

- **Lazy collectives fragment decode graphs.** Each all-reduce cuts the layer into separate cached graphs, so a
  model with cheap layers is bound by host launch time; fewer tensor inputs per MoE launch (EP) helped more than any
  device-side change.
- **The FP8 MoE op through the overload packet** costs a tensor `repr()` per layer per step, which syncs the lazy
  graph: see [moe-explicit-overload](../fixes/moe-explicit-overload.md).
- **The graph compiler drops bf16 round trips,** so a reference's rounding points need integer-op rounding if they
  must be kept.
- **Slow loads are often host-bound.** Check per-rank CPU work and synced copies before blaming storage; vllm-gaudi's
  full-tensor FP8 rescale and per-tensor syncs dominated here.
- **Batch-invariant outputs are not available** on this stack with per-token FP8 activations.

## Availability

The integration code is not published in this repository yet. It depends on model code outside vllm-gaudi: a model
plugin with the loader, frontend backports and runner patches.
