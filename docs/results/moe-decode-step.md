# Anatomy of a large-MoE decode step

Where does one decode step of a large expert-parallel MoE go on 8 Gaudi 2 cards, and which parts can be moved?
This page is a method for answering that on any EP MoE, worked through on GLM-5.3 (MXFP4 experts, TP8 + EP8, 256
routed experts, top-8, MTP with one draft token). Tracing all eight ranks overturned the working assumption: the
expert kernels were not 12.6 ms of the step but 19.5 ms, because each layer waits for the rank with the most
local routes. That wait turned out to be statistics, not a hot rank, and a different expert layout recovered
part of it.

!!! summary "At a glance"
    - **Checkpoint:** [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) (GLM-5.3 License),
      converted to MXFP4 experts. The converted checkpoint is not public; the conversion is reproducible from the
      public one (see [GLM-5 with MXFP4 experts](../ports/glm-5-mxfp4.md)).
    - **Hardware:** one node of 8 Gaudi 2 cards, TP8 + EP8, vllm-gaudi v0.24, Gaudi software 1.24.1, lazy mode,
      custom MXFP4 decode kernels not in this repository (two versions, v7 and v8).
    - **Status:** study; the changes below were gated for production.
    - **Headline:** router fast path, device-side MTP rejection and early draft: 51.64 to 57.78 tok/s at one
      stream, greedy output bit-identical (same-node A/B). EP4 x TP2 expert layout: +11% (v7 kernels) and +5% (v8
      kernels) at one stream, memory-neutral (same-node A/Bs).

## The method

1. **Trace every rank.** A torch.profiler CPU+HPU trace of a few decode-only steps on all 8 ranks, with named
   ranges for the runner's phases, once graphs are warm. Rank 0 alone hides the straggler.
2. **Build the critical path per collective:** the slowest rank's compute before each all-reduce, plus the
   collective's latency (last rank ready to first rank resumed).
3. **Split each MoE layer** into expert-kernel time per rank and the rest. Slowest minus mean rank, summed over
   layers, is the EP imbalance cost.
4. **Overlay host and device timelines** to find where the device waits for the host.
5. **Count** graph launches, nodes and recipes per step: launch-bound, node-bound or wait-bound?

## Where the time goes

One stream, greedy, about 2.8K context, step period 39.0 to 39.4 ms (8-rank trace, one complete step):

| Item | ms per step | How measured |
| --- | --- | --- |
| Expert kernels, slowest EP rank per layer | 19.5 | max over ranks per layer, summed; mean rank 9.6 |
| Attention (MLA decode) | 6.9 | same on every rank |
| MoE outside the kernels (gate, shared expert, router, combine), dense MLPs | 3.8 | slowest rank |
| All-reduce latency, 162 per step, about 20 µs each (24 KB) | 3.4 | last rank ready to first resumed |
| Host work with the device idle (step boundary, draft hand-off) | about 5.5 | host against device timeline |

At four streams the slowest-rank expert time is 50.7 ms against a mean of 30.7.

- **A lazy-mode collective ends a recipe,** so each layer replays as two recipes and two all-reduces: 322 graph
  launches per step at about 74 µs of host time each.
- **Not node-bound.** Compute engines are busy 15.4 ms of a 37.4 ms device span; the idle time is waiting at
  all-reduces, not gaps between nodes.
- **The host runs ahead during the forward** and is on the critical path only at the step boundary.

## Is the straggler a hot rank?

Traces give time per rank, not expert ids. To answer it:

1. **Record routing:** top-k ids per layer for every decode step over a realistic mix (code, chat, agent turns,
   math, seven languages, long documents) at default sampling. The recorder ran in eager mode; routing depends on
   hidden states, not graphs.
2. **Measure kernel cost per layout** on one card with real weights: time per call for T tokens and r local
   routes.
3. **Simulate:** per step and layer, count each rank's local routes, look up the cost, take the max over ranks,
   sum. Fit placements on the first 60% of steps, evaluate on the last 40%.
4. **Validate against the traces:** 18.44 ms simulated at one request against 17.9 to 19.5 traced.

The recording gave about 9.3K four-request decode steps, cut into 43.8K one-request, 20.7K two-request and 10.5K
three-request sub-steps. Simulated kernel critical path at one request (v7 kernels):

| Layout or routing | ms per step |
| --- | --- |
| EP8, contiguous experts (as served) | 18.44 |
| EP8, random expert placement | 18.43 |
| EP8, uniform independent routing | 17.37 |
| EP8 with a static placement (out of sample) | 17.33 |
| EP4 x TP2 with a static placement | 15.26 |
| Mean rank (no imbalance) | 10.11 |

- **No hot rank.** A random placement matches the contiguous one. Per expert, routing is skewed (the hottest
  expert of a layer gets 4.1x the mean), but over 32 experts the busiest rank gets only 14% more routes than the
  mean over a run.
- **Balls in bins.** One request puts 16 routes per layer (verify and draft token, top-8 each) into 8 ranks: the
  busiest gets 4.45 against a mean of 2.0. Two thirds of the gap would remain with uniform, independent routing,
  and no placement can remove that part.
- **The MTP pair overlaps.** Verify and draft tokens share 2.4 of their 8 experts (0.25 if independent), putting
  two routes on one rank. That is worth about 1.0 ms at one request, about what a static placement recovers.

## Changing the layout: EP4 x TP2

Fewer, wider bins beat a better placement. In EP4 x TP2, four groups of two ranks each hold 64 experts, each rank
half of every local expert's intermediate dimension. A route costs about half on two ranks, and 16 routes fall
into 4 bins (busiest 6.3 against a mean of 3.95). Per route, the v7 kernel costs about 47 µs at full width, 25 at
half and 17.7 at quarter (one card): each route pays a fixed part (staging the input row, loading scales) that does
not shrink. So EP2 x TP4 wins only at one request on v7 and loses to EP4 x TP2 elsewhere.

| Quiet test pods, one node | EP8 | EP4 x TP2 + placement | Measured as |
| --- | --- | --- | --- |
| Expert kernels, slowest rank, one request | 18.4 to 19.1 ms | 14.0 to 14.4 ms | same-node A/B, 8-rank traces |
| Step period, one stream | 35.4 to 36.3 ms | 31.3 to 31.4 ms | same-node A/B, with trace hooks |
| Decode, one stream, 2K greedy, v7 kernels | 57.13 tok/s | 63.58 tok/s (+11%) | same-node A/B, with trace hooks |
| Decode, one stream, 2K greedy, v8 kernels | 66.41 tok/s | 69.58 tok/s (+5%) | same-node A/B |
| Weights per rank | 52.55 GB | 52.55 GB | load log |

Faster kernels leave less imbalance to win back, so the gain halves on v8. TTFT is unchanged. Partial sums are
added in a different order, so output is not bit-identical; quality gates held: GSM8K 49/50 with the same answer
on 50/50, needles at 32K and 64K, top-1 agreement 0.988 at shared positions of greedy runs, the tool gate
unchanged, a 30-minute long-prefill stress 60/60.

One trap: selecting a placement's experts with a host `index_select` on the memory-mapped checkpoint page-faults
the file expert by expert, and loads took 1.4 to 1.9 times as long as stock. Reading each rank's contiguous share
first, then gathering from the page cache, brought it back within run-to-run spread.

## Shortening the step boundary

Three bit-identical changes took the one-stream step from 39.0 to 39.4 ms to 35.3 to 35.8 ms.

- **Router fast path (about 1.7 ms).** With `n_group = 1` and `topk_group = 1`, the grouped top-k's group stage
  selects every expert, yet costs about 25 µs per MoE layer on HPU, including a 12.9 µs DMA broadcast for a mask
  `expand`. When `0 < num_expert_group <= topk_group`, go straight to the biased top-k.
- **Early draft plus device-side rejection (about 2 ms).** Stock ends a step with: target forward, rejection
  sampler, copy sampled tokens to the host, build draft row indices on the host, enqueue the MTP draft. The device
  idles and the 1.5 ms draft runs exposed. Computing the indices on the device (last accepted row is
  `2i + accepted - 1`) lets the draft follow the sampler directly, but gains nothing alone: the HPU rejection
  sampler copies its inputs to the host and walks requests in Python. Moving that walk to the device with the same
  random draws (the host's float64 uniform rounded up to the next float32, so the float32 comparison equals the
  float64 one) removes the sync. A check mode found 0 mismatches against the stock walk in 7,000 steps per rank.

| Gate | Baseline | With the changes | Measured as |
| --- | --- | --- | --- |
| Decode, one stream, 2K greedy | 51.64 tok/s | 57.78 tok/s | same-node A/B, quiet pods |
| Decode, one stream, temperature 1.0, top-p 0.95 | 43.71 tok/s | 48.13 tok/s | same-node A/B |
| Greedy, 16 prompts x 192 tokens | | 3026/3026 tokens identical | same node |
| MTP acceptance | 0.915 | 0.913 | same node |

The 57.78 run also carried a fourth bit-identical change (a faster HPU-graph input walk) that did not move the
step period; the three changes alone gave 57.35 on another node.

## Applying it to another EP MoE

- Trace all ranks first; the mean rank is not the critical path.
- Record routing and simulate before building a placement. If a random placement matches yours, there is no hot
  rank to fix.
- Measure your expert kernel's per-route fixed cost at each slice width; it decides whether EP x TP pays.
- Hunt host syncs at the step boundary (sampler copies, draft indices).
- Measured but left alone: two full-vocabulary top-p sorts cost about 1 ms per step, and vllm-gaudi v0.24 has no
  async scheduling with speculative decoding.

## Availability

The integration code is not published in this repository yet. It depends on the GLM model code, the custom MXFP4
decode kernels and vllm-gaudi v0.24 internals, all outside the releases this cookbook covers. The MXFP4 kernel in
this repository is for gpt-oss: see [MXFP4 MoE kernel](../kernels/mxfp4-moe.md).
