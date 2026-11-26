# What did not work

Measured dead ends from the work on the [MXFP4 MoE decode kernels](mxfp4-moe.md) and on the rest of the decode step
of large MXFP4 MoE models on Gaudi 2. Each entry gives the idea, what was measured and how, and why it was dropped,
so the next attempt can start from the number instead of the idea.

Unless an entry says otherwise:

- Hardware is Gaudi 2 with Gaudi software 1.24.1 and PyTorch 2.11.0 in lazy mode.
- One-card kernel timings are paired: the candidate, the shipped kernel and an A/A copy of the shipped kernel are
  timed in one process. Each is an HPU graph of 8 chained calls on cold weights (8 copies, more than the 48 MB SRAM),
  30 rounds. A difference counts only above the larger of 1 µs and twice the A/A difference.
- "Projected" gains apply one-card µs per call to recorded GLM-5.3 routing. The setup is the slowest of 4 expert
  groups, 76 MoE calls per step, and 1, 2 and 4 streams weighted 0.4, 0.4 and 0.2. The bars were +1.0% for a new
  kernel and +0.5% for a wrapper change.

That projection overstated end-to-end gains. It overstated them by about 2x at one stream, where the step is partly
host-bound, and by about 25% at 2 to 4 streams. So an idea that only reaches the bar on one card is not worth
building.

## Folding the x permute into w13, on its own

**Idea.** The v8 w13 kernel reads x permuted per 512-k chunk, which costs one host-side permute node per MoE call.
At an assumed 8 to 10 µs launch cost per graph node, removing that node looked like -8 to -10 µs per call.

**Measured.** This was a timing-only bound: the permute node was removed and wrong results were allowed. It ran on
GLM-5.3 shapes of one EP 4 x TP 2 rank (64 local experts, K 6144, I_slice 1024) at 76 points.

- The permute costs 2.1 to 2.4 µs per call at 1 to 2 tokens and 6.2 to 6.7 µs at 3 or more.
- Per call: 150.9 to 148.4 µs at C=1 (+0.74%), 255.0 to 248.9 at C=2 (+1.37%), and 451.0 to 447.7 at C=4 (+0.50%).
- Projected: +0.94% and +1.04% in two runs, with A/A at -0.02% and -0.08%.

**Why dropped.** The mean is +0.99%, right at the bar, and a real fold can only add work. Most of the
assumed launch cost was graph replay and sync that stays. Rewriting the permute itself also stayed under the +0.5%
wrapper bar:

- a 3-D view transpose gave +0.15%;
- an int16 view or an explicit contiguous gave 0;
- an index_select gather was 3 to 14 µs slower.

Later the fold was rebuilt as part of a new w13 kernel (`mxk_w13xn`, natural-order x staged in VLM), and as such it
measured +0.97% and +1.10%. It ships in v10 because it comes with the other w13 changes. On its own it would not
have paid. On DeepSeek-V4.1-Flash shapes the no-permute bound was +0.3% to +0.8%. On Kimi (K 7168) a natural-order
bound was under +1% over the shipped v10 kernel on average, and negative at 16 tokens.

## FP8 experts on Gaudi 2

**Idea.** Store the routed experts as FP8 and use the stock FP8 MoE op instead of the MXFP4 kernels.

**Measured.**

- Setup: one card, the stock `bias_fp8_fused_weights` op against the v10 MXFP4 kernels.
- Shapes: gpt-oss-120b, K = I = 2880, 64 local experts.
- Timing: HPU graphs of 8 chained calls on distinct weight copies, 30 rounds, 4 routing seeds, on a vllm-gaudi v0.26
  image.

| Tokens | v10 MXFP4 µs | stock FP8 µs | FP8 speed vs v10 |
| --- | --- | --- | --- |
| 1 | 83.1 | 295.1 | 0.28x |
| 4 | 262.6 | 589.3 | 0.45x |
| 8 | 462.1 | 878.0 | 0.53x |
| 16 | 826.1 | 1140.7 | 0.72x |

- **Fixed cost.** The stock FP8 op costs 279 µs at 1 token with 0 local routes, against 25 µs for the v10 kernels.
  After that it adds about 20 to 30 µs per distinct expert. The v10 kernels add about 26 µs per route.
- **Other shapes.** At GLM-5.3 shapes FP8 reached 0.19x to 0.66x of v10 (1 to 16 tokens) and was never faster. On
  a 72-layer MXFP4 model (Mistral Large 4 shapes) it was faster only at 16 tokens: 1.19x, 617 against 736 µs.
- **Settings.** None helped. `chunk_size` 256 was already the best, and dropping the extra arguments was 15% to 29%
  slower.
- **Range.** Gaudi 2's e4m3 tops out at **240**, not 448. Codes quantised for a 448 range read back as NaN (11,520 of
  11,520 in the test).
- **Accuracy.** FP8 activations added a per-layer rel_l2 of 0.042 to 0.044, against 1.6e-3 to 1.7e-3 for the MXFP4
  kernels. The weight re-encode itself is exact: a per-channel FP8 copy of the gpt-oss experts matched the MXFP4
  values bit for bit.
- **bf16 op.** The stock bf16 `bias_fused_weights` op hung the card at gpt-oss shapes on every run, alone, at chains
  of 8 and of 1.

**Why dropped.** The fixed cost alone is larger than a whole MXFP4 call at decode sizes. The e4m3 range makes
the format harder to use without any speed to show for it.

## torch.compile mode and the PT2 op path

**Idea.** Compile mode should cut host time per step, and an HPU graph of a compiled model should cut it further.

**Measured.**

- Setup: two cards on a vllm-gaudi v0.30 image. The toy model has 16 layers of TP 8-sized shards with two 12 KB
  all-reduces per layer.
- Method: 6 interleaved runs per arm. The run-to-run spread was 5 to 37 µs per layer. All arms were bit-exact.

| Arm (µs per layer, one row, one step) | Total | Host |
| --- | --- | --- |
| lazy mode + HPU graph | 187.5 | 179.6 |
| compile, default | 166.0 (-11%) | 93.2 (-48%) |
| compile, single-graph flags | 175.7 | 90.5 (-50%) |
| vllm-gaudi regional compile (its default) | 221.6 (+18%) | |

- **Totals.** Pipelined, the totals moved -16% to -18% at one row and -2% to -12% at 4 rows, mostly within the
  run-to-run spread. Device work did not change (35 to 47 µs busy per layer). The rest is all-reduce latency and
  sync gaps, which compile mode does not touch.
- **Graph merging.** The single-graph and no-split flags merged nothing: still 33 recipes per step, one per
  all-reduce plus one.
- **HPU graphs.** These do not work with compile mode: `wrap_in_hpu_graph(torch.compile(m))` fails with "HPUGraph
  class is available in lazy mode only".
- **PT2 op path.** The MXFP4 kernels through the PT2 custom-op path were exact but slow. For 2 layers it took 789 µs
  at 1 token and 2064 µs at 4, against 478 and 477 in lazy mode. With the same kernel ELF, kernel wall time was 1.6x
  to 4.3x longer. The cause was not found.
- **Regional compile.** vllm-gaudi v0.30's regional compile was 9% to 25% slower than lazy mode with HPU graphs.

**Why dropped.** Host time halves, but the step does not get shorter, and the kernels themselves get slower. This
is also why the v10 kernels have no `torch.compile` op library. The v8 `--pt2` library is kept for MiMo-V2.6-Pro,
which serves in compile mode for other reasons.

## Deeper MTP drafts

**Idea.** Propose 2 or 3 draft tokens per step instead of 1, so each MoE call serves more rows.

**Measured.**

- Model and setup: GLM-5.3 on 8 cards, a vllm-gaudi v0.24-era image with the v7 kernels.
- Load: natural chat at production sampling settings.

| Drafts | C=1 tok/s | C=4 tok/s per stream | Acceptance by draft position | Step at C=1 |
| --- | --- | --- | --- | --- |
| 1 | 47.40 | 25.38 | 0.84 | 38.8 ms |
| 2 | 51.05 (1.08x) | 23.67 (0.93x) | 0.83, 0.65 | 48.6 ms |
| 3 | 51.66 (1.09x) | 22.75 (0.90x) | 0.81, 0.60, 0.41 | 54.6 ms |

- **Step cost.** Each extra draft added about 10 ms per step at 1 stream, 14 ms at 2 and 24 ms at 4.
- **Second run.** A second chat run gave 2 drafts 1.05x, 0.94x and 0.90x at 1, 2 and 4 streams.
- **Ceiling.** With acceptance 1.00 (filler text), 2 drafts would give 1.22x and 3 would give 1.41x at one stream.
- **Break-even.** A second draft pays only above about 62% second-position acceptance at 1 stream, and 75% at 4.
- **GLM-5.2.** With the decode-step levers on, 1 draft gave 50.75 tok/s at C=1 against 47.70 for 2 drafts. At C=4 it
  was 27.22 against 27.04.

**Why dropped.** Chat acceptance falls off too fast after the first position. The few percent at one stream cost
7% to 10% of four-stream capacity. More rows per call also share fewer experts than hoped: at 4 streams with one
draft, about 57 of 64 routes already go to distinct experts.

## Data-parallel attention with expert parallelism

**Idea.** Run attention data-parallel and the MoE expert-parallel, so each card does less attention work per step.

**Status.** Desk bound only; no hardware measurement. At 0.2 to 8 streams per replica the attention share per card
is too small to gain from, and the layout keeps two collectives per layer. It was not built. Treat this entry as an
estimate, not a result.

## Reduce-scatter plus all-gather instead of all-reduce

**Idea.** Replace each all-reduce with a reduce-scatter and an all-gather, so the work between them runs on a shard.

**Measured.**

- Setup: 8 cards, TP 8, a vllm-gaudi v0.30 image, lazy mode with lazy collectives.
- Method: in-place collectives inside `wrap_in_hpu_graph`, unprofiled wall clock with an A/A arm.
- Result: one reduce-scatter plus all-gather pair cost **157 to 166 µs**, against about **70 µs** for one all-reduce.
  Both are host-bound, and the pair has twice the launches. Also, `torch.empty` buffers for the pair fail inside a
  graph capture.

**Why dropped.** At decode sizes the collectives are latency-bound, so splitting one into two doubles the cost.

## A captured graph of only custom ops replays nothing

**Symptom.** A timing arm made only of the custom kernels, with the last native op removed, came out at **0.6 µs per
call** with nothing running.

- An HPU graph captured from custom ops alone replays as an empty graph.
- The eager form of the same thing runs once, then replays empty and returns a stale buffer. So a bound arm that
  comes out exact is a bug, not a result.

**What to do.** End every timed arm with one native op shared by all arms, for example `y * 1.0`, and run parity
on that output. `kernels/mxfp4-moe/bench/bench_moe.py --path` does this for the v10 paths. Those paths have no host
permute, so without the multiply they would be graphs of custom ops only. In a model, the surrounding native ops
keep the graph non-empty. The general form is in
[Platform findings](../results/platform-findings.md#an-hpu-graph-of-only-custom-ops-replays-nothing).

## Overlapping the all-reduce with other work in lazy mode

**Idea.** Start the MoE all-reduce and run independent work while it is in flight. A related idea removes the copies
around an in-place all-reduce to cut host launches.

**Measured, overlap.**

- Setup: two cards, lazy mode with lazy collectives.
- Alone, an all-reduce took 40.7 to 41.0 µs and an independent matmul 33.0 to 48.5 µs.
- An all-reduce followed by the independent matmul took **212.1 to 213.8 µs** (12 KB) and 212.4 to 214.0 µs (1 MB).
- With `async_op` and a late wait it took 220.8 to 224.2 µs.
- Followed by a dependent matmul it took 140.5 to 144.4 µs.

Lazy mode serialises the collective with the work after it. A collective also ends a recipe, so each layer costs 4
to 5 host launches.

**Measured, in-place MoE all-reduce.**

- Setup: 8 cards, TP 8, a 72-layer MXFP4 model (Mistral Large 4 shapes), a vllm-gaudi v0.30 image.
- Exactness: greedy outputs were 8/8 identical.
- Host side: launches per 2K step went from 370 to 299 and copy recipes from 142 to 0. Host launch time went from
  about 38.5 to 24.5 ms.
- Speed: device busy time was unchanged at about 20 ms. Unprofiled, one stream, 256 tokens, median of 3 (A/B/A):
  +0.1% at 2K and +0.3% at 16K without speculative decoding, +0.2% and -0.2% with it. The A/A was +0.2% and +0.5%.

**Why dropped.** At one stream, host enqueue is not on the critical path: the device waits on dependent nodes and
collectives, not on the host. The profiled gain was a profiler artifact.

## A block-grouped weight re-pack

**Idea.** Re-pack each expert, losslessly and in place, so a kernel can multiply unscaled values and apply each E8M0
scale once per 32-value block partial sum. The no-scale-multiply bound (timing only) was -9.4%, -12.3% and -15.3% per
MoE call at 1, 2 and 8 local routes. That is +8.2% projected at the serving shapes.

**Measured.**

- Setup: one card, GLM-5.3 layer 40 with 128 local experts, T 2.
- Parity: the prototype w13 kernel matched v8's rel_l2 at every route count.
- Speed: it was **slower**. w13 went 39.6 to 42.6 µs at 1 route, 62.9 to 66.1 at 2 and 202.0 to 209.9 at 8. The MoE
  call at 8 routes went 323.6 to 332.7 µs.
- Cause: the loop has fewer instructions (115 bundles per 8 chunk-rows against v8's 138), but it is bound by MAC
  latency. Two accumulator chains replace four, and two partial chains per row overflow vector local memory.

**Why dropped.** A winning version needs the vector local memory back, either by not staging the x row or by
splitting K per pass. It could at best approach the +8.2% bound. It would also need the prefill unpack and the stock
op to read the new layout, because the experts must not be stored twice. That is the only lever left once v10
used up the exact ones ([why](mxfp4-moe.md#why-the-exact-levers-without-a-re-pack-are-used-up)).

## Smaller kernel levers that did not pay

All are one-card, projected at GLM-5.3 serving shapes unless stated.

| Lever | Result | Why |
| --- | --- | --- |
| Paired horizontal sums (`MXK8_HSUM_X2`) on top of natural-order w13 | +1.02% / +0.96% against +0.97% / +1.10% without | Register allocation shifted. On its own it gave +0.59%, and it flips sign by shape (-2% on w13, +6% on w2 at I_slice 1024) |
| Grouped-route w2 alone (v9) as a switch | +3.83% / +3.86% against v8, but -0.29% to -0.35% against the v10 stack | Wins only at 4 or more repeated experts in the slowest group |
| First w13 tail-carry form (flush test as an OR of compares) | +0.45% | Under the bar; the one-branch flush replaced it |
| Merging the tuned nch-2 w2 loop with the general loop | w2 pass 238 to 246 bundles, -0.7% | One kernel per ELF keeps each one's schedule |
| Packed w2 tree without the carry at nch 6 (DeepSeek-V4.1-Flash) | -0.6% to -1.3% | The carry is what pays, not the packing |
| Exact w2 reworks that keep more live state (register block sums, per-token partials) | +1.9 µs per route from spills | Spills outweigh the saved work |
| Cheaper exact nibble lookups (3-op split, upper-half and byte-1 switches) | Inf in the output | Not exact |
| Removing the ids cast node | +0.28 / +0.43 µs, in noise | A -1 µs bound everywhere is only +0.24% |
| Expert weights sliced over TP 8 (Mistral Large 4 shapes) | Inexact (8 partial sums per expert). Projected -2.4 ms per step at 1 stream but +0.38 ms at 8; the kernel refuses I 768 | Not exact, and loses where throughput matters |
| Expert placement balancing | Routing skew is near-uniform: same-group pairs 1.454 against 1.465 for random. The best order simulated -0.27 ms, server within ±1% | No hot experts to move |
| 8, 16 or 72 layers per HPU graph (8 cards) | within 0.2% | Graph size is not the cost |
| TPC knobs (data cache prefetch, predicated commands, sync) | 0.0% each; turning the instruction cache prefetch off was 0.5% to 12% slower | No headroom there |
