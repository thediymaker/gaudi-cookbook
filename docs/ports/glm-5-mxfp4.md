# GLM-5.2 and GLM-5.3 (744B) in MXFP4 on Gaudi 2

GLM-5.2 and GLM-5.3 share one architecture: 78 layers, hidden size 6144, MLA attention with DeepSeek sparse
attention (DSA, top 2,048 keys), and 256 routed experts plus one shared expert, top-8. At about 744B parameters,
the FP8 release (704 GiB) cannot fit eight 96 GB cards, so MXFP4 experts (about 422 GB, about 53 GB per card) are
the only single-node option on Gaudi 2. Neither model ran on vllm-gaudi's native MXFP4 MoE op out of the box: the
op had only been used with gpt-oss. This port loads both on one 8-card node, fixes long context, adds speculative
decoding and replaces the stock MoE decode path with a custom TPC kernel.

!!! summary "At a glance"
    - **Checkpoints:**
        - GLM-5.2: [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4) (AMD Quark MXFP4 of
          [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)), licence MIT, repacked byte for byte.
        - GLM-5.3: quantized here from [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16),
          GLM-5.3 License. AMD has not published an MXFP4 build. The checkpoint is not public; the conversion
          is reproducible from the public BF16 release.
    - **Hardware:** 8 Gaudi 2 cards, TP 8 with expert parallel, 128K context served.
    - **Stack:** Gaudi software 1.24.1, lazy mode. The published port targets the stock `releases/v0.30.0`
      image; the earlier numbers below come from a vllm-gaudi 0.24 build, and each table says which.
    - **Status:** both serve production traffic.
    - **Headline:**
        - Long-context needles fixed: 3/3 at 16K to 128K on two fillers, where the stock path failed from 32K
          (same weights and image, test pods).
        - First TPC kernel vs stock MoE op: 23.25 to 45.6 tok/s at one stream (same-pod A/B).

## Why it needed a port

vllm-gaudi has a native fused MXFP4 MoE op (`mixture_of_experts.mxfp4_fused_weights`) that reads packed 4-bit
experts, but its only user was gpt-oss, whose model loader prepares the tensors itself. A DeepSeek-style model
reaches the op through vLLM's generic MoE loader, and four separate bugs sat on that path. Any one is fatal; two
of them fail silently.

1. **Gate/up order.** The op expects each expert's fused `w13` rows as `[gate; up]`. gpt-oss stores them
   interleaved and undoes that in its own loader. Measured on one card against a CPU dequant reference:
   concatenated `[gate; up]` gives relative error 0.0047 (cosine 0.99999); `[up; gate]` 0.091; interleaved 1.12.
2. **No fused-tensor load path for other models.** The DeepSeek-V2 loader calls `weight_loader` with two
   arguments where the routed-expert loader needs five, and the MXFP4 branch copied the global `[256, ...]`
   expert tensor into the per-rank parameter with no expert-parallel slicing. The fix defaults the arguments and
   slices by EP rank, so the model must run with `--enable-expert-parallel`.
3. **Parameter names.** Fused experts must be named `mlp.experts.routed_experts.w13_weight`. Shorter names are
   skipped without an error and the experts stay zero.
4. **Routing table dtype.** With grouped top-k, the HPU MXFP4 method did not cast `topk_ids` to int32 and called
   a `.router` attribute that `RoutedExperts` does not have. vllm-gaudi `v0.30.0` now does both correctly; the
   load-path bug is still upstream there.

## What was built

**Byte-exact repack (GLM-5.2).** AMD's Quark checkpoint and the vLLM fused format carry the same OCP MXFP4 data
(E2M1 values low nibble first, E8M0 scales, blocks of 32), so the repack copies FP4 bytes with no
dequantize-requantize step. Per MoE layer it concatenates gate and up per expert and stacks the experts; shared
experts become bf16 (vLLM leaves the shared MLP unquantized); everything else is copied. A one-card probe runs
the native op on candidate layouts against the CPU reference. At a small activation scale SiLU is nearly linear
and a swapped gate/up control still scored cosine 0.996, so the probe drives SiLU non-linear and reports
"inconclusive" when a negative control survives.

**MXFP4 encoder (GLM-5.3).** The encoder is checked against an upstream oracle: fed the same bf16 weights AMD
quantized for GLM-5.2, it reproduces 100% of AMD's weight and scale bytes (layers 3 and 29, 18 tensors). Getting
there found that Quark does not use the OCP shared-exponent rule `floor(log2(amax)) - 2`, which clamps anything
above 6.0. It bumps the exponent when `amax / 2^X >= 7`, where clamping would cost more than rounding to the next
exponent. The spec rule matches only about 80% of scale bytes and is less accurate (0.1171 vs 0.1141 relative
error against bf16). The BF16 release is the right source: MXFP4 made from it has relative error 0.1117 on a real
expert tensor, against 0.1148 when made from the FP8 release, which is itself lossy. On the served checkpoint the
native op reproduces the shipped bytes at cosine 0.99999 (one card, layers 3 and 77).

**DSA and the softmax defect.** vllm-gaudi runs DSA layers as dense MLA. That is exact up to 2,048 keys and
diluted beyond, and needles failed from 32K on varied text. The first fix ran the real indexer and masked all but
each query's top 2,048 keys. The main cause turned out to be elsewhere: vllm-gaudi calls FusedSDPA for prefill
with `softmax_mode="fast"` for every model except qwen2, and on real GLM-5.3 activations that mode is badly wrong
for some rows of deep layers.

| Worst row of 36, cosine vs fp32 | dense, `fast` | dense, `None` | top-2048 mask, `fast` |
| --- | --- | --- | --- |
| layer 38 | 0.086 | 0.99974 | 0.658 |
| layer 74 | -0.021 | 0.99989 | 0.469 |

Dense attention with the accurate `None` mode passes every needle test to 128K. One 8,192-query chunk against 70K
keys costs 27.6 ms in `fast`, 33.5 ms in `None` and 125 ms in fp32 (per-call, one card). The final DSA path is
truly sparse above 32K keys and dense with the accurate softmax below:

- The indexer splits query rows across the 8 cards, finds each row's 2,048th score by bisection, and exchanges
  bit-packed selections: 35 ms per indexer layer at 64K instead of 276 ms.
- Prefill attention, per group of four layers: early layers gather the union of 64 neighbouring queries' keys
  (3K to 9K at 64K context, 4 instead of 28 ms per layer); deeper layers, whose unions approach the whole context,
  use dense attention with the selection as a mask.
- Decode gathers exactly each sequence's 2,048 rows, so its cost no longer grows with context.
- Gathers are capped at 512 MB per call; a 9.7 GB gather hung all eight cards into the driver's 600 s timeout.
  A fully masked padding block in a decode bucket made paged attention compute 0/0, so padding blocks keep their
  original mask.

**MTP.** Both checkpoints ship one multi-token-prediction layer (layer 78), with its routed experts in BF16 in both
releases. GLM-5.3's MTP experts were encoded with the Quark-rule encoder above. GLM-5.2's were encoded earlier with
the plain OCP floor rule, before the Quark rule was found, and that is the checkpoint served; the port's GLM-5.2
repack keeps that encoder so its output matches the served bytes. The layer is served as one-draft speculative
decoding, which took eight vllm-gaudi patches. They fix: verify rows that
used the first row's KV block list; a draft residual that broadcast `[N, 1, H] + [N, H]` to `[N, N, H]`; a
rejection sampler that was greedy-only and stopped the engine at temperature above 0; greedy rows marked with
temperature -1.0, which the sampler divided by in mixed batches; and `logprobs` and structured output on the
padded verify layout. Acceptance on greedy test prompts is 0.76 to 0.82 on GLM-5.3 and 0.84 to 0.93 on GLM-5.2. A second draft costs
about 10 ms per step at one stream and pays only above about 0.9 acceptance, so GLM-5.3 stays at one. Under
speculation, async scheduling is off.

**MXFP4 TPC kernel.** The stock op unpacks every touched expert to bf16 in memory before tiny GEMMs. A custom TPC
kernel reads the served packed layout and unpacks in registers ([MXFP4 MoE decode kernels](../kernels/mxfp4-moe.md)
covers the design). Each revision was measured on its own:

| Step | Before | After | Measured as |
| --- | --- | --- | --- |
| First kernel vs stock op, GLM-5.3, 1 stream, 2K | 23.25 tok/s | 45.6 tok/s | same-pod A/B (flag off, then on) |
| v7 vs first kernel, GLM-5.3 | 45.6 tok/s | 51.6 tok/s (1.13x) | same-node A/B, greedy bit-identical |
| v8 vs v7, GLM-5.3 | 51.8 tok/s | 59.1 tok/s (1.14x) | same-node A/B, greedy bit-identical |
| v7 vs first kernel, GLM-5.2 | 42.1 tok/s | 47.8 tok/s (1.14x) | same-node A/B |

All runs use MTP with one draft and greedy decoding. The first kernel changes numerics (fp32 accumulation): greedy
outputs agree on 99.1% of top-1 tokens before the first near-tie split, the kernel is closer to an fp32
reference than the stock op, and needles and GSM8K (49/50) do not move.

**Where the decode step goes.** After the kernel, one 39 ms step at one stream (about 1.95 tokens with MTP)
splits into: expert kernels on the slowest EP rank 19.5 ms (the mean rank needs 9.6), MLA attention 6.9 ms, MoE
work outside the kernels 3.8 ms, 162 all-reduces 3.4 ms, and host time about 5.5 ms. The imbalance is
statistical: 16 routes into 8 ranks put about 4.5 routes on the busiest rank against a mean of 2. An EP 4 x TP 2
expert layout with a fitted placement narrows it (66.4 to 69.6 tok/s on the v8 kernels; same-node A/B, not
bit-identical). Details: [MoE decode step anatomy](../results/moe-decode-step.md).

## Correctness

| Gate | Result |
| --- | --- |
| Encoder vs AMD Quark, GLM-5.2 layers 3 and 29 | 100% of weight and scale bytes identical |
| Native op vs shipped bytes, 1 card | cosine 0.99999, relative error 0.004; swapped and interleaved controls fail |
| MXFP4 vs BF16, whole model, 6 texts x 512 tokens | per-token KL 0.042, top-1 agreement 91.6% |
| Needles, 3 per prompt, 16K to 128K, two fillers | 3/3 every prompt (final DSA config); stock path failed from 32K |
| GLM-5.2, same DSA settings | needles 3/3 at 16K to 128K |
| GSM8K first 50, GLM-5.3 with kernels | 49/50 |
| Tool-call checks, 40 cells | 40/40 |

Known weakness: a prompt of one word repeated tens of thousands of times makes the model loop under any accurate
attention. The `fast` softmax passed it only because its error happens to help.

## Performance

| Change, GLM-5.3 | Before | After | Measured as |
| --- | --- | --- | --- |
| Sparse DSA vs dense, decode at 64K | 72.1 ms/token | 63.6 ms/token | different pods on the same node class, idle, warm |
| Sparse DSA vs dense, TTFT at 100K | 47.8 s | 45.2 s | same |
| MTP on sparse DSA, 1 stream, 2K / 64K | 17.0 / 15.8 tok/s | 21.7 / 20.4 tok/s | two canaries on idle nodes, same spec |

GLM-5.3 on vllm-gaudi `v0.30.0` with the same kernels and patches gave no speed-up: within 1% to 2% of the 0.24
build, slightly slower, in a two-node crossover.

## What it taught about Gaudi

- **The FusedSDPA `fast` softmax can be badly wrong** on deep layers of long prompts. Any model that runs prefill
  through FusedSDPA without `VLLM_FP32_SOFTMAX` is exposed; see [platform findings](../results/platform-findings.md).
- **A loader that skips unknown names produces zero experts and fluent garbage.** Count what was loaded in both
  directions.
- Long single-node contexts hit the driver's 600 s "no progress" timeout during first-touch compiles of the
  largest decode shapes; 128K ships. The 256K single-node trials also needed the
  [rope-cpu-cache](../fixes/rope-cpu-cache.md) fix. A two-node layout that reaches 637K is in [two-node pipeline](../results/two-node-pipeline.md).
- Request-level fixes found here: [bad-words-index](../fixes/bad-words-index.md),
  [logitsprocs-rows](../fixes/logitsprocs-rows.md) (`min_p`, `logit_bias` and `min_tokens` were ignored on HPU),
  [gather-logprobs-int32](../fixes/gather-logprobs-int32.md), [prompt-logprobs](../fixes/prompt-logprobs.md) and
  [xgrammar-generation-eos-stop](../fixes/xgrammar-generation-eos-stop.md).

## Availability

The port is in [`ports/glm-5-mxfp4`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/glm-5-mxfp4),
for the stock vllm-gaudi `releases/v0.30.0` image (`058f107`, vLLM `ced6857a`): the GLM-5.3 encoder and the GLM-5.2
repack (`convert/`), 14 source patches with the fixes contract, the `glm5_hpu` package (DSA top-k attention, kernel
hooks, EP x TP expert layout and placement, decode levers), the bucket file and the GLM-5.3 placement (`serving/`),
CPU tests and serving gates. The kernels are built from
[`kernels/mxfp4-moe`](https://github.com/thediymaker/gaudi-cookbook/tree/main/kernels/mxfp4-moe) with
`--variant glm`. The recipes are [GLM-5.3 (MXFP4)](../recipes/glm-5-3-mxfp4.md) and
[GLM-5.2 (MXFP4)](../recipes/glm-5-2-mxfp4.md).

Both recipes passed their serving gates on that image, run as written on one 8-card node; GLM-5.3 carries one known
issue with `tool_choice: "required"` (see its recipe page). The prefill kernel A/B and the decode-lever A/B on that run
are in the [GLM-5.3 recipe](../recipes/glm-5-3-mxfp4.md#performance).
