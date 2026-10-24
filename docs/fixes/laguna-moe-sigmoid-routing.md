# laguna-moe-sigmoid-routing

Source: [`fixes/laguna-moe-sigmoid-routing/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/laguna-moe-sigmoid-routing)

MoE top-k routing ignores `scoring_func`, the score correction bias and `renormalize`. The fix
makes every non-grouped MoE layer route the way its config says. (The id comes from the first
model this was found on, Laguna, which uses sigmoid routing; the fix itself is generic.)

| | |
| --- | --- |
| Affects | MoE models with sigmoid scoring, an `e_score_correction_bias`, or `renormalize=False` |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/ops/hpu_fused_moe.py`, `hpu_fp8.py`, `hpu_compressed_tensors.py`, `hpu_mxfp4.py` |
| Upstream | [vllm-gaudi#1922](https://github.com/vllm-project/vllm-gaudi/pull/1922) (open) |
| Test | `test_moe_routing.py` (CPU, needs torch) |

## Problem

The monolithic MoE apply paths (unquantized, FP8, compressed-tensors W8A8 FP8 and WNA16, MXFP4)
send every layer that is not grouped top-k or custom routing through an inline branch that takes a
softmax over the router logits, picks the top k and always renormalizes. That branch never looks
at the layer's `scoring_func`, `e_score_correction_bias` or `renormalize`.

## Symptom

For sigmoid-scored MoEs with an aux-loss-free selection bias the wrong experts are selected. Prose
can stay fluent while exact tasks (counting, arithmetic, code) fail. For softmax MoEs configured
without renormalization the expert weights are scaled up, so outputs drift from the reference
implementation.

## Who is affected

Any MoE model served on HPU whose layers use sigmoid scoring, a correction bias without grouped
top-k, or `renormalize=False`. Plain renormalized softmax models (most MoEs) and grouped top-k
models (DeepSeek-style) keep their current path and output.

## The fix

`is_plain_softmax_topk(layer)` keeps the inline fast path only for plain renormalized softmax
layers. Every other layer goes through `select_experts_from_routed`, which now handles
non-grouped top-k itself, following vLLM's `fused_topk_bias`: softmax or sigmoid scores in fp32,
the correction bias used for expert selection only, weights gathered from the unbiased scores,
then optional renormalization and the layer's `routed_scaling_factor`. Grouped and custom routing
still use vLLM's `select_experts` helper.

On v0.26.0 the MXFP4 path still called `layer.router.select_experts`, which routed layers no longer
have; it now uses `select_experts_from_routed` like v0.30.0.

Note that models with `renormalize=False` change output with this fix: they now match their
reference implementation.

## Verify

```bash
python3 fixes/laguna-moe-sigmoid-routing/apply.py --check   # 0 = applied, 1 = not applied, 2 = drifted
pytest fixes/laguna-moe-sigmoid-routing
```

End to end on a sigmoid-routed MoE: exact tasks such as "count the letters in ..." or short
arithmetic go from mostly wrong to mostly right; compare with the model's published scores.
