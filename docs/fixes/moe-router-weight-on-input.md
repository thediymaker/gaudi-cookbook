# moe-router-weight-on-input

Source: [`fixes/moe-router-weight-on-input/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/moe-router-weight-on-input)

Apply the router weight to the expert input, not the output, for MoE layers that ask for it
(`apply_router_weight_on_input`), as Llama 4 does.

| | |
| --- | --- |
| Affects | Llama 4 Scout and Maverick (BF16 and compressed-tensors checkpoints) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/ops/hpu_fused_moe.py`, `vllm_gaudi/ops/hpu_compressed_tensors.py` |
| Upstream | not filed yet |
| Test | `test_moe_router_weight_on_input.py` (CPU, needs torch) |

## Problem

Llama 4's MoE builds `FusedMoE` with `apply_router_weight_on_input=True` and a top-1 sigmoid
router, so the reference computes `expert(w * x)`. The HPU monolithic MoE paths never read the
flag and pass the router weights to the Habana `mixture_of_experts` op, which applies them to the
expert output: `w * expert(x)`. With SwiGLU experts these differ (`silu(w * a) * (w * b)` against
`w * silu(a) * b`) on every MoE layer.

## Symptom

Outputs drift from the reference model. The text stays fluent, but next-token distributions
differ measurably from a float32 CPU reference, and tool calling and exact tasks suffer most.

## Who is affected

Anyone serving Llama 4 on HPU through the unquantized (BF16) or compressed-tensors (W8A8 FP8,
WNA16) MoE path. Checkpoints with `quant_method: fp8` go through `hpu_fp8.py`, which this fix does
not change. Models that do not set the flag are unchanged.

## The fix

Right before each `moe_op` call (unquantized `apply_monolithic` and `forward_oot`,
compressed-tensors W8A8 FP8 and WNA16 `apply_monolithic`), a layer with
`apply_router_weight_on_input` gets `x * w` and unit router weights. `top_k` must be 1, as
upstream requires; anything else raises an error.

Apply it after [laguna-moe-sigmoid-routing](laguna-moe-sigmoid-routing.md), which edits
the routing branch of the same methods; it also applies without it.

## Verify

```bash
python3 fixes/moe-router-weight-on-input/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=/path/to/site-packages pytest fixes/moe-router-weight-on-input
```

End to end: compare the served model's top-20 logprobs on a few prompts with a float32 CPU
reference run of the Hugging Face model. With the fix the divergence drops by an order of
magnitude.
