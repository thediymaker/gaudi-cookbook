# moe-explicit-overload

Source: [`fixes/moe-explicit-overload/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/moe-explicit-overload)

Call the per-channel FP8 MoE op by its explicit overload, so it does not force a host sync per MoE
layer.

| | |
| --- | --- |
| Affects | FP8 per-channel MoE models (for example FP8 Qwen3 MoE, GLM and DeepSeek checkpoints) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/extension/ops.py` |
| Upstream | [vllm-gaudi#1921](https://github.com/vllm-project/vllm-gaudi/pull/1921) (open) |
| Test | none (performance fix; numerics unchanged by construction) |

## Problem

`VllmMixtureOfExpertsOpFP8PerChannel.forward` calls `torch.ops.hpu.mixture_of_experts(...)`
through the overload packet. The packet tries its overloads in registration order until one binds.
Several overloads that come first declare a float `d_scale_hidden_states`; binding the Tensor
scale fails, and the schema-mismatch message PyTorch builds contains `repr(x_scale)`. Formatting a
tensor reads its values, which copies it to the host and blocks until the device is idle. The
attempt is then discarded and the next overload is tried.

## Symptom

Slow long-prompt prefill (high time to first token) and slow graph capture on FP8 MoE models. A
native profiler shows `Tensor.__repr__` and `_local_scalar_dense` under
`OpOverloadPacket.__call__` in every MoE layer. Replayed decode graphs do not run Python and are not
affected.

## Who is affected

FP8 MoE models that use the per-channel op: one blocking sync per MoE layer for every eager forward
(prompts above the graph capture size) and every graph capture.

## The fix

Name the overload the arguments bind to: `fp8_fused_weights_dynamic` for dynamic activation scales
and `fp8_fused_weights` for static scales. The kernel and its arguments are the same, so outputs do
not change.

## Verify

```bash
python3 fixes/moe-explicit-overload/apply.py --check   # 0 = applied, 1 = not applied, 2 = drifted
```

End to end: compare time to first token for a long prompt (several thousand tokens) on an FP8 MoE
model before and after; outputs with greedy decoding must be identical.
