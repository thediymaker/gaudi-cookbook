# fp8-moe-gather

Source: [`fixes/fp8-moe-gather/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/fp8-moe-gather)

Backport the gathered-expert FP8 MoE combine from vllm-gaudi v0.30.0 to v0.26.0: at low token
counts only the routed experts are read and computed, which speeds up decode on fine-grained FP8
MoE models. Opt-in, and off at runtime until `VLLM_HPU_MOE_GATHER=1` is set.

| | |
| --- | --- |
| Affects | block-quantized FP8 MoE checkpoints with dynamic activation scales and silu experts, for example Qwen3 / Qwen3.5 / Qwen3.6 MoE FP8 (performance) |
| Releases | v0.26.0 (v0.30.0 already contains the upstream code; the script reports "not needed") |
| Patches | `vllm_gaudi/ops/hpu_fp8.py` |
| Upstream | [vllm-gaudi#1731](https://github.com/vllm-project/vllm-gaudi/pull/1731) (merged, in v0.30.0) |
| Test | none (the kernel needs the Habana FP8 GEMM op; use the verify mode below) |

## Problem

The Habana `mixture_of_experts` op reads every local expert's weights on every call. At decode
with a small batch, each token routes to only `top_k` experts, so most of that memory traffic
is wasted, and single-stream decode on MoE models with many small experts is bound by it.

## Symptom

Low single-stream and small-batch decode throughput on fine-grained FP8 MoE models compared with
v0.30.0 using `VLLM_HPU_MOE_GATHER=1`. There is no correctness symptom; this is a performance
backport.

## Who is affected

v0.26.0 users serving block-quantized FP8 MoE checkpoints with dynamic activation scales, with
`VLLM_HPU_FORCE_CHANNEL_FP8` left at its default (`true`), and silu experts. Other checkpoints,
static-activation FP8 among them, always keep the stock op.

## The fix

The combine kernel and its verify helpers from vllm-gaudi v0.30.0 (`ops/hpu_moe_combine.py` and
`ops/hpu_fp8.py`) are appended to `vllm_gaudi/ops/hpu_fp8.py`, and
`HPUFp8MoEMethod.apply_monolithic` sends its expert call through the same gate v0.30.0 uses. When
the gate passes, the `g = min(local_experts, tokens * top_k)` routed experts are gathered: the
up/gate projection runs as one native FP8 GEMM and the down projection in fp32. Otherwise the
stock op runs unchanged. The gathered count is static, so the path captures into HPU graphs.

The environment variables are the same as on v0.30.0, so a configuration carries over:

| Variable | Meaning | Default |
| --- | --- | --- |
| `VLLM_HPU_MOE_GATHER` | `1`: use the gather path when it wins | `0` (stock op) |
| `VLLM_HPU_MOE_GATHER_RATIO` | use it while `tokens * top_k <= local_experts * ratio` | `0.4` |
| `VLLM_HPU_MOE_GATHER_VERIFY` | `1`: run both paths and log FP8-ULP differences above 2 ULP (slow; for checks only) | `0` |

## Verify

```bash
python3 fixes/fp8-moe-gather/apply.py --check   # 0 = applied or not needed, 1 = not applied, 2 = source drifted
```

End to end: start once with `VLLM_HPU_MOE_GATHER=1 VLLM_HPU_MOE_GATHER_VERIFY=1` and send a few
short requests. The log should show the gather path enabled and no FP8-ULP mismatch warnings.
Then restart without `VLLM_HPU_MOE_GATHER_VERIFY` and compare single-stream decode tokens per
second with `VLLM_HPU_MOE_GATHER` unset.
