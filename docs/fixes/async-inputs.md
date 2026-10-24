# async-inputs

Source: [`fixes/async-inputs/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/async-inputs)

Seed the device decode input ids from the host before the async-scheduling scatter, so a decode
request that sat out a step does not read a stale token.

| | |
| --- | --- |
| Affects | every model served with async scheduling |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | [vllm-gaudi#1919](https://github.com/vllm-project/vllm-gaudi/pull/1919) (open) |
| Test | none (host-side input preparation; covered by the end-to-end check below) |

## Problem

With async scheduling, `_create_decode_input_data` builds every decode row's input id on the host,
copies it to the device, and then replaces all decode rows with `input_ids_hpu`. `input_ids_hpu` is
a persistent device buffer that `_prepare_input_ids` writes only at the rows of requests that were
also in the previous step (and not at all when no request is in common). A decode request that was
not in the previous step therefore decodes from whatever its buffer row held last.

## Symptom

The model re-reads an old token: greedy output repeats a token (`... a, b, b ...`) or the text
derails. It happens to a running request that the scheduler left out of one step for lack of token
budget (for example behind long prompt chunks), to a request resumed after preemption, and to one
whose previous sampled token was discarded. The bug is in host-side input preparation, so lazy and
compile mode are both affected.

## Who is affected

Deployments with async scheduling enabled (`--async-scheduling`) that mix long prompts with
decoding requests, or that see preemption.

## The fix

Copy the host-side decode input ids into `input_ids_hpu` before `_prepare_input_ids` runs, as vLLM's
GPU runner does before its scatter. Rows of requests carried over from the previous step are still
overwritten with their sampled tokens on the device; every other row now holds its correct token.
Cost: one small device copy per decode step.

## Verify

```bash
python3 fixes/async-inputs/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: with async scheduling and a small `--max-num-batched-tokens`, keep several greedy
requests decoding while sending prompts that are a few times longer than the budget. Compare the
greedy outputs with the same requests sent alone; without the fix some contain a repeated token.
