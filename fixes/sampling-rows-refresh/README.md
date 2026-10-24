# sampling-rows-refresh

Re-copy the device-side sampling parameters after the input batch reorders its rows, so every
request is sampled with its own temperature, top-p, top-k, min-p and penalties.

| | |
| --- | --- |
| Affects | every model, whenever concurrent requests use different sampling parameters |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_input_batch.py` (appended wrapper) |
| Upstream | not filed yet |
| Test | `test_sampling_rows_refresh.py` (CPU; needs torch and an importable vLLM) |

## Problem

The HPU input batch keeps each row's sampling parameters twice: a CPU copy and a device copy.
`make_selective_sampling_metadata` reads the device copy and refreshes it from the CPU copy only
when `skip_copy` is false, and the runner passes `skip_copy=not batch_changed`, so the refresh
happens only on steps where a request joined or left. `swap_states`, which the runner calls after
that refresh to put decodes first and multi-token decodes last, swaps only the CPU copies.

A row that changes owner on a step where nobody joined or left therefore keeps the previous
owner's parameters on the device, and keeps them until the next join or leave.

## Symptom

- A greedy (temperature 0) request emits tokens that are not the argmax, at a neighbour's
  temperature, while its returned logprobs look normal.
- A sampled request turns greedy, or loses its `top_k` / `top_p`.
- Repetition, presence and frequency penalties use another request's values, and the cached
  prompt-token tensor used for penalties stays in the old row order.

It needs rows to be reordered on a quiet step, which happens under concurrent load when prompts
are longer than the step's token budget (chunked prefill), so it is load-dependent and hard to
reproduce with single requests.

## Who is affected

Every deployment that serves concurrent requests with different sampling parameters. Fixes that
add row swaps, such as [prompt-order](../prompt-order/README.md), make it more likely, so apply
both together.

## The fix

`swap_states` marks the batch when it moves a row. The next `make_selective_sampling_metadata`
call first re-copies every row's temperature, top-p, top-k and min-p (and the three penalties when
any request uses one) as whole tensors, which keeps shapes fixed so no new graphs are compiled,
drops the cached prompt-token tensor, and then runs the stock code. `refresh_sampling_metadata`
clears the mark. A step without a swap costs nothing extra; a step with one costs what a join or
leave step already pays.

The fix is appended to the module and wraps the `InputBatch` methods, so it composes with the
other fixes that edit the same file in either order.

## Verify

```bash
python3 fixes/sampling-rows-refresh/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/sampling-rows-refresh
```

End to end: run many greedy requests (temperature 0, `logprobs: 1`) next to steady sampled
traffic (temperature near 1) with long prompts. Without the fix some greedy responses contain
tokens whose logprob is not the highest at that position; with it none do.
