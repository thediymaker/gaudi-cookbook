# prompt-order

Source: [`fixes/prompt-order/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/prompt-order)

Run a step's prompts in order of context length, so a prefix-cache hit is never read before the
request that computes it has written it.

| | |
| --- | --- |
| Affects | every model served with prefix caching (the default) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | [vllm-gaudi#1918](https://github.com/vllm-project/vllm-gaudi/pull/1918) (open) |
| Test | `test_prompt_order.py` (stdlib only) |

## Problem

The vLLM scheduler publishes a request's KV blocks to the prefix cache when it schedules them. A
request scheduled later in the same step can therefore get cache hits on blocks that another
request only computes in that step. On GPU this is safe: the step is one forward, and every layer
writes its KV before attention reads it.

The HPU runner runs a step's prompts as separate forwards, in input-batch row order, and a request
that has context always gets a forward of its own. A new request takes the lowest free row, so the
reader can sit above the writer. The reader's forward then runs first and attends to whatever the
reused block held before.

## Symptom

A response contains a garbled copy of text from an unrelated, earlier request (for example a code
or name from someone else's prompt), or quality drops for no visible reason. It needs two requests
that share a prefix to be scheduled in the same step, with the reader in a lower row, so it shows
up under concurrent load with shared system prompts. Running the same prompts one at a time gives
the right answer.

## Who is affected

Anyone serving with prefix caching enabled, which is the vLLM default. Sync and async scheduling
are both affected. Only a request's last prompt chunk can read the stale block.

## The fix

A reader's hits always extend past what its writer has computed so far, so the reader's
`num_computed_tokens` is strictly larger than the writer's. After the runner's decodes-first
ordering, the fix sorts the prompt rows by `num_computed_tokens` (ascending, stable) with
`swap_states`, so every writer runs before its readers and all row-indexed state (sampling
parameters, logprobs, structured-output masks) follows the request. Decode rows are not moved. The
sort runs only when prefix caching is on.

Apply [sampling-rows-refresh](sampling-rows-refresh.md) as well: the extra row swaps make
the stale device sampling rows it fixes more likely.

## Verify

```bash
python3 fixes/prompt-order/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/prompt-order
```

End to end: send two requests that share a long prefix (several KV blocks) at the same moment,
where the shared part carries a distinctive string and each request asks for it back. Repeat many
times with prefix caching on; without the fix, answers occasionally quote a previous request.
