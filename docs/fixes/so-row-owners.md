# so-row-owners

Source: [`fixes/so-row-owners/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/so-row-owners)

Give grammar masks, sampling parameters and logprobs to the logits rows that actually exist in a
structured-output step.

| | |
| --- | --- |
| Affects | every model, in steps with a grammar request and a partial prompt chunk |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Order | after [structured-output-rows](structured-output-rows.md) when both are applied |
| Upstream | not filed yet |
| Test | `test_so_row_owners.py` (torch and numpy; the source tests need a runner source) |

## Problem

When a step has a grammar request, `sample_tokens` stacks the step's logits and samples them
together. vllm-gaudi computes logits only at a prompt's last position, so a prompt chunk that does
not finish its prompt produces no logits row. Three things still count it as a row:

- `apply_grammar_bitmask` places each grammar mask at the request's persistent-batch index (the GPU
  runner's rule, where every request has a row). With a partial chunk at a lower index, every later
  mask lands one row late, or past the last row.
- The sampling parameters are taken from `decode_req_ids + prompt_req_ids`, so rows after a partial
  chunk use their neighbour's parameters.
- The logprobs of those rows are filed under the wrong request.

Separately, `_prepare_prefill_inputs` gives a partial chunk one logits row when structured output
is on, but that flag still holds the previous step's value at that point, and without async
scheduling nothing discards the row: a token sampled in the middle of the prompt is returned as
the request's first output token.

## Symptom

- A plain request is forced into another request's JSON, while the grammar request samples
  unconstrained and is then terminated with a "grammar rejected tokens" error.
- `IndexError: index N is out of bounds for axis 0 with size N` on every rank, and the engine stops.
- A stray token at the start of a reply, or a grammar failure on a request that is still prefilling.

It needs a long prompt being chunked in the same step as a request that finishes its prompt, while
any grammar request is active. [prompt-order](prompt-order.md) runs prompts in order of
context length, which makes that layout common.

## Who is affected

Deployments that serve structured output or forced tool calls together with prompts longer than
the step token budget (`--max-num-batched-tokens`).

## The fix

`decode_sampled_requests + prefill_sampled_requests` is built next to the logits with one entry per
row. When that list matches the logits one to one, it is used as the row owners for the grammar
masks (a request with no row gets no mask, as in upstream vLLM), the sampling parameters and the
logprobs. Otherwise (for example several rows for one request) the step runs as before and a
warning is logged once. A partial chunk gets a logits row only when async scheduling discards it.

## Verify

```bash
python3 fixes/so-row-owners/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=/path/to/site-packages pytest fixes/so-row-owners
```

End to end: with a small `--max-num-batched-tokens`, send a prompt several times longer than the
budget, a short `json_schema` request and a few plain requests at the same time, many times over.
Without the fix some runs end in grammar errors, mixed-up outputs or an engine restart.
