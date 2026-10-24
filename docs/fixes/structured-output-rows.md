# structured-output-rows

Source: [`fixes/structured-output-rows/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/structured-output-rows)

Sample each row of a structured-output step with its own request's sampling parameters.

| | |
| --- | --- |
| Affects | every model, in any step that has a request with a grammar |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | [vllm-gaudi#1916](https://github.com/vllm-project/vllm-gaudi/pull/1916) (open) |
| Test | none (one-expression change; covered by the end-to-end check below) |

## Problem

When any request in a step uses structured output (`json_schema`, `json_object`, guided choice or
regex, `tool_choice: "required"` or a named tool), `sample_tokens` defers sampling and stacks the
step's logits decodes first (`logits_decode + logits_prompt`). The sampling metadata for those rows
is built from `prompt_req_ids + decode_req_ids`, prompts first. The logprobs bookkeeping and the
token scatter that follow use decodes first, so only the sampling parameters are out of order.

## Symptom

In a step with both a finishing prefill and decodes, every row is sampled with another request's
temperature, top-p/top-k, seed generator and penalties (the penalties also read the other
request's output tokens). A greedy request can come back random, a `logit_bias` or penalty set by
one client steers another client's tokens, and seeded requests stop being reproducible. A grammar
request stays in the batch for its whole decode, so every request that arrives meanwhile can be
hit. Some tool parsers attach a grammar to every chat request, which makes this constant.

## Who is affected

Any deployment that serves structured output or forced tool calls next to other traffic.

## The fix

Build the sampling metadata in decode-first order, matching the stacked logits. Required by
[so-row-owners](so-row-owners.md), which goes further for steps with partial prompt
chunks.

## Verify

```bash
python3 fixes/structured-output-rows/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: keep one `json_schema` request decoding while sending plain requests with
`temperature: 0` and another with a strong `logit_bias` on a common word. Without the fix the
plain requests occasionally pick up the biased word or lose determinism.
