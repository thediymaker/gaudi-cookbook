# resumed-prefill-logits

Source: [`fixes/resumed-prefill-logits/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/resumed-prefill-logits)

Sample one token, not one per recomputed position, when a request resumes after preemption with
prefix-cache hits.

| | |
| --- | --- |
| Affects | every model served with prefix caching, when requests are preempted |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_input_batch.py` |
| Upstream | [vllm-gaudi#1849](https://github.com/vllm-project/vllm-gaudi/pull/1849) (merged on main after v0.30.0) |
| Test | none (same change as upstream) |

## Problem

When a preempted request resumes, the scheduler recomputes its prompt plus the tokens it had
already generated. `InputBatch.add_request` folds those earlier outputs into `num_prompt_tokens`,
and the runner sizes a prefill's logits as `num_computed + num_scheduled - num_prompt_tokens + 1`,
which is then 1. The fold only happens when `num_computed_tokens == 0`. With prefix caching a
resumed request usually gets cache hits on its own freed blocks and resumes with
`num_computed_tokens > 0`, so the fold is skipped.

## Symptom

The runner computes a logit, and samples a token, at every recomputed output position and credits
all of them to the request:

- the already generated tail is appended again (with greedy decoding, an exact repeat of earlier
  text);
- an extra `[tokens x vocab]` logits tensor is allocated, which can run out of device memory on
  large-vocabulary models;
- with speculative decoding, the scheduler's accepted-token accounting asserts and the engine stops.

## Who is affected

Deployments with prefix caching (the default) under enough load to preempt requests, typically
long generations with a KV cache that is nearly full.

## The fix

Treat a request as recomputing whenever `num_computed_tokens < known tokens - 1`, where the known
tokens are the prompt plus the earlier outputs, as vllm-gaudi main does. Running decodes and
speculative decode requests always have `num_computed_tokens >= known tokens - 1` and are
unchanged. On a release that already contains the upstream change, `apply.py` reports
"not needed" and writes nothing.

## Verify

```bash
python3 fixes/resumed-prefill-logits/apply.py --check   # 0 = applied or not needed, 1 = not applied, 2 = source drifted
```

End to end: start the server with a small KV cache (for example a low `--gpu-memory-utilization`
or `--num-gpu-blocks-override`) so that concurrent long greedy generations get preempted, and
compare each output with the same request sent alone. Without the fix some outputs contain a
repeated block of text.
