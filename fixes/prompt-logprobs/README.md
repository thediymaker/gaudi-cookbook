# prompt-logprobs

Make `prompt_logprobs` safe and correct in the HPU model runner: no engine crash on models with a
flat hidden-state output, correct values, bounded memory.

| | |
| --- | --- |
| Affects | every model, requests with `prompt_logprobs` (or `/v1/completions` with `echo` and `logprobs`) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | crash part fixed on main by [vllm-gaudi#1839](https://github.com/vllm-project/vllm-gaudi/pull/1839) (merged, not in either release); value and memory parts not filed |
| Test | `test_prompt_logprobs.py` (CPU, needs torch) |

## Problem

`sample_tokens` keeps each prefill request's hidden states as `non_flattened_hidden_states[i]`,
assuming a `[batch, seq, hidden]` output. Several models return a flat `[batch * seq, hidden]`
tensor instead: models whose adapter flattens the input ids (many MoE models), and multimodal
models when one prompt is in the prefill batch. `[i]` then picks a single token's hidden vector,
`compute_logits` gets a 1-D tensor, and the shape error is raised in the worker, outside any
request boundary.

Three more defects sit on the same path:

- The gathered ids, logprobs and ranks are copied to the CPU with `non_blocking` copies into
  buffers of another dtype, and the ids use the int64 cat followed by an int32 cast that can read
  back wrong on HPU (see [gather-logprobs-int32](../gather-logprobs-int32/README.md)).
- The hidden states of every prefill batch are kept until the end of the step, after later
  forwards and graph replays have run.
- `compute_logits` and `log_softmax` run over every prompt token of a chunk at once. At a large
  vocabulary a long chunk needs several GB of fp32.

## Symptom

One request with `prompt_logprobs` stops the engine with
`RuntimeError: a and b must have same reduction dim` on affected models. On other models the
returned prompt logprobs can be scrambled or impossible, and long prompts can run out of device
memory.

## Who is affected

Anyone who serves `prompt_logprobs`, for example for perplexity or scoring workloads. The crash
needs a model with a flat output; the value and memory problems affect every model.

## The fix

- Each prompt-logprobs request's rows are found in the flat view of the output from the batch
  layout the runner built (row-padded or merged prefill) and cloned right after its forward. A
  layout that does not match is logged once and that batch's prompt logprobs are skipped instead
  of stopping the engine.
- Logits are computed in row chunks of `VLLM_HPU_PROMPT_LOGPROBS_ROWS` rows (default 512, a
  multiple of 128), with the tail padded to a multiple of 128 so only a few shapes compile.
- Results are copied with synchronous `.cpu()` in the device dtype, cast on the CPU, and the
  prompt-token column is written from the CPU ids.

On vllm-gaudi main the collection code has changed (#1839), so this fix reports drift there.

## Verify

```bash
python3 fixes/prompt-logprobs/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/prompt-logprobs                     # GAUDI_FIXES_ROOT=<dir with vllm_gaudi/> if not installed
```

End to end: request `prompt_logprobs: 1` on a long prompt (several chunks) and compare the summed
prompt log-likelihood with a CPU or GPU reference of the same checkpoint.
