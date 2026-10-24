# gather-logprobs-int32

Source: [`fixes/gather-logprobs-int32/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/gather-logprobs-int32)

Cast logprob token ids to int32 before concatenating them, so `top_logprobs` ids read back
correctly on HPU.

| | |
| --- | --- |
| Affects | every model, requests with `logprobs` / `top_logprobs` (and `prompt_logprobs`) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/patches.py`, `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | [vllm-gaudi#1823](https://github.com/vllm-project/vllm-gaudi/pull/1823) (open) |
| Test | none (verified on hardware; the change is a dtype reordering) |

## Problem

vllm-gaudi replaces vLLM's `Sampler.gather_logprobs` with `_hpu_gather_logprobs`. It builds the
`(tokens, k + 1)` id tensor as an int64 `torch.cat` of the sampled ids and the top-k indices and
then casts the result to int32. On HPU in lazy mode that int64 cat followed by the cast is read
back with the wrong contents. The float logprobs concatenated right next to it, and the sampled
token ids cut from the same source, are correct. Splitting the lazy graph around the cat also
avoids it, which points at graph fusion of the int64 cat with the cast.

## Symptom

Under concurrent load, `top_logprobs` responses contain ids outside the vocabulary, a first entry
that is not the sampled token, entries with probability 0 for the sampled token, or truncated
lists. An out-of-range id can also raise `OverflowError` in the detokenizer and stop the API
server (see [logprobs-decode-guard](logprobs-decode-guard.md)). Tensor-parallel ranks
disagree with each other on the same step.

## Who is affected

Any deployment where clients can request `logprobs`. Requests without logprobs never reach this
code.

## The fix

Cast both operands to int32 before the cat. Vocabulary ids fit in int32, so the result is
numerically identical and no synchronization is added. The runner's prompt-logprobs gather
(`_gather_prompt_logprobs_hpu`) builds its ids the same way and gets the same change.

## Verify

```bash
python3 fixes/gather-logprobs-int32/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: send a few hundred single-token chat requests with unique prompts and
`top_logprobs: 20` at a concurrency of 8 or more. Check that every returned id decodes, that the
first entry equals the sampled token, and that its logprob is not `-inf`.
