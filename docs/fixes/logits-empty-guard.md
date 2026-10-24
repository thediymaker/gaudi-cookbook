# logits-empty-guard

Source: [`fixes/logits-empty-guard/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/logits-empty-guard)

Skip the tensor-parallel logits gather when a step has no logits rows, so long prompts do not stop
the engine.

| | |
| --- | --- |
| Affects | every model served with tensor parallelism and sync scheduling (`--no-async-scheduling`) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/model_executor/layers/logits_processor.py` |
| Upstream | not filed yet |
| Test | none |

## Problem

With sync scheduling and no structured output, the HPU runner gives a prompt chunk that does not
finish its prompt zero logits rows. `compute_logits` still runs, on a `[0, hidden]` tensor, and the
tensor-parallel all-gather in `LogitsProcessor._get_logits` then runs over an empty tensor, which
the Gaudi collective library treats as a fatal error.

## Symptom

The engine core stops with `ValidateSyncInputTensors ... tensor_data is empty` in
`hccl::allgather_out`, and the server restarts. Any prompt longer than
`--max-num-batched-tokens` triggers it.

## Who is affected

Servers with `--tensor-parallel-size` above 1 and async scheduling off. With async scheduling (the
default) a partial chunk keeps one logits row and the gather is never empty. Single-card servers do
not gather.

## The fix

When the projected logits have no rows, `_get_logits` returns an empty `[0, org_vocab_size]`
tensor of the logits dtype before the gather. Every rank sees the same empty shape, so all ranks
skip the collective together. Non-empty batches are unchanged.

## Verify

```bash
python3 fixes/logits-empty-guard/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: on a TP 2 or larger server with `--no-async-scheduling`, send a prompt about twice as
long as `--max-num-batched-tokens`. Without the fix the engine stops; with it the request completes.
