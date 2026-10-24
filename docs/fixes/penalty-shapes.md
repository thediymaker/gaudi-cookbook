# penalty-shapes

Source: [`fixes/penalty-shapes/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/penalty-shapes)

Pad the token tensors used by repetition, presence and frequency penalties to bucketed lengths,
so a batch that uses penalties stops recompiling HPU graphs on every decode step.

| | |
| --- | --- |
| Affects | requests that set `repetition_penalty`, `presence_penalty` or `frequency_penalty` (performance) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/model_executor/layers/utils.py`, `vllm/_custom_ops.py`, `vllm/v1/sample/ops/penalties.py`, `vllm_gaudi/v1/worker/hpu_input_batch.py` |
| Upstream | not filed yet |
| Test | none (the changed arithmetic runs only on HPU tensors) |

## Problem

Penalties build two token tensors on every sampling step, each padded to exactly the longest
list in the batch:

- vLLM's `_convert_to_tensors` pads the output token ids to the longest output so far, which
  grows by one every decode step;
- the HPU input batch pads the prompt token ids to the longest running prompt.

On HPU every new tensor shape is a new graph. A batch that uses penalties therefore compiles on
nearly every step. Once shapes are stable, the arithmetic is still slow: token bin counts are
int64, which Gaudi emulates, and the repetition penalty is materialized at (rows, vocab).

## Symptom

Throughput drops sharply as soon as a request with a penalty is in the batch, and gets worse
with more concurrent streams, with long stalls while graphs compile. Requests without penalties
are fast.

## Who is affected

Deployments whose clients send penalties. On v0.26.0, penalties are only applied with
`--no-async-scheduling` (the async runner does not keep the output token lists that penalties
read), so there the fix matters for synchronous scheduling. v0.30.0 also applies penalties under
async scheduling.

## The fix

- Pad both token tensors up to a multiple of 256 (`PAD_MULTIPLE` in `apply.py`), using the
  `vocab_size` pad id the code already uses and drops, so shapes repeat. Results are unchanged.
- On HPU tensors only: count token bins in float32 (exact far beyond any context length) and
  broadcast the repetition penalty as a (rows, 1) column. Other devices keep the stock
  arithmetic.

This is a performance fix with no effect on correctness. It is not part of the default release
profiles; add it by name when your clients use penalties.

## Verify

```bash
python3 fixes/penalty-shapes/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: run 4 to 16 concurrent streams with `"repetition_penalty": 1.1` and
`--no-async-scheduling`, and compare tokens per second and the number of graph compilations
reported in the server log with and without the fix.
