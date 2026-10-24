# logitsprocs-rows

Source: [`fixes/logitsprocs-rows/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/logitsprocs-rows)

Make the `min_p`, `logit_bias` and `min_tokens` request options take effect on HPU. On stock
releases they are accepted and silently ignored.

| | |
| --- | --- |
| Affects | every model, whenever a request sets `min_p`, `logit_bias` or `min_tokens` |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_input_batch.py` |
| Upstream | not filed yet |
| Test | `test_logitsprocs_rows.py` (CPU; needs torch and an importable vLLM) |

## Problem

vLLM implements these three options as built-in logits processors. Their per-request state
arrives through `InputBatch.batch_update_builder` and `LogitsProcessor.update_state`. The HPU
input batch creates the builder but never fills it (its source carries a TODO for this), so the
processors never receive any state.

## Symptom

- `logit_bias` has no effect, even at +100 or -100.
- `min_p` has no effect on sampling.
- `min_tokens` does not stop an early end of sequence.

No error or warning is raised.

## Who is affected

Every model served on a stock release. Clients that rely on `logit_bias` to ban or force tokens,
or on `min_p` for sampling quality, get different output than on other platforms.

## The fix

The fix does not use the builder protocol, because the HPU batch reorders its rows in ways that
protocol does not track:

- the input batch keeps each request's `SamplingParams` by request id (added in `add_request`,
  dropped in `remove_request`);
- `make_selective_sampling_metadata` replaces the three built-in processors with small views
  built from those parameters and the live output-token lists, for exactly the rows being
  sampled (row `i` belongs to `req_indices[i]`). Custom logits processors pass through unchanged.
  When the logits hold several rows per request (speculative verify rows), each request's view
  covers its rows, and the `min_tokens` view handles the padded verify layout.

The views are built on the host each step and kept out of torch.compile graphs. With speculative
decoding, vLLM itself still refuses `min_p` and `logit_bias` at request validation; that is
unchanged.

The fix applies on its own or after [bad-words-index](bad-words-index.md), which edits
the same `return` statement.

## Verify

```bash
python3 fixes/logitsprocs-rows/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/logitsprocs-rows
```

End to end: send a greedy request with `"logit_bias": {"<id of a common token>": -100}` and check
that the token no longer appears; send one with `+100` for a rare token and check that the output
is that token repeated. With `"min_p": 0.99` a sampled request should match the greedy output.
