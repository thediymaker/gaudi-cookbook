# logprobs-decode-guard

Source: [`fixes/logprobs-decode-guard/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/logprobs-decode-guard)

Stop one undecodable logprob token id from shutting down the API server.

| | |
| --- | --- |
| Affects | every model, requests with `logprobs` / `top_logprobs` |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/tokenizers/detokenizer_utils.py`, `vllm/v1/engine/logprobs.py` (vLLM core) |
| Upstream | not filed (vLLM) |
| Test | `test_logprobs_decode_guard.py` (stdlib only) |

## Problem

If a platform returns a token id outside the vocabulary in logprobs, detokenizing it raises
`OverflowError: out of range integral type conversion attempted` inside `AsyncLLM.output_handler`.
That loop serves every request and has no request boundary, so the engine shuts down and every
in-flight request fails.

## Symptom

The server exits, often without a crash trace, shortly after a request that asked for
`top_logprobs`. All other in-flight requests fail with it.

## Who is affected

Any deployment that a client can send `logprobs: true` to, on a platform that can produce bad ids.
On Gaudi the known source of such ids is fixed by
[gather-logprobs-int32](gather-logprobs-int32.md); this guard is the safety net.

## The fix

This does not correct ids. It confines the damage to the response that asked for logprobs:

- `convert_ids_list_to_tokens` is wrapped at module scope (so later imports pick up the wrapper).
  When the original raises, ids outside the vocabulary become `"<bad_id>"` and the other ids of the
  same call still decode normally. The first occurrence is logged as a warning.
- `LogprobsProcessor._correct_decoded_token` returns `""` when decoding raises, which is already
  its documented value for an incomplete byte sequence.

Requests without logprobs never reach either site.

## Verify

```bash
python3 fixes/logprobs-decode-guard/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/logprobs-decode-guard
```
