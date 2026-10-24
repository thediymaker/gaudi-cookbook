# bad-words-index

Source: [`fixes/bad-words-index/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/bad-words-index)

Key `bad_words` to the rows that are actually being sampled, so one request's `bad_words` can
neither stop the engine nor ban tokens on another request's output.

| | |
| --- | --- |
| Affects | every model, whenever a request sets `bad_words` |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_input_batch.py` |
| Upstream | [vllm-gaudi#1917](https://github.com/vllm-project/vllm-gaudi/pull/1917) (open) |
| Test | `test_bad_words_index.py` (CPU; needs torch and an importable vLLM) |

## Problem

`InputBatch.make_selective_sampling_metadata` builds `SamplingMetadata` for only the rows being
sampled: a prefill's logits (often a single row) or the decode batch. It re-indexes temperature,
top-p/top-k, penalties and generators to those rows, but passes `bad_words_token_ids` through as
the whole batch's dict, keyed by persistent batch index. vLLM's `apply_bad_words` indexes the
selected logits with that key.

## Symptom

- A request with `bad_words` that sits at any batch index other than 0 and is sampled in a
  smaller selection raises `IndexError: index N is out of bounds for dimension 0 with size 1` in
  the worker, and the engine stops. With torch.compile this always happens; in lazy mode it
  happens whenever the index is past the end of the selection.
- When the index is inside the selection, the words are banned on another request's row and the
  requester's own output is not filtered.

## Who is affected

Every deployment where a client can send `bad_words` while another request is running. A single
request field can take the server down, so treat this fix as a robustness requirement for any
endpoint exposed to untrusted clients.

## The fix

Re-key the map to the selected rows, `{row i: words of req_indices[i]}`, the same way the
generators are re-keyed in the same method. Other rows and requests without `bad_words` are not
affected. The fix detects the upstream change and does nothing when it is already present.

## Verify

```bash
python3 fixes/bad-words-index/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/bad-words-index
```

End to end: start a long-running request, then send a second request with
`"bad_words": ["hello"]`. Without the fix the engine can stop with an `IndexError`; with it the
second request's output never contains the banned word and the first is unaffected.
