# mm-merge-idxcopy

Source: [`fixes/mm-merge-idxcopy/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/mm-merge-idxcopy)

Merge multimodal embeddings with `nonzero` + `index_copy_` instead of a broadcast
`masked_scatter_` that fails to compile on Gaudi under concurrent image traffic.

| | |
| --- | --- |
| Affects | vision-language models (images or video) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/models/utils.py` |
| Upstream | not filed yet |
| Test | `test_mm_merge_idxcopy.py` (CPU, needs torch and a vllm-gaudi source tree) |

## Problem

vllm-gaudi replaces vLLM's `_merge_multimodal_embeddings` with its own HPU version. For a boolean
placeholder mask it calls

```python
inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1), mm_embeds_flat)
```

`unsqueeze(-1)` makes a stride-0 broadcast mask, which the Gaudi graph compiler cannot lower at the
sizes that concurrent multi-image steps reach. `inputs_embeds` holds the whole step, so the size
grows with the number of images per scheduler step, not per request, and
`--limit-mm-per-prompt` only makes it less likely.

## Symptom

The engine dies shortly after image traffic starts, with a graph compile failure such as
`Graph compile failed ... masked_scatter_ ... synStatus 26 [Generic failure]`. Single-image tests
pass.

## Who is affected

Any multimodal model served on HPU with several image (or video) requests in flight.

## The fix

The boolean mask is turned into row indices with `nonzero`, and the embeddings are copied with
`index_copy_`, the op the function's integer-index branch already uses because it compiles on HPU.
Row placement is the same, so the result is bit-identical to `masked_scatter_`. A count mismatch
still raises the same `ValueError`. `nonzero` has a data-dependent output shape, so one graph is
compiled per distinct number of multimodal rows, which costs a few percent of throughput on
image-heavy traffic.

## Verify

```bash
python3 fixes/mm-merge-idxcopy/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/mm-merge-idxcopy
```

End to end: send several multi-image requests concurrently (for example 8 requests with 2 images
each, repeated for a few minutes); without the fix the engine stops with the compile error above.
