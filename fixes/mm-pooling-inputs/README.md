# mm-pooling-inputs

The pooling (embedding) runner runs the vision encoder and uses M-RoPE positions, so multimodal
embedding models such as Qwen3-VL-Embedding embed images instead of crashing or ignoring them.

| | |
| --- | --- |
| Affects | multimodal pooling models (`--runner pooling`), M-RoPE models in particular |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | not filed yet |
| Test | `test_mm_pooling_inputs.py` (CPU, needs a release source tree in `GAUDI_FIXES_ROOT`) |

## Problem

Two defects in the same runner:

- `_update_states` computes M-RoPE positions through `self.model.model.get_mrope_input_positions`.
  Pooling wrappers such as `Qwen3VLForEmbedding` expose that method on the model itself and have no
  `.model`, so the first request raises `AttributeError`.
- The pooling branch of `execute_model` calls `self.model.forward(input_ids=..., positions=...)`
  only. It never runs the vision encoder or merges image embeddings, so image placeholder tokens are
  embedded as text, and it passes 1D positions to M-RoPE models.

## Symptom

On the first `/v1/embeddings` request:
`AttributeError: 'Qwen3VLForEmbedding' object has no attribute 'model'`, then `EngineDeadError`.
With only the first defect worked around, every image lands on nearly the same vector (mean pairwise
cosine about 0.97, measured on an older release) and image retrieval is at chance level.

## Who is affected

Any multimodal model served with `--runner pooling` on HPU. Text-only pooling models without M-RoPE
are unaffected.

## The fix

`get_mrope_input_positions` is called on whichever of `self.model` and `self.model.model` has it.
In the pooling branch, a request uses its M-RoPE positions when the model uses them, and a request
with media builds `inputs_embeds` through `_get_model_mm_inputs`, the generation path's encoder and
merge.

## Verify

```bash
python3 fixes/mm-pooling-inputs/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/mm-pooling-inputs
```

End to end on Qwen3-VL-Embedding-8B, one Gaudi 2 card, `releases/v0.30.0`: the stock image stops on
the first request; with the fix, served vectors match a CPU FP32 transformers reference at cosine
0.9935 to 0.9961 (text) and 0.9885 to 0.9915 (images), and Flickr30k (first 300 images) caption to
image R@1 is 81.5, image to caption 96.0.
