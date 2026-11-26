# swa-window-context

Source: [`fixes/swa-window-context/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/swa-window-context)

Sliding-window layers of every interleaved model read only their window of context in chunked prefill, not
only Gemma 4's. Opt-in.

| | |
| --- | --- |
| Affects | interleaved sliding-window models other than Gemma 4 (for example Cohere2 / North-Mini-Code, Muse Glimmer) with long prompts, chunked prefill or prefix caching |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | not filed yet |
| Test | `test_swa_window_context.py` (stdlib; the source tests need a release's sources) |

## Problem

A prefill chunk that has cached context (the second and later chunks of a long prompt, or a prefix-cache hit)
gathers the context's KV blocks and runs fused SDPA over context plus chunk. For the sliding-window layers the
runner can pass only the last `sliding_window // block_size + 1` context blocks (`window_block_list`), and the
mask builder re-anchors the window for that list. The runner builds the list only when `model_type` is
`gemma4`:

```python
if self.interleaved_sliding_window and self._get_model_type() == "gemma4":
```

Every other interleaved model gathers the whole context on its sliding layers and masks it with a window as
wide as the context, on every chunk. A long prompt then costs as much as if every layer were global, and the
context-wide window mask is a path the Gemma 4 work never exercised.

## Symptom

Long prompts are much slower than the model's share of global layers suggests, and the server serves nothing
else while one prefills. Time to first token grows roughly with the square of the prompt length on every
layer instead of only on the global ones.

On Muse Glimmer (39 sliding layers with a 2,048-token window, 13 global) on v0.30.0 the full-context path
also broke the output: a 120K-token prompt returned a repeated word, every later request on that server
returned garbage, and the engine then failed with `PT_DEVMEM Allocation failed`. The
[muse-glimmer-30b](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/muse-glimmer-30b) port widens the same gate for that model only.

## Who is affected

Models whose config has interleaved sliding-window and global layers (`layer_types` with
`sliding_attention`) and whose `model_type` is not `gemma4`, when prompts are longer than the window and are
prefilled in more than one chunk or hit the prefix cache. Short prompts (one chunk, no cached context) take a
different path and are unchanged, and so is decode, which already uses a window block table for every
interleaved model.

## The fix

The condition on `model_type` is dropped at the two places that build the list: where it is sliced from each
request's context blocks, and where it is padded and copied to the device for the prefill metadata. The
attention backend and the mask builder already take the list for any model. If a model port already widened
the gate to an explicit list of model types (the muse-glimmer-30b port), that form is replaced the same way,
and the port patch accepts the fix in either order.

The fix is opt-in: it changes the prefill shapes and the attention inputs of every interleaved model, and it
has been checked on the models below only. gpt-oss is interleaved too (with attention sinks) and has not been
checked with it.

## Measured

North-Mini-Code-1.0-fp8 (Cohere2 MoE, 37 of 49 layers with a 4,096-token window) on one Gaudi 2 card,
`releases/v0.30.0` in lazy mode with the deployment's other patches, `--max-num-seqs 32`, prefix caching on,
2,048-token chunks, warm, fresh random text (no prefix-cache help). Single runs:

| | Without the fix | With the fix |
| --- | --- | --- |
| 12K-token prompt, time to first token | 2.8 s | 2.9 s |
| 48K-token prompt | 27.2 s | 14.2 s |
| 150K-token prompt | 250 s | 81 s |
| decode per stream at 1 / 8 / 32 streams | 78 / 21 / 23 tokens/s | 78 / 22 / 24 tokens/s |
| 8 short requests arriving behind two ~100K prompts, first token | 257 s | 108 s |

Greedy answers were identical to the runner without the fix on 10K, 40K and 100K-token retrieval prompts
(codes hidden at five depths), fresh, from the prefix cache and as a follow-up turn, and chosen-token
logprobs stayed within 0.06. The Muse Glimmer gate in its port is the same change for one model; with it,
120K and 127K-token needles pass on v0.30.0. The v0.26.0 runner has the same two sites and the same mask
builder; the fix was not run on a device with v0.26.0.

## Verify

```bash
python3 fixes/swa-window-context/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/swa-window-context
```

On hardware (needs a device): with your model, compare greedy answers and teacher-forced logprobs on prompts of
several times the window, prefilled in several chunks, fresh and from the prefix cache, with and without the
fix, and compare time to first token at the longest length you serve.
