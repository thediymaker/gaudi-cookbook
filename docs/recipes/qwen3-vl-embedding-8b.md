# Qwen3-VL-Embedding-8B

Qwen's 8B multimodal embedding model (text, images and screenshots in one 4,096-dimension space,
Matryoshka truncation), served through vLLM's pooling runner in BF16 on a single Gaudi 2 card.
It needs one fix that is not in the release profile, `mm-pooling-inputs`, and a chat template
that ends every prompt with the token the model card pools.

| | |
| --- | --- |
| Model | [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | BF16 (16.3 GB of weights) |
| Context | 32,768 tokens per input (262,144 native) |
| Concurrency | `--max-num-seqs 8` |
| Endpoint | `/v1/embeddings` with `messages` |
| Recipe | [`recipes/Qwen/Qwen3-VL-Embedding-8B/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-VL-Embedding-8B/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
export VLLM_SKIP_WARMUP=true
export VLLM_MULTIMODAL_BUCKETS=None
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 mm-pooling-inputs || exit 1

vllm serve Qwen/Qwen3-VL-Embedding-8B \
  --tensor-parallel-size 1 \
  --runner pooling \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --chat-template "$COOKBOOK/recipes/Qwen/Qwen3-VL-Embedding-8B/embed.jinja" \
  --limit-mm-per-prompt '{"image":32,"video":0}' \
  --hf-overrides '{"is_matryoshka": true, "matryoshka_dimensions": [64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]}'
```

- `--runner pooling` serves the checkpoint's `Qwen3VLForConditionalGeneration` as an embedding
  model with last-token pooling and L2 normalization, as the model card's sentence-transformers
  config asks.
- [`embed.jinja`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-VL-Embedding-8B/embed.jinja)
  is the checkpoint's own chat template with one change at the end: every prompt ends with
  `<|im_start|>assistant\n<|endoftext|>`, the token the model card pools. Without it a `messages`
  request ends at `<|im_end|>\n` (chat requests to `/v1/embeddings` default to no generation prompt
  and no special tokens), so the pooled vector is the newline's.
- The Matryoshka list accepts `dimensions` of 64 to 4,096 in the listed steps.
- Pooling models cannot chunk prefill, and prefix caching is off. `VLLM_MULTIMODAL_BUCKETS=None`
  leaves image shapes unbucketed.

## Required fixes

All from [Fixes](../fixes/index.md). `--profile v0.30.0` applies the release's generic fixes;
the opt-in fix is named in the launch command.

| Fix | Why this model needs it |
| --- | --- |
| [`mm-merge-idxcopy`](../fixes/mm-merge-idxcopy.md) | image embeddings are merged into the prompt by row index; the broadcast `masked_scatter_` fails to compile under concurrent images |
| [`mm-pooling-inputs`](../fixes/mm-pooling-inputs.md) (opt-in) | stock v0.30.0 stops the engine on the first request (`AttributeError: 'Qwen3VLForEmbedding' object has no attribute 'model'`); past that, the pooling path never runs the vision encoder (image placeholders are embedded as text) and uses 1D positions instead of M-RoPE |

## Client usage

Send inputs as chat `messages`, as the model card does. The system message is the task
instruction for queries and `Represent the user's input.` for documents; the user message holds
text, images, or both.

```json
{"model": "Qwen/Qwen3-VL-Embedding-8B",
 "messages": [
   {"role": "system", "content": [{"type": "text", "text": "Retrieve images that match the user's query."}]},
   {"role": "user", "content": [{"type": "text", "text": "two cats sleeping on a pink couch"}]}]}
```

An image document is a user message with an `image_url` part. Add `"dimensions": 256` (or another
listed size) for a truncated, renormalized vector.

## Verified

On vllm-gaudi `releases/v0.30.0`, one Gaudi 2 card, the launch above.

| Check | Result |
| --- | --- |
| Stock image without `mm-pooling-inputs` | engine stopped on the first request (`AttributeError`, then `EngineDeadError`) |
| Served vectors vs a CPU transformers reference (FP32, last token of the model card's prompt) | text 0.9961 and 0.9935; image 0.9885 (COCO photo) and 0.9915 (image plus text) cosine |
| COCO val2017 39769 (two cats on a pink couch) vs captions | matching caption 0.69, unrelated caption 0.13 |
| Four different images | mean pairwise cosine 0.169 (the broken path collapses every image onto nearly one vector) |
| Flickr30k 1k test, first 300 images (1,500 captions) | caption to image R@1 81.5, image to caption R@1 96.0 |
| SciFact nDCG@10 (instruction as system message) | 78.9 |
| `dimensions: 256` | 256-value vector |

## Performance

Single measurements on one card, `releases/v0.30.0`, warm, one request per input.

| Load | Result |
| --- | --- |
| Text (about 30 tokens), 1 request | 82 ms p50 |
| Text, 64 requests at 8 concurrent | 4.8 s (76 ms per input) |
| Distinct 640x480 images, 1 request | 132 ms p50 |
| Distinct 640x480 images, 64 requests at 8 concurrent | 7.2 s (113 ms per input) |
| SciFact corpus, 5,183 documents at 8 concurrent | 447 s (86 ms per document) |

Pooling prefill runs one sequence at a time on this image (see
[Qwen3-Embedding-8B](qwen3-embedding-8b.md#known-issues)), so concurrency does not raise
throughput much. Scale out with replicas.

## Known issues

- A plain-string `input` gets no chat wrapping, and it is tokenized with one more special token
  than the same text written out in the template, so its vector differs slightly (cosine 0.994 in
  one check) from the `messages` form. Pick one form per index; the model card uses messages.
- Video inputs are not enabled (`"video":0`) and were not tested.

## What not to do

- Do not serve without `embed.jinja` and rely on clients to send `add_generation_prompt` and
  `add_special_tokens`: a client that forgets gets vectors pooled on the wrong token, with no error.
- Do not mix vectors from this model with Qwen3-Embedding-8B in one index; the spaces differ.
