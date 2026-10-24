# Qwen3-Embedding-8B

Qwen's 8B instruction-aware text embedding model (4,096 dimensions, Matryoshka truncation), served
through vLLM's pooling runner in BF16 on a single Gaudi 2 card. Stock image, no fixes.

| | |
| --- | --- |
| Model | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | BF16 (15 GB of weights) |
| Context | 32,768 tokens per input |
| Concurrency | `--max-num-seqs 64` |
| Endpoint | `/v1/embeddings` |
| Recipe | [`recipes/Qwen/Qwen3-Embedding-8B/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-Embedding-8B/recipe.yaml) |

## Launch

```bash
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=true
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=64
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=1024
export VLLM_PROMPT_SEQ_BUCKET_MAX=32768
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

vllm serve Qwen/Qwen3-Embedding-8B \
  --tensor-parallel-size 1 \
  --runner pooling \
  --convert embed \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --hf-overrides '{"is_matryoshka": true}'
```

- `--runner pooling --convert embed` maps the plain `Qwen3ForCausalLM` checkpoint onto
  `Qwen3ForEmbedding` at load and installs last-token pooling with L2 normalization, which is what
  the model card's sentence-transformers config asks for. No pooler override is needed.
- `--hf-overrides '{"is_matryoshka": true}'` without a `matryoshka_dimensions` list accepts any
  `dimensions` value up to 4,096.
- Pooling models cannot chunk prefill, and prefix caching is off.

### Buckets

No bucket file. Pooling is prefill-only, so the prompt buckets are the whole HPU graph space and are
bounded explicitly: batch 1 to 64, sequence 128 to 32,768 in steps of 1,024, exponential bucketing.

## Required fixes

None. The model has no vision tower, no MRoPE and no sampling, so none of the runtime fixes apply.

## Client usage

The model is asymmetric. vLLM does not apply sentence-transformers prompt templates, so the client
adds the query instruction; documents go in bare.

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:{query}
```

Embedding queries without the prefix, or documents with it, measurably lowers retrieval quality.

## Verified

On vllm-gaudi v0.26.0, one Gaudi 2 card, the flags above, with the documented query instruction.
Retrieval scores are nDCG@10, STS is Spearman; values in parentheses are at 256 dimensions.

| Task | Score |
| --- | --- |
| SciFact | 78.4 (76.1) |
| NFCorpus | 41.3 (36.5) |
| ArguAna | 74.3 (72.5) |
| CodeSearchNet python | 96.2 (95.6) |
| Wikipedia retrieval de / hi / pt | 91.3 / 88.0 / 90.0 |
| QMSum long documents | 54.6 (49.0) |
| STSBenchmark | 93.5 (93.5) |

## Performance

Single measurements on one card, vllm-gaudi v0.26.0, warm.

| Load | Result |
| --- | --- |
| 1 input (about 30 tokens), 1 request | 58 ms |
| 32 inputs in one request | 1.61 s (50 ms per input) |
| 256 inputs in one request | 13.6 s (53 ms per input) |
| 32 concurrent single-input requests | 1.65 s wall (52 ms per input) |
| Retrieval passages, sustained | about 20 passages/s, 4,990 tokens/s; single query p50 56.8 ms |

## Known issues

- Cost per input is flat at about 50 ms however the work arrives: pooling prefill runs one sequence
  at a time on this image, so batching gives no gain. The batched prompt buckets are generated, so
  this is not a bucketing setting. Scale out with replicas.
- `VLLM_MERGED_PREFILL=true` was tried and is not faster (256 inputs in 14.5 s vs 13.6 s).

## What not to do

- Do not mix vectors from this model and from Qwen3-Embedding-4B (2,560 dimensions) in one index;
  truncate both to a common Matryoshka width if you must.
- Do not add an instruction prefix to documents.
