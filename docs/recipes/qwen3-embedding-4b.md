# Qwen3-Embedding-4B

Qwen's 4B instruction-aware text embedding model (2,560 dimensions, Matryoshka truncation), served
through vLLM's pooling runner in BF16 on a single Gaudi 2 card. Stock image, no fixes.

| | |
| --- | --- |
| Model | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | BF16 (8 GB of weights) |
| Context | 32,768 tokens per input |
| Concurrency | `--max-num-seqs 64` |
| Endpoint | `/v1/embeddings` |
| Recipe | [`recipes/Qwen/Qwen3-Embedding-4B/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-Embedding-4B/recipe.yaml) |

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

vllm serve Qwen/Qwen3-Embedding-4B \
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
  `dimensions` value up to 2,560.
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

On vllm-gaudi v0.26.0, one Gaudi 2 card, with exactly the launch command above in a container
started from the stock image, and the documented query instruction. Retrieval scores are nDCG@10,
STS is Spearman; values in parentheses are at 256 dimensions.

| Task | Score |
| --- | --- |
| SciFact | 78.1 (73.9) |
| NFCorpus | 40.9 (35.5) |
| ArguAna | 73.5 (70.3) |
| CodeSearchNet python | 95.9 (95.0) |
| Wikipedia retrieval de / hi / pt | 89.7 / 86.8 / 87.5 |
| QMSum long documents | 51.2 (44.9) |
| STSBenchmark | 93.6 (93.6) |

## Performance

Single runs on the server above (measured as: per-call, one server, warm, inputs of about 20
tokens, best of 3).

| Load | Result |
| --- | --- |
| 1 input, 1 request | 58 ms |
| 32 inputs in one request | 1.69 s (53 ms per input) |
| 256 inputs in one request | 13.5 s (53 ms per input) |

## Known issues

- Cost per input is flat at about 50 ms however the work arrives, the same as for
  [Qwen3-Embedding-8B](qwen3-embedding-8b.md): pooling prefill runs one sequence at a time on this
  image, so the smaller model is not faster per input. Scale out with replicas.

## What not to do

- Do not mix vectors from this model (2,560 dimensions) and from Qwen3-Embedding-8B (4,096) in one
  index; truncate both to a common Matryoshka width if you must.
- Do not add an instruction prefix to documents.
