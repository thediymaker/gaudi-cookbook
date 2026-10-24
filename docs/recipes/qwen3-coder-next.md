# Qwen3-Coder-Next

Qwen's 80B hybrid MoE coding model (36 Gated DeltaNet and 12 attention layers, 512 experts, 3B
active) in BF16, with tool calling, at its native 256K context on four Gaudi 2 cards. Prefix
caching is on through the [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) fix.

| | |
| --- | --- |
| Model | [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) |
| Hardware | 4x Gaudi 2 (TP 4) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | BF16 (159 GB of weights) |
| Context | 262,144 (native) |
| Concurrency | `--max-num-seqs 8` |
| Parsers | tools `qwen3_coder` |
| Recipe | [`recipes/Qwen/Qwen3-Coder-Next/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-Coder-Next/recipe.yaml) |

## Prefix caching

On the stock release, prefix-cache hits on the Gated DeltaNet layers resume from the wrong state,
so prefix caching is only correct with [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md). The fix
keeps the GDN layers in their compact per-request slots (`VLLM_COMPACT_GDN=1`) and stores a state
checkpoint at the end of every full KV block in that block's own attention page. A hit restores the
checkpoint before the next chunk, and the full KV pool stays available. With caching on, the hybrid
KV block is 384 tokens and prefill chunks end on block boundaries; prompts shorter than one block
never hit. The design and the upstream status are in
[Prefix caching on Gated DeltaNet hybrids](../results/gdn-prefix-caching.md).

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=1
export VLLM_COMPACT_GDN=1
export VLLM_SKIP_WARMUP=true
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 gdn-prefix-cache || exit 1

vllm serve Qwen/Qwen3-Coder-Next \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

- Compile mode (`PT_HPU_LAZY_MODE=0`) is required: in lazy mode the first GDN decode fails
  (`ValidateSyncInputTensors`) at TP 4 and TP 2 alike, whatever the other settings.
- No `VLLM_*_BUCKET_*` variables: they are lazy-mode graph-bucketing knobs. In compile mode
  torch.compile owns the shapes, and setting them cost throughput.
- TP 4 rather than TP 2: at TP 2 the BF16 weights leave about 12 GiB per card, and 16 concurrent
  requests at long context stopped the engine with `PT_DEVMEM OOM`.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`: with warm-up skipped every new shape compiles inside
  the model call, and the 300 s default stops the engine mid-compile. This, not
  `VLLM_RPC_TIMEOUT`, is the timeout that governs it.
- With prefix caching on, prefill chunks end on 384-token blocks: the 8,192-token budget runs as
  21 blocks (8,064 tokens) per chunk.

### Buckets

No bucket file and no bucket variables (compile mode).

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.26.0`, which also applies the release's other generic fixes (see
[Using a recipe](index.md#using-a-recipe)). The profile also runs `seeded-sampling`; this recipe
does not depend on it (the seed segfault is a lazy-mode bug).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | with prefix caching, a request that hits a block another request writes in the same step read it before it was written |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits samples again at every recomputed position, so the generated tail repeats |
| [`hybrid-decode-pad`](../fixes/hybrid-decode-pad.md) | when the hybrid KV block is split into 128-token kernel blocks, decode padding writes into a live block of another request |
| [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) (opt-in) | prefix-cache hits on the GDN layers resume from a wrong state on the stock release; the fix stores and restores GDN checkpoints in the KV pages and zeroes new blocks and slots. It needs `hybrid-decode-pad` first |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | the id corruption is lazy-mode only, but the fix is harmless here and keeps one fix set across recipes |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | at 8,192 batched tokens the stock path builds fp32 logits for a whole chunk at once (about 5 GB per chunk with a 152K vocabulary); the fix row-chunks them |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id from any source is replaced instead of stopping the API server |
| [`rope-cpu-cache`](../fixes/rope-cpu-cache.md) | Gaudi fp32 cos/sin are wrong past about 1.5e5, so RoPE tables built on the HPU are wrong from position about 196K |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of serving from out-of-step ranks |

## Verified

On vllm-gaudi v0.26.0, four Gaudi 2 cards, with exactly the launch command above in a container
started from the stock image (the fixes applied at start-up). Requests went to the
OpenAI-compatible API.

| Check | Result |
| --- | --- |
| Known-answer prompts (arithmetic, facts, translation), temperature 0 | 8/8 |
| The same prompts, 8 in flight | 8/8 |
| Tool call, `tool_choice: auto`, sampled at 0.6 | 5/5 valid `get_weather` calls with the right argument |
| Streamed tool call | name and JSON arguments assembled correctly |
| Tool result fed back | the answer uses the returned values |
| `top_logprobs: 20` | 20 entries on every token |
| `prompt_logprobs: 5` | returned; server healthy afterwards |
| `response_format` JSON schema | valid object |
| `logit_bias` +100 on one token | takes effect |
| Prefix-cache hit vs cold (`cache_salt`), greedy, 8 prompts of 6K to 126K tokens with a needle | identical text and bit-identical logprobs 8/8; cached tokens a multiple of 384 on every hit |
| Needle in a haystack, 3 depths, cold | 3/3 at 65K, 131K and 244K tokens |
| Server health after all checks | healthy, no restarts |

KV pool at start-up: 3,800,136 tokens.

## Performance

Single runs on the server above (measured as: per-call, one server; times are end to end and
include time to first token).

| Test | Result |
| --- | --- |
| One stream, short prompt, 256 tokens | 12.7 tokens/s |
| 8 streams, short prompts, 256 tokens each | 92.4 tokens/s aggregate, 11.6 per stream |
| Cold prompt of 65K tokens (needle) | 6 s (53 s for one that compiled a new shape) to the answer |
| Cold prompt of 131K tokens (needle) | 152 to 196 s (new shapes compiling on a fresh node) to the answer |
| Cold prompt of 244K tokens (needle) | 144 to 199 s to the answer |

## Known issues

- Decode is slow for a 3B-active model: the BF16 MoE op reads every expert's weights at every
  step.
- Validate with 2 or more concurrent decodes. v0.26.0 fixes a batch-collapse bug for this
  architecture that only shows at batch 2 and above; a single request passes even without it.
- The first request at a new shape on a fresh node compiles its graphs and can take minutes.

## What not to do

- Do not use lazy mode.
- Do not set `VLLM_USE_HYBRID_CACHE` by hand: v0.26.0 enables it for GDN models on its own.
- Do not move to TP 2 for a general-purpose 256K configuration.
- Do not turn on prefix caching without `gdn-prefix-cache`, and do not set `VLLM_COMPACT_GDN=0`
  with it.
