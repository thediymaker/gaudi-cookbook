# Qwen3-Coder-Next-FP8

Qwen's block FP8 release of Qwen3-Coder-Next (80B hybrid MoE: 36 Gated DeltaNet and 12 attention
layers, 512 experts, 3B active) with tool calling, at its native 256K context on two Gaudi 2 cards.
With the [`fp8-moe-gather`](../fixes/fp8-moe-gather.md) fix a decode step reads only the routed
experts, so it decodes several times faster per stream than [the BF16 recipe](qwen3-coder-next.md)
on half the cards. Prefix caching is on through the [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) fix.

| | |
| --- | --- |
| Model | [Qwen/Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) |
| Hardware | 2x Gaudi 2 (TP 2) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (block 128 x 128, dynamic activations; 80.4 GB) |
| Context | 262,144 (native) |
| Concurrency | `--max-num-seqs 8` |
| Parsers | tools `qwen3_coder` |
| Recipe | [`recipes/Qwen/Qwen3-Coder-Next-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3-Coder-Next-FP8/recipe.yaml) |

## Prefix caching

As in the [BF16 recipe](qwen3-coder-next.md#prefix-caching): prefix caching on the Gated DeltaNet
layers is only correct with [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md). At TP 2 the GDN state per
card is twice the TP 4 size, so the hybrid KV block is 640 tokens (384 in the BF16 recipe): prefill chunks end on
640-token blocks, and prompts shorter than one block never hit.

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=1
export VLLM_COMPACT_GDN=1
export VLLM_HPU_MOE_GATHER=1
export VLLM_SKIP_WARMUP=true
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 gdn-prefix-cache fp8-moe-gather || exit 1

vllm serve Qwen/Qwen3-Coder-Next-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

- TP 2: the FP8 weights take 38.7 GB per card, which leaves room for a KV pool of 3,675,057
  tokens (the BF16 recipe has 3,800,136 on four cards).
- `VLLM_HPU_MOE_GATHER=1` with [`fp8-moe-gather`](../fixes/fp8-moe-gather.md): while
  `tokens x top_k <= 512 x 0.4` (the default `VLLM_HPU_MOE_GATHER_RATIO`), the MoE layer gathers the
  routed experts and runs them as one FP8 GEMM instead of reading all 512. With top 10 routing that
  covers every decode step at 8 sequences (80 routes). Larger calls, such as prefill chunks, keep the
  stock op.
- `--max-num-batched-tokens 4096`, half the BF16 recipe's: a prefill chunk takes twice as long on two
  cards as on four, and smaller chunks keep decode steps between them short. With prefix caching on,
  chunks end on 640-token blocks (6 blocks, 3,840 tokens).
- Compile mode, no `VLLM_*_BUCKET_*` variables and `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`, as in the
  BF16 recipe and for the same reasons.

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
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | at 4,096 batched tokens the stock path builds fp32 logits for a whole chunk at once (about 2.5 GB per chunk with a 152K vocabulary); the fix row-chunks them |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id from any source is replaced instead of stopping the API server |
| [`fp8-moe-gather`](../fixes/fp8-moe-gather.md) (opt-in) | backport of v0.30.0's gathered-expert FP8 MoE: with `VLLM_HPU_MOE_GATHER=1` a decode step reads only the routed experts instead of all 512 |
| [`rope-cpu-cache`](../fixes/rope-cpu-cache.md) | Gaudi fp32 cos/sin are wrong past about 1.5e5, so RoPE tables built on the HPU are wrong from position about 196K |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of serving from out-of-step ranks |

## Verified

On vllm-gaudi v0.26.0, two Gaudi 2 cards, with exactly the launch command above in a container
started from the stock image (the fixes applied at start-up). The checkpoint is Qwen's own
`Qwen/Qwen3-Coder-Next-FP8` (revision `da6e2ed`). Requests went to the OpenAI-compatible API through
`ports/kimi-k2-7/tests/hpu/kimi_gates.py` (known answers, tools, needles, logprobs soak,
teacher-forced scoring, speed) and a few direct requests.

| Check | Result |
| --- | --- |
| Known-answer prompts, greedy, 12 | 11/12: "reverse the string 'gaudi'" answered `idaug` |
| Tool call, `tool_choice: auto`, greedy, streamed and not, 6 runs | 6/6 `get_weather` with `{"city": "Lisbon"}` |
| `response_format` JSON schema | valid object |
| `logit_bias` +100 on one token | takes effect (4/4 tokens) |
| `top_logprobs: 20` soak, 200 requests with CJK and emoji output, 4 in flight | 200/200 complete, no dead positions |
| `prompt_logprobs`, 9,146 positions of code (teacher-forced scoring) | returned for every position; NLL 1.3765 (recorded) |
| Prefix-cache hit vs cold (`cache_salt`), greedy, 4 prompts of 6K to 118K tokens with a needle | identical text and bit-identical logprobs 4/4; cached tokens a multiple of 640 on every hit |
| Needle in a haystack, 3 depths, cold | 9/9 at 65K, 131K and 244K tokens |
| Concurrent decodes (the batch-collapse case) | logprobs soak at 4 in flight clean; sweeps at 4 and 8 streams without errors |
| Server health after all checks | healthy, 0 tracebacks |

KV pool at start-up: 3,675,057 tokens.

## Performance

Single runs on the server above. Measured as: two cards, `kimi_gates.py sweep` (2 x C streamed
completions of 256 tokens with `ignore_eos` on a cached 1,000-token prefix, temperature 0.6; per
stream is 1 / median time per output token, aggregate is all tokens over the wall time), second of
two sweeps (the first one compiled the decode shapes it met: 50.2 / 39.5 / 18.2 tok/s per stream).

| Streams | Per stream | Aggregate |
| --- | --- | --- |
| 1 | 50.9 tok/s | 47.8 tok/s |
| 4 | 43.0 tok/s | 141.9 tok/s |
| 8 | 29.6 tok/s | 182.3 tok/s |

| Test | Result |
| --- | --- |
| Uncached 32K prompt, time to first token, warm (median of 3) | 5.8 s (108 s for the first, which compiled its shapes) |
| Cold needle prompt of 65K / 131K / 244K tokens, end to end | 14 to 121 s / 33 to 136 s / 189 to 388 s (the slow ones compiled new shapes) |
| Weight load from network storage | 48 minutes in this run (38.7 GB per card) |

The [BF16 recipe](qwen3-coder-next.md) measured 12.7 tokens/s for one stream and 11.6 per stream at 8
(end to end, another tool, four cards), so the two tables do not compare exactly, but FP8 with the
gathered experts is about 2.5 (8 streams) to 4 (1 stream) times faster per stream on half the cards. For context only, a
separate measurement of this checkpoint at TP 2 with another tool scored 98.1% teacher-forced top-1
agreement with the BF16 model.

## Known issues

- Every new shape compiles on first use (`VLLM_SKIP_WARMUP=true`), and in compile mode one shape can
  take minutes on a fresh node: the first long prompts and the first batch-8 decode steps after a
  start are slow. Keep the recipe cache (`PT_HPU_RECIPE_CACHE_CONFIG=<dir>,False,<size>`) on the node
  so that a restart gets compiled shapes back, and warm a new server with a few requests of the
  expected shapes before it takes traffic.
- Validate with 2 or more concurrent decodes, as for the BF16 recipe.
- Prompts shorter than one 640-token block never hit the cache.

## What not to do

- Do not use lazy mode.
- Do not turn on prefix caching without `gdn-prefix-cache`, and do not set `VLLM_COMPACT_GDN=0`
  with it.
- Do not raise `--max-num-seqs` above 8 without checking the gather limit: past
  `tokens x 10 <= 512 x VLLM_HPU_MOE_GATHER_RATIO`, decode steps go back to the stock op, which reads
  every expert.
- Do not serve this checkpoint with `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,True,<size>` on a shared node
  cache: `True` wipes the cache at every start, so every restart compiles from zero.
