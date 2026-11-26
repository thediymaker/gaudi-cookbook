# Qwen3.8-27B-FP8

Qwen's block FP8 release of Qwen3.8-27B (48 Gated DeltaNet layers and 16 attention layers, with
vision and reasoning) at its native 256K context on a single Gaudi 2 card. The FP8 weights leave
room for a KV pool more than twice the BF16 recipe's, and this recipe runs 32 sequences per card
where [the BF16 recipe](qwen3-8-27b.md) stops at 4. Prefix caching is on through the
[`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md) fix.

| | |
| --- | --- |
| Model | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (Apache 2.0) |
| Hardware | 1x Gaudi 2 (TP 1) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (block 128 x 128, dynamic activations; 30.9 GB) |
| Context | 262,144 (native) |
| Concurrency | `--max-num-seqs 32` |
| Parsers | tools `qwen3_coder`, reasoning `qwen3` |
| Recipe | [`recipes/Qwen/Qwen3.8-27B-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/Qwen/Qwen3.8-27B-FP8/recipe.yaml) |

## Prefix caching

As in the [BF16 recipe](qwen3-8-27b.md#prefix-caching): prefix caching on the Gated DeltaNet
layers is only correct with [`gdn-prefix-cache`](../fixes/gdn-prefix-cache.md), which keeps the
compact GDN slots (`VLLM_COMPACT_GDN=1`) and stores a state checkpoint in each full KV block. The
hybrid KV block is 896 tokens, so a 1,792-token batch is two blocks, and prompts shorter than one
block never hit.

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=1
export VLLM_COMPACT_GDN=1
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=8
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=32
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=4096
export VLLM_PROMPT_SEQ_BUCKET_MAX=262144
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=16384
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 gdn-prefix-cache || exit 1

vllm serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --max-num-batched-tokens 1792 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":16,"video":0}'
```

- The same geometry and settings as [Qwen3.6-27B-FP8](qwen3-6-27b-fp8.md): 32 sequences at
  `--gpu-memory-utilization 0.85` and 1,792 batched tokens. The KV pool is 623,359 tokens
  (288,446 in BF16), so two full 262,144-token requests fit at once.
- vllm-gaudi picks its block FP8 path from the checkpoint's `quantization_config`; no
  `--quantization` flag is needed. The vision tower is not quantized in the checkpoint.
- Compile mode (`PT_HPU_LAZY_MODE=0`) with the compact GDN state layout that `gdn-prefix-cache`
  needs.
- `--default-chat-template-kwargs '{"reasoning_effort": "medium"}'`: the template defaults to
  `xhigh`, which can run past `max_tokens` with empty `content`. Clients override it per request
  with `chat_template_kwargs`.
- The BF16 recipe's limit of 4 sequences at 0.60 does not apply here: this launch ran 32 streams
  and 4 concurrent 128K prompts without a stop or a preemption (see [Verified](#verified)).

### Buckets

No bucket file. Prompt batch buckets 1 to 8, decode batch buckets 1 to 32, prompt length 128 to
262,144 in steps of 4,096 with exponential bucketing, decode blocks 128 to 16,384 in steps of 1,024.

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
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | multimodal model with one prompt in a prefill batch returns flat hidden states; the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id from any source is replaced instead of stopping the API server |
| [`rope-cpu-cache`](../fixes/rope-cpu-cache.md) | Gaudi fp32 cos/sin are wrong past about 1.5e5, so RoPE tables built on the HPU are wrong from position about 196K |


## Verified

On vllm-gaudi v0.26.0, one Gaudi 2 card, with exactly the launch command above in a container
started from the stock image (the fixes applied at start-up). The checkpoint is Qwen's own
`Qwen/Qwen3.8-27B-FP8` (revision `c3a462c`). Requests went to the OpenAI-compatible API through
`ports/kimi-k2-7/tests/hpu/kimi_gates.py` (known answers, tools, needles, logprobs soak,
teacher-forced scoring, speed) and a few direct requests.

| Check | Result |
| --- | --- |
| Known-answer prompts, greedy, 12 | 11/12: "reverse the string 'gaudi'" came back `iduaG` (BF16 `iduag`); 11/12 answers word for word the same as BF16 |
| Tool call, `tool_choice: auto`, greedy, streamed and not, 6 runs | 6/6 `get_weather` with `{"city": "Lisbon"}` |
| `response_format` JSON schema | valid object |
| `logit_bias` +100 on one token | takes effect (4/4 tokens) |
| `top_logprobs: 20` soak, 200 requests with CJK and emoji output, 4 in flight | 200/200 complete, no dead positions |
| Teacher-forced top-1 agreement with the BF16 checkpoint, 9,531 positions of code (same pod, the [BF16 recipe](qwen3-8-27b.md)'s launch) | 97.29%; NLL 0.9508 against 0.9481 |
| 16 images of 448 x 448 in one prompt (`reasoning_effort: low`) | 16/16 colours named in order |
| Prefix-cache hit vs cold (`cache_salt`), greedy, 4 prompts of 6K to 118K tokens with a needle | identical text and bit-identical logprobs 4/4; cached tokens a multiple of 896 on every hit |
| Needle in a haystack, 3 depths, cold | 9/9 at 65K, 131K and 244K tokens |
| 4 needle prompts of 128K tokens at once | 4/4 found, 0 preemptions |
| Server health after all checks (three starts) | healthy, 0 tracebacks |

KV pool at start-up: 623,359 tokens (BF16: 288,446).

## Performance

Single runs on the server above, warm. Measured as: one card, `kimi_gates.py sweep` (2 x C
streamed completions of 256 tokens with `ignore_eos` on a cached 1,000-token prefix, temperature
0.6; per stream is 1 / median time per output token, aggregate is all tokens over the wall time).
The BF16 column is the [BF16 recipe](qwen3-8-27b.md)'s launch on the same pod and card, same tool.

| Streams | FP8, per stream (aggregate) | BF16, per stream (aggregate) |
| --- | --- | --- |
| 1 | 46.0 (44.6) tok/s | 35.6 (34.5) tok/s |
| 4 | 35.1 (131.5) tok/s | 29.8 (83.3) tok/s |
| 8 | 30.0 (212.5) tok/s | not possible (4 sequences) |
| 16 | 22.8 (312.4) tok/s | not possible |
| 32 | 17.9 (472.1) tok/s | not possible |

| Test | Result |
| --- | --- |
| Uncached 32K prompt, time to first token (median of 3) | 10.0 s |
| Cold needle prompt of 65K / 131K / 244K tokens, end to end | 27 to 30 s / 73 to 74 s / 217 to 218 s |
| Server start (weights from network storage, no warm-up) | 201 to 221 s to healthy |

For context only, not this checkpoint: a self-quantized block FP8 copy of the BF16 model (data-free,
the same 128 x 128 layout) measured 98.1% top-1 agreement with BF16 on a different scoring set, and
539 tokens/s aggregate at 32 streams with a 4K prompt and 768 output tokens on another benchmark
tool. Neither number was measured on Qwen's release or with the tool above.

## Known issues

- Prompts shorter than one 896-token block never hit the cache, and a hit resumes from the last
  full block before the end of the shared prefix.
- FP8 output is close to BF16, not identical: 97.3% top-1 agreement on code, and short answers can
  differ in detail (the capital letter above).
- Only `xhigh`, `medium` and `low` are legal `reasoning_effort` values for this template; `high`
  returns HTTP 400. The top-level OpenAI `reasoning_effort` field does not reach the template and is
  silently ignored: send `chat_template_kwargs`.
- The first request of each new shape compiles its graphs (`VLLM_SKIP_WARMUP=true`); expect slow
  first long prompts on a fresh node.

## What not to do

- Do not turn on prefix caching without `gdn-prefix-cache`, and do not set `VLLM_COMPACT_GDN=0`
  with it.
- Do not use lazy mode for this model.
- Do not raise `--max-num-batched-tokens` to 3,584 or more at 256K: on Qwen3.6-27B-FP8, which has
  the same geometry, a cold prefill past about 131K context then died with
  `PT_DEVMEM OOM: No enough memory for defragment`.
- Do not enable the checkpoint's MTP layer for speculative decoding: on the BF16 model it was
  measured as a net slowdown on Gaudi 2.
