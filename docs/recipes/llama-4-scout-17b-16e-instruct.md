# Llama 4 Scout 17B-16E Instruct

Meta's 109B mixture-of-experts multimodal model (17B active, 16 experts, top 1), served in BF16 at
128K context on four Gaudi 2 cards with prefix caching (images included) and a tool format the
model follows reliably.

| | |
| --- | --- |
| Model | [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) (gated) |
| Hardware | 4x Gaudi 2 (TP 4) |
| Image | vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | BF16 (217 GB of weights) |
| Context | 131,072 served (the checkpoint allows 10M) |
| Concurrency | `--max-num-seqs 16`, up to 4 images per prompt |
| Parsers | tools `llama4_lenient` (plugin from `llama4-tools-lenient`) |
| Recipe | [`recipes/meta-llama/Llama-4-Scout-17B-16E-Instruct/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/meta-llama/Llama-4-Scout-17B-16E-Instruct/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container. `llama4-tools-lenient` is not
applied as a patch: it ships the parser plugin and chat template that the flags point at.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export VLLM_SKIP_WARMUP=true
export VLLM_MULTIMODAL_BUCKETS=None
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0.5
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_HPU_MOE_WEIGHT_ON_INPUT=1
export LLAMA4_TOOL_CALL_POLICY=independent
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 \
    moe-router-weight-on-input llama4-tools-lenient || exit 1

vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 2048 \
  --max-cudagraph-capture-size 2048 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --chat-template "$COOKBOOK"/fixes/llama4-tools-lenient/llama4-tools.jinja \
  --default-chat-template-kwargs '{"fmt": "tag", "sys_mode": "meta", "defs": "meta"}' \
  --tool-parser-plugin "$COOKBOOK"/fixes/llama4-tools-lenient/llama4_tools_lenient.py \
  --tool-call-parser llama4_lenient \
  --limit-mm-per-prompt '{"image":4,"video":0}'
```

- The chat template renders tools in Meta's documented `<function=name>{json}</function>` form, puts
  Meta's tool rules first even when the client sends a system prompt, passes tool results raw and
  ends tool turns with `<|eom|>`. The kwargs `{"fmt": "tag", "sys_mode": "meta", "defs": "meta"}`
  select that behaviour.
- `LLAMA4_TOOL_CALL_POLICY=independent`: Scout tends to plan a whole task as one parallel list (read
  a file, edit text it has not seen, run the tests). The parser keeps only calls that cannot depend
  on an earlier call's result in the same turn, repairs schema-typed arguments (`"true"` to `true`,
  `null` optionals dropped), and reads every syntax Meta's decoder accepts.
- `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1`: Habana treats only the literal string `"1"` as true. Any
  other value, including `"false"`, overrides vllm-gaudi's default and turns the graph tensor
  cache on, which grows outside the graph memory budget until `PT_DEVMEM` failures.

### Buckets

No bucket file. Exponential bucketing with the image's default ranges;
`VLLM_MULTIMODAL_BUCKETS=None` leaves vision shapes unbucketed.

## Required fixes

All from [Fixes](../fixes/index.md). The ones marked opt-in are named in the launch command; the
rest come from `--profile v0.26.0`, which also applies the release's other generic fixes (see [Using
a recipe](index.md#using-a-recipe)).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching can read another request's KV in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption with prefix-cache hits samples again at every recomputed position, so the generated tail repeats |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | in lazy mode the `top_logprobs` ids come back corrupted; an out-of-range id reaching the detokenizer killed a replica |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | the first `prompt_logprobs` request kills the engine |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with `seed` can segfault a TP worker in lazy mode |
| [`moe-router-weight-on-input`](../fixes/moe-router-weight-on-input.md) (opt-in) | Llama 4 applies the top-1 router weight to the expert input (`expert(w * x)`); the HPU fused MoE applied it to the output (`w * expert(x)`), which differs through SwiGLU on every layer. Enabled by `VLLM_HPU_MOE_WEIGHT_ON_INPUT=1` |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |
| [`llama4-tools-lenient`](../fixes/llama4-tools-lenient.md) (opt-in) | the stock template and `llama4_pythonic` parser fail coding-agent tasks (prose answers, invalid Python calls once a value holds a quote or newline) |

## Verified

On vllm-gaudi v0.26.0, four Gaudi 2 cards, the flags above unless noted:

- Router fix against a float32 CPU reference built from the Hugging Face modules (next-token top-20,
  59 positions, prompts of 37, 7.6K and 9.7K tokens), KL divergence 0.003 / 0.006 / 0.007; 0.029 /
  0.354 / 0.098 without the fix (earlier image). Top-1 agreement after the fix 13/14, 14/14, 31/31.
- Coding-agent tool scenarios straight at the server: 23/24 with the template, parser and router
  fix; 0/16 with the stock template and parser and no router fix (earlier image).
- Tool reliability 120/120 sampled at 0.6, also with seeded requests; tools 7/7.
- Prefix-cache hits against cold runs: 29/32, the same as shifting the chunk boundary by one token
  with no cache (top-1 routing flips experts at near-ties). Same-step read-before-write 3 to 4 of 24
  against 4 of 24 for a no-sharing control; 0 leaks in 24 isolation rounds.
- Image prefix-cache gate (same image with new text, new image with the same text, swapped order, 4
  images, bursts): 114 checks, 0 wrong reuse, 0 wrong answers.
- Needles at 43K, 65K and 129.5K tokens: 9/9 cold and 9/9 from cache. KV driven to 100%: 9 checks
  after preemption, 0 argmax mismatches.
- Against the previous image: greedy top-1 agreement 0.9964 over 11,188 tokens, 24/24 greedy texts
  identical to 512 tokens, vision 6/6.
- 15-minute storm at saturation (48 clients, 75% seeded, streaming, client aborts): 1,991 requests,
  0 errors, 0 restarts.

## Performance

Single measurements on four cards, vllm-gaudi v0.26.0, 8K-token prompts and 400 output tokens,
warm second pass.

| Test | Result |
| --- | --- |
| 1 stream | 56.2 tokens/s |
| 4 / 8 / 16 streams, per replica | 237.3 / 402.2 / 669 tokens/s |
| TTFT at 8K, cold | 0.61 s |
| TTFT at 32K, cold / repeated (cache hit) | 4.38 s / 0.23 s |
| TTFT at 129.5K, cache hit | 0.4 s (about 54 s cold on the earlier image) |

The KV pool holds 430,592 tokens (3,364 blocks of 128).

## Known issues

- With prefix caching on, vllm-gaudi turns off contiguous paged attention: single-stream decode is
  6 to 11% slower than with caching off (17.8 vs 16.7 ms per token). From 4 streams up it is equal
  or faster.
- Remaining tool failures are model slips: a dropped quote, a typo inside JSON arguments, an edit
  with an empty `oldString`.
- Penalties are not applied under async scheduling on v0.26.0 (see
  [Qwen3-30B-A3B-2507](qwen3-30b-a3b-2507.md#known-issues)).

## What not to do

- Do not set `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` to anything but `"1"` on this model (see above).
  The same setting is not safe on every vision model; on Scout, image requests after a 130K-token
  prefill were checked and stay correct.
- Do not use `llama4_json`-style tool calls: Scout copies the schema into its arguments and wraps
  calls in code fences.
- Do not skip `moe-router-weight-on-input` because tool pass rates look acceptable without it: it
  is a numerics fix for every reply, not only tool calls.
