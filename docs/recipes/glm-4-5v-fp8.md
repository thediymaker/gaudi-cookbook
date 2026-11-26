# GLM-4.5V (FP8)

Z.ai's GLM-4.5V: a 106B-total, 12B-active MoE language model (the GLM-4.5-Air base) with a vision
encoder, served in FP8 at its full 64K context on two Gaudi 2 cards with up to 4 images per prompt.

| | |
| --- | --- |
| Model | [zai-org/GLM-4.5V-FP8](https://huggingface.co/zai-org/GLM-4.5V-FP8) |
| Hardware | 2x Gaudi 2 (TP 2) |
| Image | vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | FP8 compressed-tensors (110 GB: per-channel weights, dynamic per-token activations) |
| Context | 65,536 (native) |
| Concurrency | `--max-num-seqs 8` |
| Parsers | tools `glm45`, reasoning `glm45` |
| Recipe | [`recipes/zai-org/GLM-4.5V-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/zai-org/GLM-4.5V-FP8/recipe.yaml) |

## Launch

`$COOKBOOK` is a checkout of this repository inside the container.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export VLLM_MULTIMODAL_BUCKETS=None
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 glm4-moe-3d-input glm45v-box-tokens || exit 1

vllm serve zai-org/GLM-4.5V-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  --limit-mm-per-prompt '{"image":4,"video":0}'
```

- The FP8 checkpoint is compressed-tensors with per-channel weights and dynamic per-token
  activations; vllm-gaudi picks its FP8 per-channel MoE path from `config.json`, so no
  `--quantization` flag is needed. At 110 GB it does not fit one 96 GB card; two is the floor.
- 65,536 is the checkpoint's `max_position_embeddings`. It needs `VLLM_GRAPH_RESERVED_MEM=0.3`: at the
  default 0.1 (3.9 GiB of graph memory per card), a 48K-token prefill stopped the engine with
  `PT_DEVMEM Allocation failed` while replaying an HPU graph. At 0.3 the KV pool is 310,144 tokens (398,720
  at 0.1), still 4.7 full-length requests. An older release hung the device on a 64K prefill
  (`No progress error`); this image did not.
- `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` is required at TP 2 in lazy mode.
- Thinking is on by default; the `glm45` parser returns it in `reasoning_content`. Send
  `chat_template_kwargs: {"enable_thinking": false}` to answer directly.

### Buckets

No bucket file. Default bucketing; `VLLM_MULTIMODAL_BUCKETS=None` leaves image shapes unbucketed.

## Required fixes

All from [Fixes](../fixes/index.md). `--profile v0.30.0` applies the release's generic fixes;
the opt-in fixes are named in the launch command.

| Fix | Why this model needs it |
| --- | --- |
| [`mm-merge-idxcopy`](../fixes/mm-merge-idxcopy.md) | the multimodal embedding merge uses a broadcast `masked_scatter_` that fails to compile when several images arrive in one step |
| [`glm4-moe-3d-input`](../fixes/glm4-moe-3d-input.md) (opt-in) | stock v0.30.0 stops the engine on the first chat request: `ValueError: too many values to unpack (expected 2)` in `Glm4MoE.forward`, which assumes 2-D hidden states |
| [`glm45v-box-tokens`](../fixes/glm45v-box-tokens.md) (opt-in) | without it short answers reach the client as `<|begin_of_box|>Paris<|end_of_box|>` (4 of 4 short factual prompts, streamed and not) |

## Verified

On vllm-gaudi `releases/v0.30.0`, two Gaudi 2 cards. The table below ran at 32,768 with the default graph
reserve (KV pool 398,720 tokens); every gate below the first two rows passed on two separate pods. The 65,536
launch above was then run as written on one pod (KV pool 310,144 tokens), with the results in the second table.

| Check | Result |
| --- | --- |
| Stock image plus the profile only | engine stopped on the first chat request (`ValueError: too many values to unpack`) |
| With `glm4-moe-3d-input` only | 4/4 short factual answers wrapped in `<|begin_of_box|>...<|end_of_box|>`, streamed and not |
| The launch above (both fixes) | 0/4 markers, streamed and not |
| Text: "capital of France"; 157 x 23 + 89 worked out | Paris; 3700 |
| Reasoning split | thinking in `reasoning_content`, no `<think>` in `content` |
| Vision: read an arbitrary number from a rendered image; three coloured shapes | "4729"; red circle, blue square, green triangle |
| Text-only request right after an image request, 3 rounds | 3/3 correct |
| 4 images of 2048x2048 in one request | answered "4" |
| 16 mixed text and image requests at 8 concurrent | 16/16 correct |
| Tool call (`glm45`) | `get_weather` with `{"city": "Rome"}` |
| `top_logprobs` and `prompt_logprobs` | both returned, no bad ids |
| Needle at about 28K tokens, middle of the prompt | found |

The launch above, at 65,536 (requests through `ports/kimi-k2-7/tests/hpu/kimi_gates.py` and a few direct ones):

| Check | Result |
| --- | --- |
| Known-answer prompts, 12, greedy | 12/12 |
| Tool call (`glm45`), streamed and not | `get_weather` with `{"city": "Lisbon"}` both ways |
| 4 images, one solid colour each | colours named in order |
| `top_logprobs: 20` soak, 50 requests with CJK and emoji output | 50/50 complete |
| Needles, 3 depths, thinking off | 9/9 at 30K, 48K and 61K tokens |
| Needles, 3 depths, thinking on, `max_tokens` 4,096 | 4/6 at 48K and 61K; both misses spent the 4,096 tokens reasoning |
| 4 needle prompts of 61K tokens at once, thinking off | 4/4 |
| Uncached 60K prompt | 50.4 s to the first token (median of 3) |
| Server health after all checks (two starts) | healthy, 0 tracebacks |
| Same launch with the default `VLLM_GRAPH_RESERVED_MEM=0.1` | engine stopped on a 48K prefill (`PT_DEVMEM Allocation failed`) |

## Performance

Single measurements on two cards, `releases/v0.30.0`, at 32,768 with the default graph reserve, warm;
short chat prompt, 256 output tokens (thinking included) with `ignore_eos`, temperature 0.7. Per-stream speed is decode only (after the
first token).

| Streams | Tokens/s per stream | Tokens/s aggregate | Mean time to first token |
| --- | --- | --- | --- |
| 1 | 40.3 to 44.0 | 40 to 43 | 0.12 s |
| 4 | 32.3 to 32.5 | 123 | 0.38 to 0.39 s |
| 8 | 24.4 | 168 to 169 | 1.61 to 1.69 s |

Ranges are two runs on two pods (two nodes) with the same launch.

At 65,536 with the launch above, one run, `kimi_gates.py sweep` (2 x C streamed completions of 256 tokens with
`ignore_eos` on a cached 1,000-token prefix, thinking on; per stream is 1 / median time per output token): 48.1
tokens/s per stream at 1 stream, 29.4 (110 aggregate) at 4, 19.8 (122 aggregate) at 8. The tool and the prompt
differ from the table above, so compare within each table only.

## Known issues

- With thinking on, long prompts can use the whole `max_tokens` budget reasoning before they answer; at 48K and
  61K tokens 2 of 6 needle questions ran out at 4,096 tokens.
- Thinking adds a few hundred tokens even to one-word answers; give `max_tokens` room or turn
  thinking off per request.

## What not to do

- Do not drop `glm4-moe-3d-input`: the server starts and `/health` answers, and only the first
  real request shows the failure.
- Do not try to remove the box markers with `skip_special_tokens` or tokenizer edits: with a
  reasoning parser active the text keeps special tokens by design; the parser fix is what removes
  them.
- Do not run it on one card: 110 GB of FP8 weights do not fit 96 GB.
- Do not serve 65,536 with the default graph reserve: keep `VLLM_GRAPH_RESERVED_MEM=0.3`.
