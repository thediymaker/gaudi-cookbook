# GLM-4.5V (FP8)

Z.ai's GLM-4.5V: a 106B-total, 12B-active MoE language model (the GLM-4.5-Air base) with a vision
encoder, served in FP8 at 32K context on two Gaudi 2 cards with up to 4 images per prompt.

| | |
| --- | --- |
| Model | [zai-org/GLM-4.5V-FP8](https://huggingface.co/zai-org/GLM-4.5V-FP8) |
| Hardware | 2x Gaudi 2 (TP 2) |
| Image | vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | FP8 compressed-tensors (110 GB: per-channel weights, dynamic per-token activations) |
| Context | 32,768 served (65,536 native) |
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
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 glm4-moe-3d-input glm45v-box-tokens || exit 1

vllm serve zai-org/GLM-4.5V-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
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
- 32K is deliberate. The checkpoint advertises 65,536, but a real 64K prefill hung the device on an
  older release (`No progress error` on every card after about 32K tokens), with the bucketing
  changed in the same step. It has not been re-tested on this image.
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

On vllm-gaudi `releases/v0.30.0`, two Gaudi 2 cards, the launch above run as written (KV pool 398,720
tokens, 12.17x concurrency at 32,768). Every gate below the first two rows passed on two separate
pods.

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

## Performance

Single measurements on two cards, `releases/v0.30.0`, warm; short chat prompt, 256 output tokens
(thinking included) with `ignore_eos`, temperature 0.7. Per-stream speed is decode only (after the
first token).

| Streams | Tokens/s per stream | Tokens/s aggregate | Mean time to first token |
| --- | --- | --- | --- |
| 1 | 40.3 to 44.0 | 40 to 43 | 0.12 s |
| 4 | 32.3 to 32.5 | 123 | 0.38 to 0.39 s |
| 8 | 24.4 | 168 to 169 | 1.61 to 1.69 s |

Ranges are two runs on two pods (two nodes) with the same launch.

## Known issues

- The 64K window the checkpoint advertises is untested on this image (see the launch notes).
- Thinking adds a few hundred tokens even to one-word answers; give `max_tokens` room or turn
  thinking off per request.

## What not to do

- Do not drop `glm4-moe-3d-input`: the server starts and `/health` answers, and only the first
  real request shows the failure.
- Do not try to remove the box markers with `skip_special_tokens` or tokenizer edits: with a
  reasoning parser active the text keeps special tokens by design; the parser fix is what removes
  them.
- Do not run it on one card: 110 GB of FP8 weights do not fit 96 GB.
