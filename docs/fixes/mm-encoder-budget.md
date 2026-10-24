# mm-encoder-budget

Source: [`fixes/mm-encoder-budget/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/mm-encoder-budget)

Bound the vision-encoder batch with `VLLM_HPU_MM_ENCODER_MAX_PATCHES`, so one request with many
large images does not ask for tens of GiB in a single encoder call.

| | |
| --- | --- |
| Affects | vision-language models with variable-size images (Qwen3-VL and similar) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Switch | `VLLM_HPU_MM_ENCODER_MAX_PATCHES` (unset or 0: stock behaviour) |
| Upstream | not filed yet |
| Test | `test_mm_encoder_budget.py` (CPU, needs a release source tree in `GAUDI_FIXES_ROOT`) |

## Problem

vLLM's GPU runner encodes only the scheduler's `scheduled_encoder_inputs`, which the scheduler keeps
inside the encoder compute budget. vllm-gaudi's `HPUModelRunner._execute_mm_encoder` instead collects
every image not yet in the encoder cache, for every scheduled request, and runs them through
`embed_multimodal` as one batch. The vision tower's transient memory grows with the patches in that
batch, outside vLLM's KV accounting.

## Symptom

`PT_DEVMEM Allocation failed` inside the vision encoder (for example in
`hpu_mm_encoder_attention._forward_sdpa`) on a request with many large images, followed by
`EngineDeadError`. Small single-image tests pass.

## Who is affected

Multimodal models whose images can be large (thousands of patches each) and that allow several
images per prompt, most visibly with a low `--gpu-memory-utilization` that leaves the transient
outside the KV budget and a busy card.

## The fix

`_execute_mm_encoder` feeds `embed_multimodal` consecutive groups whose patch total (from
`image_grid_thw` or `video_grid_thw`) stays within the budget; an item larger than the budget runs
alone, and items without a grid run alone. Item order is unchanged, so outputs still line up with
their cache keys, and every image is still encoded whole. For Qwen3-VL, 65,536 patches is one
maximum-size image (16,384 tokens).

## Verify

```bash
python3 fixes/mm-encoder-budget/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/mm-encoder-budget
```

End to end on Qwen3-VL-32B-Instruct-FP8, two Gaudi 2 cards, `releases/v0.30.0`, utilization 0.55:
one request with 7 images of 4096x4096 (114,719 prompt tokens) peaked at 95,055 MiB of device
memory without the fix (one run) and at 87,580 and 91,345 MiB with
`VLLM_HPU_MM_ENCODER_MAX_PATCHES=65536` (two runs on two pods); the answer was the same. The card
holds 98,304 MiB, so the gain is headroom, and the stock run survived this request on an otherwise
idle card.
