# MiMo-V2.6-Flash-RL

Xiaomi's 309B mixture-of-experts model (15B active) at 512K tokens per request on one 8-card Gaudi 2 node.
39 of its 48 layers attend to a 128-token window with a learned sink, which vllm-gaudi v0.26.0 cannot serve on
its own, so this recipe runs in the [MiMo-V2.6-Flash port](../ports/mimo-v2-6-flash.md) image: exact
sliding-window attention with sinks, window-sized KV for the sliding layers, a tiled global-attention prefill
and the routed experts re-encoded to FP8 at load.

| | |
| --- | --- |
| Model | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (MIT), revision `5711b26` |
| Hardware | 8x Gaudi 2, one node (TP 8, experts split over the 8 ranks) |
| Image | the port image, built from vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | bf16 attention (FP8 block weights dequantised at load), FP8 e4m3 routed experts (exact re-encode of the MXFP4 checkpoint), bf16 KV |
| Context | 524,288 per request (the checkpoint allows 1,048,576) |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools `mimo`, reasoning `mimo` |
| Recipe | [`recipes/XiaomiMiMo/MiMo-V2.6-Flash-RL/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/XiaomiMiMo/MiMo-V2.6-Flash-RL/recipe.yaml) |

## Build the image

The port installs into a stock vllm-gaudi `v0.26.0` image ([Getting started](../getting-started.md)). From a
checkout of this repository:

```bash
docker build -f ports/mimo-v2-6-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> \
  -t <your-mimo-v2-6-flash-image> .
```

`install.sh` checks the release commits, applies the v0.26.0 fix profile and `vllm-long-prefill-share`, then
the port's runner patches, and installs the plugin. It stops at the first step that does not apply. See
[`ports/mimo-v2-6-flash/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/mimo-v2-6-flash/README.md).

## Launch

Inside the port image, where `$COOKBOOK` is `/opt/gaudi-cookbook`, with the Hugging Face cache mounted. The
first line stages the checkpoint (symlinks only; downloads it first if it is not in the cache).

```bash
export CHECKPOINT_DIR=/data/mimo-v2.6-flash-rl-hpu
export PT_HPU_LAZY_MODE=0
export VLLM_T_COMPILE_DYNAMIC_SHAPES=false
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=false
export MIMO_SWA_KV_GROUPS=1
export MIMO_GA_TILE=1
export VLLM_LONG_PREFILL_SHARE=0.5
export VLLM_LONG_PREFILL_SHARE_ASYNC=1
export VLLM_LONG_PREFILL_MIN_TOKENS=65536
export VLLM_MOE_CHUNK=256
export VLLM_MOE_TOKEN_BOUNDARY=2048
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

python3 "$COOKBOOK/ports/mimo-v2-6-flash/convert/stage_checkpoint.py" XiaomiMiMo/MiMo-V2.6-Flash-RL "$CHECKPOINT_DIR" --revision 5711b268169967567844e1e560e8a3966da959b1 || exit 1

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 vllm-long-prefill-share || exit 1

vllm serve $CHECKPOINT_DIR \
  --served-model-name XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --tensor-parallel-size 8 \
  --max-model-len 524288 \
  --max-num-seqs 16 \
  --long-prefill-token-threshold 2048 \
  --max-num-batched-tokens 8192 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --enable-prompt-tokens-details \
  --trust-remote-code \
  --generation-config vllm \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' \
  --chat-template "$COOKBOOK/ports/mimo-v2-6-flash/serving/chat_template.jinja" \
  --reasoning-parser mimo \
  --enable-auto-tool-choice \
  --tool-call-parser mimo
```

- `stage_checkpoint.py` writes `$CHECKPOINT_DIR` with symlinks to the checkpoint and a `config.json` with two
  edits: `quantization_config` is removed (the port dequantises the FP8 block weights and re-encodes the experts
  itself) and `layer_types` is derived from `hybrid_layer_pattern` (without it vLLM gives every layer the
  128-token window). The model is served under its public name.
- `MIMO_SWA_KV_GROUPS=1` gives the 39 sliding layers window-sized KV through vLLM's hybrid KV cache manager:
  5.33x the KV blocks. `MIMO_GA_TILE=1` runs the global layers' prefill in query tiles so a 512K prompt does
  not need a multi-gigabyte attention bias. Both are what makes four 512K requests fit.
- `PT_HPU_LAZY_MODE=0` with static shapes: torch.compile mode. Eager decode runs at about 5 tok/s, and the
  dynamic-shape compile path fails.
- `VLLM_SKIP_WARMUP=false`: the start warms up every shape for 40 to 50 minutes after a load of a few minutes.
  Without it each new shape compiles on first use and stalls requests for 10 to 30 s. Point
  `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,False,65536` at persistent storage to reuse compiled graphs across restarts.
- `--generation-config vllm` with the override keeps the model's sampling defaults (temperature 1.0, top_p 0.95)
  without the checkpoint's 2,048-token `max_new_tokens` cap.
- The chat template is the checkpoint's, with `<think>` opened in the generation prompt. Without it the reasoning
  parser sees the previous turn's `</think>` as the last marker and streams the new reasoning as content.

### Buckets

No bucket file. Bucketing uses the image defaults; with `--block-size 128` and `--max-num-batched-tokens 8192`
warmup covers prompt chunks up to 8,192 tokens against contexts up to the full window.

## Required fixes

All from [Fixes](../fixes/index.md). `install.sh` applies them when the image is built; the launch command runs
the same line again and only confirms them. The opt-in one is named on that line; the rest come from
`--profile v0.26.0`, which also applies the release's other generic fixes.

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching can read another request's KV in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-inputs`](../fixes/async-inputs.md) | a decode the scheduler leaves out of a step (behind long-prompt chunks) reads a stale token and repeats it: 42 repeated-token signatures in 190 probes without it, 0 with it |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption repeats its earlier output |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids come back corrupted |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`swa-decode-window`](../fixes/swa-decode-window.md) | the exact decode window (one extra block, the oldest block head-masked); the port's runner patch turns it on for `mimo_v2` |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |
| [`vllm-long-prefill-share`](../fixes/vllm-long-prefill-share.md) (opt-in) | one 500K prompt mid-prefill held every other stream at about 0.4 tok/s; with share 0.5 they keep 14 to 18 tok/s |

`rope-cpu-cache` is in the profile; the port also rebuilds MiMo's RoPE tables on the CPU in float64 itself,
because the fastest rotary frequency's angle is the position and Gaudi fp32 cos and sin are wrong past about
1.5e5 radians.

## Verified

This exact recipe has not been run end to end in the port image yet, so the recipe is `supported`, not
`verified`. What ran as written:

- `install.sh` in a stock vllm-gaudi `v0.26.0` image (`b059f9c`, vLLM `568afb3a`): commit check, the v0.26.0
  profile plus `vllm-long-prefill-share`, both runner patches and the plugin install all applied; the launch's fix
  step and both patches' `--check` then report "applied"; vLLM resolves `MiMoV2ForCausalLM`,
  `MiMoV2FlashForCausalLM` and `MiMoV2OmniForCausalLM` to the port's model.
- `tests/hpu/device_ops.py` on one Gaudi 2 card: the sliding-window attention with sinks matches the fp32
  reference in bf16 (worst relative error 0.0036), and 12 of 12 sampled expert tensors of the checkpoint re-encode
  to FP8 exactly and byte-identically to the CPU encode. The device's fp32 `cos(200000)` read 0.0000 against an
  exact 0.9974, the reason the RoPE tables are built on the CPU.
- `stage_checkpoint.py` on the public checkpoint (revision `5711b26`) produces the same `config.json` as the
  configuration that serves production.

The serving gates below were recorded on the production deployment of the same code and settings (not on an
image built by this `install.sh`), on 8 Gaudi 2 cards, TP 8:

- Needles at 128K / 256K / 500K: 3/3 each (depths 0.1, 0.5, 0.9). Four 500K-token requests at once: 4/4.
- Greedy outputs with window-sized KV on and off: token-identical (short chats, 12K and 30K prompts chunked across
  8,192-token steps, a 10,752-token prefix hit, and four prompts in one batch on warm engines).
- Tool calls: short suite 7/7; correct calls after a 33.8K and a 138K document.
- Prompt NLL on a 1,535-token document 0.7986 against a streaming fp32 CPU oracle at 0.7943; GSM8K 5-shot 0.910
  (200 samples).

## Performance

Recorded on the production deployment (same code and settings, server without user traffic, warm, 256-token
runs; not re-measured on this image build):

| Streams | Tokens/s per stream | Aggregate |
| --- | --- | --- |
| 1 | 33.9 | 34 |
| 4 | 33.4 | 133 |
| 16 | 32.5 | 520 |

Decode speed does not fall with context: 39 of the 48 layers read a 128-token window. The KV pool holds 56,681
blocks (7.26M tokens of global context, 11.9x concurrency at 524,288). With `VLLM_LONG_PREFILL_SHARE=0.5`, other
streams decode at 14 to 18 tok/s while one 500K prompt prefills (about 0.4 tok/s without the share; separate
server starts).

## Known issues

- At 500K tokens and depth 0.9 some seeded needles miss on both Gaudi and an H100 reference running upstream
  vLLM (3 of 4 there): the model's own boundary, not the port. Every needle at depths 0.1 and 0.5 hit up to 500K.
- The model reads imperative inserted text ("IMPORTANT NOTE: ...") as prompt injection and may decline to repeat
  it. Phrase retrieval tests as plain facts.
- Compile mode computes a new batch shape slightly differently the first time it runs. Compare outputs only on
  warm runs.
- `prompt_logprobs` over a prompt of a few hundred thousand tokens needs a multi-gigabyte logits workspace and
  can run the engine out of memory at `--gpu-memory-utilization 0.9`.
- Text only. The checkpoint's vision tower, audio encoder and MTP layers are not loaded.

## What not to do

- Do not run window-sized KV at TP4: the global and sliding layers then hold different KV heads per rank, and the
  port refuses to start.
- Do not turn prefix caching off without `VLLM_CONTIGUOUS_PA=false`: the contiguous KV layout bypasses the
  per-group block tables, and the port refuses to start.
- Do not add speculative decoding, or switch to lazy mode with HPU graphs (`PT_HPU_LAZY_MODE=1` without
  `--enforce-eager`), with window-sized KV on: the per-layer metadata swap needs regionally compiled layers, and
  the port refuses both.
- Do not drop `MIMO_GA_TILE=1` at 512K: the first long prefill with context then builds an 8.7 GB attention bias
  that warmup never profiled.
