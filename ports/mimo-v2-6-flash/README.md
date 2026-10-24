# MiMo-V2.6-Flash port

Serves [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (MIT licence) on one
8-card Gaudi 2 node with vllm-gaudi `v0.26.0`, at 512K tokens per request. The design and its evidence are in
[docs/ports/mimo-v2-6-flash.md](../../docs/ports/mimo-v2-6-flash.md); the served configuration is the recipe
[recipes/XiaomiMiMo/MiMo-V2.6-Flash-RL](../../recipes/XiaomiMiMo/MiMo-V2.6-Flash-RL/recipe.yaml), page
[docs/recipes/mimo-v2-6-flash-rl.md](../../docs/recipes/mimo-v2-6-flash-rl.md).

MiMo-V2.6-Flash is a 309B mixture-of-experts model (15B active). 39 of its 48 layers attend to a 128-token window
with a learned sink per head, query and key heads are 192 wide and value heads 128, and the routed experts are
MXFP4. vllm-gaudi v0.26.0 has no model for it, Habana's FusedSDPA ignores both sinks and windows, and the stock
runner gives every layer full-length KV.

## Contents

| Path | What it is |
| --- | --- |
| `install.sh` | The build step: checks the release commits, applies the v0.26.0 fix profile plus `vllm-long-prefill-share`, then the port's patches, then installs the plugin. Stops at the first step that does not apply. |
| `Dockerfile` | `FROM` the stock image, copies this repository to `/opt/gaudi-cookbook`, runs `install.sh`. |
| `mimo_v2_hpu/` | The vLLM plugin. `model.py` is the HPU model and loader, `attention.py` the exact sliding-window attention with sinks and the tiled global-attention prefill, `swa_kv.py` the window-sized KV groups, `ops.py` the pure torch math (tested on CPU). |
| `patches/runner_window.py` | Exact sliding windows for model type `mimo_v2` in the runner: per-sequence window lists in prefill, and the exact decode window of [swa-decode-window](../../fixes/swa-decode-window/README.md) turned on for `mimo_v2`. |
| `patches/runner_kv.py` | The runner and worker hooks for `MIMO_GA_TILE` and `MIMO_SWA_KV_GROUPS`. Inert with both unset. |
| `convert/stage_checkpoint.py` | Stages the public checkpoint without copying weights: symlinks plus a `config.json` without `quantization_config` and with `layer_types`. The recipe runs it at launch. |
| `serving/chat_template.jinja` | The checkpoint's chat template (MIT), with `<think>` opened in the generation prompt so the reasoning parser handles follow-up turns. |
| `tests/` | CPU tests (pytest; skip without torch, or without `GAUDI_FIXES_ROOT` for the patch test). |
| `tests/hpu/` | `device_ops.py`: one-card device checks. `run_gates.sh`: the serving gates. |

## Switches

Both are on in the recipe. Off, the port runs its reference path: full-length KV for every layer, and the stock
global-attention prefill.

| Variable | On | Off |
| --- | --- | --- |
| `MIMO_SWA_KV_GROUPS=1` | The sliding layers get vLLM `SlidingWindowSpec` KV groups (6 groups over 9 shared tensors): 5.33x the KV blocks. TP8 only, needs prefix caching on (the default) and torch.compile with regional compilation. | Every layer holds full-length KV (about 1.36M tokens of pool at 0.9 memory utilisation). |
| `MIMO_GA_TILE=1` | Global-layer prefill with context runs FusedSDPA per query tile with masks built per tile, and the runner skips its `[chunk, context + chunk]` bias. Needed past about 256K context. | The stock bias and prefill. |

Exact sliding-window attention with sinks, the FP8 expert re-encode and the CPU-built RoPE tables are always on:
without them the model's output is wrong, not slower.

## Build

```bash
docker build -f ports/mimo-v2-6-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> \
  -t <your-mimo-v2-6-flash-image> .
```

The base image is a stock vllm-gaudi `v0.26.0` build ([Getting started](../../docs/getting-started.md)):
vllm-gaudi `b059f9c`, vLLM `568afb3a`, installed from git checkouts so `install.sh` can read their commits.

## Serve

Use the Launch block of the [recipe page](../../docs/recipes/mimo-v2-6-flash-rl.md) inside the port image, with
`COOKBOOK=/opt/gaudi-cookbook` and the Hugging Face cache mounted. It stages the checkpoint into
`$CHECKPOINT_DIR`, confirms the fixes, and serves with TP8 at 524,288 tokens. Loading takes a few minutes from a
warm page cache; warming every shape (`VLLM_SKIP_WARMUP=false`) then takes 40 to 50 minutes. Set
`PT_HPU_RECIPE_CACHE_CONFIG=<dir>,False,65536` on persistent storage to make later starts reuse the compiled
graphs.

## Verify

```bash
pytest ports/mimo-v2-6-flash/tests                                              # CPU
GAUDI_FIXES_ROOT=<v0.26.0 site-packages copy> pytest ports/mimo-v2-6-flash/tests/test_patches.py
python3 ports/mimo-v2-6-flash/tests/hpu/device_ops.py "$CHECKPOINT_DIR"        # one card, in the port image
URL=http://localhost:8000 bash ports/mimo-v2-6-flash/tests/hpu/run_gates.sh     # against the running server
```

`run_gates.sh` prints one PASS or FAIL line per gate: decode speed at 1, 4 and 16 streams, tool calls, needles at
128K, 256K and 500K, four 500K prompts at once, and (with `KV_REFERENCE`) token-identical greedy outputs against
a server on the stock KV layout. Its header says how to record the reference.

## Limits

- Text only: the vision tower, audio encoder and MTP layers in the checkpoint are skipped at load.
- TP8 only with window-sized KV. At TP4 the global and sliding layers hold different numbers of KV heads per rank,
  so their KV pages differ and vLLM cannot share one block pool.
- No speculative decoding, no merged prefill, no pipeline parallelism, bf16 KV cache only.
- MiMo-V2.6-Pro is not covered: it keeps its experts in MXFP4 with the MXFP4 MoE kernels and needs a different
  memory setting, which this package does not include.
