# MiMo-V2.6 port (Flash and Pro)

Serves [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) and
[XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) (both MIT licence) on one 8-card
Gaudi 2 node with vllm-gaudi `v0.26.0`: Flash at 512K tokens per request, Pro at 256K. The design and its evidence
are in [docs/ports/mimo-v2-6-flash.md](../../docs/ports/mimo-v2-6-flash.md); the served configurations are the
recipes [MiMo-V2.6-Flash-RL](../../recipes/XiaomiMiMo/MiMo-V2.6-Flash-RL/recipe.yaml) (page
[docs/recipes/mimo-v2-6-flash-rl.md](../../docs/recipes/mimo-v2-6-flash-rl.md)) and
[MiMo-V2.6-Pro-RL](../../recipes/XiaomiMiMo/MiMo-V2.6-Pro-RL/recipe.yaml) (page
[docs/recipes/mimo-v2-6-pro-rl.md](../../docs/recipes/mimo-v2-6-pro-rl.md)). The directory keeps the Flash name it
was published under.

MiMo-V2.6-Flash is a 309B mixture-of-experts model (15B active). 39 of its 48 layers attend to a 128-token window
with a learned sink per head, query and key heads are 192 wide and value heads 128, and the routed experts are
MXFP4. vllm-gaudi v0.26.0 has no model for it, Habana's FusedSDPA ignores both sinks and windows, and the stock
runner gives every layer full-length KV.

MiMo-V2.6-Pro is the same architecture at 1.02T parameters (42B active): 70 layers, 60 of them sliding, 128 query
heads and 8 KV heads, 384 experts. The same code serves it. What differs is the expert path: FP8 experts would not
fit one node, so Pro keeps the checkpoint's MXFP4 bytes on the device and runs decode-sized MoE calls on the
[MXFP4 MoE kernels](../../kernels/mxfp4-moe/README.md) (`glm` variant, with the torch.compile op library).

## Contents

| Path | What it is |
| --- | --- |
| `install.sh` | The build step: checks the release commits, applies the v0.26.0 fix profile plus `vllm-long-prefill-share`, then the port's patches, installs the plugin, and builds the MXFP4 MoE kernels (`build.sh --variant glm --pt2` into `/opt/gaudi-cookbook-mimo/mxk`). Stops at the first step that does not apply. |
| `Dockerfile` | `FROM` the stock image, copies this repository to `/opt/gaudi-cookbook`, runs `install.sh`. |
| `mimo_v2_hpu/` | The vLLM plugin. `model.py` is the HPU model and loader, `attention.py` the exact sliding-window attention with sinks and the tiled global-attention prefill, `swa_kv.py` the window-sized KV groups, `mxk.py` the MXFP4 kernel switch for Pro, `ops.py` the pure torch math (tested on CPU). |
| `patches/runner_window.py` | Exact sliding windows for model type `mimo_v2` in the runner: per-sequence window lists in prefill, and the exact decode window of [swa-decode-window](../../fixes/swa-decode-window/README.md) turned on for `mimo_v2`. |
| `patches/runner_kv.py` | The runner and worker hooks for `MIMO_GA_TILE` and `MIMO_SWA_KV_GROUPS`. Inert with both unset. |
| `convert/stage_checkpoint.py` | Stages the public checkpoint without copying weights: symlinks plus a `config.json` without `quantization_config` and with `layer_types`. The recipe runs it at launch. |
| `serving/chat_template.jinja` | The checkpoint's chat template (MIT), with `<think>` opened in the generation prompt so the reasoning parser handles follow-up turns. |
| `tests/` | CPU tests (pytest; skip without torch, or without `GAUDI_FIXES_ROOT` for the patch test). |
| `tests/hpu/` | `device_ops.py`: one-card device checks. `mxk_ops.py`: one-card check of Pro's kernel path in compile mode on real checkpoint layers. `run_gates.sh`: the serving gates (`PROFILE=pro` for Pro). |

## Switches

All are on in the recipes that use them. Off, the port runs its reference path: full-length KV for every layer,
the stock global-attention prefill, and the stock MXFP4 MoE op.

| Variable | On | Off |
| --- | --- | --- |
| `MIMO_SWA_KV_GROUPS=1` | The sliding layers get vLLM `SlidingWindowSpec` KV groups (6 groups over 9 shared tensors): 5.33x the KV blocks. TP8 only, needs prefix caching on (the default) and torch.compile with regional compilation. | Every layer holds full-length KV (about 1.36M tokens of pool at 0.9 memory utilisation). |
| `MIMO_GA_TILE=1` | Global-layer prefill with context runs FusedSDPA per query tile with masks built per tile, and the runner skips its `[chunk, context + chunk]` bias. Needed past about 256K context. | The stock bias and prefill. |
| `MIMO_MXK=1` (Pro) | MoE calls of at most `MIMO_MXK_MAX_TOKENS` tokens (default 16, at most 16) and 128 routes run `mxk_w13x` + `mxk_w2x` from the build in `MIMO_MXK_LIB` (default `/opt/gaudi-cookbook-mimo/mxk`). Needs MXFP4 experts and compile mode. | Every MoE call on the stock MXFP4 op. |

`MIMO_EXPERT_DTYPE` selects the experts' device format: `fp8` (default, Flash: an exact re-encode at load) or
`mxfp4` (Pro: the checkpoint bytes). It is a format choice, not a speed-up switch.

With `MIMO_MXK=1` the plugin points the graph compiler at the kernel library in every vLLM process before a device
opens (`GC_KERNEL_PATH`, with every stock kernel forwarded through `MXK_TPC_FWD_LIB`) and moves
`PT_HPU_RECIPE_CACHE_CONFIG` to a directory named after the kernel build. After loading, each worker calls the
kernels once eagerly: the PT2 ops register their kernels with the bridge on that call, and a compiled graph does not
make it. The engine log then shows `[mimo-mxk] enabled: mxk_tpc glm-v8-<hash>, E_local=48 K=6144 I=2048`. A missing
library, lazy mode or a layer outside the kernels' shapes keeps the stock op and logs why.

Exact sliding-window attention with sinks, the FP8 expert re-encode (with `fp8` experts) and the CPU-built RoPE
tables are always on: without them the model's output is wrong, not slower.

### The kernel variant

Pro's per-rank expert shapes at TP8 are K 6144 (hidden), I 2048 (expert intermediate), 48 of 384 experts, top-8,
`[gate; up]` rows and plain SiLU. The `glm` variant covers K as a multiple of 512 up to 6144, I as a multiple of 512
up to 3072, at most 128 local experts, 16 tokens and 128 routes per call
([Supported shapes](../../docs/kernels/mxfp4-moe.md#limitations-and-pitfalls)), so Pro uses `--variant glm` and has
no variant of its own. Its kernels were gated at GLM's 64 experts x I 1024; `mxk_ops.py` and
`kernels/mxfp4-moe/tests/run_parity.sh --variant glm --e-local 48 --inter 2048` check them at Pro's shapes.
`--pt2` adds `libmxk_op_pt2.so`, the same ops on the PT2 custom-op API, because Pro serves in torch.compile mode and
the lazy-mode op library does not trace.

## Build

```bash
docker build -f ports/mimo-v2-6-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> \
  -t <your-mimo-v2-6-flash-image> .
```

The base image is a stock vllm-gaudi `v0.26.0` build ([Getting started](../../docs/getting-started.md)):
vllm-gaudi `b059f9c`, vLLM `568afb3a`, installed from git checkouts so `install.sh` can read their commits.

## Serve

Use the Launch block of a recipe page ([Flash](../../docs/recipes/mimo-v2-6-flash-rl.md),
[Pro](../../docs/recipes/mimo-v2-6-pro-rl.md)) inside the port image, with `COOKBOOK=/opt/gaudi-cookbook` and the
Hugging Face cache mounted. It stages the checkpoint into `$CHECKPOINT_DIR`, confirms the fixes, and serves with TP8
(Flash at 524,288 tokens, Pro at 262,144). Flash loads in a few minutes from a warm page cache; warming every shape
(`VLLM_SKIP_WARMUP=false`) then takes 40 to 50 minutes. Pro reads 566 GB and takes longer on both (see its page).
Set `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,False,65536` on persistent storage to make later starts reuse the compiled
graphs.

## Verify

```bash
pytest ports/mimo-v2-6-flash/tests                                              # CPU
GAUDI_FIXES_ROOT=<v0.26.0 site-packages copy> pytest ports/mimo-v2-6-flash/tests/test_patches.py
python3 ports/mimo-v2-6-flash/tests/hpu/device_ops.py "$CHECKPOINT_DIR"        # one card, in the port image
URL=http://localhost:8000 bash ports/mimo-v2-6-flash/tests/hpu/run_gates.sh     # against the running server
```

For Pro, in the port image, before the server takes the cards:

```bash
python3 ports/mimo-v2-6-flash/tests/hpu/mxk_ops.py "$CHECKPOINT_DIR"                          # one card
PT_HPU_LAZY_MODE=1 bash kernels/mxfp4-moe/tests/run_parity.sh --variant glm --lib /opt/gaudi-cookbook-mimo/mxk \
  --e-local 48 --inter 2048 --out /tmp/parity-pro                                               # one card, lazy ops
URL=http://localhost:8000 PROFILE=pro bash ports/mimo-v2-6-flash/tests/hpu/run_gates.sh       # the running server
```

`run_gates.sh` prints one PASS or FAIL line per gate: decode speed (Flash at 1, 4 and 16 streams, Pro at 1, 4 and 8),
tool calls, needles (Flash at 128K, 256K and 500K, Pro at 128K and 240K), four prompts of the longest needle size at
once, and (with `KV_REFERENCE`) token-identical greedy outputs against a server on the stock KV layout. Its header
says how to record the reference. `mxk_ops.py` compares the kernels (compiled, as the model calls them) and the
stock op with a CPU fp32 reference on two layers and two ranks of the checkpoint.

## Limits

- Text only: the vision tower, audio encoder and MTP layers in the checkpoint are skipped at load.
- TP8 only with window-sized KV. At TP4 the global and sliding layers hold different numbers of KV heads per rank,
  so their KV pages differ and vLLM cannot share one block pool.
- No speculative decoding, no merged prefill, no pipeline parallelism, bf16 KV cache only.
- The MXFP4 kernels run in compile mode only (`libmxk_op_pt2.so`), and only on decode-sized calls; prefill keeps the
  stock op. Pro needs `--gpu-memory-utilization 0.8` (see its recipe page).
