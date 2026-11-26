# Ripple-65B port

Serves [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) (Apache-2.0) on four
Intel Gaudi 2 cards, from a stock vllm-gaudi `releases/v0.30.0` image (`058f107`, vLLM `ced6857a`, Gaudi software
1.24.1). The BF16 checkpoint is read as published. Design, gates and measurements:
[docs/ports/ripple-65b.md](../../docs/ports/ripple-65b.md). Recipe:
[recipes/ricursiveAI/Ripple-65B-Preview](../../recipes/ricursiveAI/Ripple-65B-Preview/recipe.yaml) and its page
[docs/recipes/ripple-65b-preview.md](../../docs/recipes/ripple-65b-preview.md).

Upstream vLLM added the model in [vllm#65745](https://github.com/vllm-project/vllm/pull/65745), after v0.30.0. This
port is a backport of that pull request to the vLLM of `releases/v0.30.0`, plus what Gaudi needs.

## What is in it

| Path | What it does |
| --- | --- |
| `ripple_hpu/` | The vLLM plugin (entry point `vllm.general_plugins: ripple_hpu`). `model.py`: the model on vllm-gaudi's stock attention (2-D tensors, one shared rotary module, per-layer windows checked at build). `config.py`: `RicursiveConfig` from the PR. `parser.py`: the PR's `ripple` reasoning and tool parser on v0.30.0's parser engine, plus the tool grammar for `tool_choice` auto. `swa_kv.py`: window-sized KV groups for the 64 sliding layers. `runner_hooks.py`: FusedSDPA query tiles and fast softmax for long full-layer prompt attention. `graph_budget.py`: a bound on the device memory decode HPU graphs keep. |
| `patches/parser_streams_all.py` | The PR's DelegatingParser change in vLLM: reasoning that opens after content streams as reasoning. |
| `patches/runner_window.py` | Exact 1024-token sliding windows for model type `ricursive` in the runner, on top of the cookbook fixes `swa-decode-window` and `swa-window-context`. |
| `patches/swa_kv_hooks.py` | Runner and worker hooks for `swa_kv.py` (the same hooks as the muse-spark-open port). |
| `serving/chat_template.jinja` | The checkpoint's tools-aware chat template (`tokenizer_config.json` carries an older one without tools). |
| `install.sh`, `Dockerfile` | The build. |
| `tests/` | CPU tests and the build gates; `tests/hpu/` the device gates. |

All patches follow the [fix contract](../../fixes/README.md): every anchor must match, nothing is written otherwise
(exit 2), `--check` reports the state, a second run reports "applied".

Not included: DFlash2 speculative decoding (Ricursive's `Ripple-65B-Preview-DFlash2` drafter). It was built for this
model on Gaudi and measured 3.1x per stream at one stream, but under serving load its verify steps captured a new HPU
graph per shape until a TP rank ran out of device memory and the others hung; it stays out until its verify shapes
are fixed and a long soak passes.

## Build

From the repository root, on top of your stock image (built as in
[Getting started](../../docs/getting-started.md)):

```bash
docker build -f ports/ripple-65b/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-ripple-65b-image> .
```

The Dockerfile copies the repository to `/opt/gaudi-cookbook` and runs `install.sh`, which:

1. checks that the image is vllm-gaudi `058f107` with vLLM `ced6857a` and stops otherwise;
2. applies the cookbook's `v0.30.0` fix profile plus `swa-window-context`
   (`fixes/apply_all.py --profile v0.30.0 swa-window-context`). The runner patches are written against the fixed
   runner, so the fixes come first; the recipe's launch step runs the same command again and finds every fix
   applied;
3. patches vLLM's DelegatingParser, then the runner (exact windows, then the window-sized KV hooks);
4. installs `ripple_hpu` with pip (no index, no dependencies: everything it needs is in the image);
5. runs `tests/build_gates.py`: fixes and patches applied, plugin registered, every module imports, the checkpoint
   config parses, the chat template renders tools and reasoning, the parser splits them, the vllm-gaudi internals the
   hooks wrap are unchanged, and the patched runner's window masks are exact.

## Serve

Use the Launch block of the [recipe page](../../docs/recipes/ripple-65b-preview.md) in the port image.

## Switches

Each speed-up or memory change has a switch; with every switch at its "off" value the port runs the reference path.
The recipe sets all of them explicitly.

| Variable | Recipe | Off | What it does |
| --- | --- | --- | --- |
| `RIPPLE_SWA_KV` | `1` | `0` | Window-sized KV groups for the 64 sliding-window layers: only the 16 full layers keep full-length KV (16,384 instead of 81,920 B per token per card at TP 4). Outputs identical to full-length KV. |
| `RIPPLE_CONSTRAIN_AUTO_TOOLS` | `1` | `0` | `tool_choice` auto gets the hermes tool grammar even when no tool is `strict`, so a call names an offered tool and its arguments follow that tool's schema. Tools marked `strict: false` keep free arguments; requests with `response_format` or structured outputs are left as upstream leaves them. `0` is upstream (grammar only with a strict tool). |
| `RIPPLE_FAST_SOFTMAX_MIN_KV` | `16384` | `0` | Fast softmax for the prompt attention of the 16 full layers once a chunk sees this many keys. Sliding layers, short prompts and decode keep `VLLM_FP32_SOFTMAX`. |
| `RIPPLE_FSDPA_WS_GIB`, `RIPPLE_FSDPA_Q_TILE` | `2`, `256` | `0` | Biased FusedSDPA prompt calls whose workspace would exceed 2 GiB run as 256-row query tiles (same attention). |
| `RIPPLE_GRAPH_BUDGET_GIB` | `auto` | `off` | Decode HPU graphs are dropped least recently used first when their predicted memory would pass the budget (`auto`: free device memory at the first decode minus `RIPPLE_GRAPH_MARGIN_GIB`, default 8). `off` is vllm-gaudi's unbounded cache. |
| `RIPPLE_PREFILL_INFLIGHT` | `1` | `0` | In prefills of 256 tokens or more the host waits for layer N-1 before launching layer N, so queued lazy graphs cannot pile up activations past free device memory. |
| `RIPPLE_FP8_VIEW_FIX` | (default `1`) | `0` | Block-FP8 checkpoints only (not the recipe): the FP8 linear returns its GEMM output for 2-D inputs instead of a view that splits the decode graph around each all-reduce. |

vllm-gaudi's FusedSDPA slicer (`VLLM_HPU_FSDPA_SLICE_ENABLED`, on by default with `VLLM_BUCKETING_STRATEGY=pad`) is
refused at start: its merge returns NaN when a key slice is all bucket padding, and it would cut a sliding layer's
window mask.

## Verify

CPU tests (skip what they cannot run; the parser tests need the port image and vLLM's source tree):

```bash
pytest ports/ripple-65b/tests
GAUDI_FIXES_ROOT=<site-packages of releases/v0.30.0> pytest ports/ripple-65b/tests/test_patches.py
```

Device gates, against a server started with the recipe's command:

```bash
RIPPLE_URL=http://127.0.0.1:8000 ports/ripple-65b/tests/hpu/run_gates.sh
```

It prints one PASS or FAIL line per gate: `smoke`, `determinism`, `reasoning`, `dvp` (decode against prefill),
`needle` (16K up to the server's `max_model_len`), `copy`, `tools`, `isolation`, `stress` (15 decodes next to a
prompt near `max_model_len`), then an INFO line with decode speed at 1, 4 and 16 streams. The lengths follow the
server's `max_model_len`; all gates take about 15 minutes at the recipe's 131,072.
