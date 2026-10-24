# Beam-1.0 port

Serves [reflection/Beam-1.0-501B-A23B-FP8](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8)
(Apache-2.0) on one node of eight Intel Gaudi 2 cards, from a stock vllm-gaudi `releases/v0.30.0`
image (`058f107`, vLLM `ced6857a`, Gaudi software 1.24.1). The checkpoint is read as published: no
conversion and no extra disk. Design, gates and measurements:
[docs/ports/beam-1-0.md](../../docs/ports/beam-1-0.md). Recipe:
[recipes/reflection/Beam-1.0-501B-A23B-FP8](../../recipes/reflection/Beam-1.0-501B-A23B-FP8/recipe.yaml)
and its page [docs/recipes/beam-1-0-501b-a23b-fp8.md](../../docs/recipes/beam-1-0-501b-a23b-fp8.md).

## What is in it

| Path | What it does |
| --- | --- |
| `beam_hpu/` | The vLLM plugin (entry point `vllm.general_plugins: beam_hpu`). `vllm_model.py`, `attention.py`, `moe.py`, `numerics.py`: the model on vllm-gaudi's stock attention, fused MoE and compressed-tensors FP8 methods. `loader.py`, `load_plan.py`: load format `beam`. `parser.py`, `frontend.py`: the `beam` reasoning and tool parsers (a backport of vLLM PR #62052) and request corrections. `config.py`: `BeamConfig` from vLLM PR #62074. |
| `vendor/` | The platform-neutral files of [vLLM PR #62052](https://github.com/vllm-project/vllm/pull/62052) the parsers build on (response-template framework and grammar), with their sha256 in `vendor/MANIFEST`. |
| `patches/vllm_pr_62052.py` | Installs `vendor/` into the vLLM tree. |
| `patches/beam_sliding_window.py` | Exact 512-token sliding windows for model type `beam` in the runner, on top of the cookbook fix `swa-decode-window`. |
| `patches/grammar_device_mask.py` | Applies the response-grammar bitmask on the device instead of copying the logits to the host every step, on top of `so-row-owners`. |
| `serving/tp8-seqs16-256k.txt` | Bucket file for 16 sequences at 256K: vllm-gaudi's prompt buckets plus a 256K point, and coarse decode buckets so the captured decode graphs stay bounded. |
| `install.sh`, `Dockerfile` | The build. |
| `tests/` | CPU tests and the build gates; `tests/hpu/` the device gates. |

All patches follow the [fix contract](../../fixes/README.md): every anchor must match, nothing is
written otherwise (exit 2), `--check` reports the state, a second run reports "applied".

## Build

From the repository root, on top of your stock image (built as in
[Getting started](../../docs/getting-started.md)):

```bash
docker build -f ports/beam-1-0/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-beam-1-0-image> .
```

The Dockerfile copies the repository to `/opt/gaudi-cookbook` and runs `install.sh`, which:

1. checks that the image is vllm-gaudi `058f107` with vLLM `ced6857a` and stops otherwise;
2. applies the cookbook's `v0.30.0` fix profile (`fixes/apply_all.py --profile v0.30.0`). The two
   runner patches are written against the fixed runner, so the profile comes first; the recipe's
   launch step runs the same command again and finds every fix applied;
3. installs the vendored PR #62052 files, then the two runner patches;
4. installs `beam_hpu` with pip (no index, no dependencies: everything it needs is in the image);
5. runs `tests/build_gates.py`: fixes and patches applied, plugin registered, every module imports,
   the checkpoint config parses as compressed-tensors FP8, the chat template renders.

## Serve

Use the Launch block of the [recipe page](../../docs/recipes/beam-1-0-501b-a23b-fp8.md) in the port
image. The checkpoint is 503 GB; with `--load-format beam` a start reads it at storage speed (about
6 minutes at 1.4 GB/s).

## Switches

Each speed-up has a switch; with every switch at its "off" value the port runs the reference path.
The recipe sets all of them explicitly.

| Variable | Recipe | Off | What it does |
| --- | --- | --- | --- |
| `--load-format beam` | on | `auto` (stock loader) | Ranged, parallel reads of only the rank's expert rows, the Gaudi 2 FP8 rescale on the kept region only, one device copy per expert stack instead of per tensor. Same model state as `auto`. |
| `BEAM_EXACT_ROUNDING` | `0` | `1` | `1` forces every bf16 rounding point of the reference with integer ops, which the graph compiler cannot drop; `0` uses plain casts (about one bf16 ulp at the dropped points, smaller graphs, faster decode). Both pass the gates. |
| `BEAM_GRAMMAR_DEVICE_MASK` | `1` | `0` | Grammar bitmask applied on the device (bit-exact); `0` is vllm-gaudi's host round trip. |
| `BEAM_PREFILL_INFLIGHT` | `1` | `0` | In prefills of 256 tokens or more the host waits for layer N-1 before launching layer N, so queued lazy graphs cannot pile up activations past free HBM. |
| `BEAM_FSDPA_WS_GIB`, `BEAM_FSDPA_Q_TILE` | `2`, `256` | `0` | Biased FusedSDPA prompt calls whose workspace would exceed 2 GiB run as 256-row query tiles (same attention; without it, prompts with 32K or more of cached context ran out of device memory). |
| `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT` | `1` | `0` | Upstream's switch: the response grammar constrains every assistant turn. `0` leaves unconstrained turns free (tools and `response_format` keep their grammar). |
| `VLLM_BEAM_STREAM_TOOL_ARGS` | `1` | `0` | Tool calls are announced when they start and string arguments stream as written; `0` sends each call whole when it closes, as upstream does. |
| `BEAM_LOAD_THREADS` | `16` (default) | | Reader threads per rank for `--load-format beam`; 12 if the pod's CPU quota throttles. |

## Verify

CPU tests (skip what they cannot run; the parser tests need the port image):

```bash
pytest ports/beam-1-0/tests
GAUDI_FIXES_ROOT=<site-packages of releases/v0.30.0> pytest ports/beam-1-0/tests/test_patches.py
```

Device gates, against a server started with the recipe's command:

```bash
BEAM_URL=http://127.0.0.1:8000 ports/beam-1-0/tests/hpu/run_gates.sh
```

It prints one PASS or FAIL line per gate: `smoke`, `ppl` (prompt logprobs against fp32 and
per-token-FP8 CPU references of the same checkpoint, shipped in `tests/hpu/reference/`), `dvp`
(decode against prefill), `needle` (16K to 255K), `copy` (130K and 255K), `tools`, `isolation`,
`stress` (15 decodes next to a 250K prompt), then an INFO line with decode speed.
