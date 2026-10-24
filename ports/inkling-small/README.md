# Inkling-Small port

[thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) (276B total, 12B active,
Apache-2.0) on Gaudi 2, text only, for vllm-gaudi `releases/v0.30.0` (`058f107`, vLLM `ced6857a`). vLLM's Inkling
model is CUDA/ROCm only; this port runs the published BF16 checkpoint, unconverted, on one 8-card node at 256K
tokens per request. Design and evidence: [docs/ports/inkling-small.md](../../docs/ports/inkling-small.md). Recipe:
[recipes/thinkingmachines/Inkling-Small](../../recipes/thinkingmachines/Inkling-Small/recipe.yaml).

## Contents

| Path | What it is |
| --- | --- |
| `install.sh` | The only build step: checks the release commits, installs the plugin, applies the runner patch. Fails closed. |
| `Dockerfile` | `FROM ${BASE}`, copies this repository to `/opt/gaudi-cookbook`, runs `install.sh`. |
| `inkling_hpu/` | The vLLM general plugin: config (`config.py`), model (`model.py`), rel-bias helpers (`ops.py`), conv state (`conv_state.py`), `inkling_py` parsers (`parsers.py`). |
| `patches/runner_decode_window.py` | Gives Inkling's sliding-window layers one more decode block (fix contract: `--check`, `--root`, exit 2 on drift). |
| `tests/` | CPU tests (pytest). The rel-bias and conv-state tests need only torch; the stack and parser tests need vLLM and skip without it. |
| `tests/hpu/run_gates.sh` | Device gates against a running server; one PASS or FAIL line per gate. |

No bucket files and no kernels: the recipe uses the release's default buckets and stock ops.

## Build

```bash
docker build -f ports/inkling-small/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <inkling-image> .
```

The base must be built as in [Getting started](../../docs/getting-started.md) from `releases/v0.30.0` at `058f107`.
`install.sh` reads the commits from the source checkouts (or the `+g<sha>` version suffix) and stops on any other.

## Serve

The recipe's launch command runs inside the port image: export its environment, run the v0.30.0 fix profile plus
`vllm-long-prefill-share`, then `vllm serve thinkingmachines/Inkling-Small ...`. See
[docs/recipes/inkling-small.md](../../docs/recipes/inkling-small.md).

The model refuses to start on configurations it cannot serve: block size other than 128, contiguous PA, merged
prefill, prefill batch above 1, FP8 KV cache, speculative decoding, or a KV pool smaller than one `--max-model-len`
request.

## Switches

All read from the environment at startup. Off is the reference path; the recipe turns all of them on.

| Variable | On | Off |
| --- | --- | --- |
| `INKLING_DECODE_WINDOW_PLUS1=1` | Local layers get `512 / 128 + 1` decode blocks, so decode sees the whole window (the rel-bias mask cuts it at 512 exactly). | 4 blocks: decode sees 385 to 512 of the 512 keys while prefill sees 512. |
| `INKLING_MOE_SINGLE_REDUCE=1` | Routed and shared experts are summed per rank and all-reduced once per layer. The model refuses to start if FusedMoE does not grant `skip_final_all_reduce`. | FusedMoE reduces its own output and the shared experts are reduced separately. |
| `INKLING_LOCAL_WINDOW_PREFILL=1` | In chunked prefill and prefix-cache hits, local layers fetch and bias only the last `ceil(512 / 128) + 1` context blocks. | Every layer fetches and biases the whole context. |
| `INKLING_STEP_HOIST=1` | Conv-state slots, decode rel-bias indices and the prefill window context are computed once per forward. | Recomputed in every layer; identical results. |
| `INKLING_PREFILL_BIAS_BUDGET` | Bytes per prefill bias sub-block (default 1 GiB; the recipe uses 768 MiB, which keeps long-context graph replay inside HBM). | |

## Verify

```bash
pip install pytest   # not in the stock image
INKLING_MODEL_DIR=<checkpoint dir> SERVER_LOG=<server log> bash ports/inkling-small/tests/hpu/run_gates.sh        # quick tier
INKLING_MODEL_DIR=<checkpoint dir> bash ports/inkling-small/tests/hpu/run_gates.sh long                           # 200K / 250K tier
```

On CPU, without a device:

```bash
pytest ports/inkling-small/tests                                            # torch only; vLLM tests skip
GAUDI_FIXES_ROOT=<v0.30.0 sources> pytest ports/inkling-small/tests/test_inkling_patch.py
```

## Fixes

The generic bugs this model hits are fixes in [`fixes/`](../../fixes/README.md), applied by the recipe at container
start: the v0.30.0 profile (prompt order, sampling rows, structured-output rows, bad words, logits processors,
async inputs, resumed prefill, logprobs, seeded sampling, rank exceptions and the rest) plus the opt-in
`vllm-long-prefill-share`. The runner patch in `patches/` exists only for this model and leaves the
`swa-decode-window` fix's anchor intact, so the two apply in either order.
