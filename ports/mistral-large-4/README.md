# Mistral Large 4 port

Serves [amd/Mistral-Large-4-1T-A52B-Quark-MXFP4](https://huggingface.co/amd/Mistral-Large-4-1T-A52B-Quark-MXFP4)
(1T MoE, 52B active, text only) on one 8-card Gaudi 2 node with vllm-gaudi `releases/v0.30.0`, 128K tokens per
request and 8 sequences, with DSpark speculative decoding, an int8 latent KV cache, prefix caching and KV offload to
host memory. The design and its evidence are in the write-up,
[docs/ports/mistral-large-4.md](../../docs/ports/mistral-large-4.md); the served configuration is the recipe
[docs/recipes/mistral-large-4.md](../../docs/recipes/mistral-large-4.md).

The checkpoint is under the Mistral Ecosystem Open-Weight License (modified MIT): a Model as a Service business
needs a separate commercial licence from Mistral for commercial use; internal use is exempt.

## What is here

| Path | What it is |
| --- | --- |
| `install.sh` | The only build step. Runs inside the stock image and stops at the first step that does not apply. |
| `Dockerfile` | `FROM` the stock image; copies this repository to `/opt/gaudi-cookbook` and runs `install.sh`. |
| `vendor/` | The five frontend and config files of [vllm#64534](https://github.com/vllm-project/vllm/pull/64534) (params.json adapter, tokenizer, renderer, `/tokenize` protocol, reasoning and tool parser), each the stock v0.30.0 file plus exactly the pull request's hunks (a 3-way merge; `vendor/MANIFEST` says how and pins every hash). Apache-2.0 with their vLLM headers. |
| `patches/install_pr_files.py` | Installs the vendored files over the stock v0.30.0 files, each only over the recorded stock bytes. |
| `patches/kernel_env_hook.py` | A hook at the top of `vllm_gaudi/__init__.py` that registers the kernel libraries in every process before the first HPU op (`ML4_MXK=0` makes it a no-op). |
| `ml4_hpu/` | The vLLM general plugin `mistral_large4_hpu`: the HPU model (MLA with the sparse indexer, shared KV layers, SSMax, LeRoPE), the ranged per-rank loader, the virtual expert split, the Quark MXFP4 claim, the int8 latent KV codec, decode and prompt HPU graphs, the DSpark drafter, prefix-cache and offload support, and the start-up warm-up module. `ref/` is the pure-torch reference the CPU tests compare against. |
| `kernels/k1/` | Model-only TPC kernels for the decode step (fused indexer, sparse attention gather, node-cut kernels) and `build_k1.sh`. |
| `tests/` | CPU tests (pytest; skip without torch) and `build_gates.py`, which `install.sh` runs. |
| `tests/hpu/` | Device gates: `run_gates.sh` against a running server, `run_kernel_gates.sh` on one free card. |

Every patch follows the [fix contract](../../fixes/README.md#contract): it checks every anchor or hash before
writing, `--check` reports the state (0 applied, 1 not applied, 2 the code does not match), and a second run is a
no-op.

## Build

From the repository root, on a host with the stock image built as in [Getting started](../../docs/getting-started.md):

```bash
docker build -f ports/mistral-large-4/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> \
  -t <your-mistral-large-4-image> .
```

`install.sh` runs, in order:

1. checks that the image has vllm-gaudi `058f107` and vLLM `ced6857a`;
2. installs mistral-common 1.13.0 (`--no-deps`, sha256-pinned; the image's 1.11.7 has no `low` and `medium`
   reasoning efforts), so the build needs network access to PyPI;
3. installs the vendored files of vllm#64534;
4. installs the `mistral_large4_hpu` plugin and the kernel environment hook;
5. builds the MXFP4 MoE kernels with `kernels/mxfp4-moe/build.sh --variant ml4` into
   `/opt/gaudi-cookbook-ml4/mxk` (both library gates run inside the build);
6. builds the decode-step kernels with `kernels/k1/build_k1.sh` into `/opt/gaudi-cookbook-ml4/k1`;
7. runs `tests/build_gates.py --kernels`: vendored files, plugin registration, the params.json adapter on a fixture,
   the quantisation claim, both kernel libraries and the hook.

It also works in a running container started from the stock image, with the repository at any path. The release's
generic fixes are applied by the recipe's launch step, not here.

## Serve

Download the checkpoint to a local directory (the plugin reads `consolidated-*.safetensors` and `params.json`
directly), then use the Launch block of the [recipe page](../../docs/recipes/mistral-large-4.md) inside the port
image. `vllm serve` gets that directory and `--served-model-name amd/Mistral-Large-4-1T-A52B-Quark-MXFP4`.

Each rank reads only its slices of the 578 GB checkpoint (71.7 GiB of weights per card, 63.7 GiB of them experts).
The warm-up module (`python3 -m mistral_large4_hpu.warm &`) waits for the server, sends every prompt width, decode
shape and follow-up turn once, and writes `/tmp/ml4-warm.done`; point a readiness probe at that file. It fails open
after `ML4_WARMUP_MAX_S` (2700 s by default).

## Switches

All are read from the environment at start. The recipe sets the ones in the first column; the rest are on by
default in the code.

| Variable | Recipe | What it does | Off |
| --- | --- | --- | --- |
| `ML4_MXK` | default `1` | MoE through the `ml4` MXFP4 kernels (v8 per route, decode up to 16 tokens; prefill unpack) | `0`: the stock MXFP4 op on the same virtual experts, the reference path |
| `ML4_MXK_V9G` | `1` | The grouped-route kernels for 2 to 64 tokens: each local expert unpacked once per unit of up to 2 routes. Bit-identical to v8. | `0`: v8 per route |
| `ML4_PLACEMENT` | `B` | The two halves of each expert (I 6144 split into 2 x 3072) on different ranks | `A`: both halves on one rank |
| `ML4_SPEC` | `dspark` | The checkpoint's DSpark head drafts 3 tokens per step in the n-gram slot of `--speculative-config` | Unset, and drop `--speculative-config` |
| `ML4_KV_FP8`, `ML4_KV_CODEC`, `ML4_INDEX_U8` | `1`, `i8-t128-f32`, `1` | Int8 latent KV in 128-wide tiles with fp32 scales (656 B per token per layer) and byte indexer keys (lossless) | Unset all three: bf16 KV; the pool then holds about half as many tokens, use a 64K window |
| `ML4_LAYER_GRAPHS`, `ML4_LAYER_GROUP` | `1`, `8` | Decode replays HPU graphs of 8 layers (9 graphs per step) | `0`: eager decode, about 13x slower |
| `ML4_INDEX_KERNEL`, `ML4_ATTN_KERNEL` | `1` | Decode-step kernels K1 (fused indexer) and K2 (sparse attention gather), exact | `0` |
| `ML4_NUMERICS_RW`, `ML4_R5_*`, `ML4_R6_*` | as in the recipe | Fewer graph nodes per layer: the indexer's integer helpers rewritten, and node-cut TPC kernels (RMS tail, RoPE, router, fake-FP8, SSMax, latent pack), each bitwise equal to the native ops | Unset; never `ML4_R5_ROUTER_MODE=4` (inexact) |
| `ML4_LOAD_FAST` | `1` | Pipelined loading: the next layers staged on the host while one is copied to the card, same bytes | `0`: one layer at a time |
| `ML4_PREFILL_GRAPHS` | default `1` | Prompt HPU graphs for 64 to 768-row chunks | `0` |
| `ML4_VERIFY_ROWS` | default `1` | DSpark verify as `[request, 4]` rows, one indexer pass per request | `0` |
| `ML4_ATTN_FSDPA`, `ML4_INDEX_SPLIT`, `ML4_BT_PROMPT_HALF` | default `1` | Long-prompt speed: bf16 FusedSDPA prompt attention, indexer rows split over ranks, 3/4-power-of-two key widths | `0` each |
| `ML4_SPEC_GRAMMAR` | default `1` | DSpark keeps drafting for grammar requests (tools, JSON): drafts cut at the first invalid token | `0` |
| `ML4_WARMUP` | default `1` | The warm-up module sends its requests | `0`: writes the done file at once |
| `ML4_VISION` | default `0` | The checkpoint's vision tower | `1` is refused: not in this port |

## Verify

```bash
python3 -m pytest ports/mistral-large-4/tests                    # CPU; skips what it cannot run here
ports/mistral-large-4/tests/hpu/run_kernel_gates.sh              # one free card, before the server starts
BASE=http://127.0.0.1:8000 ports/mistral-large-4/tests/hpu/run_gates.sh   # against a running, warmed server
```

`run_kernel_gates.sh` runs the MoE kernel parity (`PARITY PASS` for the v8 and grouped paths with the grouped path
bit-identical to v8, `PREFILL PASS`, `SIG_DEVICE PASS`) and the bitwise checks of the decode-step kernels.
`run_gates.sh` prints one PASS or FAIL line per gate: 12 golden prompts, greedy repeat, tool calls (plain and
streamed), needles at 8K, 32K, 64K and 127K tokens, 100 logprob requests, and teacher-forced scoring. `ml4_gates.py
sweep <context> 1,4` measures decode speed at 1 and 4 streams. With `REF_DIR` (an earlier run's output, for example
with `ML4_MXK=0`), golden and teacher-forced results are compared across the two servers.

## Not included

- The vision tower (the checkpoint has one; the port serves text only).
- 256K context. It ran with context parallelism over the 8 cards (pool 589,824 tokens), but short requests waited
  behind a 256K prefill and short-context decode lost 17% to 22%, so it is not in this port.
- The fp32 CPU reference that produced the perplexity bars in the write-up: it streams the checkpoint through `ref/`
  one layer at a time and needs a large CPU host.
- Async scheduling with DSpark: exact at 1 stream, not bit-equal to the synchronous path at 4 and 8 streams, so
  `--no-async-scheduling` stays.
