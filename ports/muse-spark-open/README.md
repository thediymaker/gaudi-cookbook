# Muse Spark Open port

Serves [amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4)
(816B MoE, 42B active, text only) on one 8-card Gaudi 2 node with vllm-gaudi `releases/v0.30.0`, 256K tokens per
request and 4 sequences. The design and its evidence are in the write-up,
[docs/ports/muse-spark-open.md](../../docs/ports/muse-spark-open.md); the served configuration is the recipe
[docs/recipes/muse-spark-open.md](../../docs/recipes/muse-spark-open.md).

**Built with Muse Spark Open.** The checkpoint is under the Muse Model License (Muse Spark Open version), which asks
every product using the model to show that notice.

## What is here

| Path | What it is |
| --- | --- |
| `install.sh` | The only build step. Runs inside the stock image and stops at the first step that does not apply. |
| `Dockerfile` | `FROM` the stock image; copies this repository to `/opt/gaudi-cookbook` and runs `install.sh`. |
| `vendor/` | Verbatim copies of the platform-neutral new files of the upstream draft [vllm#61879](https://github.com/vllm-project/vllm/pull/61879) (config classes, processor, reasoning and tool parsers), Apache-2.0 with their vLLM headers. `vendor/MANIFEST` pins the PR head and each file's sha256. |
| `patches/vllm_pr_61879.py` | Installs the vendored files into the vLLM tree (refuses a file whose hash differs). |
| `patches/runner_window.py` | Runner patch for model type `muse_spark`: the window gate, the exact 2048-key decode window and the prefill off-by-one. Written on top of the [swa-decode-window](../../fixes/swa-decode-window/README.md) fix. |
| `patches/swa_kv_hooks.py` | 13 runner and 2 worker hooks for window-sized KV groups (`MUSE_SPARK_SWA_KV`). |
| `patches/kernel_env_hook.py` | A hook at the top of `vllm_gaudi/__init__.py` that points `GC_KERNEL_PATH` at the kernel library before the first HPU op, only when `MUSE_SPARK_MXFP4_KERNELS=1`. |
| `muse_spark_hpu/` | The vLLM general plugin: the HPU model, the ranged per-rank loader, the latent MoE, the Quark MXFP4 quantisation claim, windowed KV, long-context attention, the forced-tool-choice parser and the structured-output gate. |
| `serving/buckets-256k-seqs4.txt` | The bucket file (`VLLM_BUCKETING_FROM_FILE`). |
| `tests/` | CPU tests (pytest; skip without torch or vLLM) and `build_gates.py`, which `install.sh` runs. |
| `tests/hpu/` | Device gates against a running server: `run_gates.sh`. |

Every patch follows the [fix contract](../../fixes/README.md#contract): it checks every anchor before writing,
`--check` reports the state (0 applied, 1 not applied, 2 the code does not match), and a second run is a no-op.

## Build

From the repository root, on a host with the stock image built as in [Getting started](../../docs/getting-started.md):

```bash
docker build -f ports/muse-spark-open/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> \
  -t <your-muse-spark-open-image> .
```

`install.sh` runs, in order:

1. checks that the image has vllm-gaudi `058f107` and vLLM `ced6857a`;
2. applies the cookbook's `v0.30.0` fix profile (the runner patches are written on top of it);
3. installs the vendored files of vllm#61879;
4. patches the HPU runner (`runner_window.py`, `swa_kv_hooks.py`);
5. installs the `muse_spark_hpu` plugin and the kernel environment hook;
6. builds the MXFP4 decode kernels with `kernels/mxfp4-moe/build.sh --variant muse` into
   `/opt/muse-spark-open/mxfp4-kernels` (both kernel gates run inside the build);
7. runs `tests/build_gates.py --kernels`: patches applied, plugin registered, parsers and quantisation method
   present, the window masks of the patched runner, the kernel library and its environment hook.

It also works in a running container started from the stock image, with the repository at any path.

## Serve

Use the Launch block of the [recipe page](../../docs/recipes/muse-spark-open.md) inside the port image. The launch
step runs `fixes/apply_all.py --profile v0.30.0 topk-subset-sampler vllm-long-prefill-share` again: the profile
part reports "applied", and the two opt-in fixes are added.

Weights load with the port's own loader: each rank reads only its slices (70.8 GB) of the 509 GB checkpoint. The
load took 61 minutes (3,665 s) from network storage on the verification run; the AccThread of each worker is busy
for most of it, so faster storage helps less than it seems.

## Switches

All are read from the environment at start. The recipe sets every one of them.

| Variable | Recipe | What it does | Off |
| --- | --- | --- | --- |
| `MUSE_SPARK_MOE_TP_SPLIT` | `4` | Experts EP2 x TP4 at TP 8: two groups of four cards, each card a quarter of every local expert. | `1` is EP8 (whole experts per card), 33% slower per stream at batch 1 |
| `MUSE_SPARK_SWA_KV` | `1` | The 46 sliding layers get window-sized KV groups; 3.875x more tokens in the same pool. | Full-length KV in every layer (the pool then holds 4x less) |
| `MUSE_SPARK_LONG_ATTN` | `1` | Prompt chunks with more than 64K keys use a compact key-validity row and query tiles instead of a dense bias. | vllm-gaudi's dense bias, `[bs, 1, q, keys]` bf16: about 2 GiB plus a 1 GiB mask per 4096-token chunk at 256K |
| `MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS` | `5408` | Decode batches above this many KV blocks run without an HPU graph (each captured graph keeps about 0.5 MiB per block). | `0`: no cap |
| `MUSE_SPARK_DECODE_LAYER_GROUP` | `62` | One graph cut per 62 decoder layers in decode graphs instead of one per layer: 313 to 252 launches per step. | `1` (vllm-gaudi's cut per layer) |
| `MUSE_SPARK_MXFP4_KERNELS` | `1` | MXFP4 decode kernels for the expert layers (decode steps of up to 4 tokens). | `0`: the stock MXFP4 op, the reference path |
| `MUSE_SPARK_MEMLOG` | `1` | `muse_spark mem:` lines with device memory from TP rank 0. | No log lines |

Prefill always runs the stock MoE op. With `MUSE_SPARK_MXFP4_KERNELS=0` the port runs its reference path; the page
lists which gates ran in each mode.

## Verify

```bash
python3 -m pytest ports/muse-spark-open/tests                 # CPU; skips what it cannot run here
MUSE_URL=http://127.0.0.1:8000 ports/muse-spark-open/tests/hpu/run_gates.sh     # against a running server
```

`run_gates.sh` prints one PASS or FAIL line per gate: perplexity against a CPU reference
(`tests/hpu/reference/muse_ppl_reference.json.gz`), decode against prefill, needles at 2K, 32K and the served window,
tool calls, reasoning effort, greedy repeatability, and decode speed at 1 and 4 streams. To compare the kernel switch,
run it once with `MUSE_SPARK_MXFP4_KERNELS=0`, restart the server with `1`, and run again with
`--greedy-ref <first run's muse_greedy.json>`.

## Not included

- The vision and audio towers (the checkpoint has them; the port serves text only).
- Pipeline parallelism and FP8 variants.
- The CPU reference that produced the perplexity reference file: it streams the 509 GB checkpoint through a
  reference implementation one layer at a time and needs a large CPU host. The reference file records its
  numerics.
