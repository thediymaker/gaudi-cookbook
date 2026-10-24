# deepseek-v4-1-flash

DeepSeek-V4.1-Flash on Gaudi 2: a vLLM model plugin, one source patch, a model-only TPC kernel and the checkpoint
conversion. The design and its evidence are in [docs/ports/deepseek-v4-1-flash.md](../../docs/ports/deepseek-v4-1-flash.md);
the served configuration is [recipes/deepseek-ai/DeepSeek-V4.1-Flash-1017](../../recipes/deepseek-ai/DeepSeek-V4.1-Flash-1017/recipe.yaml)
with its page [docs/recipes/deepseek-v4-1-flash-1017.md](../../docs/recipes/deepseek-v4-1-flash-1017.md).

Base: vllm-gaudi `releases/v0.30.0` (`058f107`) with vLLM `ced6857a`, Gaudi software 1.24.1, lazy mode. One replica
is one 8-card node (attention TP 8, experts EP 8), 131,072-token context.

## Contents

| Path | What it is |
| --- | --- |
| `dsv41_hpu/` | The vLLM general plugin (entry point `dsv41_hpu`). Registers `DeepseekV41ForCausalLM` and the `dsv41_hpu_mxfp4` quant config the converted checkpoint declares, hooks the HPU runner (per-request block tables, block allocation numbers, Engram hash rows), corrects the frontend's reasoning-effort aliases, and caps prefix-cache hits for decoder replay. |
| `patches/dsv41-kernel-env/` | Puts a call to `dsv41_hpu.mxk.setup_env()` at the top of `vllm_gaudi/__init__.py`, so every process registers the kernel library before the device opens. Fix contract (`--check`, exit 2 on drift). |
| `kernels/dsv41_sinkhorn/` | The mHC Sinkhorn loop (20 iterations on 4x4 matrices) as one TPC kernel, built into the MXFP4 kernel library with `--extra-kernel`. |
| `kernels/image_gate.py` | Build gate: op library, kernel library version, served layer covered, env hook on, off and with the library missing. |
| `convert/` | `convert_to_hpu_mxfp4.py` (the lossless re-layout the plugin serves) and `validate_conversion.py` (checks it by value with its own decoders). |
| `serving/` | `check_checkpoint.py` (fail closed unless the path holds a converted checkpoint) and `warmup.py` (compiles every prefill and decode shape at start and writes a readiness file). |
| `tests/` | CPU tests (pytest): numerics, top-k, the stack (chunking invariance, switches, decode graphs), decoder replay, cache tags, the conversion and the kernel-env patch. |
| `tests/hpu/` | Device gates against a running server: `run_gates.sh`, and `kernel_ab.py` to compare a kernel-off and a kernel-on run. |

Generic runner bugs the model depends on are fixes in [`fixes/`](../../fixes/README.md), applied at launch with the
v0.30.0 profile. The plugin refuses to start if `prompt-order`, `resumed-prefill-logits`, `gather-logprobs-int32` or
`prompt-logprobs` is missing.

## 1. Convert the checkpoint

On any machine with Python 3.10+ and torch (CPU only, about 520 GB of free disk next to the 510 GB source):

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir SRC
python3 ports/deepseek-v4-1-flash/convert/convert_to_hpu_mxfp4.py --src SRC --dst DST --part all \
  --source deepseek-ai/DeepSeek-V4.1-Flash-1017@<revision>
python3 ports/deepseek-v4-1-flash/convert/validate_conversion.py --src SRC --dst DST --part all   # VALIDATE_OK
```

Nothing is requantized: expert bytes are copied (gate and up rows interleaved, -0 written as +0), FP8 linears become
bf16 exactly, Engram tables are split into hash buckets byte for byte. The parts (`experts --layers 0-4`, `mtp`,
`dense`, `engram --layer 1`, `engram --layer 14`, then `finalize`) can run separately or in parallel; each writes a
manifest with the sha256 of every tensor, and `finalize` checks that every source tensor went somewhere exactly once.
The original `deepseek-ai/DeepSeek-V4.1-Flash` converts with the same commands.

## 2. Build the image

From the repository root, on a stock vllm-gaudi v0.30.0 image built as in [Getting started](../../docs/getting-started.md):

```bash
docker build -f ports/deepseek-v4-1-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> \
  -t <your-deepseek-v4-1-flash-image> .
```

`install.sh` checks the release commits, installs the plugin, applies `patches/dsv41-kernel-env`, builds the MXFP4
kernels (variant `dsv41`) with the Sinkhorn kernel into `/opt/mxfp4-moe/lib`, and runs the image gate. It stops at
the first step that fails.

## 3. Serve

Use the Launch block of the [recipe page](../../docs/recipes/deepseek-v4-1-flash-1017.md). It checks the checkpoint,
starts `serving/warmup.py` in the background, applies the v0.30.0 fix profile and runs `vllm serve` on
`$DSV41_CHECKPOINT`. The warm-up writes `/tmp/dsv41-warmup.done` when every shape is compiled (or after
`DSV41_WARMUP_MAX_S`); point a readiness probe at that file.

## 4. Verify

After the warm-up, on an otherwise idle server:

```bash
ports/deepseek-v4-1-flash/tests/hpu/run_gates.sh --log <server log> --kernel on --out gates-on
```

For the kernel A/B, restart the server with `DSV41_MXK=0`, run the gates again with `--kernel off --out gates-off`,
then `python3 ports/deepseek-v4-1-flash/tests/hpu/kernel_ab.py gates-off gates-on`. Set `DSV41_TAG_CHECK=count` on
the server to have the `TAGS` gate check that no request read a cache row it does not own.

CPU tests: `pytest ports/deepseek-v4-1-flash/tests` (needs torch and numpy; about 3 minutes).

## Switches

Every speed-up is a switch, off by default in the plugin; the recipe turns on what is served. With all of them off
the plugin runs its reference path.

| Variable | Default | Served | What it does |
| --- | --- | --- | --- |
| `DSV41_LAYER_GRAPHS` | 0 | 1 | Decode replays HPU graphs of layer groups (eager otherwise) |
| `DSV41_LAYER_GROUP` | 1 | 10 | Layers per decode graph |
| `DSV41_DET_TOPK` | 0 | 1 | Indexer top-k with a stable order on ties |
| `DSV41_TOPK_SELECT` | sort | thresh | Prefill: sort-free threshold top-k, the same sets |
| `DSV41_PREFILL_ATTN` | 0 | 1 | Prefill: banded window attention and the compressed part under one softmax |
| `DSV41_INDEX_BF16_DOT` | 0 | 1 | Prefill: the index dot as a bf16 GEMM (bit-identical: both operands are fake-FP4) |
| `DSV41_INDEX_STEP_TILES` | 8 | 8 | Prefill: index scoring split into this many graph tiles |
| `DSV41_REPLAY_KEEP` | 0 | 140 | Decoder replay: layers 21 to 39 run on the last N tokens of each prompt chunk |
| `DSV41_PREFILL_HEADROOM_GIB` | 6 | 6 | Bound on prefill work the host launches ahead of the device (0 = off) |
| `DSV41_MHC_SLICED` | 0 | 1 | Decode: Sinkhorn on slices instead of full-matrix copies |
| `DSV41_HC_SINKHORN` | torch | kernel | The Sinkhorn loop as one TPC kernel (needs `DSV41_MXK=1`) |
| `DSV41_MHC_TPC_DOT` | 0 | 1 | The fp32 mHC mix as a TPC multiply and sum instead of an MME GEMM |
| `DSV41_FQ_FASTBITS` | 0 | 1 | Power-of-two helpers as bit masks |
| `DSV41_LEAN_OPS` | 0 | 1 | Fewer temporaries in cache addressing |
| `DSV41_DECODE_HOIST` | 0 | 1 | Per-step addressing once per decode graph instead of once per layer |
| `DSV41_ROPE_FUSED` | 0 | 1 | RoPE tables gathered once |
| `DSV41_INDEX_DECODE` | 0 | 1 | Decode: index scores for the new token only |
| `DSV41_WOA_LINEAR` | 0 | 1 | Decode: the grouped output projection as one batched linear |
| `DSV41_MXK` | 0 | 1 | MXFP4 decode kernels and the Sinkhorn kernel; 0 = stock ops |
| `DSV41_MXK_LIB` | `/opt/mxfp4-moe/lib` | | Where install.sh put the kernel build |
| `DSV41_TAG_CHECK` | mask | mask | Cache-row ownership check: `mask`, `count` (also counts violations, for the gates) or `off` |

The warm-up reads `DSV41_WARMUP` (1 = on), `DSV41_WARMUP_MAX_SEQS` (the serve `--max-num-seqs`), `DSV41_WARMUP_MAX_S`
and `DSV41_WARMUP_DONE` (the readiness file).

## Not included

The DSpark draft model (speculative decoding with the checkpoint's own drafter) is built and gated but not served;
its code is not in this port.

## Licence

Apache-2.0, like the rest of this repository. The model weights are MIT (DeepSeek). The decoder replay follows the
method described for DeepSeek's own engine, [DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0); no
code is copied from it.
