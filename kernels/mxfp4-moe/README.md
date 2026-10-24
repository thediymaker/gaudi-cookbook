# MXFP4 MoE kernels for Gaudi 2

TPC kernels that run the routed experts of an MXFP4 mixture-of-experts layer straight from the packed weights
(E2M1 values + E8M0 scales, the layout vLLM already keeps on the device), with no bf16 unpack round trip. They
replace the stock `mixture_of_experts.*mxfp4_fused_weights` ops for decode-sized calls, and for GLM-5.x also unpack
experts once for prefill. One kernel core builds a variant per model family.
Full write-up: [docs/kernels/mxfp4-moe.md](../../docs/kernels/mxfp4-moe.md).

| | |
| --- | --- |
| Hardware | Gaudi 2 (`tpc-clang -march=gaudi2`). Gaudi 3 is not built or tested. |
| Software | Gaudi software 1.24.1, vllm-gaudi `v0.26.0` or `releases/v0.30.0` (same torch and plugin), lazy mode |
| Integration | gpt-oss: [`python/`](python/) here, switch `MXFP4_TPC_MOE=1`. Other models: their port under [`ports/`](../../ports/README.md) builds a variant and calls the ops. |
| Scope | Decode: at most 32 tokens (gpt-oss) or 16 tokens (other variants) and 128 routes per call. Prefill keeps the stock op, except the GLM unpack path. |

## Variants

`build.sh --variant <name>` builds one variant. A serving image carries exactly one.

| Variant | Model | Kernels (GUIDs) | torch ops | Per-rank shapes it was gated at |
| --- | --- | --- | --- | --- |
| `gptoss` (default) | `openai/gpt-oss-120b`, `-20b` | `mxk_w13xib`, `mxk_w2xb` | `mxk_w13xib`, `mxk_w2xb` | K = I = 2880, top-4 of 128, interleaved w13 + bias, 64 local experts (TP2 + EP) |
| `glm` | GLM-5.3, GLM-5.2 (MXFP4) | `mxk_w13x`, `mxk_w2x`, `mxk_deq` | `mxk_w13x` (`interleaved=False`), `mxk_w2x`, `mxk_deq` | K 6144, top-8 of 256, [gate; up] w13, 64 experts x I 1024 (EP4 x TP2) |
| `dsv41` | DeepSeek-V4.1-Flash | `mxk_w13xi`, `mxk_w2x` (+ the port's `dsv41_sinkhorn`) | `mxk_w13x` (`interleaved=True`), `mxk_w2x` | K 5120, I 2304, top-6 of 384, interleaved w13, 48 local experts (EP8) |
| `muse` | Muse Spark Open | `mxk_w13x`, `mxk_w2xr` | `mxk_w13x` (`interleaved=False`), `mxk_w2xr` | K 4096, top-8 of 256, 128 experts x I 1024 (EP2 x TP4) |
| `kimi` | Kimi-K2.7 (MXFP4) | `mxk_w13xt`, `mxk_w2x` | `mxk_w13x` (`interleaved=False`), `mxk_w2x` | K 7168, I 2048, top-8 of 384, [gate; up] w13, 48 local experts (EP8) |
| `none` | a port that needs only its own kernel (GLM-5.3-Flash) | the `--extra-kernel` GUIDs only | the port's own ops | needs at least one `--extra-kernel` |

`mxk_w13xt` is `mxk_w13x` for rows longer than the 6144 values the kernel can stage in vector local memory: the
first 12 chunks of 512 come from local memory, the rest from the tensor (`MXK8_XTAIL` in `src/mxk8_core.h`).

## Build interface

```bash
kernels/mxfp4-moe/build.sh [--variant gptoss|glm|dsv41|muse|kimi] [--extra-kernel DIR]... [OUT_DIR]
```

Run it inside the serving image (or a Dockerfile stage `FROM` it); the default `OUT_DIR` is `dist/`. It writes:

| File | What it is |
| --- | --- |
| `libmxk_tpc.so` | The kernel library for `GC_KERNEL_PATH`. It serves the variant's kernels and forwards every other GUID to the stock `libtpc_kernels.so`, so `GC_KERNEL_PATH` stays one file and the recipe cache keeps working. Its version string is `mxk_tpc <variant>-v8-<hash>`. |
| `libmxk_op_lazy.so` | The variant's ops under `torch.ops.custom_op`, lazy mode. Load it with `torch.ops.load_library`. |
| `mxk_lib_gate` | The library gate binary: `mxk_lib_gate <libmxk_tpc.so> <stock lib> <guid>...` |
| `MANIFEST` | `version`, `variant`, `kernels`, `extra_kernels`, toolchain versions, `elf_sha256` and `code_sha256` per kernel, `sha256` of both libraries |

The build fails unless both gates pass:

- **library gate:** the stock GUIDs are listed unchanged and in order, followed by exactly the build's GUIDs; the
  six graph-compiler entry points are exported; `GetLibVersion` matches stock;
- **op gate** (`python/mxk_op_gate.py`): the version matches `MANIFEST`, the op library loads, it registers the ops
  of every GUID and no op of another variant.

`code_sha256` is the hash of the kernel's machine code (`.text` and `.tpc_metadata`, `python/tpc_elf_code.py`).
Unlike `elf_sha256` it does not change with comments or paths, which tpc-clang embeds in the ELF. Equal code hashes
mean the same kernel code; `tpc_elf_code.py --lib libmxk_tpc.so` prints them for any built library.

Environment: `TPC_CLANG` (compiler path), `MXK_BUILD_DIR` (scratch directory, default `/tmp/mxk_<variant>_v8_build`),
`MXK_STOCK_LIB` (stock kernel library for the gate, default `/usr/lib/habanalabs/libtpc_kernels.so`).

### Model-only kernels: `--extra-kernel DIR`

A port adds a TPC kernel of its own to the same library (one file in `GC_KERNEL_PATH`) with `--extra-kernel DIR`,
repeatable. `DIR` holds three files:

| File | Contract |
| --- | --- |
| `<guid>.c` | The TPC-C source. Its file name is the GUID. Self-contained: compiled as `tpc-clang -march=gaudi2 -O2 -x c++ <guid>.c` with no include path. |
| `glue.inc` | C++ included into `libmxk_tpc.so` after the shared helpers. It defines `static GlueCodeReturn inst_<guid>(HabanaKernelParams*, HabanaKernelInstantiation*)`. `ELF(<guid>)` is declared for it; it may use `copyElf`, `checkTypes`, `allRequired`, `tpcPoints` and `setIndexSpace` from `src/mxk_tpc_lib.cpp`. |
| `op.cpp` | A translation unit linked into `libmxk_op_lazy.so` that registers `torch.ops.custom_op.<guid>` with `TORCH_LIBRARY_FRAGMENT(custom_op, m)` and `TORCH_LIBRARY_IMPL(custom_op, HPU, m)` through `hpu_custom_op.h`. The op has the GUID's name. |

The extra GUIDs follow the variant's GUIDs in the library, both gates expect them, and their sources are part of the
version hash. Keep the input signature of an extra op distinct from the MoE ops' (see the lazy-mode pitfalls below).

## Op reference

All ops take device tensors in the served layout: `w` u8 `[E, N, K/2]` (two E2M1 values per byte, low nibble = even
k) and `s` u8 `[E, N, K/32]` (one E8M0 scale per 32 values), for the `E` local experts `e_min .. e_min + E - 1`.
`ids` are global expert ids, i32 `[Q]` with `Q = T * topk`; routes to other ranks' experts add nothing.
`Ip` is I rounded up to 512.

```text
mxk_w13x(xp, ids, w13, s13, topk, e_min, alpha, limit, beta, ascale, interleaved) -> ap bf16 [Q, Ip]
mxk_w2x(ap, ids, tw, w2, s2, topk, e_min)                                          -> bf16 [T, K]
mxk_w2xr(ap, ids, w2, s2, topk, e_min)                                             -> bf16 [Q, K]
mxk_w13xib(xp, ids, w13, s13, b13k, topk, e_min, alpha, limit, beta, ascale)       -> ap bf16 [Q, Ip]
mxk_w2xb(ap, ids, tw, w2, s2, b2k, topk, e_min)                                    -> bf16 [T, K]
mxk_deq(w, s, rowperm=0)                                                           -> bf16 [.., N, K]
```

- `xp` is the activation `x` bf16 `[T, K]` permuted per 512-k chunk:
  `xp = x.reshape(T, K // 512, 128, 4).transpose(-1, -2).reshape(T, K)` (K a multiple of 512). `mxk_w13xib` takes
  `x` zero-padded to a multiple of 512 first (`mxfp4_tpc_moe_rt.permute_x`).
- `w13` holds `[gate (I); up (I)]` rows (`interleaved=False`) or `[g0, u0, g1, u1, ...]` (`True`). The kernel
  computes `a = min(g, limit) * sigmoid(alpha * min(g, limit)) * (clamp(u, -limit, limit) + beta) * ascale` in fp32:
  SiLU is `alpha 1, limit 3.0e38, beta 0, ascale 1`; gpt-oss SwiGLU-OAI is `1.702, 7, 1, 1`. `ap` comes out in the
  permuted column order `mxk_w2x` reads; rows of non-local routes are not written.
- `tw` is bf16 `[Q]`, the routing weight of each route (any routed scaling factor already applied).
  `mxk_w2x` returns the routing-weighted sum over each token's local routes. `mxk_w2xr` returns one row per route,
  zero for routes whose expert is not local; call it with `topk=1`.
- `mxk_w13xib` / `mxk_w2xb` take the gpt-oss biases re-laid once at load time by `mxfp4_tpc_moe_rt.bias13` and
  `bias2` (f32 `[E, 4, Ip/2]` and `[E, 2, K/2]`).
- `mxk_deq` unpacks one expert (`[N, K/2]`, the stock bf16 MoE op needs one tensor per expert) or a stack to exact
  bf16 in natural order, for the GLM prefill path.

How each model calls them (the port does this in its MoE forward):

| Variant | Decode call (T tokens, at most 16) |
| --- | --- |
| `glm` | `ap = mxk_w13x(xp, ids, w13, s13, 8, e_min, 1.0, 3.0e38, 0.0, 1.0, False)`; `y = mxk_w2x(ap, ids, tw, w2, s2, 8, e_min)` |
| `glm` prefill (256 to 4096 tokens) | `l13 = [mxk_deq(w13[e], s13[e], 0) for e in range(E)]`, the same for `w2`, `htorch.core.mark_step()`, then `torch.ops.hpu.mixture_of_experts(x, ids, tw, w12=l13, w3=l2, permuted_weights=True, activation="silu", experts_min=e_min, experts_max=e_max, chunk_size=..., total_experts=...)` with the stock op's chunk arguments |
| `dsv41` | scales served with E8M0 + 7 on w13: `ap = mxk_w13x(xp, ids, w13, s13, 6, e_min, 2**-7, 10 * 2**7, 0.0, 2**-14, True)`; `y = mxk_w2x(ap, ids, tw, w2, s2, 6, e_min)` with `tw` unscaled by the stock op's `2**-14` |
| `muse` | `ap = mxk_w13x(xp, ids, w13, s13, 8, e_min, 1.0, 3.0e38, 0.0, 1.0, False)`; `rows = mxk_w2xr(ap, ids, w2, s2, 1, e_min)` |
| `kimi` | `ap = mxk_w13x(xp, ids, w13, s13, 8, e_min, 1.0, 3.0e38, 0.0, 1.0, False)`; `y = mxk_w2x(ap, ids, tw, w2, s2, 8, e_min)` |
| `gptoss` | `python/mxfp4_tpc_moe_rt.moe_decode` |

Before the first HPU op of every process, set `GC_KERNEL_PATH=<lib>/libmxk_tpc.so` (keep the previous single value
in `MXK_TPC_FWD_LIB`) and `PT_HPU_LAZY_ACC_PAR_MODE=0`, and move `PT_HPU_RECIPE_CACHE_CONFIG` to a directory per
kernel version. `mxfp4_tpc_moe_rt.setup_env` does exactly this for gpt-oss; ports do the same.

## Layout

| Path | What it is |
| --- | --- |
| `src/mxk8_core.h`, `src/mxk8_w13.h`, `src/mxk8_w2.h` | TPC-C kernel core: dequant + dot, w13 GEMV + gated activation, w2 GEMV (sum or per route) |
| `src/mxk8_w13x.c`, `mxk8_w13xi.c`, `mxk8_w13xt.c`, `mxk8_w13xib.c`, `mxk8_w2x.c`, `mxk8_w2xr.c`, `mxk8_w2xb.c` | One file per kernel: the compile-time configuration of the headers |
| `src/mxk_deq.c` | The prefill unpack kernel (GLM) |
| `src/mxk_tpc_lib.cpp`, `src/mxk_tpc.map` | Kernel library glue, compiled per variant with `-DMXK_HAS_<guid>`; forwards every stock GUID |
| `src/mxk_op_lazy.cpp` | The torch ops, compiled per variant the same way |
| `src/mxk_lib_gate.cpp` | Library gate |
| `build.sh` | Builds a variant into `dist/` (or a given directory) and writes a `MANIFEST` |
| `python/mxfp4_tpc_moe_rt.py`, `python/install_mxfp4_tpc_moe.py` | gpt-oss runtime and its fail-closed installer into vllm-gaudi |
| `python/mxk_harness.py` | Variant shapes, CPU fp32 reference, stock and kernel runners for the tests and the benchmark |
| `python/mxk_op_gate.py`, `python/tpc_elf_code.py` | Op gate; kernel code hashes |
| `tests/test_*.py` | CPU tests of the host-side layout code (pytest, no device) |
| `tests/parity_hpu.py`, `tests/run_parity.sh` | One-card parity per variant against the stock op and a CPU fp32 reference |
| `bench/bench_moe.py` | Per-call micro benchmark per variant, stock vs kernels |

Compiled libraries and recipe caches are build outputs and are not committed (see `.gitignore`).

## Toolchain

Everything comes from the Habana `pytorch-installer` base image (1.24.1); nothing proprietary is copied here:

- `tpc-clang` (TPC C/C++ compiler 2.7.33 in the 1.24.1 base) at `/opt/habanalabs/bin/tpc-clang`. Set
  `TPC_CLANG` if it is elsewhere.
- `tpc_kernel_lib_interface.h` in `/usr/include/habanalabs`.
- `hpu_custom_op.h` and `libhabana_pytorch_plugin.so` from the installed `habana_frameworks` package.
- `g++`, `objcopy` and `strip`, plus `python3` with the image's torch.

The op library links against the image's torch (`2.11.0a0+git009b5f6` and habana-torch-plugin `1.24.1.482` in both
the `v0.26.0` and the `v0.30.0` images). Rebuild it for any other image.

## gpt-oss: build, install, enable

Inside the serving image (or a Dockerfile stage built from it):

```bash
cd kernels/mxfp4-moe
./build.sh /opt/mxfp4-moe/lib                 # gptoss variant: kernels, glue, op library, gates, MANIFEST
python3 python/install_mxfp4_tpc_moe.py       # fails closed if vllm-gaudi does not match
```

Serve with the kernels on:

```bash
export PT_HPU_LAZY_MODE=1 HPU_MXFP4_NATIVE=1 MXFP4_TPC_MOE=1   # MXFP4_TPC_MOE_LIB=/opt/mxfp4-moe/lib is the default
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --enable-expert-parallel ...
```

`HPU_MXFP4_NATIVE=1` keeps the experts packed. That is the stock vllm-gaudi switch the kernels hook into. With the
flag on, the runtime sets `GC_KERNEL_PATH`, `MXK_TPC_FWD_LIB`, `PT_HPU_LAZY_ACC_PAR_MODE=0` and a recipe-cache
directory per kernel build itself, before the device opens. Rollback: set `MXFP4_TPC_MOE=0`. The installed hooks
are then inert and the engine runs exactly like stock.

## Verify

1. One card, before enabling any variant:
   `tests/run_parity.sh --variant <name> [--checkpoint <snapshot>]` must end with `PARITY PASS` and
   `SIG_DEVICE PASS` (for `glm` also `PREFILL PASS`). Without `--checkpoint` it uses random weights of the
   variant's shapes.
2. gpt-oss engine log: `MXFP4 TPC MoE installed: mxk_tpc gptoss-v8-<hash>` and
   `MXFP4 TPC MoE enabled (E_local=... K=2880 I=2880, <= 32 tokens, ...)`.
3. In the container, `grep -l libmxk_tpc.so /proc/*/maps` lists the engine worker processes.
4. Speed: `bench/bench_moe.py --variant <name> --impl stock --json s.json`, then `--impl ours --json o.json`, then
   `--compare s.json o.json`.

## Lazy-mode pitfalls

**Identical input signatures.** Lazy-mode legacy custom ops are cached by input signature, not by GUID: in one
process, a second custom op with the same input shapes silently returns the first op's result. Within each variant
the ops differ in tensor count or shapes, each op library holds one variant only (the op gate checks it), and
`tests/parity_hpu.py --impl sig` runs every op of the variant twice with identical shapes and different data in one
lazy graph and compares with the calls run alone. Benchmarks and parity runs use one implementation per process.

**Views from an earlier graph.** A legacy custom op that is given a slice view (for example one expert,
`w13[e]`) created in an earlier lazy graph, before a `mark_step`, can read the wrong data. Native ops read the
same view correctly. Take the view in the same graph as the call, as the `glm` prefill path does
(`mxk_deq(w13[e], s13[e], 0)` inside the MoE forward). The `sig` test for `glm` takes its expert views that way.
