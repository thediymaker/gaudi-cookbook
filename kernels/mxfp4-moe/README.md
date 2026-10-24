# MXFP4 MoE decode kernels for Gaudi 2

Two TPC kernels that run the routed experts of an MXFP4 mixture-of-experts layer straight from the packed
weights (E2M1 values + E8M0 scales, the layout vLLM already keeps on the device), with no bf16 unpack round trip.
They replace the stock `mixture_of_experts.bias_mxfp4_fused_weights` op for decode calls of up to 32 tokens.
Full write-up: [docs/kernels/mxfp4-moe.md](../../docs/kernels/mxfp4-moe.md).

| | |
| --- | --- |
| Hardware | Gaudi 2 (`tpc-clang -march=gaudi2`). Gaudi 3 is not built or tested. |
| Software | Gaudi software 1.24.1, vllm-gaudi `v0.26.0` (hook points also checked on `releases/v0.30.0`), lazy mode |
| Models | `openai/gpt-oss-120b` (measured). `openai/gpt-oss-20b` has the same layer shapes and is expected to work, but was not measured. |
| Scope | Decode only: at most 32 tokens and 128 routes per MoE call. Prefill keeps the stock op. |
| Switch | `MXFP4_TPC_MOE=1`. Off by default; anything unexpected keeps the stock op. |

## Layout

| Path | What it is |
| --- | --- |
| `src/mxk8_core.h`, `src/mxk8_w13.h`, `src/mxk8_w2.h` | TPC-C kernel sources: dequant + dot core, w13 GEMV + bias + SwiGLU-OAI, w2 GEMV + bias + routing-weight sum |
| `src/mxk8_w13xib.c`, `src/mxk8_w2xb.c` | The two gpt-oss kernels (compile-time configuration of the headers) |
| `src/mxk_tpc_lib.cpp`, `src/mxk_tpc.map` | Kernel library glue for the graph compiler. It forwards every stock GUID, so it is the only file in `GC_KERNEL_PATH`. |
| `src/mxk_op_lazy.cpp` | `torch.ops.custom_op.mxk_w13xib` / `mxk_w2xb` for lazy mode |
| `src/mxk_lib_gate.cpp` | Build gate: the library lists all stock GUIDs plus the two kernels |
| `build.sh` | Builds everything into `dist/` (or a given directory) and writes a `MANIFEST` |
| `python/mxfp4_tpc_moe_rt.py` | Runtime: env setup, bias relayout, the vLLM hooks |
| `python/install_mxfp4_tpc_moe.py` | Installs the runtime into vllm-gaudi; exits 1 if any anchor does not match |
| `python/mxk_harness.py` | Shared helpers for the tests and the benchmark |
| `tests/test_layout_cpu.py` | CPU test of the host-side layout code (pytest, no device) |
| `tests/parity_hpu.py`, `tests/run_parity.sh` | One-card parity against the stock op and a CPU fp32 reference |
| `bench/bench_moe.py` | Per-call micro benchmark, stock vs kernels |

Compiled libraries and recipe caches are build outputs and are not committed (see `.gitignore`).

## Toolchain

Everything comes from the Habana `pytorch-installer` base image (1.24.1); nothing proprietary is copied here:

- `tpc-clang` (TPC C/C++ compiler 2.7.33 in the 1.24.1 base) at `/opt/habanalabs/bin/tpc-clang`. Set
  `TPC_CLANG` if it is elsewhere.
- `tpc_kernel_lib_interface.h` in `/usr/include/habanalabs`.
- `hpu_custom_op.h` and `libhabana_pytorch_plugin.so` from the installed `habana_frameworks` package.
- `g++`, `objcopy` and `strip`, plus `python3` with the image's torch.

The op library links against the image's torch (`2.11.0a0+git009b5f6` and habana-torch-plugin `1.24.1.482` in the
measured image). Rebuild it for any other image.

## Build, install, enable

Inside the serving image (or a Dockerfile stage built from it):

```bash
cd kernels/mxfp4-moe
./build.sh /opt/mxfp4-moe/lib                 # kernels, glue, op library, gate, MANIFEST
python3 python/install_mxfp4_tpc_moe.py       # fails closed if vllm-gaudi does not match
```

Serve with the kernels on:

```bash
export PT_HPU_LAZY_MODE=1 HPU_MXFP4_NATIVE=1 MXFP4_TPC_MOE=1   # MXFP4_TPC_MOE_LIB=/opt/mxfp4-moe/lib is the default
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --enable-expert-parallel ...
```

`HPU_MXFP4_NATIVE=1` keeps the experts packed. That is the stock vllm-gaudi switch the kernels hook into. With the
flag on, the runtime sets three things itself, before the device opens:

- `GC_KERNEL_PATH=<lib>/libmxk_tpc.so`, with the previous value kept in `MXK_TPC_FWD_LIB`;
- `PT_HPU_LAZY_ACC_PAR_MODE=0`;
- a recipe-cache directory per kernel build.

Rollback: set `MXFP4_TPC_MOE=0`. The installed hooks are then inert and the engine runs exactly like stock.

## Verify

1. The engine log shows `MXFP4 TPC MoE installed: mxk_tpc mxfp4-moe-v8-<hash>` and
   `MXFP4 TPC MoE enabled (E_local=... K=2880 I=2880, <= 32 tokens, ...)`.
2. In the container, `grep -l libmxk_tpc.so /proc/*/maps` lists the engine worker processes.
3. One card: `tests/run_parity.sh [--checkpoint <gpt-oss-120b snapshot>]` must end with `PARITY PASS` and
   `SIG_DEVICE PASS`.
4. Speed: `bench/bench_moe.py --impl stock --json s.json`, then `--impl ours --json o.json`, then `--compare s.json o.json`.
