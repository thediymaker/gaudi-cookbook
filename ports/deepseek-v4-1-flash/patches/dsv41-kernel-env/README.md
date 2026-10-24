# dsv41-kernel-env

Register the port's kernel library in every vLLM process before the Gaudi device opens.

| | |
| --- | --- |
| Affects | DeepSeek-V4.1 served with this port and `DSV41_MXK=1` |
| Releases | v0.30.0 |
| Patches | `vllm_gaudi/__init__.py` |
| Test | `ports/deepseek-v4-1-flash/tests/test_kernel_env_patch.py` (stdlib only) |

## Problem

The graph compiler reads `GC_KERNEL_PATH` once, when a process opens the device. The MXFP4 decode kernels and the
Sinkhorn kernel live in `libmxk_tpc.so`, so that library must be on `GC_KERNEL_PATH` in the API server, the engine
core and every tensor-parallel worker before their first HPU op. The model plugin is imported too late in the workers
for that, and exporting the variable by hand is easy to get wrong (the library forwards every other kernel to the
stock library, which it finds through `MXK_TPC_FWD_LIB`, and the recipe cache must not mix graphs compiled with and
without it).

## The fix

`vllm_gaudi/__init__.py` is the platform plugin that vLLM imports in every process before the device opens. The patch
puts a call to `dsv41_hpu.mxk.setup_env()` at its top. `setup_env()` does nothing unless `DSV41_MXK=1`; then it
sets `GC_KERNEL_PATH`, `MXK_TPC_FWD_LIB` and `PT_HPU_LAZY_ACC_PAR_MODE=0`, and tags the recipe-cache directory with
the library version. An exception in it never breaks the platform import: the model logs that the kernels are off and
runs the stock op.

The patch is DeepSeek-only (it imports the port's plugin), so it lives in the port and not in `fixes/`. It follows
the fix contract: the anchor must match exactly once, nothing is written on drift, and a marker makes it idempotent.

## Verify

```bash
python3 ports/deepseek-v4-1-flash/patches/dsv41-kernel-env/apply.py --check   # 0 applied, 1 not yet, 2 drift
```
