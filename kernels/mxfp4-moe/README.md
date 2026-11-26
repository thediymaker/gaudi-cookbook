# MXFP4 MoE kernels for Gaudi 2

TPC kernels that run the routed experts of an MXFP4 mixture-of-experts layer straight from the packed weights
(E2M1 values + E8M0 scales, the layout vLLM already keeps on the device), with no bf16 unpack round trip. They
replace the stock `mixture_of_experts.*mxfp4_fused_weights` ops for decode-sized calls, and for GLM-5.x also unpack
experts once for prefill. One kernel core builds a variant per model family. Each variant has a
v10 kernel set except `muse` and `ml4`: faster decode kernels with bit-identical output, built into the same library next to the v8 ones.
Full write-up: [docs/kernels/mxfp4-moe.md](../../docs/kernels/mxfp4-moe.md); measured dead ends:
[docs/kernels/what-did-not-work.md](../../docs/kernels/what-did-not-work.md).

| | |
| --- | --- |
| Hardware | Gaudi 2 (`tpc-clang -march=gaudi2`). Gaudi 3 is not built or tested. |
| Software | Gaudi software 1.24.1, vllm-gaudi `v0.26.0` or `releases/v0.30.0` (same torch and plugin), lazy mode; `torch.compile` mode with the `--pt2` op library |
| Integration | gpt-oss: [`python/`](python/) here, switch `MXFP4_TPC_MOE=1` (v10: also `MXFP4_TPC_MOE_V10=1`). Other models: their port under [`ports/`](../../ports/README.md) builds a variant and calls the ops. |
| Scope | Decode: at most 32 tokens (gpt-oss) or 16 tokens (other variants) and 128 routes per call; the grouped-route kernels take up to 64 tokens and 512 routes. Prefill keeps the stock op, except the GLM and Mistral Large 4 unpack path. |

## Variants

`build.sh --variant <name>` builds one variant. A serving image carries exactly one.

| Variant | Model | v8 kernels (GUIDs) | v10 kernels added by `--kernels v10` (the default) | Per-rank shapes it was gated at |
| --- | --- | --- | --- | --- |
| `gptoss` (default) | `openai/gpt-oss-120b`, `-20b` | `mxk_w13xib`, `mxk_w2xb` | `mxk_w13xibn`, `mxk_w2gn`, `mxk_w2cb` | K = I = 2880, top-4 of 128, interleaved w13 + bias, 64 local experts (TP2 + EP) |
| `glm` | GLM-5.3, GLM-5.2 (MXFP4); MiMo-V2.6-Pro (v8 only) | `mxk_w13x`, `mxk_w2x`, `mxk_deq` | `mxk_w13xn`, `mxk_w2xc`, `mxk_w2gp`, `mxk_w2c` (+ the `mxk_w2g` ELF) | K 6144, top-8 of 256, [gate; up] w13, 64 experts x I 1024 (EP4 x TP2) |
| `dsv41` | DeepSeek-V4.1-Flash | `mxk_w13xi`, `mxk_w2x` (+ the port's `dsv41_sinkhorn`) | `mxk_w13xin`, `mxk_w2gn`, `mxk_w2c` (+ the `mxk_w2gp` ELF) | K 5120, I 2304, top-6 of 384, interleaved w13, 48 local experts (EP8) |
| `muse` | Muse Spark Open | `mxk_w13x`, `mxk_w2xr` | none: v8 only | K 4096, top-8 of 256, 128 experts x I 1024 (EP2 x TP4) |
| `kimi` | Kimi-K2.7 (MXFP4) | `mxk_w13xt`, `mxk_w2x` | `mxk_w13xk`, `mxk_w2xc`, `mxk_w2gp`, `mxk_w2c` (+ the `mxk_w2g` ELF) | K 7168, I 2048, top-8 of 384, [gate; up] w13, 48 local experts (EP8) |
| `ml4` | Mistral Large 4 (MXFP4) | `mxk_w13x`, `mxk_w2x`, `mxk_deq`, plus the grouped-route `mxk_w13g`, `mxk_w2g`, `mxk_w2c` | none: v8 only | K 6144, I 3072 (each expert of I 6144 split into two virtual experts), top-8 of 256, [gate; up] w13, 32 local experts (EP8) |
| `none` | a port that needs only its own kernel (GLM-5.3-Flash) | the `--extra-kernel` GUIDs only | none | needs at least one `--extra-kernel` |

Every GUID registers a torch op of the same name, except `mxk_w13xi` and `mxk_w13xt`, which run through the
`mxk_w13x` op (`interleaved=True` / the kimi build). An ELF in parentheses is built into the library without a GUID:
the glue serves it for shapes its GUID's main kernel does not cover (see [v8 and v10](#v8-and-v10)).

### v8 and v10

A `--kernels v10` build (the default for every variant that has a v10 set) holds the v8 kernels, compiled from the
same sources to the same machine code as a `--kernels v8` build, plus the v10 kernels. The integration picks the path
once per process from the ops the library registers, so the same image runs v8 (the switch off) or v10 (the switch
on), and rollback is the switch. Every v10 path gives the v8 path's output bit for bit; `tests/run_parity.sh` checks
that on every case.

| Path | Decode call | Variants |
| --- | --- | --- |
| v8 | (permute) -> v8 w13 -> `mxk_w2x` / `mxk_w2xb` | all |
| stack3 | v10 w13 -> grouped-route w2 (`mxk_w2gp` / `mxk_w2gn`, fp32 rows per route) -> combine (`mxk_w2c` / `mxk_w2cb`) | glm, kimi, dsv41, gptoss |
| stack2 | v10 w13 -> `mxk_w2xc` (one node) | glm, kimi |

What the v10 kernels change (details and measurements in
[the kernel page](../../docs/kernels/mxfp4-moe.md#v10-decode-kernels)):

- **w13:** the 4 row sums of a pass go through a packed reduction tree (all 4 rows share one vector after two
  butterfly steps) whose tail is carried into the next pass, and the batched-epilogue flush test is one branch
  (`MXG_W13C 2`, `MXG_FB 2` in `src/mxk10_w13.h`). For glm, dsv41 and gptoss the kernel also reads x in natural
  order and permutes it while staging it in vector local memory (`MXKX_NATX`), which removes the host permute node.
  kimi keeps the permuted x (natural-order staging needs 14 KB of VLM at K 7168) and reads its last chunk pair from
  global memory.
- **w2:** the routes are walked expert-outer, so two routes of the same expert are unpacked once (`mxk9_w2.h`), with
  the same packed and carried tree on the one-route passes and the two-route half passes (`mxk9_w2q.h` at nch 2,
  `mxk9_w2n.h` at nch 4 and 6), and a second kernel sums the per-route rows per token in `mxk_w2x`'s order. That
  second node is what keeps the per-token summation order (the sums of a whole step do not fit vector local memory).
  `mxk_w2xc` is the one-node alternative at nch 2.

The `ml4` build is v8 plus the grouped-route kernels the Mistral Large 4 port calls for 2 to 64 tokens per call
(multi-stream decode and speculative verify rows): `mxk_w13g` (the v8 w13 math with the routes of one local expert
unpacked together, x rows of grouped units read from an SRAM scratch tensor), `mxk_w2g` (fp32 rows per route) and
`mxk_w2c` (the per-token sum in `mxk_w2x`'s order). Each gives the v8 output bit for bit; the parity run checks the
`grouped` path against v8 on every case up to 64 tokens, with v8 called in 16-token pieces above 16.

Not on v10: `ml4` (no v10 set; a v10 build for this model was measured at about +1% end to end, inside run-to-run
noise), `muse` (its v10 build was faster at 3 to 4 streams but about 1.6% slower per token at one stream end
to end, and was never gated for serving), MiMo-V2.6-Pro (it builds
`glm` with `--pt2`, and the v10 kernels have no torch.compile ops), and GLM-5.3-Flash (`none`).

MiMo-V2.6-Pro uses the `glm` build: its per-rank shapes at EP8 (K 6144, I 2048, top-8 of 384, 48 local experts) are
inside the `glm` caps (K a multiple of 512 up to 6144, I a multiple of 512 up to 3072), its rows are `[gate; up]` and
its activation is plain SiLU. It serves in `torch.compile` mode, so its port builds with `--pt2`, and the port's
`tests/hpu/mxk_ops.py` checks the kernels at those shapes on checkpoint layers.

`mxk_w13xt` is `mxk_w13x` for rows longer than the 6144 values the kernel can stage in vector local memory: the
first 12 chunks of 512 come from local memory, the rest from the tensor (`MXK8_XTAIL` in `src/mxk8_core.h`).

## Build interface

```bash
kernels/mxfp4-moe/build.sh [--variant gptoss|glm|dsv41|muse|kimi|ml4] [--kernels v8|v10] [--extra-kernel DIR]... [--pt2] [OUT_DIR]
```

`--kernels` defaults to `v10` for every variant that has a v10 set and to `v8` for `muse` and `none`; `--kernels v10`
on those fails. With `--pt2` it defaults to `v8`, and `--kernels v10 --pt2` fails.

Run it inside the serving image (or a Dockerfile stage `FROM` it); the default `OUT_DIR` is `dist/`. It writes:

| File | What it is |
| --- | --- |
| `libmxk_tpc.so` | The kernel library for `GC_KERNEL_PATH`. It serves the variant's kernels and forwards every other GUID to the stock `libtpc_kernels.so`, so `GC_KERNEL_PATH` stays one file and the recipe cache keeps working. Its version string is `mxk_tpc <variant>-<v8|v10>-<hash>`. |
| `libmxk_op_lazy.so` | The variant's ops under `torch.ops.custom_op`, lazy mode. Load it with `torch.ops.load_library`. |
| `libmxk_op_pt2.so` | With `--pt2`: the same ops and schemas on the PT2 custom-op API, with Meta implementations, for `torch.compile` mode (`PT_HPU_LAZY_MODE=0`). Call each op once eagerly per process before the first compiled graph that uses it: the PT2 API registers the op's kernel with the bridge on that call. Not built for `gptoss` or with `--extra-kernel`. A process loads one of the two op libraries. |
| `mxk_lib_gate` | The library gate binary: `mxk_lib_gate <libmxk_tpc.so> <stock lib> <guid>...` |
| `MANIFEST` | `version`, `variant`, `kernel_set` (v8 or v10), `kernels`, `fallback_elfs` (ELFs without a GUID), `extra_kernels`, toolchain versions, `elf_sha256` and `code_sha256` per kernel, `sha256` of both libraries |

The build fails unless both gates pass:

- **library gate:** the stock GUIDs are listed unchanged and in order, followed by exactly the build's GUIDs; the
  six graph-compiler entry points are exported; `GetLibVersion` matches stock;
- **op gate** (`python/mxk_op_gate.py`): the version matches `MANIFEST`, the op library loads, it registers the ops
  of every GUID and no op of another variant. With `--pt2` it runs on both op libraries.

`code_sha256` is the hash of the kernel's machine code (`.text` and `.tpc_metadata`, `python/tpc_elf_code.py`).
Unlike `elf_sha256` it does not change with comments or paths, which tpc-clang embeds in the ELF. Equal code hashes
mean the same kernel code; `tpc_elf_code.py --lib libmxk_tpc.so` prints them for any built library.

Environment: `TPC_CLANG` (compiler path), `MXK_BUILD_DIR` (scratch directory, default `/tmp/mxk_<variant>_v8_build`
for both kernel sets),
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

v10 ops (same tensors; `ids2` / `tw2` are the `[T, topk]` views of `ids` / `tw`):

```text
mxk_w13xn(x, ids, w13, s13, topk, e_min, alpha, limit, beta, ascale, False)         -> ap  (x natural order; glm)
mxk_w13xk(xp, ids, w13, s13, topk, e_min, alpha, limit, beta, ascale, False)        -> ap  (xp permuted, K <= 7168; kimi)
mxk_w13xin(x, ids, w13, s13, topk, e_min, alpha, limit, beta, ascale, True)         -> ap  (x natural order; dsv41)
mxk_w13xibn(x, ids, w13, s13, b13k, topk, e_min, alpha, limit, beta, ascale)        -> ap  (x natural, unpadded; gptoss)
mxk_w2xc(ap, ids, tw, w2, s2, topk, e_min)                                          -> bf16 [T, K]  (mxk_w2x's contract)
mxk_w2gp(ap, ids2, w2, s2, e_min, gmax)                                             -> part f32 [Q, K]
mxk_w2gn(ap, ids2, w2, s2, e_min, gmax)                                             -> part f32 [Q, K]
mxk_w2c(part, ids2, tw2, e_min, e_local)                                            -> bf16 [T, K]
mxk_w2cb(part, ids2, tw2, b2k, e_min, e_local)                                      -> bf16 [T, K]
```

- Each v10 w13 op has the contract, shape caps and output of its v8 twin (`mxk_w13x`, `mxk_w13xt`, `mxk_w13xi`,
  `mxk_w13xib`), except the x layout noted. `mxk_w2xc` serves nch 2 (I <= 1024) with its own kernel and every other
  I with the variant's `mxk_w2x` ELF.
- `mxk_w2gp` serves nch 2 with the packed kernel and other I with the `mxk_w2g` ELF; `mxk_w2gn` serves nch 4 and 6
  (1024 < I <= 3072) and, in the dsv41 build, nch 2 with the `mxk_w2gp` ELF. Both need Q = T * topk <= 512, topk <= 16,
  I <= 3072. `gmax` caps the routes of one expert unpacked together; 2 is the gated setting.
- `mxk_w13g(xp, ids2, w13, s13, e_min, gmax, alpha, limit, beta, ascale) -> ap` and
  `mxk_w2g(ap, ids2, w2, s2, e_min, gmax) -> part` (the ml4 build) are the grouped-route forms of `mxk_w13x` and of
  `mxk_w2x`'s first half: Q = T * topk <= 512, topk <= 16, E <= 128, K % 512 == 0 and K <= 6144, I % 512 == 0 and
  I <= 3072. `mxk_w2g` serves any nch with the plain grouped kernel.
- `part` is private to the pair: feed it to `mxk_w2c` (or `mxk_w2cb` with the gpt-oss bias) with the same `ids2`.
  `e_local` is the number of local experts (`w2.shape[0]`).

How each model calls them (the port does this in its MoE forward):

| Variant | Decode call (T tokens, at most 16) |
| --- | --- |
| `glm` | `ap = mxk_w13x(xp, ids, w13, s13, 8, e_min, 1.0, 3.0e38, 0.0, 1.0, False)`; `y = mxk_w2x(ap, ids, tw, w2, s2, 8, e_min)` |
| `glm` prefill (256 to 4096 tokens) | `l13 = [mxk_deq(w13[e], s13[e], 0) for e in range(E)]`, the same for `w2`, `htorch.core.mark_step()`, then `torch.ops.hpu.mixture_of_experts(x, ids, tw, w12=l13, w3=l2, permuted_weights=True, activation="silu", experts_min=e_min, experts_max=e_max, chunk_size=..., total_experts=...)` with the stock op's chunk arguments |
| `dsv41` | scales served with E8M0 + 7 on w13: `ap = mxk_w13x(xp, ids, w13, s13, 6, e_min, 2**-7, 10 * 2**7, 0.0, 2**-14, True)`; `y = mxk_w2x(ap, ids, tw, w2, s2, 6, e_min)` with `tw` unscaled by the stock op's `2**-14` |
| `muse` | `ap = mxk_w13x(xp, ids, w13, s13, 8, e_min, 1.0, 3.0e38, 0.0, 1.0, False)`; `rows = mxk_w2xr(ap, ids, w2, s2, 1, e_min)` |
| `kimi` | `ap = mxk_w13x(xp, ids, w13, s13, 8, e_min, 1.0, 3.0e38, 0.0, 1.0, False)`; `y = mxk_w2x(ap, ids, tw, w2, s2, 8, e_min)` |
| `ml4` (T <= 16) | as `kimi`, on the 32 local virtual experts; 17 to 64 tokens in 16-token pieces |
| `ml4` grouped (2 <= T <= 64) | `ap = mxk_w13g(xp, ids2, w13, s13, e_min, 2, 1.0, 3.0e38, 0.0, 1.0)`; `y = mxk_w2c(mxk_w2g(ap, ids2, w2, s2, e_min, 2), ids2, tw2, e_min, 32)` |
| `ml4` prefill (256 to 4096 tokens) | as `glm` prefill |
| `gptoss` | `python/mxfp4_tpc_moe_rt.moe_decode` |
| v10 (stack3), glm / kimi / dsv41 | `ap = <v10 w13>(x or xp, ids, w13, s13, ...)` with the v8 activation parameters; `y = mxk_w2c(mxk_w2gp(ap, ids2, w2, s2, e_min, 2), ids2, tw2, e_min, E)` (`mxk_w2gn` for dsv41) |
| v10 (stack2), glm / kimi | `ap` as above; `y = mxk_w2xc(ap, ids, tw, w2, s2, topk, e_min)`, for calls outside the grouped limits |
| v10, `gptoss` | `python/mxfp4_tpc_moe_rt.moe_decode_v10` |

Before the first HPU op of every process, set `GC_KERNEL_PATH=<lib>/libmxk_tpc.so` (keep the previous single value
in `MXK_TPC_FWD_LIB`) and `PT_HPU_LAZY_ACC_PAR_MODE=0`, and move `PT_HPU_RECIPE_CACHE_CONFIG` to a directory per
kernel version. `mxfp4_tpc_moe_rt.setup_env` does exactly this for gpt-oss; ports do the same.

## Layout

| Path | What it is |
| --- | --- |
| `src/mxk8_core.h`, `src/mxk8_w13.h`, `src/mxk8_w2.h` | TPC-C kernel core: dequant + dot, w13 GEMV + gated activation, w2 GEMV (sum or per route) |
| `src/mxk8_w13x.c`, `mxk8_w13xi.c`, `mxk8_w13xt.c`, `mxk8_w13xib.c`, `mxk8_w2x.c`, `mxk8_w2xr.c`, `mxk8_w2xb.c` | One file per kernel: the compile-time configuration of the headers |
| `src/mxk_deq.c` | The prefill unpack kernel (GLM) |
| `src/mxk10_core.h`, `src/mxk10_w13.h`, `src/mxk10_w2core.h`, `src/mxk10_w2.h` | v10 w13 (natural-order staging, carried packed tree) and v10 one-node w2 at nch 2 |
| `src/mxk9_core.h`, `src/mxk9_pass.h`, `src/mxk9_pass2.h`, `src/mxk9_w2.h`, `src/mxk9_w2q.h`, `src/mxk9_w2n.h` | Grouped-route w2 (expert-outer, per-route fp32 rows) and its combine; the carried packed tree at nch 2 and at nch 4 / 6 |
| `src/mxk9_w13.h`, `src/mxk9_w13g.c` | Grouped-route w13 + gated activation (ml4 build), per route the v8 output |
| `src/mxk10_w13xn.c`, `mxk10_w13xk.c`, `mxk10_w13xin.c`, `mxk10_w13xibn.c`, `mxk10_w2xc.c`, `mxk9_w2g.c`, `mxk9_w2gp.c`, `mxk9_w2gn.c`, `mxk9_w2c.c`, `mxk9_w2cb.c` | One file per v10 kernel |
| `src/mxk_tpc_lib.cpp`, `src/mxk_tpc.map` | Kernel library glue, compiled per variant with `-DMXK_HAS_<guid>`; forwards every stock GUID |
| `src/mxk_op_lazy.cpp` | The torch ops, compiled per variant the same way |
| `src/mxk_op_pt2.cpp` | The same ops on the PT2 custom-op API for `torch.compile` mode (`--pt2`) |
| `src/mxk_lib_gate.cpp` | Library gate |
| `build.sh` | Builds a variant into `dist/` (or a given directory) and writes a `MANIFEST` |
| `python/mxfp4_tpc_moe_rt.py`, `python/install_mxfp4_tpc_moe.py` | gpt-oss runtime and its fail-closed installer into vllm-gaudi |
| `python/mxk_harness.py` | Variant shapes, CPU fp32 reference, stock and kernel runners for the tests and the benchmark |
| `python/mxk_op_gate.py`, `python/tpc_elf_code.py` | Op gate; kernel code hashes |
| `tests/test_*.py` | CPU tests of the host-side layout code and of the two op libraries' schemas (pytest, no device) |
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
export MXFP4_TPC_MOE_V10=1                                     # optional: the v10 kernels, same output bit for bit
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --enable-expert-parallel ...
```

`HPU_MXFP4_NATIVE=1` keeps the experts packed. That is the stock vllm-gaudi switch the kernels hook into. With the
flag on, the runtime sets `GC_KERNEL_PATH`, `MXK_TPC_FWD_LIB`, `PT_HPU_LAZY_ACC_PAR_MODE=0` and a recipe-cache
directory per kernel build itself, before the device opens. Rollback: set `MXFP4_TPC_MOE=0`. The installed hooks
are then inert and the engine runs exactly like stock. `MXFP4_TPC_MOE_V10=0` (or unset) goes back from v10 to the v8
kernels in the same library.

## Verify

1. One card, before enabling any variant:
   `tests/run_parity.sh --variant <name> --lib <build> [--checkpoint <snapshot>]` must end with `PARITY PASS` and
   `SIG_DEVICE PASS` (for `glm` also `PREFILL PASS`). Without `--checkpoint` it uses random weights of the
   variant's shapes. On a v10 build it runs every path (`PARITY PASS (paths v8, stack2, stack3)`, one
   `SIG_DEVICE PASS (path <p>)` per path) and fails unless each v10 path is bit-identical to v8 on every case.
2. gpt-oss engine log: `MXFP4 TPC MoE installed: mxk_tpc gptoss-v10-<hash>, decode v8 (...)` (or `decode v10`) and
   `MXFP4 TPC MoE enabled (E_local=... K=2880 I=2880, <= 32 tokens, ...)`.
3. In the container, `grep -l libmxk_tpc.so /proc/*/maps` lists the engine worker processes.
4. Speed: `bench/bench_moe.py --variant <name> --impl stock --json s.json`, then `--impl ours --json o.json` (add
   `--path stack3` for the v10 path), then `--compare s.json o.json`.

## Lazy-mode pitfalls

**Identical input signatures.** Lazy-mode legacy custom ops are cached by input signature, not by GUID: in one
process, a second custom op with the same input shapes silently returns the first op's result. A v10 op can have
its v8 twin's signature (`mxk_w2xc` and `mxk_w2x` always do), so the integrations pick one path per process and
never call both for the same weights. Within each variant
the ops differ in tensor count or shapes, each op library holds one variant only (the op gate checks it), and
`tests/parity_hpu.py --impl sig` runs every op of the variant twice with identical shapes and different data in one
lazy graph and compares with the calls run alone. Benchmarks and parity runs use one implementation per process.

**A graph of only custom ops.** An HPU graph captured from custom ops alone replays as an empty graph and returns
a stale buffer. The v10 paths have no native node (no host permute), so `bench/bench_moe.py --path` ends each call
with a multiply by 1.0. In a model the surrounding native ops make the graph non-empty.

**Views from an earlier graph.** A legacy custom op that is given a slice view (for example one expert,
`w13[e]`) created in an earlier lazy graph, before a `mark_step`, can read the wrong data. Native ops read the
same view correctly. Take the view in the same graph as the call, as the `glm` prefill path does
(`mxk_deq(w13[e], s13[e], 0)` inside the MoE forward). The `sig` test for `glm` takes its expert views that way.
