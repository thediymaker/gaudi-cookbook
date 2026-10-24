# MXFP4 MoE decode kernels

Two Gaudi 2 TPC kernels run the routed experts of an MXFP4 mixture-of-experts layer directly from the packed
weights, with no bf16 unpack round trip. For decode calls of up to 32 tokens they replace the stock
`torch.ops.hpu.mixture_of_experts.bias_mxfp4_fused_weights` op that vllm-gaudi uses for gpt-oss. Prefill keeps the
stock op.

Sources, build script, vLLM hook, tests and benchmark:
[`kernels/mxfp4-moe/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/kernels/mxfp4-moe).

!!! summary "At a glance"
    - **Model:** `openai/gpt-oss-120b`, measured. `openai/gpt-oss-20b` has the same layer shapes (K = I = 2880,
      top-4) and is expected to work, but has not been measured.
    - **Stack:** Gaudi 2, Gaudi software 1.24.1, vllm-gaudi `v0.26.0`, lazy mode, `HPU_MXFP4_NATIVE=1`.
    - **One card:** 3x to 10x faster per MoE call, and closer to an fp32 reference than the stock op
      (rel_l2 1.57e-3 to 1.75e-3, against 4.08e-3 to 7.17e-3).
    - **End to end:** 1.1x to 3.2x decode tok/s per stream. The largest gains are at 4 to 16 requests in flight.
      TTFT is unchanged.
    - **Switch:** `MXFP4_TPC_MOE=1`. Off by default; any mismatch keeps the stock op.

## Coverage

| Model / checkpoint | Status |
| --- | --- |
| `openai/gpt-oss-120b` | Covered and measured: TP1, TP2 + EP and TP4 + EP |
| `openai/gpt-oss-20b` | Expected to work, not measured. Same K, I, top-k and layout; 32 experts |
| gpt-oss with tensor-parallel experts (TP > 1 without `--enable-expert-parallel`) | Not tested. Per-rank shapes the kernels do not support (I not a multiple of 32) fall back to the stock op |
| Other MXFP4 MoE models | Not covered by this package; see [Other MXFP4 models](#other-mxfp4-models) |

## The bottleneck

With `HPU_MXFP4_NATIVE=1`, vllm-gaudi keeps gpt-oss experts packed on the device: 4-bit E2M1 values, two per byte,
plus one E8M0 scale per 32 values. gpt-oss-120b's weights take 60.8 GiB. The MoE layer calls
`bias_mxfp4_fused_weights`. The graph compiler lowers that op into:

1. a `cast_packed_mxfp4_to_bf16` TPC node for each touched expert, which writes the expert's weights to memory as
   bf16 (4x the packed bytes);
2. a GEMM on the matrix engine for each expert, with tiny shapes at decode (one to a few rows);
3. DMA copies between the pieces.

A device trace of a large MXFP4 MoE model at batch 4 shows the scale of this. One decode step ran about 124,000
cast calls (about 8 µs each, 47.6 ms of TPC time in a 97 ms step), 33,000 small GEMMs (21.6 ms) and 49,000 DMA
copies. HBM can deliver about 2.2 TB/s per card, but the cast path moves only 0.2 to 0.5 TB/s of packed weights.
Gaudi software 1.24.2 does not change this lowering.

For gpt-oss-120b on stock vllm-gaudi (TP4 + EP, `--max-num-seqs 32`), the MoE dominates every decode step above
batch 1:

| Batch | Step period | MoE | Attention | Note |
| --- | --- | --- | --- | --- |
| 1 | 17.4 ms | 8.4 ms (41%) | 1.75 ms | host-launch bound (183 graph launches, ~15 ms host time) |
| 4 | 50 ms | 36.9 ms (75%) | 2.8 ms | |
| 16 | 101 ms | 81.6 ms (82%) | 5.1 ms | |
| 32 | 114 ms | 94.1 ms (78%) | 8.5 ms | |

The stock MoE time grows almost linearly with the number of distinct experts touched, so it is the expert unpack
that costs, not the arithmetic.

## Design

Each MoE layer runs two kernel launches and one small host-side transpose instead of the unpack, GEMM and copy
chain:

```text
x [T, 2880] bf16 --pad to 3072, permute per 512-k chunk--> xp
xp --mxk_w13xib(w13, s13, bias13)--> a [T*4 routes, 3072] bf16   (gate/up GEMV + bias + SwiGLU-OAI)
a  --mxk_w2xb(w2, s2, bias2, routing weights)--> y [T, 2880] bf16 (down GEMV + bias + weighted sum per token)
```

**Read the served layout.** The kernels take the tensors vLLM already holds, with no repack and no second copy:

- `w13` is u8 `[E, 2I, K/2]` with gate and up rows interleaved;
- `s13` is u8 `[E, 2I, K/32]`;
- `w2` is u8 `[E, K, I/2]` and `s2` is u8 `[E, K, I/32]`.

Only the biases are re-laid once at load time, as f32 in the kernels' column order. That is about 0.8 MB per layer
and rank at 32 local experts.

**Unpack in registers.** For each 512-value chunk of a weight row the kernel:

1. issues one 256-byte load;
2. splits the nibbles;
3. maps E2M1 to bf16 with the TPC lookup table (the same table the stock cast uses);
4. builds a per-lane bf16 scale from 16 scale bytes and multiplies (exact in bf16);
5. accumulates in fp32.

The activation row is pre-permuted on the host so that lane i of sub-vector p holds element 512c + 4i + p. The four
nibble planes of 128 packed 16-bit lanes then line up with four activation vectors, with no lane shuffles.

**Fused epilogues.**

- `mxk_w13xib` adds the bias in fp32 and applies SwiGLU-OAI in fp32: gate capped at 7, up clamped to ±7,
  `gate * sigmoid(1.702 * gate) * (up + 1)`. It stores bf16 in the permuted order that the w2 kernel reads. The
  activation runs once per run of up to 32 column pairs on whole vectors, not once per pair. That matters because,
  run once per pair, the two sigmoids cost about a quarter of a pass.
- `mxk_w2xb` keeps fp32 per-token sums in vector local memory. It adds the routing-weighted bias once per route and
  writes bf16 `[T, K]`.

**Routing inside the kernel.** A scalar prologue counting-sorts the routes by local expert. Routes to experts on
other expert-parallel ranks are skipped, so they cost nothing. The index space is one point per TPC (24 on
Gaudi 2). Each TPC owns a fixed slice of output rows and walks every local route, so the work split does not depend
on the routing.

**Software pipelining.** All passes of a TPC (4 weight rows of one expert against one activation row) form one
stream. The last chunk pair of a pass issues the loads for the first pair of the next pass, which may belong to a
different expert.

**Why it is issue-bound.** Each 512-value chunk-row costs about 17 vector instructions: nibble split, 4 lookups,
scale build, 4 multiplies and 4 MACs. At gpt-oss shapes one local route (w13 + w2, about 13 MB packed with scales)
takes about 30 µs. That is about 0.44 TB/s of packed bytes, about 5x above the HBM read floor. Timing variants
(wrong results on purpose) show where the time goes. On the shapes of another MXFP4 model, a w13 pass with every
load removed still took about 25 µs per expert, against 34 µs with loads and a 9 µs HBM floor. So the loop is
limited mostly by vector issue, not by memory, and the next step would be a layout that needs fewer vector
operations per value. That would mean repacking the weights, which these
kernels avoid on purpose so that the experts are never stored twice.

**One kernel library file.** `libmxk_tpc.so` serves the two kernels and forwards every other GUID to the stock
`libtpc_kernels.so`. With the recipe cache on (`PT_HPU_RECIPE_CACHE_CONFIG`), the bridge accepts `GC_KERNEL_PATH`
only as a single regular file, so a two-library list is not an option. The library answers the six entry points
the graph compiler reads, and passes the remaining stock entry points straight through.

## Parity

The parity test has three parts:

- **Setup:** one card, layer 18 of `openai/gpt-oss-120b` with real weights, activations at RMS 1 and 4 (RMS 4
  drives the SwiGLU clamps), and T = 1 to 32 tokens.
- **Routing:** random top-4 routing, every route local ("hot"), and the layer's own router on the activation rows.
- **Reference:** CPU fp32 with dequantised weights and a bf16 intermediate.

| Shape | Kernel rel_l2 vs fp32 | Stock rel_l2 vs fp32 | Kernel vs stock |
| --- | --- | --- | --- |
| 32 local experts (one rank of TP4 + EP) | 1.57e-3 to 1.75e-3 | 4.37e-3 to 7.17e-3 | 4.4e-3 to 7.3e-3 |
| 128 local experts (one card, TP1) | 1.57e-3 to 1.73e-3 | 4.08e-3 to 5.69e-3 | 4.4e-3 to 5.9e-3 |

The kernel is about 3x closer to fp32 than the stock op in every case. It accumulates in fp32, applies bias and
activation in fp32, and rounds to bf16 only once, for the intermediate.

Two kernel calls with identical input shapes in one lazy graph (T = 1 to 32) give bit-identical results to the same
calls run alone (see [pitfalls](#limitations-and-pitfalls)).

End to end, the output is not bit-identical to the stock op:

- At temperature 0, 2 of 16 completions of 192 tokens were identical. Every first divergence was a near-tie, with a
  top-1 vs top-2 margin of at most 0.125 nats.
- The chosen-token logprob difference had a median of 0.002 and a p99 of 0.14.
- Task gates did not move (next section).

## Measured results

### Per MoE call, one card

Layer 18 of `openai/gpt-oss-120b`, real weights, x RMS 1. Each value is µs per call inside an HPU graph (a chain
of 8 calls, median of 30 replays).

| Tokens | Routing | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 1 | 246.8 | 51.7 | 4.8x |
| 2 | random | 2 | 529.2 | 81.2 | 6.5x |
| 4 | random | 4 | 1138.3 | 141.5 | 8.0x |
| 8 | random | 7 | 2363.2 | 234.8 | 10.1x |
| 16 | random | 17 | 3000.7 | 533.9 | 5.6x |
| 32 | random | 31 | 3460.6 | 952.5 | 3.6x |
| 32 | model router | 37 | 3337.8 | 1129.2 | 3.0x |
| 8 | all local | 32 | 4096.4 | 967.6 | 4.2x |

A fresh build of this repository's sources on a stock vllm-gaudi v0.26.0 image (Gaudi software 1.24.1) reproduces
these timings within 3% and passes `tests/run_parity.sh` on the same layer: kernel relative error against the fp32
reference about 1.6e-3, stock op about 5e-3, at every token count from 1 to 32.

The table above is the 32-local-expert shape (one rank of TP4 + EP). The 128-local-expert shape (TP1) gives:

| Tokens | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- |
| 1 | 4 | 536.4 | 142.8 | 3.8x |
| 4 | 16 | 2416.4 | 495.9 | 4.9x |
| 32 | 128 | 13751.1 | 3824.0 | 3.6x |

The kernel costs about 30 µs per local route at any token count. The stock op costs about 250 to 300 µs per touched
expert at small counts, but amortises the unpack when several routes share an expert. So the speed-up peaks around
8 tokens and narrows to 3x to 3.6x at 32.

### End to end, gpt-oss-120b

**Setup** (the same for every table in this section):

- **Hardware:** Gaudi 2 (HL-225).
- **Image:** vllm-gaudi `v0.26.0` built as in [Getting started](../getting-started.md) on the 1.24.1 base. The
  image also carried unrelated request-handling fixes that do not touch the MoE path.
- **Environment:** lazy mode and `HPU_MXFP4_NATIVE=1`.
- **Engine:** `--max-model-len 131072 --block-size 128 --enable-chunked-prefill --gpu-memory-utilization 0.9
  --max-num-seqs 32`.
- **Load:** prompts of about 3.2K tokens with a unique prefix each (no prefix-cache hits), 1024 output tokens,
  greedy, `ignore_eos`.
- **Method:** A/B on the same node and image, with `MXFP4_TPC_MOE` off and then on.
- **Values:** decode tok/s per stream, median, with the aggregate per engine in brackets.

**TP4 + EP (4 cards):**

| In flight | Stock | Kernel | Speed-up | TTFT stock / kernel |
| --- | --- | --- | --- | --- |
| 1 | 56.95 | 62.55 | 1.10x | 0.61 / 0.61 s |
| 4 | 20.14 (78) | 59.55 (220) | 2.96x | 1.94 / 1.90 s |
| 16 | 9.98 (150) | 22.43 (319) | 2.25x | 6.57 / 6.06 s |
| 32 | 8.88 (253) | 14.20 (390) | 1.60x | 13.20 / 11.93 s |

With the kernel the step period is 13 to 15 ms at batch 1, 15 to 16 ms at batch 4, 36 ms at batch 16 and 56 to
65 ms at batch 32.

**TP2 + EP (2 cards):**

| In flight | Stock | Kernel | Speed-up |
| --- | --- | --- | --- |
| 1 | 49.87 | 67.04 | 1.34x |
| 4 | 16.59 (64) | 52.41 (188) | 3.16x |
| 16 | 5.51 (83) | 14.77 (208) | 2.68x |
| 32 | 4.84 (140) | 9.30 (253) | 1.92x |

**TP1 (1 card, 128 local experts):**

| In flight | Stock | Kernel |
| --- | --- | --- |
| 1 | 40.41 | 88.24 |
| 4 | not measured | 32.32 (115) |
| 16 | not measured | 8.56 (120) |
| 32 | not measured | 5.47 (149) |

TP1 with the stock op is impractical because of graph compile time. The first batch-1 decode step ran 10 minutes
after the first request, and the batch-4 shape was still compiling after more than 60 minutes. With the kernels,
each new decode shape compiled in 4 to 7 minutes.

Per card at 32 in flight with the kernels: TP1 149, TP2 126 and TP4 98 tok/s. At equal streams per card, TP2 with
the kernels beat TP4 at every load measured.

**Why batch 1 gains least.** A batch-1 step is mostly host time (graph launches), so removing about 6 ms of MoE
device time shows up only partly.

**Gates with the kernels on** (TP4 + EP; TP2 + EP gave the same results):

| Gate | Kernel | Stock |
| --- | --- | --- |
| Harmony tool calls (one-shot, `required`, after a tool result, second call) | 8/8 each | 8/8 each |
| `json_schema` structured output | 6/6 | 6/6 |
| Two needles in ~112K-token prompts | both found | both found |
| GSM8K, first 50, greedy | 50/50 | 49/50 |
| 25.6-minute mixed stress at 16 in flight (chat, tools, JSON, long QA to 82K tokens) | 428 requests, 0 errors, 0 preemptions | |

## Limitations and pitfalls

- **Decode only.** A call runs on the kernels only if it has at most 32 tokens
  (`MXFP4_TPC_MOE_MAX_TOKENS`, at most 32) and at most 128 routes. Prefill, mixed batches and decode batches
  larger than 32 keep the stock op. With `--max-num-seqs` above 32, the larger decode buckets use the stock op.
- **Supported shapes.** K and I must be at most 3072 and multiples of 32, with at most 128 local experts. Layers
  must have gpt-oss biases and interleaved gate/up rows. Anything else keeps the stock op, and the runtime logs why.
- **Lazy mode only.** The op library uses the legacy `hpu_custom_op.h` API. The PT2 custom-op API fails in lazy
  mode when the op's input comes from a native op in the same lazy graph. Eager and `torch.compile` serving are
  not covered.
- **`PT_HPU_LAZY_ACC_PAR_MODE=0` is required.** With the lazy accumulation thread on, a custom op whose input is a
  view of a tensor produced in the same lazy graph fails intermittently. The runtime sets the variable when the flag
  is on. With the kernels off, this setting gave bit-identical output and the same speed and TTFT.
- **`GC_KERNEL_PATH` must be one file.** With `PT_HPU_RECIPE_CACHE_CONFIG` set, the bridge rejects a list of kernel
  libraries ("Path is not a regular file") and device init fails. The variable must also be set before the first
  HPU op, which is why the hook sits at the top of the `vllm_gaudi` package import. If `GC_KERNEL_PATH` already
  lists several libraries, only the stock library is forwarded.
- **Recipe cache keys ignore the kernel library.** A cache written by one kernel build could hand its recipes to
  another. The runtime moves the cache to a sibling directory named after the kernel version (`<dir>_mxk-<version>/`).
- **Identical input signatures.** Lazy-mode legacy custom ops are cached by input signature, not by GUID. In one
  process, a second custom op with the same input shapes silently returns the first op's result. The two kernels
  here have 5 and 6 tensor inputs, so they cannot collide. Benchmarks must still run one implementation per process.
  Any new op added to this library needs a distinct signature.
- **Rebuild per image.** The op library links against the image's torch and habana-torch-plugin. The kernel ELFs
  target Gaudi 2 only.
- **Not bit-identical to stock.** Greedy outputs diverge at bf16 near-ties (see [Parity](#parity)). Compare greedy
  runs only on an otherwise idle engine: batching with other requests changes the output even on the stock path.
- **The library must be present before the device opens.** If the flag is on but the libraries are missing, the
  runtime logs it and stays on the stock op. Check the log lines below after every image change.

tpc-clang pitfalls met while writing the kernels (Gaudi software 1.24.1, tpc-clang 2.7.33):

- templates crash the front end;
- loops on a constant parameter of an inlined function can be miscompiled, so variants are stamped out with the
  preprocessor;
- vector local memory is 16 KB, and static arrays plus spills beyond that fail at compile time;
- a loop that stores a constant zero to local memory becomes a `memset` the backend cannot lower, so the kernels
  store a runtime zero;
- for a 16-bit shuffle the source-half bit must be set in both bytes (`0x2020`);
- two back-to-back partial tensor stores with the same size and offset drop the second one;
- the glue must reject unexpected dtypes, or the graph compiler re-types the output and corrupts it silently.

## Build, enable and verify

The package has its own [README](https://github.com/thediymaker/gaudi-cookbook/tree/main/kernels/mxfp4-moe). In
short:

**Build** inside the serving image. The Habana `pytorch-installer` base ships everything the build needs:
`tpc-clang` at `/opt/habanalabs/bin/tpc-clang`, `tpc_kernel_lib_interface.h` in `/usr/include/habanalabs`, and
`hpu_custom_op.h` in the `habana_frameworks` package.

```dockerfile
FROM <your-vllm-gaudi-v0.26.0-image>
COPY kernels/mxfp4-moe /opt/mxfp4-moe/src
RUN /opt/mxfp4-moe/src/build.sh /opt/mxfp4-moe/lib \
 && python3 /opt/mxfp4-moe/src/python/install_mxfp4_tpc_moe.py
```

`build.sh` compiles the two kernels with `tpc-clang -march=gaudi2 -O2` and embeds the ELFs in `libmxk_tpc.so`. It
also builds `libmxk_op_lazy.so` against the image's torch and runs a gate: every stock GUID must be present
unchanged, plus the two kernels. Finally it writes a `MANIFEST` with versions and sha256 sums. The installer copies
the runtime into `vllm_gaudi/ops/` and adds two hooks. If any of its 11 anchors in vllm-gaudi is missing, it exits 1
before writing anything.

**Enable:**

```bash
export PT_HPU_LAZY_MODE=1 HPU_MXFP4_NATIVE=1 MXFP4_TPC_MOE=1
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 131072 --block-size 128 --enable-chunked-prefill \
  --tool-call-parser openai --reasoning-parser openai_gptoss --enable-auto-tool-choice
```

The remaining engine flags, and the fixes gpt-oss needs on `v0.26.0`, are in the
[gpt-oss-120b recipe](../recipes/gpt-oss-120b.md).

| Variable | Default | Meaning |
| --- | --- | --- |
| `MXFP4_TPC_MOE` | `0` | `1` turns the kernels on |
| `MXFP4_TPC_MOE_MAX_TOKENS` | `32` | Per-call token cap (at most 32) |
| `MXFP4_TPC_MOE_LIB` | `/opt/mxfp4-moe/lib` | Directory with `libmxk_tpc.so` and `libmxk_op_lazy.so` |

**Verify:**

1. The engine log shows `MXFP4 TPC MoE installed: mxk_tpc mxfp4-moe-v8-<hash>` and
   `MXFP4 TPC MoE enabled (E_local=64 K=2880 I=2880, <= 32 tokens, ...)` (64 local experts at TP2 + EP).
2. `grep -l libmxk_tpc.so /proc/*/maps` inside the container lists the engine worker processes.
3. On one card, `kernels/mxfp4-moe/tests/run_parity.sh --checkpoint <gpt-oss-120b snapshot>` must report
   `PARITY PASS` and `SIG_DEVICE PASS`. Without `--checkpoint` it uses random weights of the same shapes.
4. For speed, run `kernels/mxfp4-moe/bench/bench_moe.py` once per implementation, then `--compare`.

**Rollback:** set `MXFP4_TPC_MOE=0`. The hooks are then inert and the engine serves exactly as stock.

## Other MXFP4 models

The kernel headers also build, through compile-time switches:

- a SiLU variant without bias for `[gate; up]` row layouts;
- a per-route w2 output for models that apply a per-route operation before the token sum.

With model-specific integrations these were used on other MXFP4 MoE models. On an MXFP4 repack of GLM-5.3 (8
cards, TP8 + EP8, one MTP draft token, `--max-num-seqs 4`), single-stream decode went from 23.3 to 59.1 tok/s at a
2K prompt, and the 4-stream aggregate from 53.5 to 120.3 tok/s. Those integrations depend on model code outside
vllm-gaudi and are not part of this package.

Public MXFP4 checkpoints in other formats are not covered either. Checkpoints with `quant_method: quark` do not
load as packed MXFP4 on stock vllm-gaudi `v0.26.0`, because `quark` is not in the HPU platform's supported
quantizations.

## Upstream

The natural home for this work is the Gaudi software stack itself: either a lowering of
`mixture_of_experts.mxfp4_fused_weights` / `bias_mxfp4_fused_weights` for small token counts that fuses the
unpack into a GEMV, or a native MXFP4 MoE decode kernel in the TPC kernel library. Every MXFP4 model on Gaudi would
then gain without a custom build step. The grouped-dequant MoE kernels added in 1.24.2 cover INT4/UINT4 only.

A second option is an optional custom op in vllm-gaudi, behind a flag like the one here. That needs a TPC build
step or prebuilt kernel ELFs per Gaudi software release, which vllm-gaudi does not have today.

These platform changes would make custom kernels like this one simpler to ship:

- accept a list in `GC_KERNEL_PATH` when the recipe cache is on, so a library does not need to forward stock GUIDs;
- include the kernel libraries in the recipe cache key;
- key lazy-mode custom-op recipes by GUID as well as by input signature;
- support custom ops on views of tensors produced in the same lazy graph with the accumulation thread on.
