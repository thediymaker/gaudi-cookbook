# MXFP4 MoE decode kernels

**Up to 3x faster MXFP4 MoE decode on Gaudi 2.** The stock MXFP4 path caps large MoE models at about 20 tokens/s
per stream. These kernels took GLM-5.3 from 23.3 to 59.1 tokens/s per stream and gpt-oss-120b up to 3.2x per stream, with
lower numerical error than the stock op, and also run DeepSeek-V4.1-Flash, GLM-5.2 and Muse Spark Open (see
[Beyond gpt-oss](#beyond-gpt-oss) for how each number was measured).

Two Gaudi 2 TPC kernels run the routed experts of an MXFP4 mixture-of-experts layer directly from the packed
weights, with no bf16 unpack round trip. For small decode calls they replace the stock
`torch.ops.hpu.mixture_of_experts` MXFP4 ops: for gpt-oss, calls of up to 32 tokens replace the
`bias_mxfp4_fused_weights` op that vllm-gaudi uses. The package builds one variant per model family (gpt-oss,
GLM-5.x, DeepSeek-V4.1-Flash, Muse Spark Open, Kimi-K2.7), each with its own parity test and benchmark. The
vLLM hook for gpt-oss ships here; the other models call the kernels from their [ports](../ports/index.md).

Sources, build script, gpt-oss vLLM hook, tests and benchmark:
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
| GLM-5.3, GLM-5.2 (MXFP4 checkpoints) | Variant `glm`: decode kernels and the prefill unpack, parity and per-call timings on one card in this package. The model hook is in the port |
| DeepSeek-V4.1-Flash (MXFP4 checkpoint) | Variant `dsv41`: decode kernels, parity and per-call timings on one card in this package. The port adds its `dsv41_sinkhorn` kernel with `--extra-kernel` and holds the model hook |
| Muse Spark Open | Variant `muse`: decode kernels, parity and per-call timings on one card in this package. The model hook is in the port |
| Kimi-K2.7 (MXFP4 checkpoint) | Variant `kimi`: decode kernels, parity and per-call timings on one card in this package. New kernel for K = 7168; no end-to-end measurement yet |
| MiMo-V2.6-Pro (MXFP4 checkpoint) | Measured end to end (see [Beyond gpt-oss](#beyond-gpt-oss)) with the `glm` kernels and a `torch.compile` op library that is not in this package; no variant of its own |

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

**Every variant.** `tests/run_parity.sh` on one card with each model's real expert weights, at the per-card shape
of the layout named. Measured as: one Gaudi 2 card, vllm-gaudi `v0.30.0` image (Gaudi software 1.24.1), the build
of this repository, random top-k routing over all experts and every route local ("hot"), T = 1 up to the
variant's token cap, CPU fp32 reference as above. gpt-oss was also run on a vllm-gaudi `v0.26.0` image with the
same results.

| Variant | Model, layer | Per-card shape | Kernel rel_l2 vs fp32 | Stock rel_l2 vs fp32 |
| --- | --- | --- | --- | --- |
| `gptoss` | gpt-oss-120b, layer 18 (also its own router) | TP2 + EP: 64 of 128 experts, K = I = 2880 | 1.62e-3 to 1.71e-3 | 4.34e-3 to 6.25e-3 |
| `glm` | GLM-5.3 MXFP4, layer 40 | EP 4 x TP 2: 64 of 256 experts, K 6144, I 1024 | 1.65e-3 to 1.66e-3 | 4.65e-3 to 5.15e-3 |
| `dsv41` | DeepSeek-V4.1-Flash MXFP4, layer 3 | EP 8: 48 of 384 experts, K 5120, I 2304 | 1.64e-3 to 1.68e-3 | 3.75e-3 to 3.75e-2 |
| `muse` | Muse Spark Open, layer 30 | EP 2 x TP 4: 128 of 256 experts, K 4096, I 1024 | 1.66e-3 | 4.52e-3 to 4.91e-3 |
| `kimi` | Kimi-K2.7 MXFP4, layer 3 | EP 8: 48 of 384 experts, K 7168, I 2048 | 1.65e-3 to 1.67e-3 | 4.75e-3 to 5.52e-3 |

All five print `PARITY PASS` and `SIG_DEVICE PASS`. The stock op's largest DeepSeek-V4.1-Flash errors (2.4e-2 and
3.8e-2) are the 8-token calls with all 48 experts touched; the kernel stays at 1.7e-3 there.

For the `glm` prefill path (`mxk_deq` per expert, then the stock bf16 MoE op), the same run at 256, 1024 and 4096
tokens gave rel_l2 5.14e-3 to 5.22e-3 against fp32, equal to the stock MXFP4 op, and differed from the stock op by
1.2e-4 to 1.8e-4. The unpacked bf16 weights of the experts checked (first, middle, last) are bit-identical to an
exact CPU unpack (`PREFILL PASS`).

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
of 8 calls, median of 30 replays). Measured as: per-call, one card, `bench/bench_moe.py --variant gptoss
--e-local 32` with the build of this repository on a vllm-gaudi `v0.26.0` image (Gaudi software 1.24.1), stock
op and kernel in separate processes.

| Tokens | Routing | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 1 | 246.0 | 51.5 | 4.8x |
| 2 | random | 2 | 528.7 | 82.4 | 6.4x |
| 4 | random | 4 | 1153.3 | 141.6 | 8.1x |
| 8 | random | 7 | 2362.9 | 235.8 | 10.0x |
| 16 | random | 17 | 3003.3 | 532.3 | 5.6x |
| 32 | random | 31 | 3437.5 | 952.3 | 3.6x |
| 1 | all local | 4 | 523.9 | 140.9 | 3.7x |
| 8 | all local | 32 | 4082.0 | 968.2 | 4.2x |

An earlier build of the same kernel code, timed the same way with the layer's own router at 32 tokens (37 local
routes), gave 3337.8 µs stock and 1129.2 µs kernel (3.0x).

The table above is the 32-local-expert shape (one rank of TP4 + EP). The 64-local-expert shape (one rank of
TP2 + EP), measured the same way on the same image but with random weights of the layer's shape:

| Tokens | Routing | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 3 | 427.2 | 113.9 | 3.8x |
| 2 | random | 4 | 761.3 | 141.5 | 5.4x |
| 4 | random | 7 | 1466.2 | 231.2 | 6.3x |
| 8 | random | 16 | 3238.3 | 499.8 | 6.5x |
| 16 | random | 31 | 5713.0 | 947.4 | 6.0x |
| 32 | random | 65 | 6863.7 | 1975.1 | 3.5x |
| 1 | all local | 4 | 525.6 | 142.4 | 3.7x |
| 8 | all local | 32 | 4562.4 | 972.4 | 4.7x |

The 128-local-expert shape (TP1), from an earlier build of the same kernel code timed the same way:

| Tokens | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- |
| 1 | 4 | 536.4 | 142.8 | 3.8x |
| 4 | 16 | 2416.4 | 495.9 | 4.9x |
| 32 | 128 | 13751.1 | 3824.0 | 3.6x |

The kernel costs about 30 µs per local route at any token count. The stock op costs about 250 to 300 µs per touched
expert at small counts, but amortises the unpack when several routes share an expert. So the speed-up peaks around
8 tokens and narrows to 3x to 3.6x at 32.

### Per-call timings by variant

Measured as: per-call, one card, `bench/bench_moe.py --variant <name>` with the build of this repository on a
vllm-gaudi `v0.30.0` image (Gaudi software 1.24.1), random weights of the variant's per-card shape, stock op and
kernel in separate processes, µs per call inside an HPU graph (a chain of 8 calls, median of 30 replays).

**`kimi`** (Kimi-K2.7, EP 8: 48 of 384 experts, K 7168, I 2048, top-8; stock op `mxfp4_fused_weights`):

| Tokens | Routing | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 1 | 390.2 | 68.7 | 5.7x |
| 2 | random | 1 | 432.1 | 77.8 | 5.6x |
| 4 | random | 4 | 1043.3 | 217.5 | 4.8x |
| 8 | random | 6 | 1589.7 | 310.6 | 5.1x |
| 16 | random | 12 | 3002.7 | 589.5 | 5.1x |
| 1 | all local | 8 | 2055.2 | 398.2 | 5.2x |
| 8 | all local | 64 | 9687.8 | 2976.3 | 3.3x |

**`muse`** (Muse Spark Open, EP 2 x TP 4: 128 of 256 experts, K 4096, I 1024 per card, top-8; per-route w2
output; the stock path is the per-route equivalent built from `mxfp4_fused_weights`):

| Tokens | Routing | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 3 | 381.2 | 66.9 | 5.7x |
| 2 | random | 8 | 764.4 | 139.7 | 5.5x |
| 4 | random | 16 | 1457.7 | 260.0 | 5.6x |
| 1 | all local | 8 | 694.4 | 138.7 | 5.0x |
| 4 | all local | 32 | 2301.5 | 489.2 | 4.7x |

**`dsv41`** (DeepSeek-V4.1-Flash, EP 8: 48 of 384 experts, K 5120, I 2304, top-6; stock op
`bias_mxfp4_fused_weights`):

| Tokens | Routing | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 1 | 368.4 | 61.1 | 6.0x |
| 2 | random | 1 | 447.4 | 62.1 | 7.2x |
| 4 | random | 2 | 654.7 | 108.1 | 6.1x |
| 8 | random | 4 | 1129.9 | 192.1 | 5.9x |
| 16 | random | 12 | 2499.6 | 521.8 | 4.8x |
| 1 | all local | 6 | 1344.1 | 264.2 | 5.1x |
| 8 | all local | 48 | 7974.2 | 2007.8 | 4.0x |

**`glm`** (GLM-5.3 and GLM-5.2, EP 4: 64 local experts, K 6144, I 1024 per card, top-8; stock op
`mxfp4_fused_weights`):

| Tokens | Routing | Local routes | Stock µs | Kernel µs | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 1 | 273.0 | 42.2 | 6.5x |
| 2 | random | 5 | 598.7 | 128.6 | 4.7x |
| 4 | random | 9 | 979.3 | 219.5 | 4.5x |
| 8 | random | 14 | 1525.4 | 331.6 | 4.6x |
| 16 | random | 29 | 2672.5 | 659.9 | 4.0x |
| 1 | all local | 8 | 928.8 | 191.9 | 4.8x |
| 8 | all local | 64 | 4691.9 | 1397.5 | 3.4x |

**`gptoss`** on `v0.30.0`, 64 local experts: every row is within 1% of the 64-local-expert table above (T1 3.8x,
T8 6.4x, T32 3.5x).

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

- **Decode only.** Prefill keeps the stock op (`glm` adds an unpack kernel for it, see
  [Prefill](#prefill-unpack-once-then-the-stock-op)). For gpt-oss, a call runs on the kernels only if it has at most 32 tokens
  (`MXFP4_TPC_MOE_MAX_TOKENS`, at most 32) and at most 128 routes. Prefill, mixed batches and decode batches
  larger than 32 keep the stock op. With `--max-num-seqs` above 32, the larger decode buckets use the stock op.
- **Supported shapes.** For gpt-oss, K and I must be at most 3072 and multiples of 32, with at most 128 local
  experts. Layers must have gpt-oss biases and interleaved gate/up rows. Anything else keeps the stock op, and the
  runtime logs why. The other variants check their limits in the kernel glue, and a call outside them fails to
  compile instead of falling back, so the model hook must check first:

    | Variant | K | I per card | Tokens per call |
    | --- | --- | --- | --- |
    | `glm`, `muse` | multiple of 512, at most 6144 | multiple of 512, at most 3072 | at most 16 (`muse`: the port uses 4) |
    | `dsv41` | multiple of 512, at most 6144 | multiple of 256, at most 3072 | at most 16 |
    | `kimi` | multiple of 512, at most 8192 | multiple of 512, at most 3072 | at most 16 |

    Every variant also needs at most 128 routes per call and at most 128 local experts.
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
  Any new op added to this library needs a distinct signature. `tests/parity_hpu.py --impl sig` checks every
  variant for this.
- **Views from an earlier graph.** A legacy custom op that is given a slice view (one expert, `w13[e]`) created in
  an earlier lazy graph can read the wrong data; native ops read the same view correctly. Take the view in the same
  graph as the call, as the `glm` prefill path does.
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

The package has its own [README](https://github.com/thediymaker/gaudi-cookbook/tree/main/kernels/mxfp4-moe) with
the full op reference. In short:

**Build** inside the serving image, one variant per image. The Habana `pytorch-installer` base ships everything the
build needs: `tpc-clang` at `/opt/habanalabs/bin/tpc-clang`, `tpc_kernel_lib_interface.h` in
`/usr/include/habanalabs`, and `hpu_custom_op.h` in the `habana_frameworks` package.

```bash
kernels/mxfp4-moe/build.sh [--variant gptoss|glm|dsv41|muse|kimi] [--extra-kernel DIR]... OUT_DIR
```

| Variant | Kernels | Model |
| --- | --- | --- |
| `gptoss` (default) | `mxk_w13xib`, `mxk_w2xb` | gpt-oss, through `python/` in this package |
| `glm` | `mxk_w13x`, `mxk_w2x`, `mxk_deq` | GLM-5.3 and GLM-5.2, decode and the prefill unpack |
| `dsv41` | `mxk_w13xi`, `mxk_w2x` | DeepSeek-V4.1-Flash (its port adds `dsv41_sinkhorn` with `--extra-kernel`) |
| `muse` | `mxk_w13x`, `mxk_w2xr` | Muse Spark Open |
| `kimi` | `mxk_w13xt`, `mxk_w2x` | Kimi-K2.7 MXFP4 |

`build.sh` compiles the variant's kernels with `tpc-clang -march=gaudi2 -O2` and embeds the ELFs in `libmxk_tpc.so`,
which forwards every stock GUID. It builds `libmxk_op_lazy.so` with the variant's `torch.ops.custom_op` ops against
the image's torch, and writes a `MANIFEST` with versions and sha256 sums. The build fails unless two gates pass: the
library lists every stock GUID unchanged followed by exactly the variant's GUIDs, and the op library registers the
variant's ops and no op of another variant. `--extra-kernel DIR` adds a model-only TPC kernel a port supplies
(`<guid>.c`, `glue.inc`, `op.cpp`) to the same two libraries, so `GC_KERNEL_PATH` stays one file.

The `MANIFEST` also records a `code_sha256` per kernel: the hash of its machine code (the `.text` and
`.tpc_metadata` sections). tpc-clang embeds the source text and the build command in the ELF, so the ELF hash
changes with any comment or path, but the code hash does not. Built from this repository, every kernel of the
`gptoss`, `glm`, `dsv41` and `muse` variants has the same code hash as the kernels of the same name in the
libraries the measurements on this page ran on.

For gpt-oss, build and install in one image stage:

```dockerfile
FROM <your-vllm-gaudi-v0.26.0-image>
COPY kernels/mxfp4-moe /opt/mxfp4-moe/src
RUN /opt/mxfp4-moe/src/build.sh /opt/mxfp4-moe/lib \
 && python3 /opt/mxfp4-moe/src/python/install_mxfp4_tpc_moe.py
```

The installer copies the runtime into `vllm_gaudi/ops/` and adds two hooks. If any of its 11 anchors in vllm-gaudi
is missing, it exits 1 before writing anything. The other variants are installed by their ports, which call the
ops from the model's MoE forward.

**Enable (gpt-oss):**

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

1. On one card, before enabling any variant:
   `kernels/mxfp4-moe/tests/run_parity.sh --variant <name> --lib <build> [--checkpoint <dir>]` must report
   `PARITY PASS` and `SIG_DEVICE PASS`, and `PREFILL PASS` for `glm`. Without `--checkpoint` it uses random
   weights of the variant's serving shapes.
2. gpt-oss: the engine log shows `MXFP4 TPC MoE installed: mxk_tpc gptoss-v8-<hash>` and
   `MXFP4 TPC MoE enabled (E_local=64 K=2880 I=2880, <= 32 tokens, ...)` (64 local experts at TP2 + EP).
3. `grep -l libmxk_tpc.so /proc/*/maps` inside the container lists the engine worker processes.
4. For speed, run `kernels/mxfp4-moe/bench/bench_moe.py --variant <name>` once per implementation, then `--compare`.

**Rollback (gpt-oss):** set `MXFP4_TPC_MOE=0`. The hooks are then inert and the engine serves exactly as stock.

## Beyond gpt-oss

The same kernel core runs other MXFP4 MoE models. Compile-time switches in the headers stamp out the other
kernels of the package:

- `mxk_w13x`: SiLU without bias for `[gate; up]` row layouts (GLM-5.x, Muse Spark Open);
- `mxk_w13xi`: the same for interleaved gate/up rows, with a clamped SwiGLU (DeepSeek-V4.1-Flash);
- `mxk_w13xt`: `mxk_w13x` for K above 6144, reading the activation row past the first 6144 values from memory
  instead of vector local memory (Kimi-K2.7, K = 7168);
- `mxk_w2x`: the down projection with the weighted sum per token;
- `mxk_w2xr`: a per-route w2 output for models that apply a per-route operation before the token sum (Muse Spark
  Open).

`build.sh --variant` picks the set each model needs (see [Build](#build-enable-and-verify)).
[Per-call timings by variant](#per-call-timings-by-variant) has one-card numbers.

With model-specific integrations, the kernels serve six more models on one 8-card Gaudi 2 node. How each number
was measured is part of the result, so the table says it:

| Model | Setup | Stock op | With the kernel | Gain | How it was measured |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3 (MXFP4 checkpoint) | TP 8 + EP 8, one MTP draft token, 2K prompt, 1 stream | 23.3 tok/s | 59.1 tok/s | 2.5x | Three same-node A/Bs chained: first kernel vs stock 1.96x, then two kernel revisions 1.13x and 1.14x |
| DeepSeek-V4.1-Flash (MXFP4 checkpoint) | TP 8, 1 stream | 20.7 tok/s | 37.6 tok/s | 1.8x | Same-pod A/B, the published recipe with `DSV41_MXK` off and on |
| GLM-5.2 (MXFP4 checkpoint) | TP 8 + EP 8, one MTP draft token, 1 stream | 23.1 tok/s | 47.8 tok/s | 2.1x | Same benchmark on a stock pod and on kernel pods; not a same-node A/B |
| Muse Spark Open | TP 8, 1 stream, 256 greedy tokens | 22.7 tok/s | 40.8 tok/s | 1.8x | Same-pod A/B, the published recipe with `MUSE_SPARK_MXFP4_KERNELS` off and on |
| Muse Spark Open | TP 8, 4 streams, aggregate | 43.3 tok/s | 102.8 tok/s | 2.4x | Same |
| Kimi-K2.7-Code (MXFP4 checkpoint) | TP 8 + EP 8, 16K context, 1 stream | 17.9 tok/s | 44.5 tok/s | 2.5x | Same-pod A/B, the published recipe with `KIMI_MXK` off and on |
| Kimi-K2.7-Code (MXFP4 checkpoint) | TP 8 + EP 8, 16K context, 8 streams, per stream | 7.5 tok/s | 14.8 tok/s | 2.0x | Same |
| MiMo-V2.6-Pro (MXFP4 checkpoint) | 8 cards, 1 stream | 18.1 tok/s | 39.7 tok/s | 2.2x | Same node and configuration: stock pod, then kernel pod |

- **GLM-5.3** keeps the same tool-call, needle and GSM8K results as the stock op. Greedy outputs agree on 99.1% of
  top-1 tokens before the first near-tie split, and the kernel is the closer of the two to an fp32 reference. With
  decode-step and expert-placement work on top of the kernel, single-stream decode reaches 69.6 tok/s.
- **DeepSeek-V4.1-Flash** gains less because its MoE is about a third of the decode step; the rest is attention
  and launch overhead.
- **Muse Spark Open** is a full port to Gaudi 2: latent MoE with a norm on every expert output, and sliding plus
  global attention. It matches the reference model's perplexity within 0.4% (top-1 agreement 97.6%) and serves on
  one 8-card node.

The kernels, their parity tests and their benchmarks are in this package. The model hooks depend on model code
outside vllm-gaudi and live in the ports, described under [Ports](../ports/index.md).

### Prefill: unpack once, then the stock op

Decode is where the kernels pay off, but on large MoE models the routed experts are also most of the prefill. On
GLM-5.3 (EP 8) they take 52% to 65% of the time to first token. A third kernel, `mxk_deq`, unpacks the routed
experts' MXFP4 weights straight to bf16 in one pass, and the stock bf16 MoE op then runs the GEMMs. The output is
bit-identical to the stock path for prompts of 300 to 33K tokens, and time to first token drops by 22% to 26%
(2K prompt 0.64 to 0.49 s, 8K prompt 2.01 to 1.49 s). Measured as: a baseline pod and a pod with the prefill
path on two nodes with the same configuration; the decode-side differences between them do not touch prefill. Under an EP 4 x TP 2 layout the gain is about
half and the output is no longer bit-identical: with the GLM-5.3 recipe (EP 4 x TP 2), the 2K prompt went from 0.64 to
0.52 s (19%) and the 8K prompt from 1.96 to 1.76 s (10%), and greedy output on 16 prompts x 192 tokens stayed
token-identical. Measured as: same pod, server restarted with `GLM_MXK_PREFILL` flipped, median of 3. `mxk_deq` is part of the `glm` variant, and
`tests/parity_hpu.py --variant glm` checks it (`PREFILL PASS`); the prefill hook is in the GLM port.

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
