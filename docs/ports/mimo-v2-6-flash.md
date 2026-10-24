# MiMo-V2.6-Flash on Gaudi 2

MiMo-V2.6-Flash is a 309B-parameter mixture-of-experts model (15B active) whose attention is mostly a 128-token
sliding window with a learned attention sink. Query and key heads are 192 wide, value heads 128. None of that has
a stock HPU path, and two Gaudi kernel behaviours silently break it. This port serves it on one Gaudi 2 node in
production at 512K tokens per request, with window-sized KV for the sliding layers.

!!! summary "At a glance"
    - **Checkpoint:** [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), MIT.
      Routed experts are OCP MXFP4 in the checkpoint; production re-encodes them to FP8 e4m3 at load, exactly
      (every one of the 36,096 expert tensors passes a round-trip check). Nothing is converted offline.
    - **Hardware:** 8 Gaudi 2 cards, one node, TP8 with the experts split over the 8 ranks. 512K tokens per
      request, 16 sequences.
    - **Stack:** Gaudi software 1.24.1, vllm-gaudi `v0.26.0` (vLLM `568afb3a`), torch.compile mode (`PT_HPU_LAZY_MODE=0`) with static
      shapes.
    - **Status:** serving production traffic.
    - **Headline:**
        - 33.9 / 33.4 / 32.5 tok/s per stream at 1 / 4 / 16 streams (production configuration on an unrouted
          canary, warm, 256-token runs).
        - Needles 3/3 at 500K, and 4 of 4 with four 500K prompts at once (one server).
        - Window-sized KV gives 5.33x the token capacity of stock vllm-gaudi with identical outputs.

## Why it needed a port

- vllm-gaudi v0.26.0 has no MiMo-V2 model. Upstream vLLM's `mimo_v2.py` targets GPUs.
- 39 of the 48 layers use a 128-token window with a per-head sink logit. Habana's FusedSDPA ignores both the
  `sinks` and the `window_size` arguments, and the stock paged decode adds the sink once per KV block.
- `HPUAttentionImpl` takes one head size; MiMo has qk 192 and v 128.
- vllm-gaudi gives every layer full-length KV, so the sliding layers store context they never read.
- The RoPE table is built on the device, and its rows past about 196K positions are wrong.

## What was built

| Model piece | On Gaudi 2 |
| --- | --- |
| qk 192 / v 128 heads | V zero-padded to 192 before attention, the extra 64 output columns dropped after. Exact (a zero V column adds nothing); costs a third more V cache |
| SWA 128 with sinks, prefill | own banded tiles with an fp32 softmax that includes the sink |
| SWA 128 with sinks, decode | window masked to exactly the last 128 positions; each of a sequence's n blocks gets the sink logit minus log(n), so the n copies sum to one sink term |
| 9 global layers, prefill with context | tiled global-attention prefill (below) |
| partial RoPE (64 of 192 dims) | cos/sin table rebuilt on the CPU in float64 |
| 256 experts, top 8, sigmoid with bias | fp32 gate, selection bias for the choice only; experts on the HPU MoE op, partial sums all-reduced |
| FP8 block attention weights | dequantised to bf16 at load |

The sink matters: q·k logits sit near -25 and the sinks near +1, so the sink normally takes almost all the
attention mass. Dropping it, as FusedSDPA does, is not a small error.

### Windowed KV groups

The sliding layers emit vLLM's `SlidingWindowSpec`, and vLLM core's hybrid KV cache manager does the rest: 6 KV
groups (global 9 layers, sliding 8/8/8/8/7) over 9 shared tensors, freeing of out-of-window blocks, the
sliding-window prefix-hit rule and preemption. The port adds a helper module, 13 small runner anchors and 2 worker
anchors:

- allocate once per shared tensor and bind it to every layer that shares it (the stock loop gives each layer its
  own tensors, 5.33x over budget);
- per-group slot mappings and window block lists, built with the runner's own buffer builder so shapes and
  buckets stay unchanged;
- swap each sliding layer's metadata view in the model's layer loop, which runs outside the compiled regions, so
  there are no new graph variants.

Global context costs 6,912 B per token per card instead of 36,864. The server reports 56,774 usable blocks against
10,645 without the change, 7.27M tokens. Prefill and greedy outputs are identical with the change on and off. It is
TP8 only: at TP4 the global and sliding layers hold different KV-head counts per rank, so their pages differ.

### Tiled global-attention prefill

For a chunk with context, the runner builds one `[chunk, context + chunk]` bias: at 512K with 8192-token chunks
that is 8.7 GB of bf16 plus 4.4 GB of bool, never profiled. The global layers instead split the chunk's queries
into tiles, build each tile's mask directly with `where`, and call FusedSDPA once per tile against the full keys
(`is_causal=False`, since the mask carries causality). Softmax is per query row, so no merge is needed. The tile
size keeps rows x keys x heads at or below 2³⁰ elements, about 2 GiB of scores. Budgeting rows x keys alone asked
for an 8.6 GiB tile at 400K and ran out of memory.

### The RoPE fix for large positions

The first 256K runs found the right needle but copied it with a duplicated digit. A GPU reference run with
upstream vLLM on the same prompts was clean. Probing q and k around positions 196K to 205K showed every rotary
frequency accurate to about 5e-4 rad except the fastest, whose angle equals the position in radians: its error was
random (median π/2). Gaudi fp32 cos/sin are wrong past about 1.5e5 radians (`cos(200000)` is off by 0.997), and vLLM built the table on the device.
Rebuilding it on the CPU in float64 brings the rotary error to bf16 level (0.27%). The generic fix is
[rope-cpu-cache](../fixes/rope-cpu-cache.md).

### The async duplicated-token fix

Under async scheduling, vllm-gaudi takes every decode row's input from a persistent device buffer that is only
written for requests that ran in the previous step. A decode the scheduler leaves out of one step comes back
reading a stale token, and greedy decoding emits a token twice. The long-prefill share makes that reachable here.
Seeding the decode rows from the host ids first ([async-inputs](../fixes/async-inputs.md)) removed it:

| Arm (four 70K prompts + one decode, greedy probes) | Probes | Repeated-token signature |
| --- | --- | --- |
| no fix | 190 | 42, all in probes that sat out a step |
| fix | 184 | 0 |
| fix + exact share under async | 282 | 0 |

## Correctness

Against a streaming fp32 CPU oracle running the reference math.

| Gate | Result |
| --- | --- |
| Prompt NLL, 1,535-token document (oracle 0.7943) | 0.7959 to 0.7966 TP4 MXFP4; 0.7986 TP8 FP8; 0.7959 production configuration |
| Decode vs prefill, 200 greedy tokens re-scored | mean abs Δ 0.02 to 0.05 nats, including prefix-cache hits |
| Needles 8K to 128K, 3 depths per rung | 12/12 |
| Needles at 207.5K and 242.3K, seeded | 12/12 after the RoPE fix (7/12 before; GPU reference 18/18 at 172.8K to 242.3K) |
| Needles at 128K / 256K / 500K | 3/3 / 3/3 / 3/3 |
| Tool calls: short suite; after a 33.8K and a 138K document | 7/7; correct |
| GSM8K 5-shot, flexible extract, 200 samples | 0.890 TP4 MXFP4; 0.910 TP8 FP8 |

At 500K and depth 0.9 some seeded needles miss on both platforms (H100 with upstream vLLM 3 of 4, Gaudi 2 of 4),
and fp32 softmax does not change the count. That is the model's own boundary; every needle at depths 0.1 and 0.5
hit up to 500K. MiMo also reads imperative "IMPORTANT NOTE" needles as prompt injection, so needles are plain facts.

## Performance

| Measurement | Result | Measured as |
| --- | --- | --- |
| Decode per stream at 1 / 4 / 16 streams, production configuration | 33.9 / 33.4 / 32.5 tok/s (aggregate 34 / 133 / 520) | unrouted canary, warm, 256-token runs |
| Decode at 1 stream, short / 128K / 258K context, TP8 FP8 at 256K max | 36.7 / 38.4 / 37.8 tok/s | one dev-pod server |
| Others' decode while one 500K prompt prefills: threshold 4096 only / plus share 0.5 | 0.4 / 14 to 18 tok/s | separate server starts, one dev pod |
| Others' speed kept during a 200K prefill, with 1 and with 3 others | stock share: 34% and 31%; exact under async: 50% and 49% | different conditions: separate canary restarts on different nodes |
| MiMo-V2.6-Pro, MXFP4 MoE kernels (canary only), 1 / 4 streams per stream | 18.1 / 8.8 to 39.7 / 23.5 tok/s | same-node A/B, stock canary pod then kernel canary pod |

Decode speed does not fall with context: 39 of the 48 layers read a 128-token window. The exact share costs the
long prompt a third more time to first token (178 to 237 s with one other stream). The Pro row uses
[XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) at TP8 with the
[MXFP4 MoE kernels](../kernels/mxfp4-moe.md); perplexity was unchanged. It is a canary result, not production.

## What it taught about Gaudi

- **FusedSDPA (Gaudi software 1.24.1) silently ignores `sinks` and `window_size`.** Sliding-window or sink models
  need their own prefill path or a masked bias. See [platform findings](../results/platform-findings.md).
- **Gaudi fp32 cos/sin are wrong past about 1.5e5 radians**, so any RoPE table built on the device fails beyond
  about 196K positions on the fastest frequency. Build tables on the CPU.
- **Paged decode counts a sink once per block**, not once per sequence. Rescaling each block's sink by log(n)
  keeps the stock block layout exact; [sink-once](../fixes/sink-once.md) covers gpt-oss.
- **Compile mode computes a new batch shape differently the first time it runs.** Compare outputs on warm runs
  only.
- vLLM core's hybrid KV manager already works for HPU sliding-window layers; the missing pieces are all in the
  runner. [Muse Spark Open](muse-spark-open.md) reuses the same design on vllm-gaudi v0.30.0.

## Availability

The integration code (HPU model, windowed-KV helper and runner patches) is not published in this repository yet.
It depends on model code outside vllm-gaudi: vLLM's MiMo-V2 model definition and the MiMo checkpoint layout. The
generic pieces are published as [fixes](../fixes/index.md).
