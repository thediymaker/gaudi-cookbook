# Two-node pipeline parallel (PP2 x TP8)

A 744B GLM-5.3 does not fit one 8-card Gaudi 2 node in FP8, which is why the single-node build serves MXFP4
experts ([GLM-5.2 and GLM-5.3 in MXFP4](../ports/glm-5-mxfp4.md)). Splitting the model by layers across two
ordinary nodes lets FP8 experts fit, roughly doubles the KV pool, and needs no scale-out fabric: only the hidden
state crosses between stages, over the host NIC. vllm-gaudi has no working pipeline parallelism, so this needed a
runtime patch. The result is a long-context and accuracy tier: a 637K-token needle passes, and FP8 expert weights
are 2.65% from BF16 against 11.18% for MXFP4. Its per-stream speed edge over the single-node build was measured
before the MXFP4 TPC kernel and no longer holds.

!!! summary "At a glance"
    - **Model:** GLM-5.3, FP8 per-channel experts quantized from
      [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16) (GLM-5.3 License). The checkpoint is
      not public; the conversion is reproducible from the public BF16 release.
    - **Hardware:** two 8-card Gaudi 2 nodes. Stage 0 = layers 0 to 41 and the embedding, stage 1 = layers 42 to
      77 and the LM head; TP 8 + EP 8 inside each node; 100 Gbit/s host NIC between them.
    - **Stack:** Gaudi software 1.24.1, a vllm-gaudi 0.24 build (vLLM 0.24), lazy mode, plus a pipeline patch.
    - **Status:** canary and study. Not in production.
    - **Headline:**
        - KV pool 709,888 tokens against about 346K on one node (both at `--gpu-memory-utilization 0.86`).
        - Needles 3/3 at 512K, 600K and 637,640 tokens.
        - FP8 expert weight error 2.65% against 11.18% for MXFP4 (all 19,456 experts, against BF16).

## Problem

In vllm-gaudi 0.24, `v0.26.0` and `v0.30.0`, the HPU model runner never passes `intermediate_tensors` to the model
and nothing sends activations between stages, so pipeline parallelism does not run. Tensor parallelism across two
nodes would need the RoCE scale-out fabric, which many Gaudi 2 installations do not have cabled on
every node. Pipeline
parallelism moves only one hidden-state row per token between stages, about 12 to 24 KB.

## Approach

**Runtime patch.** Two nodes form one vLLM multiprocess executor (`--nnodes 2`, the second node headless, no Ray).
Stage outputs travel as CPU tensors over gloo on the host NIC, about 1 ms per decode step. Crossing top-of-rack
switches measured 94 Gbit/s and about 30 µs extra, so the two nodes need no pairing rule. Each stage loads only its
own layers. Required settings:

- the stage split must start on a layer that owns a DSA indexer (layers that reuse the previous indexer's
  selection cannot open a stage), hence 42 / 36 rather than 39 / 39;
- `--no-async-scheduling`, which costs about 10% at one stream;
- the scheduler admits at most half the running requests per microbatch.

**FP8 per-channel checkpoint.** Routed experts (and the MTP layer) are FP8 e4m3 with one fp32 scale per output
channel (amax / 240, the Gaudi 2 maximum), in compressed-tensors format, with dynamic per-token FP8 activations.
Everything else is the same BF16 as the single-node build. A validator checks every expert code against an
independent round-to-nearest-even grid. It loads on the stock HPU FP8 MoE method with
`VLLM_SCALE_ADJUSTMENT=false`, `--enable-ep-weight-filter` and the
[moe-explicit-overload](../fixes/moe-explicit-overload.md) fix, without which FP8 graph replay is host-bound from
two tokens. Weights take 47.8 and 43.8 GB per card on the two stages.

**Why the KV pool only doubles.** The MLA latent cache is replicated on every TP rank. Splitting layers across two
nodes halves the per-card cost of a token, so the pool grows about 2x, not 16x. Many concurrent 1M-token streams
would need data-parallel attention or a host-memory KV tier.

**Long context.** Past 262K keys stage 0 ran out of device memory in the DSA prefill (indexer scratch, then
dense-masked attention). Row-budget limits in the DSA patch fixed it without changing the key selection or
anything below 32K keys, at a cost of 2% to 5% TTFT at 32K to 128K. The long-context canary runs at utilization
0.80 (pool 660,352 tokens).

**MTP under PP.** vLLM's config check refuses it: the draft model inherits the pipeline size, and the DeepSeek MTP
model does not implement `SupportsPP`. A second patch runs the draft on the last stage only, with a bucket file
for the verify shapes (without it, 256K at 8 streams fell from 20.7 to 8.8 tok/s).

## Measured results

| Gate | Result |
| --- | --- |
| Needles, 3 per prompt | 3/3 at 40K, 80K, 122K, 128K, 256K, 512K, 600K and 637,640 tokens |
| Six back-to-back ~250K requests | all served, no hang (the single-node 256K hang did not reproduce) |
| Golden prompts, loops | 5/5 correct, 0/10 loops (single node: 0/10) |
| Tool calling | 16/16 shapes and 33/33 reliability calls |
| Teacher-forced NLL, 20,308 positions of code-repository text | 1.6483 against 1.6511 for the single-node MXFP4 build |
| MTP under PP | gates pass (golden, loops, prompt logprobs, tools 30/30, needles at 256K), acceptance 0.78 to 0.85 |

Accuracy against BF16 (whole model on one card, 6 real texts x 512 tokens, the real HPU MoE ops):

| | FP8 per channel | MXFP4 |
| --- | --- | --- |
| Expert weight error, all 19,456 experts | 2.65% | 11.18% |
| Routed-expert output error on real hidden states, mean of 76 layers | 4.8% | 15.1% |
| Per-token KL from BF16 / top-1 agreement | 0.019 / 93.6% | 0.042 / 91.6% |
| MTP greedy acceptance (BF16 0.771) | 0.773 | 0.767 |

About 30% of the FP8 output error is the weights; the rest is the per-token FP8 activations.

| Speed | Result | Measured as |
| --- | --- | --- |
| FP8 PP2, 2K context, 1 / 4 / 32 streams, per stream | 30.5 / 27.8 / 15.5 tok/s (495 aggregate at 32) | canary pair, MTP off, warm |
| MXFP4 PP2, same | 14.9 / 11.8 / 4.5 tok/s | same pair, MXFP4 checkpoint |
| FP8 PP2 + MTP, 1 stream, 2K to 256K | 39 to 46 tok/s, 1.3x to 1.6x over no MTP | a second pair with the MTP patch |
| Async scheduling under PP, 1 stream | +11% | A/B on the second pair |
| TTFT, FP8 PP2, 2K / 32K / 128K | 0.29 / 4.55 / 30.7 s | canary pair, warm |
| Decode at 640K window, 1 stream, 128K / 256K context | 20.8 / 15.9 tok/s | canary pair, utilization 0.80 |
| MoE layer, 1 / 32 tokens, one card | FP8 0.11 / 0.53 ms; stock MXFP4 op 0.50 / 4.53 ms | per-call, one card |

MXFP4 at PP2 runs at the single node's per-card rate: the speed gain came from FP8 experts, not from pipeline
parallelism. Prefill is MoE-bound, which is why FP8 also shortens TTFT.

**The per-stream speed edge no longer holds.** The pair's 30.5 tok/s (39 to 46 with MTP) was compared against
the single-node MXFP4 build before its TPC kernel, which then ran 17.0 tok/s at one stream (21.7 with MTP). The
first kernel alone took the single node from 23.25 to 45.6 tok/s at one stream with MTP (same-pod A/B), and later
revisions are faster still. The pair remains ahead on context length, KV capacity and expert accuracy.

The same runtime ported to vllm-gaudi `v0.30.0` for Muse Spark Open, from an FP8 checkpoint, gave the same
picture: needles to 1M tokens and a 4.19M-token KV pool against 1.38M, but 27.3 tok/s at one user against 33.5
for the single-node MXFP4 kernel build (different conditions: separate pods and runs). See
[Muse Spark Open](../ports/muse-spark-open.md).

## Limits

- A two-node replica must restart, drain and report readiness as a unit. A design using an
  [AIBrix](https://github.com/vllm-project/aibrix) StormService with a pod group of two (routing only to the
  leader) passed failure tests and a 10-hour soak, but has not been deployed.
- Async scheduling and MTP together under PP were not built.
- Runs are not bit-reproducible across instances (cold vs cached numerics).
- At the real traffic mix, prefill limits a pair to about 8 concurrent users at a comfortable TTFT.

## Availability

The pipeline patches and the FP8 conversion are not published in this repository yet. They depend on model code
outside vllm-gaudi.
