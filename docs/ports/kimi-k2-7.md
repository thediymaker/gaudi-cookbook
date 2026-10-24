# Kimi-K2.7-Code on Gaudi 2

Kimi-K2.7-Code is a 1T-parameter multimodal MoE model: a DeepSeek-V3-style text tower (61 layers, 384 routed
experts, MLA attention) plus a vision tower. At this size it does not fit one 8-card Gaudi 2 node in any servable
format, so it runs tensor-parallel across two nodes over RoCE. The work here was less a new model implementation than
making the checkpoint load, fit and run fast on vllm-gaudi: a repack of AMD's Quark MXFP4 release into vLLM's fused
MXFP4 layout, then an exact conversion of the routed experts to FP8 per-channel, which doubled single-stream decode.
It serves production traffic, text only, at 131,072 tokens of context.

!!! summary "At a glance"
    - **Checkpoint:** [moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code), modified MIT
      licence. The served weights are converted from
      [amd/Kimi-K2.7-Code-MXFP4](https://huggingface.co/amd/Kimi-K2.7-Code-MXFP4) (modified MIT); the converted
      checkpoints are not published, and both conversions are reproducible from that public checkpoint.
    - **Hardware:** 16 Gaudi 2 cards on **two nodes connected by RoCE**: TP 16 with expert parallelism (24 experts
      per card), 131,072-token context, 8 sequences, 266,240-token KV pool per replica.
    - **Stack:** Gaudi software 1.24.1, lazy mode, vllm-gaudi built from source at commit `90815f2` with vLLM 0.24.0
      (older than the `v0.26.0` release), Ray for the cross-node workers.
    - **Status:** serving production traffic, text only.
    - **Headline:** per-stream decode at a 16K context, MXFP4 experts vs FP8 experts: 17.9 vs 36.4 to 38.2 tok/s at
      one stream, about 6.5 vs 20.7 to 21.4 tok/s at 8 streams (different conditions, see Performance).

## Why it needed a port

- **The INT4 release did not fit.** At TP 16, vllm-gaudi's compressed-tensors INT4 path repacks weights on the device
  after loading and ran out of HBM on 16 x 96 GB. TP 32 worked but cost four nodes per replica.
- **MXFP4 fits at TP 16** (35.75 GiB of weights per card) because vllm-gaudi's native MXFP4 MoE op dequantizes in the
  TPC without materialising bf16 experts. But AMD's MXFP4 release is a per-expert Quark checkpoint, which stock
  vllm-gaudi does not load as packed MXFP4.
- **Kimi's grouped top-k routing** was not wired into the vllm-gaudi MXFP4 MoE method of that build, and the fused
  expert tensors were not sliced per EP rank at load.

## What was built

**Quark MXFP4 repack.** A host-only tool rewrites the AMD checkpoint into vLLM's `gpt_oss_mxfp4` layout: per-expert
tensors fused per layer, gate and up rows concatenated, E8M0 block-32 scales, so the format is auto-detected. Attention
(FP8) and the dense and shared MLPs are dequantized to bf16; the router, `lm_head` and vision tower stay bf16. A
one-card layout probe confirmed the concatenated layout on the device (relative error 0.5% against a dequantized
reference). Two startup patches make it serve: the grouped top-k router (sigmoid scoring with correction bias) for
the MXFP4 method, and EP slicing of the fused `[384, ...]` expert tensors at load.

**Why MXFP4 decode was slow.** The compiled MoE recipes show that the stock MXFP4 op runs a fixed number of expert
slots per rank, `min(8T, local experts)` for T tokens, whatever the routing. Each slot is a DMA of one expert's packed
weights, a TPC unpack to bf16 and an MME GEMM. HBM bandwidth was about 5% used; the TPC unpack, about 0.19 ms per
touched expert per layer, is the cost.

**Exact MXFP4 to FP8 per-channel conversion.** FP8 experts feed the MME directly. The conversion is exact, not a
re-quantization:

- An MXFP4 element is an E2M1 value times a power-of-two E8M0 scale per 32 values. For each output row the tool picks
  one power-of-two channel scale, 2^k with k = (exponent of the row's largest element) - 7, so every element becomes an
  E4M3 normal with exponent in [-5, 7] and magnitude at most 192.
- Gaudi 2's E4M3 has a maximum of 240 (exponent field 15 is not finite), so 192 fits. This needs the E8M0 scales
  within a row to span at most 9 binades; all 259.5 million channels do, and 99.96% span at most 2.
- vllm-gaudi halves FP8 weights at load by default (scale adjustment); the -5 floor keeps the halved copy normal, and
  the deployment turns the halving off anyway.
- The result loads through vllm-gaudi's stock compressed-tensors FP8 method (per-channel weights, dynamic per-token
  activations), so no loader patch is needed. Activations become FP8 per token: W8A8 instead of W4A16.
- 969 GiB in total, 63.42 GiB per card at TP 16. At TP 8 the experts alone would need 118 GiB per card, so FP8 is a
  two-node configuration.

**Runtime fixes on top.** The FP8 MoE op called through the overload packet stringified a device tensor per MoE layer,
which syncs the lazy graph: [moe-explicit-overload](../fixes/moe-explicit-overload.md). Logprobs ids were corrupted by
an int64 concatenation in lazy mode ([gather-logprobs-int32](../fixes/gather-logprobs-int32.md)), and Kimi's tiktoken
tokenizer raises `KeyError` rather than `OverflowError` on a bad id, so the decode guard
([logprobs-decode-guard](../fixes/logprobs-decode-guard.md)) needed that case too. vLLM 0.24 called a Triton top-k/top-p
kernel at batch 8 and above, which fails on HPU (patched to take the torch path), and Kimi runs with prefix caching, so it carries
[prompt-order](../fixes/prompt-order.md).

**EAGLE3, rejected.** The public draft head
[novita/kimi-k2.7-code-eagle3-mla](https://huggingface.co/novita/kimi-k2.7-code-eagle3-mla) (MIT) runs on HPU after
several runner fixes, but vllm-gaudi's rejection sampler is greedy-only, so serving it forces greedy decoding on every
request, and at 8 streams it is no faster than plain FP8. Replicas are shared, so production runs plain FP8.

**Text only.** Every new image size is a new lazy-mode graph, and very large images stalled the vision prefill long
enough to trip the engine's step timeout. The deployment runs `--language-model-only`, so the vision tower is not built
and image requests get a 400.

**Two nodes.** Each replica is a pair of 8-card nodes with RoCE scale-out links configured on both (per-port IPs and a
24-entry `gaudinet.json`). Each pair is its own Ray cluster with its own HCCL coordinator address.

## Correctness

| Gate | Result |
| --- | --- |
| FP8 encoding tests (including adversarial rows, the -5 floor edge, E8M0 0/254/255, the Gaudi 2 halving replayed) | 22/22; seven deliberate mutations of the tool each fail at least one test |
| Converted checkpoint vs AMD's per-expert MXFP4 tensors | sampled experts element-exact; every layer "384 experts exact" |
| Tool calls, FP8 experts | pass |
| Logprobs soak, 200 requests with `top_logprobs` | 200/200 clean |
| Needles | 6/6 up to 100K |
| Greedy goldens | all answers correct; wording differs from MXFP4 in places (W8A8 activations) |

## Performance

Warm sweep at a 16K-token shared context, per-stream decode tok/s. Measured as: different conditions. MXFP4 ran on a
production replica on one node pair; FP8 ran on a canary on another pair with the same configuration otherwise. Values
marked ~ are approximate in the source.

| Streams | MXFP4 experts | FP8 experts |
| --- | --- | --- |
| 1 | 17.9 | 36.4 to 38.2 |
| 2 | 11.9 | 36.2 to 36.4 |
| 4 | ~8.1 | 29.1 to 29.2 |
| 8 | ~6.5 | 20.7 to 21.4 |

| Measurement | MXFP4 | FP8 | Measured as |
| --- | --- | --- | --- |
| One MoE layer, T = 1 / 4 / 8 tokens | 0.31 / 1.07 / 1.43 ms | 0.10 / 0.29 / 0.26 ms | per-call, one card |
| FP8 TTFT, fresh 2K prompt, before / after the overload fix | | 5.1 s / 0.43 s | same canary, before and after the patch |
| EAGLE3 on FP8, per stream at 1 / 2 / 4 / 8 streams | | 60.2 / 46.2 / 31.9 / 19.9 tok/s (plain FP8: 36.4 / 36.4 / 29.1 / 20.7) | same canary pair and sweep, separate server starts |

## What it taught about Gaudi

- **The MXFP4 MoE op cost is set by slots, not routing.** On a single-node TP 8 MXFP4 test, passing the global expert count
  to the op (`VLLM_MOE_CHUNK=256`) let it skip empty slots: 60 / 94 / 159 ms to 55 / 75 / 100 ms per token at 1 / 2 /
  4 streams with identical greedy output on 14 of 14 prompts (same-node A/B).
- **FP8 KV cache with unit scales hurt long context.** On the same node, `fp8_inc` KV found needles in 1 of 2 runs
  at about 200K and 0 of 2 at about 240K, against 2 of 2 and 1 of 2 with bf16 KV. Production keeps bf16 KV.
- **FP8 per-channel experts load slowly over NFS** (54 to 55 minutes cold, against 4 to 6 from local NVMe), because
  each node reads about 527 GiB and the page cache cannot hold it next to the engine.
- More in [platform findings](../results/platform-findings.md).

## Availability

The integration code is not published in this repository yet. It depends on code outside vllm-gaudi: the two
checkpoint conversion tools and startup patches for the model's routing and loading on the build used. It requires a
RoCE-connected pair of 8-card Gaudi 2 nodes.
