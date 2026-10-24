# GLM-5.3-Flash on Gaudi 2

GLM-5.3-Flash is a 320B-total, 18B-active multimodal MoE model built from four subsystems that rarely appear
together: Kimi Delta Attention (KDA, a linear-attention recurrence with per-channel decay), DeepSeek sparse
attention (DSA) over an MLA cache with no RoPE, manifold hyper-connections (mHC, four residual streams mixed
through a Sinkhorn projection) and a 288-expert MoE. When this port started, no vLLM backend could run it, so the
model itself had to be written for vLLM before any Gaudi work could begin. It now serves text and images on one
8-card Gaudi 2 node with a 256K-token window, and decode speed does not fall as the context grows.

!!! summary "At a glance"
    - **Checkpoint:** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), licence MIT.
      Block-scaled FP8 (128 x 128), 306 GiB. Loads unmodified.
    - **Hardware:** 8 Gaudi 2 cards, TP 8 with expert parallel (36 experts per card). 256K context served,
      512K tested.
    - **Stack:** Gaudi software 1.24.1, vllm-gaudi `v0.26.0`, lazy mode, out-of-tree model plugin.
    - **Status:** in service; code and recipe published (see [Availability](#availability)).
    - **Headline:**
        - Single-stream decode 22.8 tok/s at 2K context and 24.2 tok/s at 262K (same image and code; 262K on a
          pod rebuilt with the larger window).
        - Decode step 32.5 to 24.5 ms at one stream, greedy output bit-identical (same-pod A/B).
        - An 8,192-token prompt chunk 2.2 to 1.0 s (different pods, same command and image).

## Why it needed a port

At the start of this work a search of vLLM `main` and the `v0.28.0` release for `glm5_next`, `Glm5Next` and
`GLM-5.3` returned no hits, nothing in the open pull requests added it, and vllm-gaudi had nothing. The only
implementation anywhere was the reference model in Hugging Face transformers (`modeling_glm5_next.py`), and the
`transformers` 5.14.1 in the serving image did not include it. Upstream vLLM has since added GPU support
([vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906)), with Triton KDA kernels for NVIDIA
and AMD. Gaudi has no Triton driver, so that path does not run here, and vllm-gaudi has no HPU KDA path.

| Piece | Layers | Shape |
| --- | --- | --- |
| KDA linear attention | 34 | 64 heads x 128, short conv 4, gate lower bound -5.0 |
| DSA over NoPE MLA | 11 | `kv_lora_rank` 512, `qk_rope_head_dim` 0, `index_topk` 2048 |
| mHC | all 45 | 4 streams, 20 Sinkhorn iterations |
| MoE | 42 | 288 routed + 1 shared, top-8, sigmoid scoring |
| Vision tower | 24 blocks | patch 14, merge 2, clamped SwiGLU |

Because `qk_rope_head_dim` is 0, the attention layers carry no positional signal. Every bit of token order in
this model comes from the KDA recurrence, which shapes how its defects look.

## What was built

The model is an out-of-tree vLLM package registered through a `vllm.general_plugins` entry point, so it registers
in every engine and worker process. It reuses vLLM's DeepSeek-V2 MoE and MLP, its mHC ops, its MLA wrapper and
the parameter plumbing of its Kimi KDA layer. The config is vendored and checked against the real `config.json`
(53 fields).

**KDA kernels in PyTorch.** vllm-gaudi's gated-delta-rule kernel takes a per-head scalar decay; KDA's is
per channel. The reference chunked form materialises an `exp(g_i - g_j)` decay mask of 8.6 GB at 4K tokens. The
cheap factorisation into two matmuls overflows fp32, because the within-chunk cumulative decay reaches about -300
(22,912 non-finite entries in one chunk on real gate values). The kernel anchors the exponent at a block boundary,
so both factors land in (0, 1] and the product is an exact matmul. All tensors stay at rank 4 or less: a rank-7
tensor in the same graph as the MoE op fails with "ndim is currently not supported for MOE". The gate is GLM's
bounded sigmoid, not the softplus gate of vLLM's Kimi layer. Reusing the Kimi gate would have loaded and produced
fluent text, with retention over 64 tokens off by twelve orders of magnitude.

**Prefill scan.** In lazy mode the stock loop over 64-token chunks is re-recorded on the host for every prompt
chunk (about 37 ms per KDA layer at 8,192 tokens). The carry between chunks is affine, `S[i+1] = M[i] S[i] + B[i]`,
so the states entering all chunks come from a Hillis-Steele scan: 7 levels of batched matmuls at 128 chunks. Every
`M[i]` is a product of contractions, so the scan has no cancellation problem. Host recording at 8,192 tokens drops
from 31.8 to 5.0 ms and aten ops per call from 2,121 to 412 (per-call, one card).

**DSA.** The 11 DSA layers run as dense MLA with the indexer skipped. For prompts up to 2,048 tokens this is the
same function: the indexer selects `min(2048 / 4, ceil(S / 4))` key pools, which is every pool. Beyond that it is
an approximation; long-context needles still pass to 512K (below). NoPE MLA needed one line in vllm-gaudi:
`k_pe.view(-1, 1, 0)` cannot infer `-1` from zero elements, so the leading dimension is passed explicitly.

**Fused mHC Sinkhorn.** At decode, mHC was the largest single cost: about 16 ms of each step, from 90 Sinkhorn
calls of roughly 100 small lazy ops each. A TPC kernel runs the 20-iteration 4 x 4 Sinkhorn in one launch. To be
bit-identical it reproduces the device's reduction order (pairwise for one axis, sequential for the other) and
its divide (`x * reciprocal(y)`). Per `mhc_pre` + `mhc_post` pair at one row: 198 to 75 µs (per-call, one card).
Marking one graph per decode step instead of one per layer removes 44 launches, which matters once the kernel has
made the step host-bound.

**Vision.** The tower is ported from upstream's own classes and is bit-exact against them (0.0 at fp32 with
explicit-matmul attention; the shipped SDPA path differs only by accumulation order). It runs eager, because
vllm-gaudi's vision graph wrap is gated on Gemma 3 and on an attribute named `vision_tower`. Images cost compile
time through the prompt-length buckets they create, not through image geometry, so the deployment caps image
tokens at 2,048 and bounds the graph cache with `--max-cudagraph-capture-size 2048`. Image merging uses the
[mm-merge-idxcopy](../fixes/mm-merge-idxcopy.md) fix.

**Serving shape.** vLLM derives a 640-token KV block (the attention page is raised to at least the KDA state page)
and its autosizing over-commits by about one MLA layer, so KV is pinned with `--num-gpu-blocks-override`.
Prefix caching is off and the layer refuses to start with it, because the KDA layer does not store per-block
states. Expert parallel replaced expert slicing across TP 8 (48 to 32 ms per token at one stream; different
conditions: an idle server before, a second server after), together with the
[moe-explicit-overload](../fixes/moe-explicit-overload.md) fix. The [async-inputs](../fixes/async-inputs.md) fix
is required under async scheduling.

## Correctness

The defect hunt ended at these root causes:

| Defect | Effect | Fix |
| --- | --- | --- |
| `(I - A)^-1` built by nilpotent doubling | exact in algebra, catastrophic cancellation in fp32 on the real `A`; rows 32 to 63 of every chunk wrong, so prompts past about 50 tokens read as garbled | blocked forward substitution, 16 x 16 blocks, 4 sequential stages |
| bucket padding folded into the recurrence | the layer override skipped upstream's trim; batched prompts shared one state | mask q/k/v, gate and beta; fold to `[n_seqs, seq_len]` |
| NaN in the padded conv tail | `0 * NaN = NaN`; one NaN value row poisons every query of the first MLA layer | `nan_to_num` after masking |
| conv-state write through strided views of the KV cache | graph replay fails ("Neither storage attached") | one `copy_` into the whole tensor |
| constants allocated inside capture | garbage output, or a crash on the second prompt bucket | allocate at layer init |
| `swiglu_limit` 10.0 ignored | unbounded SwiGLU on clamped weights | `SiluAndMulWithClamp` on dense and shared experts; routed experts stay on the fused FP8 op, where the clamp binds on about 1 element in 10^5 to 10^7 (measured through layer 11) |
| missing `is_hybrid` | every KDA layer silently skipped | set it; count KDA calls per rank |
| flat `A_log` checkpoint tensor | TP 8 load fails; TP 1 hides it | reshape before sharding |

| Gate | Result |
| --- | --- |
| KDA kernel vs float64 recurrence, real layer-0 inputs | max relative error 4.8e14 before the inverse fix, 2.3e-4 after |
| mHC vs upstream transformers, against float64 | `post` 2.7e-7, `comb` 2.9e-7 |
| Needles, depths 10/50/90% | 12/12 at 8K to 129K; 3/3 at 262K; 3/3 at 516,900 tokens |
| Two distant facts in one answer | correct at 129K (facts 84% of the context apart), 260K and 517K |
| Tool call after a long document | correct at 129K, 260K and 517K |
| Decode fast path vs stock, greedy | 12/12 single streams and 56/56 lock-step batched streams identical |
| Prefill scan vs stock | 20/20 greedy prompts byte-identical; 3 of 56 lock-step batched streams split at near-ties |
| Vision | image suite 7/7; video (opt-in, not served) 5/6, the miss is the model's colour naming, checked at pixel level |

## Performance

| Metric | Before | After | Measured as |
| --- | --- | --- | --- |
| Single-stream decode by context | 22.8 tok/s at 2K | 22.5 at 32K, 21.8 at 129K, 24.2 at 262K, 23.7 at 517K | same image and code, warm; 262K and 517K on pods rebuilt for the window; before the expert-parallel change |
| TTFT, warm | 0.5 s at 2K | 70 s at 129K, 163 s at 262K, 429 s at 517K | same runs |
| Decode step, 1 / 32 streams, 2K | 32.5 / 57.3 ms | 24.5 / 44.2 ms | same-pod A/B (server relaunched per arm); Sinkhorn kernel + one graph per step |
| Aggregate decode, 32 streams, 2K | 559 tok/s | 724 tok/s | same |
| 8,192-token prompt chunk, no prior context | 2,207 ms | 975 ms | different pods on two nodes, same command and image; the scan arm also had the async-inputs fix |
| TTFT 16K / 64K / 260K | 5.2 / 20.6 / 87.4 s | 2.1 / 9.5 / 67.3 s | same |

Above about 127K tokens of context the padded MLA attention, which the KDA host time used to hide, now sets the
prefill step, which is why 260K gains less.

## What it taught about Gaudi

- **A parity test with generated inputs tests your imagination.** The unstable inverse passed every synthetic
  test (`0.3 * randn` gives 1e-15 error); only the real model's `A` exposed it. Running the whole reference model
  on CPU on the failing prompt, and comparing against upstream's code rather than a self-written reference, is
  what settled it.
- **Write into graph inputs only as whole tensors**, and allocate constants before the first capture. Exercise
  at least two prompt buckets after any graph-mode change.
- **HPU `max` skips NaN**, so per-tensor absmax checks miss poisoned tensors; count non-finite positions instead.
  Values read inside a captured graph are not trustworthy diagnostics.
- **vllm-gaudi's synced weight loader drops the inner loader's return value**, so `return_success` reads `None`
  and completeness checks built on it mis-report every expert. Check bytes, not bookkeeping.
- **Source patches must run as separate processes before `vllm serve`.** Applied from a plugin, they edit files
  already imported in that process.
- The decode-step anatomy method and the FusedSDPA findings on the larger GLM models are in
  [GLM-5.2 and GLM-5.3 in MXFP4](glm-5-mxfp4.md) and [platform findings](../results/platform-findings.md).
  Request-level fixes this deployment carries include [logprobs-decode-guard](../fixes/logprobs-decode-guard.md)
  and [gather-logprobs-int32](../fixes/gather-logprobs-int32.md).

## Availability

The port is published in [`ports/glm-5-3-flash`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/glm-5-3-flash):
the model plugin, the KDA kernels, the Sinkhorn TPC kernel, four source patches, CPU tests and the device gates. It
installs into a stock vllm-gaudi `v0.26.0` image; the served configuration is the
[GLM-5.3-Flash recipe](../recipes/glm-5-3-flash.md). The port carries no video path and no MTP head, and upstream's
GPU kernels from [vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906) are not used; the
[port README](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/glm-5-3-flash/README.md#differences-from-upstream-vllm)
lists the differences.
