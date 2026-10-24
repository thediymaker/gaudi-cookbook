# Kimi-K2.7-Code (MXFP4)

Moonshot AI's Kimi-K2.7-Code (1T-parameter MoE, 32B active, MLA attention, 384 routed experts top 8) served
text-only from AMD's MXFP4 release on one 8-card Gaudi 2 node at 128K context. The
[kimi-k2-7 port](../ports/kimi-k2-7.md) repacks the release into vLLM's fused MXFP4 layout, lets the loader split
the fused experts by expert group, and runs decode-sized MoE calls on the `kimi` build of the
[MXFP4 MoE kernels](../kernels/mxfp4-moe.md) behind one switch, `KIMI_MXK`.

| | |
| --- | --- |
| Model | [amd/Kimi-K2.7-Code-MXFP4](https://huggingface.co/amd/Kimi-K2.7-Code-MXFP4) (from [moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code)), repacked by the port |
| Hardware | 8x Gaudi 2 (TP 8, `--enable-expert-parallel`: 48 experts per card) |
| Image | the kimi-k2-7 port image on vllm-gaudi `releases/v0.30.0` at `058f107`, vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | routed experts MXFP4 (E2M1 with an E8M0 scale per 32), everything else BF16: 659 GB, 67.3 GB per card |
| Context | 131,072 served (262,144 native), KV pool 260,480 tokens |
| Concurrency | `--max-num-seqs 8` |
| Parsers | reasoning and tools `kimi_k2` |
| Licence | modified MIT: MIT, plus one condition: a product or service with more than 100 million monthly active users or more than 20 million US dollars of monthly revenue must display "Kimi K2.7 Code" prominently in its user interface |
| Recipe | [`recipes/amd/Kimi-K2.7-Code-MXFP4/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/amd/Kimi-K2.7-Code-MXFP4/recipe.yaml) |

## Convert the checkpoint

AMD's release stores one Quark tensor per expert, which vLLM does not load as packed MXFP4. Repack it once, on
CPUs (about 650 GB in, 659 GB out; expert bytes are copied, not re-quantized):

```bash
python3 ports/kimi-k2-7/convert/repack_quark_mxfp4.py --src <amd/Kimi-K2.7-Code-MXFP4 snapshot> --dst $KIMI_MXFP4_DIR
python3 ports/kimi-k2-7/convert/repack_quark_mxfp4.py --src <snapshot> --dst $KIMI_MXFP4_DIR --verify   # VERIFY_OK
```

The FP8 attention and the dense and shared MLPs become BF16; the output keeps the release's `LICENSE`.

## Build the image

From a checkout of this repository, on top of your stock `releases/v0.30.0` image (built as in
[Getting started](../getting-started.md)):

```bash
docker build -f ports/kimi-k2-7/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-kimi-k2-7-image> .
```

The build copies the repository to `/opt/gaudi-cookbook`, checks the image commits, applies the port's two patches,
installs the `kimi_hpu` package and builds the kernels (`kernels/mxfp4-moe/build.sh --variant kimi`) into
`/opt/gaudi-cookbook-kimi/mxk`. Before the first start with `KIMI_MXK=1`, run the kernel parity check on one free
card (it must print `[kimi] PARITY PASS` and `[kimi] SIG_DEVICE PASS`):

```bash
bash /opt/gaudi-cookbook/kernels/mxfp4-moe/tests/run_parity.sh --variant kimi --lib /opt/gaudi-cookbook-kimi/mxk \
  --checkpoint $KIMI_MXFP4_DIR --out /tmp/parity
```

## Launch

Inside the port image (`$COOKBOOK` is `/opt/gaudi-cookbook` there).

```bash
export KIMI_MXFP4_DIR=/models/Kimi-K2.7-Code-MXFP4
export KIMI_MXK=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_SKIP_WARMUP=true
export VLLM_EXPONENTIAL_BUCKETING=false
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_QUERY_BUCKET_MIN=512
export VLLM_PROMPT_QUERY_BUCKET_STEP=512
export VLLM_PROMPT_QUERY_BUCKET_MAX=2048
export VLLM_PROMPT_CTX_BUCKET_MIN=0
export VLLM_PROMPT_CTX_BUCKET_STEP=256
export VLLM_PROMPT_CTX_BUCKET_MAX=1024
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=8
export VLLM_DECODE_BLOCK_BUCKET_MIN=256
export VLLM_DECODE_BLOCK_BUCKET_STEP=512
export VLLM_DECODE_BLOCK_BUCKET_MAX=8192
export VLLM_MOE_CHUNK=256
export VLLM_MOE_TOKEN_BOUNDARY=2048
export PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

# Fixes: the v0.30.0 profile. Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 || exit 1

vllm serve $KIMI_MXFP4_DIR \
  --served-model-name amd/Kimi-K2.7-Code-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code
```

- `--enable-expert-parallel` is required: each card loads its block of 48 experts from the fused tensors, and the
  loader refuses to split one expert across cards.
- `KIMI_MXK=1` runs MoE calls of up to 16 tokens (every decode step at 8 sequences) on the kernels; prefill keeps
  the stock op. `KIMI_MXK=0` is the stock op, with nothing else changed. The engine log shows
  `[kimi-mxk] enabled: mxk_tpc kimi-v8-<hash>, E_local=48 K=7168 I=2048` when it is on.
- `VLLM_MOE_CHUNK=256` and `VLLM_MOE_TOKEN_BOUNDARY=2048` let the stock op skip expert slots no token routes to;
  the output is unchanged.
- `--language-model-only` serves text only: the vision tower is not built, and image requests get HTTP 400.
- Linear buckets keep the shape set small; with `VLLM_SKIP_WARMUP=true` the first requests of each new shape
  compile (up to a few minutes on a cold recipe cache).
- The first start reads 659 GB: 21.5 minutes from network storage in the run below.

## Required fixes

The `v0.30.0` profile, applied at launch.

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching is on by default |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids are read back corrupted otherwise |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | Kimi's tiktoken tokenizer raises `KeyError` on a bad id, which would stop the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with a `seed` would segfault a tensor-parallel worker |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | a failed rank stops the engine instead of serving garbage |

The port adds two patches of its own: the loader change for fused MXFP4 experts under expert parallelism (generic
for DeepSeek-style MXFP4 checkpoints) and the two hooks that install the kernel switch.

## Verified

Pod with 8 Gaudi 2 cards on one node, the port built by `install.sh` on the stock `releases/v0.30.0` image, the
Launch block above, `tests/hpu/run_gates.sh`. The same server build was started twice on the same pod: once with
`KIMI_MXK=0` (stock MXFP4 op), then with `KIMI_MXK=1`.

| Check | `KIMI_MXK=0` | `KIMI_MXK=1` |
| --- | --- | --- |
| Kernel parity, real layer-3 experts, T = 1 to 16 (`run_parity.sh --variant kimi`) | | PARITY PASS, SIG_DEVICE PASS (error vs fp32 reference 1.7e-3, stock op 4.8e-3 to 5.5e-3) |
| Golden answers, 12 prompts, greedy | 12/12 | 12/12 (both arms all correct) |
| Tool call, `kimi_k2` parser, streaming and not | pass | pass |
| Needles, 3 depths at 32K, 64K and 120K | 9/9 | 9/9 |
| `top_logprobs=20` soak with CJK and emoji output, 200 requests | 200/200 clean | 200/200 clean |
| Teacher-forced NLL, 8,198 positions of code | 1.2878 | 1.2888; top-1 agrees with the stock op on 7,826 of 8,198 positions (0.955) |
| Image request | HTTP 400 (`--language-model-only`) | |

## Performance

Decode at a 16,000-token cached prefix, 256 tokens per request, 2 x C requests at C streams (`kimi_gates.py sweep`).
Per stream is 1 / median time per output token; aggregate is all tokens over the wall time, prefill included.
Measured as: one 8-card node, same pod and server build, `KIMI_MXK=0` then `KIMI_MXK=1`, second of two sweeps
(the first gave 18.0 / 13.1 / 10.4 / 6.3 and 44.8 / 34.3 / 21.0 / 12.2 tok/s per stream).

| Streams | Stock op, per stream (aggregate) | Kernels, per stream (aggregate) |
| --- | --- | --- |
| 1 | 17.9 (17.3) tok/s | 44.5 (40.6) tok/s |
| 2 | 12.9 (24.8) tok/s | 34.5 (61.9) tok/s |
| 4 | 9.8 (36.6) tok/s | 23.4 (79.0) tok/s |
| 8 | 7.5 (56.0) tok/s | 14.8 (95.5) tok/s |

| Measurement | Stock op | Kernels | Measured as |
| --- | --- | --- | --- |
| TTFT, uncached 16,000-token prompt | 6.49 s | 6.48 s | same pod, median of 3, server otherwise idle |
| Weight load from network storage | 21.5 min (cold) | 9.9 min on rank 0 (warm page cache); the last rank finished 15 min after start | same pod, first and second start |

Two-node FP8 (TP 16 over RoCE, 16 cards) on an older vllm-gaudi build ran 36.4 to 38.2 / 36.2 to 36.4 / 29.1 to 29.2
/ 20.7 to 21.4 tok/s per stream at 1 / 2 / 4 / 8 streams with the same 16K shape. Measured as: different conditions
(another build, another sweep tool, a different pair of nodes); see the [write-up](../ports/kimi-k2-7.md).
One node with the kernels is faster at one stream (44.5 against 36.4 to 38.2 tok/s) and needs half the cards; two
nodes with FP8 experts are faster per stream from two streams up (at 8 streams 14.8 against 20.7 to 21.4 tok/s).
Per card at 8 streams (per-stream rate x 8 / cards) the one-node build gives about 14.8 tok/s per card
against 10.4 to 10.7. FP8 experts also make prefill
cheaper (no unpack), but the two builds' TTFT was not measured at the same shape. The two-node build needs a
RoCE-connected pair and is not verified on a release, so the recipe is the one-node build.

## Two nodes, FP8 experts

The MXFP4 experts convert exactly to per-channel FP8 (`ports/kimi-k2-7/convert/mxfp4_to_fp8.py`, checked by
`validate_fp8.py`), which feeds the MME without unpacking. FP8 experts need 63.4 GiB per card at 16 cards, so that
configuration is TP 16 across two nodes with RoCE scale-out links between them. It was measured on an older
vllm-gaudi build with Ray (see the [write-up](../ports/kimi-k2-7.md)) and is not verified on `v0.30.0`, whose image
has no Ray.

## Known issues

- Prefill keeps the stock MXFP4 op: an uncached 16K-token prompt takes about 6.5 s, with the kernels on or off.
- The first request of each new bucket shape compiles a graph (`VLLM_SKIP_WARMUP=true`); expect a slow first long
  prompt after every restart with an empty recipe cache.
- Output with `KIMI_MXK=1` is not bit-identical to the stock op (the kernels round differently; against an fp32
  reference they are closer, 1.7e-3 against 4.8e-3 to 5.5e-3). The gates compare answers and teacher-forced
  scores instead: 95.5% top-1 agreement, NLL within 0.001.
- No vision: `--language-model-only` is required (see What not to do).

## What not to do

- Do not serve without `--enable-expert-parallel`: the load stops with an error naming the expert split.
- Do not point `KIMI_MXK_LIB` at another variant of the kernel build: each variant is compiled for one model's
  shapes, and only the `kimi` build passed the parity check for this model.
- Do not drop `--language-model-only`: image prompts of new sizes each compile a graph, and very large images
  stall the vision prefill past the engine's step timeout.
