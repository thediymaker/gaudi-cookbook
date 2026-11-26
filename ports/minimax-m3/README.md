# MiniMax-M3 port

Serves [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) (427B total, 23B active, 128 routed experts top-4
plus one shared expert, SwiGLU-OAI activation) text-only on one 8-card Gaudi 2 node, from
[amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) re-encoded to per-channel FP8 experts.

vllm-gaudi v0.26.0 already has the HPU MiniMax-M3 model, the expert-parallel MoE and the `minimax_m3_py` tool
parser. What it cannot do is run the AMD checkpoint: `quark` is not a supported quantization on HPU, and its
compressed-tensors FP8 MoE op has no SwiGLU-OAI activation. The port closes that gap with a conversion and two
source patches, plus an off-by-default KV cache guard; there is no plugin package. Write-up: [docs/ports/minimax-m3.md](../../docs/ports/minimax-m3.md).
Recipe: [recipes/amd/MiniMax-M3-MXFP4](../../recipes/amd/MiniMax-M3-MXFP4/recipe.yaml).

## Contents

| Path | What it does |
| --- | --- |
| `convert/repack.py` | MXFP4 to per-channel FP8 for the routed experts, shared experts to BF16, everything else copied. Exact: every chunk is re-decoded in float64 and compared, or the run stops |
| `convert/fp8_codec.py` | The re-encoding numerics (pure Python reference and a torch twin) and a raw safetensors reader and writer |
| `convert/validate.py` | Independent check of a converted checkpoint against its source, with its own reader and decoders |
| `patches/swigluoai_fp8_moe.py` | Runs SwiGLU-OAI experts of `HPUCompressedTensorsW8A8Fp8MoEMethod` on Habana's `bias_fp8_fused_weights` kernel (`VLLM_M3_FP8_MOE=native`) or on a bf16 reference path (`dequant`) |
| `patches/reasoning_prompt_scope.py` | Fixes vLLM's `minimax_m3` reasoning parser, which streams the reasoning as content because the prompt quotes the thinking markers |
| `patches/kv_finite_writes.py` | `VLLM_M3_KV_FINITE=1` replaces non-finite K/V values with 0 before they are written to the KV cache, so one bad write cannot reach other requests through masked reads. Off by default |
| `install.sh`, `Dockerfile` | Check the image commits, apply the patches, re-check them |
| `tests/` | CPU tests (pytest): re-encoding exactness against an independent oracle, the quantization config under vLLM's matching rules, both patches against v0.26.0 sources (`GAUDI_FIXES_ROOT`), the w13 interleave |
| `tests/hpu/run_gates.sh` | Serving gates: golden prompts, greedy loops, needles to 125K, logprobs, streamed reasoning and tool calls, speed |

## 1. Convert the checkpoint

Download `amd/MiniMax-M3-MXFP4` (about 284 GiB) and run the converter in any Python with torch (CPU is enough). The
workers are independent and can run in parallel, for example as jobs sharing one output directory:

```bash
SRC=/models/MiniMax-M3-MXFP4 DST=/models/MiniMax-M3-FP8
python3 ports/minimax-m3/convert/repack.py --src $SRC --self-test            # synthetic and real rows
for L in 3-8 9-14 15-20 21-26 27-32 33-38 39-44 45-50 51-55 56-59; do
  python3 ports/minimax-m3/convert/repack.py --src $SRC --dst $DST --layers $L --threads 16
done
python3 ports/minimax-m3/convert/repack.py --src $SRC --dst $DST --rest
python3 ports/minimax-m3/convert/repack.py --src $SRC --dst $DST --finalize
python3 ports/minimax-m3/convert/validate.py --src $SRC --dst $DST --part inventory
python3 ports/minimax-m3/convert/validate.py --src $SRC --dst $DST --part rest
python3 ports/minimax-m3/convert/validate.py --src $SRC --dst $DST --part scales --layers 3-59
python3 ports/minimax-m3/convert/validate.py --src $SRC --dst $DST --part sample --layers 3-59
```

The output is 411 GiB. One layer range of six layers takes about 20 minutes on 16 cores.

## 2. Build the image

From the repository root, on a stock vllm-gaudi v0.26.0 image built as in
[Getting started](../../docs/getting-started.md):

```bash
docker build -f ports/minimax-m3/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> -t <your-minimax-m3-image> .
```

`install.sh` stops if the image is not vllm-gaudi `b059f9c` with vLLM `568afb3a`, or if a patch anchor does not
match.

## 3. Serve and verify

Launch with the recipe's command ([docs/recipes/minimax-m3.md](../../docs/recipes/minimax-m3.md)): it applies the
cookbook's v0.26.0 fix profile, then serves `$CHECKPOINT_DIR` under the name `amd/MiniMax-M3-MXFP4`. Then, inside the
container:

```bash
SERVER_LOG=/path/to/server.log ports/minimax-m3/tests/hpu/run_gates.sh http://127.0.0.1:8000 amd/MiniMax-M3-MXFP4
```

## Switches

| Variable | Values | Meaning |
| --- | --- | --- |
| `VLLM_M3_KV_FINITE` | `0` (default, recipe), `1` | `1`: every K/V write replaces NaN and infinity with 0 (bf16 KV cache). A guard, not a fix for the source of a bad value; see below |
| `VLLM_M3_FP8_MOE` | `native` (recipe), `dequant` | `native`: Habana's fused FP8 kernel with SwiGLU-OAI inside, FP8 activations per token. `dequant`: the reference path, experts decoded to bf16 and run in PyTorch with bf16 activations (numerically the MXFP4 checkpoint in bf16; several times slower at decode) |

The patch reads vllm-gaudi's existing `VLLM_MINIMAX_M3_MOE_DECODE_GATHER`, `VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS` and
`VLLM_MINIMAX_M3_MOE_TOKEN_TILE` in `dequant` mode and the stock `VLLM_MOE_CHUNK` and `VLLM_MOE_TOKEN_BOUNDARY` hints
in `native` mode.

## Non-finite KV guard

Attention on HPU reads more than a request's own keys: padded rows and padded block lists point at a shared
padding block, and masked positions are removed by adding a large negative bias. A NaN or infinity written to the
KV cache therefore does not stay in the request that wrote it (NaN plus any bias is NaN). A MiniMax-M3 server
running a block-sparse attention path, which reads other sequences' blocks under a mask (not part of this port),
went from one such write to every request returning token 0 until it was restarted. The origin of the first
non-finite value was not found. With `VLLM_M3_KV_FINITE=1` every K/V write keeps finite values only, at the cost
of one `torch.where` per layer and step. It was not needed and not observed with this port's dense attention, and
the recipe's results were measured with it off; turn it on for long-running servers if you see a server that
answers every request with the same token.

## Licence

The port is Apache-2.0. The model weights are under the MiniMax Community License: a non-commercial grant;
commercial use requires displaying "Built with MiniMax M3" and a notice to (or, above a revenue threshold, prior
authorization from) MiniMax. The license also lists prohibited uses. Read it before serving the model.
