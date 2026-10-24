# GLM-5.3-Flash port

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total, 18B active, MIT) on Gaudi 2, text
and images, for the vllm-gaudi `v0.26.0` release (`b059f9c`, vLLM `568afb3a`). The release has no GLM-5.3-Flash
model and its `transformers` 5.14.1 has no `glm5_next` config, so this port supplies both and runs the published
block-FP8 checkpoint, unconverted, on one 8-card node at 256K tokens per request. Design and evidence:
[docs/ports/glm-5-3-flash.md](../../docs/ports/glm-5-3-flash.md). Recipe:
[recipes/zai-org/GLM-5.3-Flash](../../recipes/zai-org/GLM-5.3-Flash/recipe.yaml).

## Contents

| Path | What it is |
| --- | --- |
| `install.sh` | The only build step: checks the release commits, applies the four patches, builds the mHC Sinkhorn kernel, installs the plugin, re-checks. Fails closed. |
| `Dockerfile` | `FROM ${BASE}`, copies this repository to `/opt/gaudi-cookbook`, runs `install.sh`. |
| `glm5_next_hpu/` | The vLLM general plugin (entry point `10_glm5_next_hpu`). |
| `glm5_next_hpu/glm5_next_hpu/config.py` | `glm5_next` config classes (registered with `AutoConfig`), defaults as published. |
| `glm5_next_hpu/glm5_next_hpu/hf/` | The image and video processors and the processor from Hugging Face transformers `main` (vendored, Apache-2.0, headers kept). |
| `glm5_next_hpu/glm5_next_hpu/kda.py`, `kda_layer.py` | Kimi Delta Attention in PyTorch: chunked prefill (loop or scan), recurrent decode, GLM's bounded-sigmoid gate, and the layer on top of vLLM's Kimi KDA layer. |
| `glm5_next_hpu/glm5_next_hpu/model.py` | The model: NoPE MLA attention (DSA indexer skipped, see below), mHC streams, vLLM's DeepSeek-V2 MoE, the hybrid-state plumbing. |
| `glm5_next_hpu/glm5_next_hpu/vision.py`, `multimodal.py` | The vision tower (SDPA, eager) and vLLM's multimodal processor for images. |
| `glm5_next_hpu/glm5_next_hpu/mhc.py` | The switch for the Sinkhorn kernel: registers the library and rebinds vLLM's `mhc_pre_torch`. |
| `patches/` | Four source patches (fix contract: anchors, `--check`, `--root`, exit 2 on drift, idempotent). |
| `kernels/` | The mHC Sinkhorn TPC kernel (`glm_mhc_sinkhorn/`) and `build.sh`, which builds it with `kernels/mxfp4-moe/build.sh`. |
| `serving/chat_template.jinja` | The checkpoint's chat template with image placeholders (one change, noted in its header). |
| `tests/` | CPU tests (pytest): patches (stdlib only), KDA kernels and the mHC wrapper (torch; the wrapper also needs vLLM), config. |
| `tests/hpu/` | Device gates: `run_gates.sh` against a running server, and the Sinkhorn kernel parity gate on a free card. |

### Patches

| Patch | Target | Why |
| --- | --- | --- |
| `nope_mla.py` | `vllm_gaudi/attention/backends/hpu_attn.py` | With `qk_rope_head_dim` 0, `k_pe.view(-1, 1, 0)` cannot infer `-1` from zero elements; the leading dimension is passed explicitly. |
| `mla_model_type.py` | `vllm/transformers_utils/model_arch_config_convertor.py` | Adds `glm5_next_text` to the MLA model types, so vLLM sizes the KV cache for MLA. |
| `cache_group_idx.py` | `vllm_gaudi/v1/worker/hpu_model_runner.py` | The runner hands KV cache group indices to linear-attention layers named `.mixer` or `.linear_attn`; GLM's KDA layers are `.self_attn`, shared with its MLA layers. |
| `mhc_kernel_env.py` | `vllm_gaudi/__init__.py` | Registers the Sinkhorn kernel library (`GC_KERNEL_PATH`) at package import, before any process opens its device. Does nothing unless `GLM5_NEXT_MHC_SINKHORN=kernel`. |

All four apply on top of the v0.26.0 fix profile and `glm47-streaming-text-markers`, before or after them.

## Build

```bash
docker build -f ports/glm-5-3-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> -t <glm-5-3-flash-image> .
```

The base must be built as in [Getting started](../../docs/getting-started.md) from the `v0.26.0` tag (`b059f9c`), so
that vllm-gaudi and vLLM are git checkouts: `install.sh` reads both commits from them and stops on any other. The
kernel build needs the `none` variant of `kernels/mxfp4-moe/build.sh` (no MXFP4 kernels, only the extra kernel).

## Serve

The recipe's launch command runs inside the port image with `COOKBOOK=/opt/gaudi-cookbook`: export its environment,
run the v0.26.0 fix profile plus `glm47-streaming-text-markers`, then `vllm serve zai-org/GLM-5.3-Flash ...`. See
[docs/recipes/glm-5-3-flash.md](../../docs/recipes/glm-5-3-flash.md).

The model refuses to start with prefix caching: the KDA layer implements only the non-caching state contract.

## Switches

Read from the environment at startup. Off is the reference path; the recipe turns all of them on.

| Variable | On | Off (default) |
| --- | --- | --- |
| `GLM5_NEXT_VISION=1` | Builds the vision tower and admits images (up to `--limit-mm-per-prompt`). | Text only; vision tensors are skipped at load. |
| `GLM5_NEXT_IMAGE_MAX_TOKENS` | Tokens per image after merging (default 2048; the checkpoint allows 8000). Larger images are resized down. | |
| `GLM5_NEXT_KDA_SCAN=1` | The KDA state entering each 64-token chunk comes from a log-depth scan over the chunks (batched matmuls). | A loop over chunks, re-recorded on the host for every prompt chunk. |
| `GLM5_NEXT_DECODE_LAYER_GROUP=45` | Decode marks one lazy-mode step after this many layers (45 = once per forward). | `1`: a step after every layer. |
| `GLM5_NEXT_MHC_SINKHORN=kernel` | The 20-iteration 4 x 4 Sinkhorn of `mhc_pre` runs as one TPC kernel, for calls of at most 1,024 tokens. Larger calls (prompt chunks) and non-bf16 inputs use the stock op. | `torch`: vLLM's `mhc_pre_torch` unchanged. |
| `GLM5_NEXT_MHC_LIB` | Directory of the kernel build (default `ports/glm-5-3-flash/kernels/dist`, written by `install.sh`). If the libraries are missing the stock loop runs and one warning says so. | |

The kernel reproduces the device's reduction order and divide, so with it on the results are bit-identical to the
stock loop (gate `sinkhorn`). The scan and the decode grouping change summation order only; their gates are in the
write-up.

## Verify

Against a server started with the recipe's command (Pillow is needed for the image gate; it is in the stock image):

```bash
bash ports/glm-5-3-flash/tests/hpu/run_gates.sh                     # all six gates, about 8 minutes
bash ports/glm-5-3-flash/tests/hpu/run_gates.sh golden tools        # a subset
```

The Sinkhorn kernel gate takes one free card and no server:

```bash
bash ports/glm-5-3-flash/tests/hpu/run_gates.sh sinkhorn
```

On CPU, without a device:

```bash
pip install pytest   # not in the stock image
pytest ports/glm-5-3-flash/tests
GAUDI_FIXES_ROOT=<directory holding v0.26.0 vllm/ and vllm_gaudi/> pytest ports/glm-5-3-flash/tests/test_patches.py
GLM5_NEXT_CHECKPOINT=<checkpoint dir> pytest ports/glm-5-3-flash/tests/test_config.py
```

## Fixes

The generic bugs this model hits are fixes in [`fixes/`](../../fixes/README.md), applied by the recipe at container
start: the v0.26.0 profile (sampling and structured-output rows, async inputs, hybrid decode padding, logprobs,
seeded sampling, the FP8 MoE overload, image merging, rank exceptions, the xgrammar end-of-turn stop and the rest)
plus the opt-in `glm47-streaming-text-markers`. The patches in `patches/` exist only for this model.

## Differences from upstream vLLM

Upstream vLLM added GLM-5.3-Flash for NVIDIA and AMD GPUs after this port was written
([vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906)). None of that code runs on Gaudi, and
none of it is vendored here:

| Piece | Upstream (GPU) | This port (Gaudi 2) |
| --- | --- | --- |
| KDA | Triton chunked and recurrent kernels | PyTorch: block-anchored chunk form, blocked `(I - A)^-1`, optional scan; GLM's bounded-sigmoid gate |
| DSA | sparse MLA with the indexer selecting key pools (FlashInfer, aiter) | dense MLA, indexer skipped: the same function up to 2,048 prompt tokens, an approximation beyond (needle gates to the served window) |
| mHC | vLLM's CUDA, TileLang and Triton mHC kernels | vLLM's PyTorch mHC ops, with the Sinkhorn loop optionally on one TPC kernel |
| MTP draft head | supported | not loaded |
| Video | supported | not served (images only) |
| Config and processors | in vLLM's `transformers_utils` | vendored config; processors from transformers |
