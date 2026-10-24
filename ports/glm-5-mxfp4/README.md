# GLM-5.x MXFP4 port (GLM-5.2 and GLM-5.3)

Serves GLM-5.2 and GLM-5.3 (744B MoE, DeepSeek sparse attention, one MTP layer) with MXFP4 routed experts on one
8-card Gaudi 2 node at 128K context, on a stock vllm-gaudi `releases/v0.30.0` image (`058f107`, vLLM `ced6857a`,
Gaudi software 1.24.1, lazy mode). The design and its history are in
[docs/ports/glm-5-mxfp4.md](../../docs/ports/glm-5-mxfp4.md); the recipes are
[GLM-5.3](../../docs/recipes/glm-5-3-mxfp4.md) and [GLM-5.2](../../docs/recipes/glm-5-2-mxfp4.md).

## Contents

| Path | What |
| --- | --- |
| `install.sh`, `Dockerfile` | the build: commit check, cookbook fixes, `glm5_hpu`, port patches, kernels, final checks |
| `glm5_hpu/` | Python package the patches call into: `dsa.py` (top-k sparse attention), `mxfp4_moe.py` (kernel hooks), `moe_layout.py` (EP x TP layout and placement), `decode_lean.py` (decode levers) |
| `patches/` | 14 source patches with the fixes/ contract, applied in order by `patches/apply_all.py` |
| `convert/` | `glm53_bf16_to_mxfp4.py` (encode GLM-5.3 from BF16), `glm52_quark_to_mxfp4.py` (repack AMD's GLM-5.2), the encoder (`mxfp4.py`) and a `--verify` mode that compares against an existing checkpoint |
| `serving/` | `tp8-seqs4-128k.txt` (bucket file of both recipes), `glm53-ep4-placement.json` (GLM-5.3 expert placement) |
| `tests/` | CPU tests (encoder, DSA core, decode levers, expert layout, patches on real sources) |
| `tests/hpu/` | serving gates: `run_gates.sh` prints one PASS or FAIL line per gate |

The kernels come from [`kernels/mxfp4-moe`](../../kernels/mxfp4-moe/README.md): `install.sh` runs
`build.sh --variant glm /opt/gaudi-cookbook-glm5/mxk` (decode kernels `mxk_w13x`, `mxk_w2x` and the prefill unpack
`mxk_deq`).

### Patches

Ten of the patches are generic (any DeepSeek-style MXFP4 model, or any speculative decoding on vllm-gaudi) and are
candidates for [`fixes/`](../../fixes/README.md); they live here until they are moved. The other four exist for this
model.

| Patch | Generic | What |
| --- | --- | --- |
| `mxfp4_fused_load` | yes | load fused MXFP4 experts of non-gpt-oss models under expert parallelism |
| `fsdpa_softmax_mode` | yes | `VLLM_HPU_FSDPA_SOFTMAX_MODE` for the prefill FusedSDPA (`fast` is badly wrong on deep layers) |
| `mtp_tuple` | yes | unpack DeepSeekMTP's tuple output in the HPU draft proposer |
| `spec_draft_prefill` | yes | MTP drafting with chunked prefill: no crash on partial chunks, draft KV for every chunk |
| `spec_verify_blocks` | yes | per-row block lists for verify rows, padded slots to the pad block |
| `mtp_residual_shape` | yes | the MTP layer's final residual add no longer mixes rows on HPU decode |
| `spec_sampling` | yes | rejection sampling at temperature above 0; greedy rows in mixed batches |
| `spec_logprobs`, `spec_penalties`, `spec_structured_output` | yes | logprobs, penalties and grammars under speculative decoding |
| `glm_dsa` | no | dense MLA for the DSA layers plus the hooks of `glm5_hpu.dsa` |
| `mxk_hooks` | no | the hooks of `glm5_hpu.mxfp4_moe` |
| `moe_layout` | no | the hooks of `glm5_hpu.moe_layout` |
| `decode_lean` | no | the hooks of `glm5_hpu.decode_lean` |

## Switches

Every switch is off by default; off runs the stock vllm-gaudi path. The recipes turn all of them on.

| Variable | Effect | Greedy output vs off |
| --- | --- | --- |
| `VLLM_HPU_FSDPA_SOFTMAX_MODE=None` | accurate FusedSDPA softmax in prefill | different (fixes long prompts) |
| `GLM_DSA_SPARSE=1` | top-2048 sparse attention above 32K keys | identical below 32K keys |
| `GLM_MXK=1` | MXFP4 decode kernels (calls of at most 16 tokens) | close, not identical (fp32 accumulation) |
| `GLM_MXK_PREFILL=1` | prefill unpack kernel (calls of 256 to 4096 tokens) | identical |
| `GLM_MOE_TP=2` | EP 4 x TP 2 expert layout | close, not identical (bf16 rounding of the split down projection) |
| `GLM_MOE_PLACEMENT=<json>` | per-layer expert placement (GLM-5.3 file in `serving/`) | identical routing; layout as above |
| `GLM_ROUTER_FAST=1`, `GLM_EARLY_DRAFT=1`, `GLM_SPEC_DEVICE_REJECT=1` | decode levers (host work off the step) | identical |

## 1. Convert the checkpoint

CPU only; the scripts need Python, torch and safetensors.

GLM-5.2 (MIT): repack [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4). The routed experts are copied
byte for byte; the MTP layer's experts, which AMD left in BF16, are quantized with the encoder the served checkpoint
used (`quantize_mxfp4_floor`, see `convert/mxfp4.py`).

```bash
python3 ports/glm-5-mxfp4/convert/glm52_quark_to_mxfp4.py --src AMD_DIR --dst OUT_DIR --layers 3-78
python3 ports/glm-5-mxfp4/convert/glm52_quark_to_mxfp4.py --src AMD_DIR --dst OUT_DIR --rest
python3 ports/glm-5-mxfp4/convert/glm52_quark_to_mxfp4.py --src AMD_DIR --dst OUT_DIR --finalize
```

GLM-5.3 (GLM-5.3 License): encode [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), not the FP8
repo. The encode is CPU-bound; run the layer ranges as parallel workers, then `--rest` and `--finalize`:

```bash
for r in 3-12 13-21 22-30 31-39 40-48 49-57 58-66 67-78; do
  python3 ports/glm-5-mxfp4/convert/glm53_bf16_to_mxfp4.py --src BF16_DIR --dst OUT_DIR --layers $r --threads 16 &
done
python3 ports/glm-5-mxfp4/convert/glm53_bf16_to_mxfp4.py --src BF16_DIR --dst OUT_DIR --rest
wait
python3 ports/glm-5-mxfp4/convert/glm53_bf16_to_mxfp4.py --src BF16_DIR --dst OUT_DIR --finalize
```

The GLM-5.3 License allows using, modifying and distributing the weights and derivative works, provided the
copyright and permission notice is included (`--finalize` copies the release's `LICENSE` next to the weights).
Operators of a model-as-a-service business above 10 billion US dollars of yearly revenue must pass Z.AI's security
review before commercial use. Read the licence itself before you redistribute.

`--verify REF_DIR` in place of `--dst` writes nothing and compares every tensor a worker would write with an existing
checkpoint, byte for byte.

## 2. Build the image

```bash
docker build -f ports/glm-5-mxfp4/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-glm-5-mxfp4-image> .
```

`install.sh` can also run in a container started from the stock image: `bash ports/glm-5-mxfp4/install.sh`.

## 3. Serve and verify

Launch with the recipe's command (the Launch block of the recipe page), then:

```bash
BASE=http://127.0.0.1:8000 ports/glm-5-mxfp4/tests/hpu/run_gates.sh OUT_DIR [REF_DIR]
```

CPU tests: `pytest ports/glm-5-mxfp4/tests` (they skip without torch); with
`GAUDI_FIXES_ROOT=<a releases/v0.30.0 site-packages copy>` the patch test applies the whole install order twice.

## Limits

- 8 cards, TP 8 with `--enable-expert-parallel` (required: the fused experts are sliced by expert group).
- 128K served context; `--max-num-seqs 4`. The DSA indexer buffers are allocated after vLLM's memory profiling,
  so keep `--gpu-memory-utilization 0.86`.
- `GLM_DSA_SPARSE=1` needs KV defragmentation off (the default when prefix caching is on) and refuses to start
  otherwise.
- Async scheduling is off under speculative decoding.
