# kimi-k2-7: Kimi-K2.7-Code on one 8-card Gaudi 2 node

Kimi-K2.7-Code (1T-parameter MoE: a DeepSeek-V3-style text model with 61 layers, 384 routed experts top 8, MLA
attention, plus a vision tower) served text-only from AMD's MXFP4 release, at 131,072 tokens of context on one
8-card node. vLLM at the `releases/v0.30.0` commit already has the model and the `kimi_k2` parsers, and vllm-gaudi
has a native MXFP4 MoE method with Kimi's routing. The port adds what is missing around them:

| Path | What it does |
| --- | --- |
| `convert/repack_quark_mxfp4.py` | Repacks [amd/Kimi-K2.7-Code-MXFP4](https://huggingface.co/amd/Kimi-K2.7-Code-MXFP4) (one Quark tensor per expert) into vLLM's fused MXFP4 layout. Expert bytes are copied, not re-quantized; FP8 attention and the dense and shared MLPs become BF16. `--verify` checks sampled experts byte for byte. |
| `patches/mxfp4_fused_load.py` | Lets the DeepSeek loader load fused MXFP4 experts under expert parallelism (each rank copies its block of the 384 experts). |
| `kimi_hpu/` + `patches/mxk_hooks.py` | The MXFP4 decode kernel hook, behind `KIMI_MXK`: decode-sized MoE calls run on the `kimi` build of [`kernels/mxfp4-moe`](../../kernels/mxfp4-moe/README.md) instead of the stock op. Off (`KIMI_MXK=0`) is the stock op. |
| `convert/mxfp4_to_fp8.py`, `convert/validate_fp8.py` | The two-node alternative: an exact re-encoding of the MXFP4 experts as per-channel FP8 (compressed-tensors), and an independent checker. See [Two nodes, FP8 experts](#two-nodes-fp8-experts). |
| `install.sh`, `Dockerfile` | The image build: checks the release commits, applies the two patches, installs `kimi_hpu`, builds the `kimi` kernel variant into `/opt/gaudi-cookbook-kimi/mxk`. |
| `tests/` | CPU tests of the FP8 encoding (`pytest ports/kimi-k2-7/tests`); `tests/hpu/run_gates.sh` runs the serving gates against a running server. |

Generic fixes come from [`fixes/`](../../fixes/README.md): the recipe applies the `v0.30.0` profile at launch.

## 1. Convert the checkpoint

On CPUs, once (about 650 GB in, 659 GB out):

```bash
python3 ports/kimi-k2-7/convert/repack_quark_mxfp4.py --src <amd/Kimi-K2.7-Code-MXFP4 snapshot> --dst <MXFP4 dir>
python3 ports/kimi-k2-7/convert/repack_quark_mxfp4.py --src <snapshot> --dst <MXFP4 dir> --verify   # VERIFY_OK
```

The output keeps the release's `LICENSE` (modified MIT, see [Licence](#licence)).

## 2. Build the image

```bash
docker build -f ports/kimi-k2-7/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-kimi-k2-7-image> .
```

`install.sh` prints one line per step and stops at the first one that does not apply. The kernel build ends with
`mxk_tpc kimi-v8-<hash>: ... PASS` and `[mxk-op-gate] PASS`.

## 3. Check the kernels on one card

Before serving with `KIMI_MXK=1`, in the port image with one free card and the converted checkpoint:

```bash
bash /opt/gaudi-cookbook/kernels/mxfp4-moe/tests/run_parity.sh --variant kimi --lib /opt/gaudi-cookbook-kimi/mxk \
  --checkpoint <MXFP4 dir> --out /tmp/parity
```

It must end with `[kimi] PARITY PASS` and `[kimi] SIG_DEVICE PASS`.

## 4. Serve and run the gates

Launch with the recipe's command ([docs/recipes/kimi-k2-7-code.md](../../docs/recipes/kimi-k2-7-code.md)), then:

```bash
BASE=http://127.0.0.1:8000 ports/kimi-k2-7/tests/hpu/run_gates.sh /tmp/gates
```

One PASS or FAIL line per gate: golden answers, tool calls (streaming and not), needles at 32K, 64K and 120K,
a `top_logprobs` soak with CJK output, and teacher-forced scoring. To compare two settings of the same server
build, run once with `KIMI_MXK=0`, restart with `KIMI_MXK=1` and pass the first output directory as the second
argument: golden and teacher-forced results are then compared too.

## The kernel switch

| Variable | Default | Meaning |
| --- | --- | --- |
| `KIMI_MXK` | `0` | `1` runs MoE calls of at most `KIMI_MXK_MAX_TOKENS` tokens (and at most 128 routes) on the kernels |
| `KIMI_MXK_MAX_TOKENS` | `16` | per-call token cap, 1 to 16; covers decode at `--max-num-seqs 8` |
| `KIMI_MXK_LIB` | `/opt/gaudi-cookbook-kimi/mxk` | where `install.sh` put `libmxk_tpc.so` and `libmxk_op_lazy.so` |

With the switch on, every vLLM process registers the kernel library with the graph compiler before it opens a
device (`GC_KERNEL_PATH`, with every stock kernel forwarded), moves the recipe cache to a directory named after the
kernel build, and sets `PT_HPU_LAZY_ACC_PAR_MODE=0`. The engine log shows `[kimi-mxk] installed` and
`[kimi-mxk] enabled: mxk_tpc kimi-v8-<hash>, E_local=48 K=7168 I=2048`. Anything unexpected (libraries missing, a
layer of another shape) keeps the stock op for that layer and logs why. Prefill keeps the stock op.

## Two nodes, FP8 experts

The MXFP4 experts can be re-encoded as per-channel FP8 exactly (each row gets one power-of-two scale; every value
lands on an E4M3 normal of at most 192, inside Gaudi 2's range of 240). FP8 experts run on the MME directly, but
they need 63.4 GiB per card at 16 cards (118 GiB per card at 8), so this is a two-node configuration:

```bash
# one process per layer range (any job runner), plus one for the other tensors, then finalize
python3 ports/kimi-k2-7/convert/mxfp4_to_fp8.py --src <MXFP4 dir> --dst <FP8 dir> --layers 1-6
python3 ports/kimi-k2-7/convert/mxfp4_to_fp8.py --src <MXFP4 dir> --dst <FP8 dir> --rest
python3 ports/kimi-k2-7/convert/mxfp4_to_fp8.py --src <MXFP4 dir> --dst <FP8 dir> --finalize
for part in inventory rest scales sample; do
  python3 ports/kimi-k2-7/convert/validate_fp8.py --src <MXFP4 dir> --dst <FP8 dir> --part $part
done
```

It loads on vllm-gaudi's stock compressed-tensors FP8 MoE method with `--enable-ep-weight-filter`,
`VLLM_SCALE_ADJUSTMENT=false` and the [moe-explicit-overload](../../fixes/moe-explicit-overload/README.md) fix,
at tensor parallel 16 across two nodes with RoCE scale-out links between them. That configuration was measured on
an older vllm-gaudi build with Ray and is not verified in this repository on `v0.30.0`; the
[write-up](../../docs/ports/kimi-k2-7.md) has its numbers and how they compare.

## Licence

Kimi-K2.7-Code and AMD's MXFP4 release are under Moonshot AI's modified MIT licence: MIT, plus one condition. A
commercial product or service that uses the model (or a derivative, such as the converted checkpoints here) and has
more than 100 million monthly active users or more than 20 million US dollars of monthly revenue must display
"Kimi K2.7 Code" prominently in its user interface. The port's code is Apache-2.0.
