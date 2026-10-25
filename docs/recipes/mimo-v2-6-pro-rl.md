# MiMo-V2.6-Pro-RL

Xiaomi's 1.02T mixture-of-experts model (42B active) at 256K tokens per request on one 8-card Gaudi 2 node. It
has the same architecture as [MiMo-V2.6-Flash](mimo-v2-6-flash-rl.md), at a larger size: 60 of its 70 layers attend
to a 128-token window with a learned sink. It runs in the same [MiMo-V2.6 port](../ports/mimo-v2-6-flash.md) image.
The difference from Flash is the experts. Pro keeps them in MXFP4, because FP8 experts would not fit on one node,
and runs decode on the [MXFP4 MoE kernels](../kernels/mxfp4-moe.md).

| | |
| --- | --- |
| Model | [XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) (MIT), revision `73875d0` |
| Hardware | 8x Gaudi 2, one node (TP 8, experts split over the 8 ranks: 48 of 384 per card) |
| Image | the port image, built from vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | bf16 attention (FP8 block weights dequantised at load), MXFP4 routed experts (the checkpoint's bytes), bf16 KV |
| Kernels | MXFP4 MoE decode kernels, `glm` variant with the `torch.compile` op library (`MIMO_MXK=1`) |
| Context | 262,144 per request (the checkpoint allows 1,048,576) |
| Concurrency | `--max-num-seqs 8` |
| Parsers | tools `mimo`, reasoning `mimo` |
| Recipe | [`recipes/XiaomiMiMo/MiMo-V2.6-Pro-RL/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/XiaomiMiMo/MiMo-V2.6-Pro-RL/recipe.yaml) |

## Build the image

The port installs into a stock vllm-gaudi `v0.26.0` image ([Getting started](../getting-started.md)). It is the
same image as for Flash. From a checkout of this repository:

```bash
docker build -f ports/mimo-v2-6-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> \
  -t <your-mimo-v2-6-image> .
```

`install.sh` runs these steps and stops at the first one that does not apply:

1. check the release commits;
2. apply the v0.26.0 fix profile and `vllm-long-prefill-share`;
3. apply the port's runner patches and install the plugin;
4. build the MXFP4 MoE kernels with `kernels/mxfp4-moe/build.sh --variant glm --pt2` into
   `/opt/gaudi-cookbook-mimo/mxk`.

The kernel build needs `tpc-clang`, which is in the Habana base image, and runs its library gate and op gates. See
[`ports/mimo-v2-6-flash/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/mimo-v2-6-flash/README.md).

## Launch

Inside the port image, where `$COOKBOOK` is `/opt/gaudi-cookbook`, with the Hugging Face cache mounted. The
first line stages the checkpoint (symlinks only; downloads it first if it is not in the cache: 566 GB of weights).

```bash
export CHECKPOINT_DIR=/data/mimo-v2.6-pro-rl-hpu
export PT_HPU_LAZY_MODE=0
export VLLM_T_COMPILE_DYNAMIC_SHAPES=false
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=false
export MIMO_EXPERT_DTYPE=mxfp4
export MIMO_MXK=1
export MIMO_SWA_KV_GROUPS=1
export MIMO_GA_TILE=1
export VLLM_LONG_PREFILL_SHARE=0.5
export VLLM_LONG_PREFILL_SHARE_ASYNC=1
export VLLM_LONG_PREFILL_MIN_TOKENS=65536
export VLLM_MOE_CHUNK=256
export VLLM_MOE_TOKEN_BOUNDARY=2048
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

python3 "$COOKBOOK/ports/mimo-v2-6-flash/convert/stage_checkpoint.py" XiaomiMiMo/MiMo-V2.6-Pro-RL "$CHECKPOINT_DIR" --revision 73875d00b30a89ef8cc353a0b60b0e9f9561952d || exit 1

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 vllm-long-prefill-share || exit 1

vllm serve $CHECKPOINT_DIR \
  --served-model-name XiaomiMiMo/MiMo-V2.6-Pro-RL \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --long-prefill-token-threshold 2048 \
  --max-num-batched-tokens 8192 \
  --block-size 128 \
  --gpu-memory-utilization 0.8 \
  --enable-prompt-tokens-details \
  --trust-remote-code \
  --generation-config vllm \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' \
  --chat-template "$COOKBOOK/ports/mimo-v2-6-flash/serving/chat_template.jinja" \
  --reasoning-parser mimo \
  --enable-auto-tool-choice \
  --tool-call-parser mimo
```

- `stage_checkpoint.py` writes `$CHECKPOINT_DIR` with symlinks to the checkpoint and a `config.json` with two
  edits. It removes `quantization_config`, because the port dequantises the FP8 block weights itself and loads the
  MXFP4 experts as raw bytes. It derives `layer_types` from `hybrid_layer_pattern`; without it vLLM gives every
  layer the 128-token window. The model is served under its public name.
- `MIMO_EXPERT_DTYPE=mxfp4` keeps the routed experts in the checkpoint's MXFP4. They take about 67 GiB of weights per
  card. The Flash recipe's FP8 re-encode would double the expert memory, which does not fit on one node.
- `MIMO_MXK=1` runs MoE calls of up to 16 tokens (decode) on the `mxk_w13x` and `mxk_w2x` kernels, which read the
  packed experts directly. Larger calls (prefill) stay on the stock MXFP4 op. The engine log shows
  `[mimo-mxk] enabled: mxk_tpc glm-v8-<hash>, E_local=48 K=6144 I=2048`. Set `MIMO_MXK=0` to roll back to the stock op.
- `MIMO_SWA_KV_GROUPS=1` gives the 60 sliding layers window-sized KV through vLLM's hybrid KV cache manager. Global
  context then costs one layer in seven of the stock layout (10 global layers instead of 70 full-length ones).
  `MIMO_GA_TILE=1` runs the global layers' prefill in query tiles, so a long prompt with context does not need a
  multi-gigabyte attention bias.
- `PT_HPU_LAZY_MODE=0` with static shapes: torch.compile mode. The window-sized KV needs it, and the kernels run
  through their `torch.compile` op library. The dynamic-shape compile path fails.
- `--gpu-memory-utilization 0.8`: vLLM applies the fraction to the memory left after the weights, and the higher
  values failed (see [Known issues](#known-issues)).
- `VLLM_SKIP_WARMUP=false`: the start warms every shape after the load, so that new shapes do not compile while
  requests wait. Point `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,False,65536` at persistent storage to reuse compiled
  graphs across restarts. With `MIMO_MXK=1` the port moves the cache to a sibling directory named after the
  kernel build.
- `--generation-config vllm` with the override keeps the model's sampling defaults (temperature 1.0, top_p 0.95)
  without the checkpoint's 2,048-token `max_new_tokens` cap.
- The chat template is the checkpoint's, with `<think>` opened in the generation prompt. Pro and Flash share it.

### Buckets

No bucket file. Bucketing uses the image defaults; with `--block-size 128` and `--max-num-batched-tokens 8192`
warmup covers prompt chunks up to 8,192 tokens against contexts up to the full window.

## Required fixes

All from [Fixes](../fixes/index.md), the same set as Flash. `install.sh` applies them when the image is built; the
launch command runs the same line again and only confirms them. The opt-in one is named on that line; the rest
come from `--profile v0.26.0`, which also applies the release's other generic fixes.

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching can read another request's KV in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | structured-output steps sample each row with another request's parameters |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-inputs`](../fixes/async-inputs.md) | a decode the scheduler leaves out of a step (behind long-prompt chunks) reads a stale token and repeats it |
| [`async-discard-counter`](../fixes/async-discard-counter.md) | `AttributeError` on the first preemption under async scheduling |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption repeats its earlier output |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids come back corrupted |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`swa-decode-window`](../fixes/swa-decode-window.md) | the exact decode window (one extra block, the oldest block head-masked); the port's runner patch turns it on for `mimo_v2` |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |
| [`vllm-long-prefill-share`](../fixes/vllm-long-prefill-share.md) (opt-in) | a long prompt mid-prefill no longer holds every other stream to a crawl; measured on Flash, not on Pro |

`rope-cpu-cache` is in the profile; the port also rebuilds MiMo's RoPE tables on the CPU in float64 itself (see
the [port write-up](../ports/mimo-v2-6-flash.md#the-rope-fix-for-large-positions)).

## Verified

Run on one 8-card Gaudi 2 node from the stock vllm-gaudi `v0.26.0` image (`b059f9c`, vLLM `568afb3a`), with
`install.sh` run as the Dockerfile runs it and this page's Launch block as written.

- `install.sh`: commit check, the v0.26.0 profile plus `vllm-long-prefill-share`, both runner patches, the plugin
  install, and the kernel build `--variant glm --pt2`. The library gate (3,253 GUIDs: the 3,250 stock ones plus
  `mxk_w13x`, `mxk_w2x`, `mxk_deq`) passed, and the op gate passed on both op libraries.
- `tests/hpu/mxk_ops.py` on one card, in compile mode with the `torch.compile` op library, on layers 1 and 69 for
  TP8 ranks 0 and 7: 28 of 28 decode cases (1 to 16 tokens) within the parity bound. The kernels' worst relative
  error against an fp32 CPU reference was 1.71e-3, against 2.3e-3 to 4.8e-3 for the stock op. Compiled output equals
  eager output bit for bit.
- `kernels/mxfp4-moe/tests/run_parity.sh --variant glm --e-local 48 --inter 2048` (lazy-mode op library, random
  weights of Pro's per-rank shapes): `PARITY PASS`, `PREFILL PASS`, `SIG_DEVICE PASS`.
- `tests/hpu/device_ops.py`: sliding-window attention with sinks within bound in bf16 at Pro's head layout (16 query
  heads, 1 KV head per rank; worst relative error 0.0036).
- `stage_checkpoint.py` on the public checkpoint (revision `73875d0`): 130 shards, 60 sliding and 10 global layers.

Serving gates, `PROFILE=pro tests/hpu/run_gates.sh` against this page's command:

| Gate | Result |
| --- | --- |
| Decode, 256 greedy tokens (floor 12 tok/s per stream; the stock MoE op stays under it at 4 and 8 streams) | 40.3 / 27.5 / 15.1 tok/s per stream at 1 / 4 / 8 streams |
| Tool calls: one call with arguments, choosing between tools, no call, using a result, `tool_choice: "required"`, a call after a 32.9K-token document | 6/6 |
| Needles at 240K tokens x depths 0.1 / 0.5 / 0.9 (greedy) | 3/3 |
| Four 240K-token prompts at once | 4/4, 899 s |
| Needles at 128K tokens x depths 0.1 / 0.5 / 0.9 (greedy) | 2/3. At depth 0.9 the model called the line about the access code inserted text and declined to repeat it (see Known issues) |
| Window-sized KV against the stock layout | not completed: the test pod was preempted while the stock-layout server was loading. The same comparison on [MiMo-V2.6-Flash](mimo-v2-6-flash-rl.md#verified), with the same port code, was token-identical on single requests |

On a test deployment of the same model code and command line (installed from its own copies, kernel library
built from the same `glm` kernel code), the same server measured with the stock MoE op and then with the kernels:

| Gate | Stock MoE op | With the kernels |
| --- | --- | --- |
| Prompt NLL, 255-token text (fp32 CPU oracle 0.6708) | 0.6797 (perplexity 1.973) | 0.6797 (perplexity 1.973) |
| Decode vs teacher-forced scoring, 200 greedy tokens, mean abs difference | 0.035 nats | 0.031 nats |
| GSM8K 5-shot, flexible extract, 200 samples | not run | 0.925 |

## Performance

Per MoE call on one card (`mxk_ops.py`, 48 experts of layer 1 or 69, compiled, warm): 53 to 56 us with the kernels
against 353 to 357 us for the stock op at 1 token, and 247 to 248 us against 2,612 to 2,613 us at 8 tokens.

End to end, from the decode gate above (one run, warm server, 256 greedy tokens, TP 8):

| Streams | Tokens/s per stream | Aggregate |
| --- | --- | --- |
| 1 | 40.3 | 40 |
| 4 | 27.5 | 110 |
| 8 | 15.1 | 121 |

On the test deployment, the stock MoE op ran 18.1 / 8.8 / 6.2 tok/s per stream at 1 / 4 / 8 streams, against 39.7 /
23.5 / 16.2 with the kernels (same server, same node).

- The KV pool held about 2.0M tokens, 7.6 requests at the full 262,144-token window.
- A needle request at about 240K tokens took 3.8 to 6.4 minutes from submission to answer, most of it prefill.
- Starting from storage that was not in the page cache took about 35 minutes to load and 45 minutes to warm up on
  the test deployment.

## Known issues

- Pro sometimes reads an inserted fact as prompt injection and declines to repeat it, even when the fact is phrased
  plainly ("For the record, the access code ... is ..."). In the gates above, the needle at 128K and depth 0.9 was
  refused this way; on the test deployment, one of three needles at 240K on the stock-op server. The needle gates
  count a refusal as a miss.
- In development runs of the port (earlier code, before window-sized KV), `--gpu-memory-utilization 0.9` ran out of
  memory at a 1.5K-token prefill, and 0.85 failed on a prefix-cached prefill with "synLaunch Sections' validation
  failure". 0.8 is the value the test deployment ran.
- Compile mode computes a new batch shape slightly differently the first time it runs. Compare outputs only on
  warm runs.
- The kernels sum in a different order than the stock op, so greedy outputs with `MIMO_MXK=1` and `MIMO_MXK=0` can
  part ways at a near-tie. Compare quality (perplexity, needles, GSM8K), not tokens.
- Text only. The checkpoint's vision tower, audio encoder, MTP layers and draft model are not loaded.

## What not to do

- Do not set `MIMO_EXPERT_DTYPE=fp8` for Pro: the experts would need twice the memory and do not fit on one node.
- Do not use another kernel variant, and do not turn `MIMO_MXK=1` on in lazy mode. The port only loads a `glm`
  build with its `torch.compile` op library, and otherwise keeps the stock op.
- Do not run window-sized KV at a TP other than 8: the global and sliding layers then hold different KV heads per
  rank, and the port refuses to start.
- Do not turn prefix caching off without `VLLM_CONTIGUOUS_PA=false`: the contiguous KV layout bypasses the
  per-group block tables, and the port refuses to start.
- Do not add speculative decoding, or switch to lazy mode with HPU graphs, with window-sized KV on: the per-layer
  metadata swap needs regionally compiled layers, and the port refuses both.
- Do not drop `MIMO_GA_TILE=1`. Without it, the first long prefill with context builds an attention bias that
  warmup never profiled: 4.3 GB of bf16 for an 8,192-token chunk against 256K of context.
