# Muse Spark Open (MXFP4)

An 816B-parameter mixture-of-experts model (42B active) with a latent MoE and a norm on each expert's output,
served text only from AMD's published Quark MXFP4 checkpoint at 256K tokens per request on one 8-card Gaudi 2 node.
vllm-gaudi has no path for this model; the [Muse Spark Open port](../ports/muse-spark-open.md) adds one.

**Built with Muse Spark Open.**

| | |
| --- | --- |
| Model | [amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4](https://huggingface.co/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4), loaded as published (no conversion) |
| Hardware | 8x Gaudi 2 (attention TP 8, experts EP2 x TP4) |
| Image | vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1, plus [`ports/muse-spark-open`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/muse-spark-open) |
| Precision | MXFP4 experts (layers 1 to 60) run W4A16; BF16 experts in layers 0 and 61; everything else BF16. 509 GB |
| Context | 262,144 served (1,048,576 native) |
| Concurrency | `--max-num-seqs 4` |
| Parsers | reasoning `muse`, tools `muse` (from [vllm#61879](https://github.com/vllm-project/vllm/pull/61879)) |
| Licence | Muse Model License (Muse Spark Open version): products using the model show "Built with Muse Spark Open" |
| Recipe | [`recipes/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4/recipe.yaml) |

## Build the image

From a checkout of this repository, on a stock vllm-gaudi `releases/v0.30.0` image built as in
[Getting started](../getting-started.md):

```bash
docker build -f ports/muse-spark-open/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-muse-spark-open-image> .
```

`install.sh` checks the image's commits, applies the `v0.30.0` fix profile, installs the vendored vllm#61879 files,
patches the runner, installs the plugin, builds the MXFP4 decode kernels (`kernels/mxfp4-moe`, variant `muse`) and
runs the build gates. See [`ports/muse-spark-open/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/muse-spark-open/README.md).

## Launch

Inside the port image (`$COOKBOOK` is `/opt/gaudi-cookbook` there). Mount persistent storage at `/data` so the
compiled graphs survive restarts.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_RECIPE_CACHE_CONFIG=/data/muse-spark-recipe-cache,False,32768
export VLLM_CONTIGUOUS_PA=false
export VLLM_SKIP_WARMUP=true
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_MOE_CHUNK=256
export VLLM_MOE_TOKEN_BOUNDARY=65536
export VLLM_FP32_SOFTMAX=true
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_HPU_PROMPT_LOGPROBS_ROWS=256
export VLLM_BUCKETING_FROM_FILE=/opt/gaudi-cookbook/ports/muse-spark-open/serving/buckets-256k-seqs4.txt
export VLLM_LONG_PREFILL_SHARE=0.25
export VLLM_LONG_PREFILL_SHARE_ASYNC=1
export VLLM_LONG_PREFILL_MIN_TOKENS=32768
export MUSE_SPARK_MOE_TP_SPLIT=4
export MUSE_SPARK_SWA_KV=1
export MUSE_SPARK_LONG_ATTN=1
export MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS=5408
export MUSE_SPARK_DECODE_LAYER_GROUP=62
export MUSE_SPARK_MXFP4_KERNELS=1
export MUSE_SPARK_MEMLOG=1

mkdir -p "${PT_HPU_RECIPE_CACHE_CONFIG%%,*}"

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 \
    topk-subset-sampler vllm-long-prefill-share || exit 1

vllm serve amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --block-size 128 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --max-cudagraph-capture-size 4 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --language-model-only \
  --reasoning-parser muse \
  --enable-auto-tool-choice \
  --tool-call-parser muse \
  --chat-template-content-format openai \
  --num-gpu-blocks-override 10752 \
  --long-prefill-token-threshold 1024
```

- The `MUSE_SPARK_*` switches are the port's; each is described in the
  [port README](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/muse-spark-open/README.md#switches).
  `MUSE_SPARK_MXFP4_KERNELS=0` runs the stock MXFP4 MoE op instead of the decode kernels (the reference path;
  restart the server to switch).
- `--language-model-only` is required: the port serves text only and refuses to start without it.
- `--num-gpu-blocks-override 10752`: with window-sized KV a block id costs 1 MiB per card (16 global layers keep
  full-length KV), so 10,752 blocks hold 4 x 256K tokens of global-layer context.
- `--max-cudagraph-capture-size 4` turns HPU graphs on; decode graphs are captured per bucket at first use
  (`VLLM_SKIP_WARMUP=true`), and the recipe cache keeps them between restarts. The kernel switch keeps its own
  cache directory next to it.
- `VLLM_HPU_PROMPT_LOGPROBS_ROWS=256` is the [prompt-logprobs](../fixes/prompt-logprobs.md) slice: the 202,048-token
  vocabulary in fp32 over a 4096-token chunk does not fit otherwise.
- `VLLM_LONG_PREFILL_SHARE=0.25` with `--long-prefill-token-threshold 1024` keeps decodes running while a 256K
  prompt reads in ([vllm-long-prefill-share](../fixes/vllm-long-prefill-share.md)).
- The load reads only each card's slices of the checkpoint (70.8 GB). It took 61 minutes from network storage on the
  verification run; `VLLM_ENGINE_READY_TIMEOUT_S=7200` leaves room for it.

### Buckets

The bucket file ships with the port, `ports/muse-spark-open/serving/buckets-256k-seqs4.txt`, loaded with
`VLLM_BUCKETING_FROM_FILE`. Decode buckets are a fixed set (each captured decode graph keeps about 0.5 MiB per block
of its bucket for the life of the process); prompt buckets add exact 1024- and 2048-token queries for the
long-prompt chunks. The file's comments give the arithmetic.

## Required fixes

All from [Fixes](../fixes/index.md). `topk-subset-sampler` and `vllm-long-prefill-share` are opt-in and named in the
launch command; the rest come from `--profile v0.30.0`, which `install.sh` has already applied once (the launch step
then reports them as applied).

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching can read another request's KV in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | every chat turn carries a grammar; rows must keep their own sampling parameters |
| [`so-row-owners`](../fixes/so-row-owners.md) | grammar masks on the right rows when a long prompt is chunked |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`async-inputs`](../fixes/async-inputs.md) | stale decode input ids under async scheduling duplicate tokens |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption repeats its generated tail |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids come back corrupted in lazy mode |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | `prompt_logprobs` (the perplexity gate) runs out of memory over a 4096-token chunk |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with `seed` can segfault a TP worker in lazy mode |
| [`topk-subset-sampler`](../fixes/topk-subset-sampler.md) (opt-in) | top-k / top-p sampling without sorting the whole vocabulary each step |
| [`swa-decode-window`](../fixes/swa-decode-window.md) | the port's window patch for model type `muse_spark` is written on top of it |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |
| [`vllm-long-prefill-share`](../fixes/vllm-long-prefill-share.md) (opt-in) | a 256K prompt otherwise takes whole steps from running decodes |

## Verified

Run on one 8-card Gaudi 2 node from the stock `releases/v0.30.0` image, with `install.sh` run as the Dockerfile
runs it, the launch fix step, and this page's Launch block. The device gates are
`ports/muse-spark-open/tests/hpu/run_gates.sh`; the CPU tests are `pytest ports/muse-spark-open/tests` (33 passed;
the patch chain on a clean release copy needs `GAUDI_FIXES_ROOT`, and passed with it).

With `MUSE_SPARK_MXFP4_KERNELS=0` (the stock MXFP4 op, the port's reference path):

| Gate | Result |
| --- | --- |
| Build gates (`install.sh` step 7, without the kernel check) | 8/8 |
| Perplexity vs the CPU reference, 5 passages, 17,408 tokens (per passage: within 1.5%, top-1 at least 0.97, mean KL at most 0.02) | worst ratio +0.27%, worst top-1 0.980, worst KL 0.0013. Per passage: Sherlock Holmes +0.001%, six.py +0.05%, regex HOWTO +0.09%, Flatland -0.003%, original prose +0.27% |
| Decode vs prefill, 8 x 512 greedy tokens from 2,100 to 2,779-token prompts (gate 0.05) | 0.0058 mean abs, max 2.31 at one token |
| Needles at 2K / 32K / 258K tokens x depths 0.1 / 0.5 / 0.9 (gate: every code found, at least 8/9 in the answer, control clean) | 9/9 found, 8/9 answered, 32K control clean; 258K 3/3 answered, 438 to 463 s each. The one refusal (2K at 0.1, inside the opening of "A Scandal in Bohemia") called the needle inserted text and named the code in its reasoning (2/2 repeats) |
| Tool calls: `auto`, `required`, `auto` streamed | pass: `get_weather {"city": "Paris", "unit": "celsius"}` each time |
| `reasoning_effort` low and high | pass (1,029 and 2,195 reasoning characters) |
| Greedy repeat, 16 prompts x 256 tokens, fresh `cache_salt` | 16/16 identical |

With `MUSE_SPARK_MXFP4_KERNELS=1` (this page's command), the same pod with the server restarted from the final
image:

| Gate | Result |
| --- | --- |
| Perplexity vs the CPU reference, same 5 passages | worst ratio +0.22%, worst top-1 0.983, mean KL 0.0011 |
| Decode vs prefill, 8 x 512 greedy tokens (gate 0.05) | 0.0025 mean abs, max 0.58 at one token |
| Needles at 2K / 32K / 258K tokens x depths 0.1 / 0.5 / 0.9 | 9/9 found, 9/9 answered, 32K control clean |
| Tool calls: `auto`, `required`, `auto` streamed | pass: `get_weather {"city": "Paris", "unit": "celsius"}` each time |
| `reasoning_effort` low and high | pass (1,634 and 2,520 reasoning characters) |
| Greedy repeat, 16 prompts x 256 tokens, fresh `cache_salt` | 16/16 identical |

Greedy against the switch-off run, same 16 prompts (reported, not gated): 4/16 identical. Of the 12 first splits,
11 sit at near-ties of the switch-off output (top-2 gaps 0 to 0.29 nats); one sits at a gap of 1.18 ("Reasons"
vs "Specific" at token 122). The stock op alone moves a logprob by up to 2.31 between decode and prefill (the
decode-vs-prefill gate above), and the kernels' perplexity and KL against the CPU reference are at least as close
as the stock op's. Peak device memory in use was 82.9 GiB of 94.6 GiB on rank 0, 11.8 GiB headroom.

## Performance

Decode, 256 greedy tokens after a 9-token prompt (`run_gates.sh` speed gate; second of two runs at 1 stream,
aggregate over the window where all 4 streams decode):

| Kernel switch | 1 stream | 4 streams, aggregate | Measured as |
| --- | --- | --- | --- |
| `MUSE_SPARK_MXFP4_KERNELS=0` | 22.7 tok/s | 43.3 tok/s | one server, this recipe otherwise |
| `MUSE_SPARK_MXFP4_KERNELS=1` | 40.8 tok/s | 102.8 tok/s | same pod, server restarted with the switch flipped |

Recorded in the [write-up](../ports/muse-spark-open.md#performance) under other conditions: the decode kernels took
sampled 2K-prompt decode from 22.3 to 33.5 tok/s at 1 stream and from 10.9 to 23.3 tok/s per stream at 4 (different
conditions: same deployment and clients, image and flag swapped); decode layer grouping took greedy 1-stream
decode from 34.5 to 39.7 tok/s (same-pod A/B). A needle prompt at the served window takes 438 to 463 s end to end
(kernel switch off, measured here).

## Known issues

- Weight loading is slow: 61 and 46 minutes on the two verification loads, bound by each worker's lazy-mode AccThread
  dispatching tens of thousands of per-expert copies as much as by storage. Keep the recipe cache on persistent
  storage so at least graph compilation is not repeated.
- A needle prompt at the served window takes several minutes end to end (346 to 461 s recorded in the
  [write-up](../ports/muse-spark-open.md#performance)); prefill of a 256K prompt is the cost, not decode.
- Greedy output is repeatable on one server with a fresh `cache_salt` (16/16). Across the kernel switch it need
  not be bit-identical: the kernels and the stock op round differently, so paths split, mostly at near-ties. The
  greedy gate with `--greedy-ref` reports each split and the switch-off top-2 gap there (on the verification run,
  one of 12 splits was at a gap of 1.18).
- A needle inside a famous public text can be refused: the model may call the inserted sentence "not part of the
  original" and keep the answer in its reasoning.
- vLLM logs `Error when parsing AST of Jinja template` once at start: the content-format detection cannot parse the
  chat template. `--chat-template-content-format openai` sets the format explicitly, so it has no effect.
- The checkpoint's vision and audio towers are not served.

## What not to do

- Do not drop `--language-model-only` or try image inputs: the port has no vision path.
- Do not set `MUSE_SPARK_MOE_TP_SPLIT=1` (EP8, whole experts per card): each all-reduce waits for the card whose
  experts drew the most routes, and batch-1 decode lost 33% per stream.
- Do not turn off `MUSE_SPARK_SWA_KV` at 256K: every layer then keeps full-length KV, and the pool holds 3.875x fewer
  tokens than the 4 x 256K this recipe promises.
- Do not raise `MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS` or add long decode buckets without accounting for the about
  0.5 MiB per block each captured decode graph keeps.
- Do not reuse a recipe cache directory across images or configurations.
