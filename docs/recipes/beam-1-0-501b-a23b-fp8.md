# Beam-1.0-501B-A23B-FP8

Reflection's 501B mixture-of-experts reasoning and coding model (23B active; 52 layers, 256 experts,
top 8 with group-limited routing; 43 sliding-window and 9 full-attention layers), from the official
compressed-tensors FP8 checkpoint as published, at 256K context on one 8-card Gaudi 2 node. vllm-gaudi
has no Beam model, so this recipe runs in the image of the [beam-1-0 port](../ports/beam-1-0.md),
built from the stock `releases/v0.30.0` image.

| | |
| --- | --- |
| Model | [reflection/Beam-1.0-501B-A23B-FP8](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8) (Apache-2.0) |
| Hardware | 8x Gaudi 2 (TP 8 attention, `--enable-expert-parallel`: 32 experts per card) |
| Image | the beam-1-0 port image on vllm-gaudi `releases/v0.30.0` at `058f107`, vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | FP8 checkpoint (503 GB): per-channel FP8 weights, dynamic per-token FP8 activations |
| Context | 262,144 served (1,048,576 native); KV pool 776,448 tokens per replica |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools and reasoning `beam` (from the port) |
| Recipe | [`recipes/reflection/Beam-1.0-501B-A23B-FP8/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/reflection/Beam-1.0-501B-A23B-FP8/recipe.yaml) |

## Build the image

From a checkout of this repository, on top of your stock `releases/v0.30.0` image (built as in
[Getting started](../getting-started.md)):

```bash
docker build -f ports/beam-1-0/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-beam-1-0-image> .
```

The build copies the repository to `/opt/gaudi-cookbook` (so `$COOKBOOK` is set there), checks the
image commits, applies the `v0.30.0` fix profile, installs the vendored vLLM PR #62052 parser files,
patches the runner (exact sliding windows for Beam, grammar bitmask on the device), installs the
`beam_hpu` plugin and runs the build gates. Details: [`ports/beam-1-0/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/beam-1-0/README.md).

## Launch

In the port image. `$COOKBOOK` is `/opt/gaudi-cookbook` there.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_CONTIGUOUS_PA=false
export VLLM_SKIP_WARMUP=true
export VLLM_FP32_SOFTMAX=true
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_BUCKETING_FROM_FILE=/opt/gaudi-cookbook/ports/beam-1-0/serving/tp8-seqs16-256k.txt
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT=1
export VLLM_BEAM_STREAM_TOOL_ARGS=1
export BEAM_EXACT_ROUNDING=0
export BEAM_PREFILL_INFLIGHT=1
export BEAM_FSDPA_WS_GIB=2
export BEAM_FSDPA_Q_TILE=256
export BEAM_GRAMMAR_DEVICE_MASK=1

# Fixes: the v0.30.0 profile, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 || exit 1

vllm serve reflection/Beam-1.0-501B-A23B-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --load-format beam \
  --dtype bfloat16 \
  --block-size 128 \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --max-cudagraph-capture-size 4 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --reasoning-parser beam \
  --enable-auto-tool-choice \
  --tool-call-parser beam
```

- `--load-format beam` is the port's loader: it reads only each rank's expert rows from the
  published files, in parallel, and copies each expert stack to the device once. The model state is
  identical to the stock loader's. A start is bound by storage reads (about 6 minutes at 1.4 GB/s for
  503 GB); the stock loader took 36 minutes on the same node.
- `--enable-expert-parallel`: each card's FP8 MoE call takes 32 whole experts instead of 256 slices.
  Decode is bound by host graph launches, and the smaller call is a cheaper launch (+23% per stream).
- `--max-cudagraph-capture-size 4` turns HPU graphs on. Eager decode runs at 0.68 tokens/s; the
  decode graphs are required.
- `--enable-prefix-caching` and `--async-scheduling` cost nothing measurable here and are the mode
  every check below ran in. Prefix caching saves the whole prefill of a repeated conversation prefix.
- The fix step finds every fix already applied (the port's `install.sh` applies the same profile
  before its own runner patches), and stops if the image is not the port image.
- `VLLM_GRAPH_RESERVED_MEM=0.4` leaves 12.84 GiB per card for graphs; the KV pool gets the rest of
  `--gpu-memory-utilization 0.90`.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`: a 250K-token prompt prefills next to running decodes,
  for 95 to 110 seconds.
- The `BEAM_*` and `VLLM_BEAM_*` switches are the port's; each is described in the port README. The
  recipe sets every one explicitly.

### Buckets

`VLLM_BUCKETING_FROM_FILE` loads `ports/beam-1-0/serving/tp8-seqs16-256k.txt` from the port image:
vllm-gaudi's own prompt buckets for 4,096-token chunks plus a top point at 256K, and coarse decode
buckets (batch 1, 2, 4, 8 and 16, four or five block sizes each), so the number of captured decode
graphs, each of which holds device memory, stays bounded.

## Required fixes

All from [Fixes](../fixes/index.md), all in the `v0.30.0` profile; the port's runner patches build
on `swa-decode-window` and `so-row-owners`, so `install.sh` applies the profile first.

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching is on: a request must not read cached blocks another prompt in the same step has not written yet |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | the response grammar is on for every chat turn, so every step with a finishing prompt is a structured-output step |
| [`so-row-owners`](../fixes/so-row-owners.md) | grammar masks land on the right rows when a long prompt is chunked next to grammar requests |
| [`async-inputs`](../fixes/async-inputs.md) | async scheduling is on |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption must not repeat its earlier output |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids are read back corrupted on this release otherwise |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | the perplexity gate uses `prompt_logprobs`, which crashes or returns wrong values without it |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | one undecodable logprob id would stop the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with a `seed` would segfault a tensor-parallel worker |
| [`swa-decode-window`](../fixes/swa-decode-window.md) | exact sliding window in paged decode; the port extends it to model type `beam` |
| [`moe-explicit-overload`](../fixes/moe-explicit-overload.md) | the per-channel FP8 MoE op is called through its overload packet otherwise: a host sync per MoE layer per eager step (eager decode 0.65 tokens/s) |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | a failed rank stops the engine instead of serving garbage |

## Verified

On the image built by `ports/beam-1-0/install.sh` from the stock `releases/v0.30.0` image, eight Gaudi 2
cards, served with the Launch block above (plus `--host`, `--port`, `--served-model-name`), checked with
`ports/beam-1-0/tests/hpu/run_gates.sh`:

| Gate | Result |
| --- | --- |
| Build gates (`tests/build_gates.py`) | 8/8: fix profile and port patches applied, plugin registered, config and template |
| CPU tests in the image (`pytest ports/beam-1-0/tests`) | 103 passed, 2 skipped (they need `GAUDI_FIXES_ROOT`; they pass on a release copy) |
| Smoke: 20 chat prompts, temperature 0 | 20/20 correct, finish `stop` 20/20, reasoning 20/20, no template markup |
| Perplexity vs the per-token FP8 CPU reference, 3 passages (257 to 1,699 tokens) | -0.09% / +0.23% / +0.04%; top-1 98.35% / 97.04% / 94.53% (floor 98.47% / 97.24% / 96.48%, gate floor - 2 points); KL 0.0071 / 0.0053 / 0.0102 (gate 0.02) |
| Decode vs prefill, 8 x 512 greedy tokens past the window | mean \|Δlogprob\| 0.032 (gate 0.05) |
| Needles at 16K, 64K, 129K and 255K tokens x depths 0.1 / 0.5 / 0.9 | 12/12 in the answer; control without the needle clean |
| Copy test (a passage restarted at the end must continue exactly) | 64/64 tokens at 129,938 and at 254,938 tokens |
| Tools: auto, required, named, none (each streaming and not), parallel calls, `json_schema`, `reasoning_effort` none | 11/11 |
| Isolation: 8 different requests at once, each with its own code | own code 8/8, foreign content 0 |
| Stress: 15 streaming decodes of 600 tokens while a 250K-token prompt prefills | 15/15 complete, server healthy |

The perplexity figures are identical, to every printed digit, to those of the configuration's first
build, which carried its own runner fixes instead of the cookbook's profile. The checkpoint load on
this run took 40 to 50 minutes because the shared storage delivered about 0.2 GB/s at the time (the
ranks waited on reads for 98% of it); see Performance for the load on unloaded storage.

## Performance

Single runs on eight Gaudi 2 cards.

| Test | Result | Measured as |
| --- | --- | --- |
| Decode, 1 stream / 16 streams (completions, 256 tokens) | 28.5 to 29.5 tok/s / 410 tok/s aggregate (25.2 per stream) | this recipe, the gate run above (two runs) |
| 250K-token prompt next to 15 running decodes | 108.7 s | this recipe, the gate run above |
| Copy test prompt, 129,938 / 254,938 tokens, end to end | 34.5 s / 106.6 s | this recipe, the gate run above |
| Chat, 2K-token prompts, 768 tokens out, grammar on: 1 / 4 / 16 streams | 29.0 tok/s per stream, TTFT 0.52 s / 100.7 tok/s aggregate / 286.4 tok/s aggregate | the same configuration at 128K context on its first build |
| Weight load, `--load-format beam` vs stock `auto` | 356 s vs 2,177 s, identical model state on all 8 ranks | same-node A/B, TP 8, earlier build |

Decode is bound by host graph launches, not by the device: a step costs about the same at 1 and 4
sequences, so 16 sequences add little per step. The [port write-up](../ports/beam-1-0.md) has the
step-by-step speed-ups and the expert-parallel study.

## Known issues

- Outputs are not bit-identical across batch compositions or prefix-cache hits: per-token FP8
  activations turn kernel-shape differences into token flips at near-ties (top-1/top-2 gaps of 0.05
  to 0.58 nats). Two runs of the same request alone are identical; no foreign content appeared under
  any batching.
- The model always reasons. `reasoning_effort` minimal, low, medium, high, xhigh and max set the
  template's budget (8 to 512); `none` is served as `minimal` instead of an HTTP 400, and the answer
  still carries a short reasoning. Integer budgets go through `chat_template_kwargs`.
- The template's `allow_missing_thinking` stays at its default (true), so multi-turn requests from
  clients that drop earlier reasoning work. Tool arguments in history that are not objects are
  coerced to `{}` by vLLM v0.30.
- A tool call cut off by `max_tokens` inside a streamed string argument leaves the client with
  argument text that does not parse; non-streaming responses drop that parameter.
- The always-on response grammar costs about 10% of aggregate throughput at 8 to 16 concurrent chat
  requests (the engine defers a grammar step's sampling); `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT=0`
  frees unconstrained turns, but the grammar is what keeps long agentic tool calls well formed.

## What not to do

- Do not set `--chat-template-content-format openai`: the Beam template takes string content, and
  every chat request then fails with HTTP 400.
- Do not run with `--enforce-eager`: eager decode is host-bound at 0.68 tokens/s.
- Do not combine `beam` with another reasoning or tool parser; the port refuses the combination.
- Do not set `PT_HPU_SDPA_QKV_SLICE_MODE_FWD`: its sliding-window behaviour is not verified, and the
  model refuses to start with it.
- Do not set `BEAM_FSDPA_WS_GIB=0` at this context length: prompts with 32K or more of cached
  context then grow FusedSDPA's workspace past the free device memory, and the engine returned NaN
  logits or died.
