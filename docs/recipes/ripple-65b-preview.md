# Ripple-65B-Preview

Ricursive's 65B dense model (80 layers: 64 with a 1024-token sliding window and RoPE, 16 full-attention layers
without positional encoding; gated attention, value residuals), from the official BF16 checkpoint as published, at
128K context on four Gaudi 2 cards. vLLM added the model after v0.30.0 and vllm-gaudi has no path for it, so this
recipe runs in the image of the [ripple-65b port](../ports/ripple-65b.md), built from the stock `releases/v0.30.0`
image.

| | |
| --- | --- |
| Model | [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) (Apache-2.0), revision `d58dbcf` |
| Hardware | 4x Gaudi 2 (TP 4) |
| Image | the ripple-65b port image on vllm-gaudi `releases/v0.30.0` at `058f107`, vLLM `ced6857a`, Gaudi SW 1.24.1 |
| Precision | BF16 checkpoint (130 GB, 30.2 GiB of weights per card) |
| Context | 131,072 served (the config allows 2M positions, the model card says 1M; 262,144 was run too, see Known issues); KV pool 2,068,666 tokens per replica |
| Concurrency | `--max-num-seqs 16` |
| Parsers | tools and reasoning `ripple` (from the port) |
| Recipe | [`recipes/ricursiveAI/Ripple-65B-Preview/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/ricursiveAI/Ripple-65B-Preview/recipe.yaml) |

## Build the image

From a checkout of this repository, on top of your stock `releases/v0.30.0` image (built as in
[Getting started](../getting-started.md)):

```bash
docker build -f ports/ripple-65b/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-ripple-65b-image> .
```

The build copies the repository to `/opt/gaudi-cookbook` (so `$COOKBOOK` is set there), checks the image commits,
applies the `v0.30.0` fix profile plus `swa-window-context`, patches vLLM's DelegatingParser (vLLM PR #65745) and the
runner (exact sliding windows for model type `ricursive`, hooks for window-sized KV), installs the `ripple_hpu` plugin
and runs the build gates. Details: [`ports/ripple-65b/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/ripple-65b/README.md).

## Launch

In the port image. `$COOKBOOK` is `/opt/gaudi-cookbook` there.

```bash
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_CONTIGUOUS_PA=false
export VLLM_SKIP_WARMUP=true
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_FP32_SOFTMAX=true
export VLLM_BUCKETING_STRATEGY=pad
export VLLM_HPU_FSDPA_SLICE_ENABLED=false
export VLLM_GRAPH_RESERVED_MEM=0.3
export RIPPLE_SWA_KV=1
export RIPPLE_CONSTRAIN_AUTO_TOOLS=1
export RIPPLE_FAST_SOFTMAX_MIN_KV=16384
export RIPPLE_FSDPA_WS_GIB=2
export RIPPLE_FSDPA_Q_TILE=256
export RIPPLE_GRAPH_BUDGET_GIB=auto
export RIPPLE_PREFILL_INFLIGHT=1

# Fixes: the v0.30.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 swa-window-context || exit 1

vllm serve ricursiveAI/Ripple-65B-Preview \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --block-size 128 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --chat-template /opt/gaudi-cookbook/ports/ripple-65b/serving/chat_template.jinja \
  --reasoning-parser ripple \
  --enable-auto-tool-choice \
  --tool-call-parser ripple \
  --exclude-tools-when-tool-choice-none
```

- `--chat-template`: the checkpoint ships two templates. `tokenizer_config.json` has an older one without tools;
  `chat_template.jinja` is the tools-aware one, copied unchanged into the port.
- `--exclude-tools-when-tool-choice-none`: with `tool_choice: "none"` and the tools still in the prompt, Ripple writes
  a tool call anyway, the parser removes it, and the answer comes back empty. Without the tools in the prompt it
  answers in text.
- `RIPPLE_SWA_KV=1` gives the 64 sliding layers window-sized KV: only the 16 full layers keep full-length KV, 16,384
  instead of 81,920 B per token per card. Outputs are identical with it on and off.
- `VLLM_BUCKETING_STRATEGY=pad`, `VLLM_HPU_FSDPA_SLICE_ENABLED=false` and `RIPPLE_FAST_SOFTMAX_MIN_KV=16384` are the
  long-prompt settings: pad buckets stop padding every context past 128K to 258K, and the full layers' prompt
  attention against 16K or more keys uses the fast softmax. A 250K-token prompt read in about 3 minutes instead of
  7.7 (at `--max-model-len 262144`). Pad buckets turn vllm-gaudi's FusedSDPA slicer on, which the port refuses (it returns NaN on padded slices and
  cuts the window mask), so the variable is set explicitly.
- `VLLM_GRAPH_RESERVED_MEM=0.3` leaves 17.4 GiB per card for graphs captured at run time and eager prefill
  workspaces; the KV pool gets the rest of `--gpu-memory-utilization 0.90`. At the default 0.1 a server ran out of
  device memory after 29 minutes of mixed traffic.
- `RIPPLE_GRAPH_BUDGET_GIB=auto` bounds the device memory that decode HPU graphs keep (free memory at the first decode
  minus 8 GiB). Without a bound, every new decode shape keeps its graph until one TP rank fails an allocation and the
  others hang in the next all-reduce.
- `VLLM_FP32_SOFTMAX=true` halves the engine's distance to an fp32 reference in the parity gates.
- The fix step finds every fix already applied (the port's `install.sh` applies the same set before its own runner
  patches), and stops if the image is not the port image.
- The `RIPPLE_*` switches are the port's; each is described in the port README. The recipe sets every one explicitly.

## Required fixes

From [Fixes](../fixes/index.md). All are in the `v0.30.0` profile except `swa-window-context`, which the launch names.
The port's runner patch builds on `swa-decode-window` and `swa-window-context`.

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching is on: a request must not read cached blocks another prompt in the same step has not written yet |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | every request with tools carries a grammar (`RIPPLE_CONSTRAIN_AUTO_TOOLS`), so rows must keep their own sampling parameters |
| [`so-row-owners`](../fixes/so-row-owners.md) | grammar masks land on the right rows when a long prompt is chunked next to grammar requests |
| [`async-inputs`](../fixes/async-inputs.md) | async scheduling is on |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption must not repeat its earlier output |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | logprob ids are read back corrupted otherwise (the stock release returned id -1 in Ripple's prompt logprobs) |
| [`prompt-logprobs`](../fixes/prompt-logprobs.md) | the decode-vs-prefill gate uses `prompt_logprobs`, which crashes or returns wrong values without it |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | one undecodable logprob id would stop the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with a `seed` would segfault a tensor-parallel worker |
| [`swa-decode-window`](../fixes/swa-decode-window.md) | exact sliding window in paged decode; the port extends it to model type `ricursive` |
| [`swa-window-context`](../fixes/swa-window-context.md) | prefill with cached context reads only the window on the 64 sliding layers instead of the whole context (opt-in) |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | a failed rank stops the engine instead of serving garbage |

## Verified

On the image built by `ports/ripple-65b/install.sh` from the stock `releases/v0.30.0` image, four Gaudi 2 cards,
served with the Launch block above (plus `--host`, `--port`, `--served-model-name`), checked with
`ports/ripple-65b/tests/hpu/run_gates.sh`. The weights were the published files of revision `d58dbcf` (identical to
the earlier revision `b172323` except for the added `generation_config.json`).

| Gate | Result |
| --- | --- |
| Build gates (`tests/build_gates.py`) | 10/10: fixes and port patches applied, plugin registered, config, template and parser, wrapped internals unchanged, window masks exact; a second `install.sh` run reports every step applied |
| CPU tests in the image (`pytest ports/ripple-65b/tests`) | 46 passed, 1 skipped (it needs `GAUDI_FIXES_ROOT`; it passes on a release copy) |
| Smoke: 20 chat prompts, temperature 0 | 20/20 correct, finish `stop` 20/20, no template markup (the model reasoned on 3 of 20) |
| Greedy determinism | alone 3/3 identical, including prefix-cache hits; 4 concurrent copies: 1 identical, 3 split at exact top-2 ties (margin 0.0) |
| Reasoning split | 6/6: reasoning filled, content free of markup, streamed split equals non-streamed, a second turn after a reasoned turn splits again |
| Decode vs prefill, 8 x 512 greedy tokens past the window | mean \|Δlogprob\| 0.0055 (gate 0.05) |
| Needles at 16K, 64K and 124.5K tokens x depths 0.1 / 0.5 / 0.9, 3 samples each at the checkpoint's default sampling | 9/9 positions with 2 or more of 3 (25/27 answers carry the number); control without the needle clean |
| Copy test (a passage restarted at the end of a 129,010-token prompt must continue exactly) | 251/251 characters, same tokens |
| Tools: auto, required, named, none (each streaming and not), parallel calls, `json_schema` | 10/10 |
| Isolation: 8 different requests at once, each with its own code | own code 8/8, foreign content 0 |
| Stress: 15 streaming decodes of 600 tokens while a 119,072-token prompt prefills | 15/15 complete, server healthy |

The needle gate samples because Ripple reasons before it answers: at temperature 0 it reasoned its way to "not
found" on 2 of 9 tries at 16K (the direct answer still had about 30% of the first token), while every needle from
64K up was found. Answers that give only the number count as found, since the question asks for the number only.

## Performance

Single runs on four Gaudi 2 cards.

| Test | Result | Measured as |
| --- | --- | --- |
| Decode, 1 / 4 / 16 streams (completions, 256 tokens, greedy) | 34.1 / 34.1 / 33.1 tok/s per stream; 530 tok/s aggregate at 16 | this recipe, the gate run above; two earlier runs of the same build at `--max-model-len 262144` gave 32.5 to 32.9 at 1 stream and 27.9 to 32.5 per stream at 4 and 16 |
| Chat decode with a tool offered (`tool_choice` auto, so a grammar request) vs none, 4 streams | 28.4 vs 32.9 tok/s per stream | this recipe, single run; at 1 stream 30.6 vs 30.3 |
| 119,072-token prompt next to 15 running decodes | 58.4 s | this recipe, the gate run above |
| Copy test prompt, 129,010 tokens, end to end | 65.1 s | this recipe, the gate run above |
| 250K-token prompt next to 15 running decodes | 166.7 s | this build at `--max-model-len 262144` |
| 250K-token prompt alone | about 180 s (462 s with exponential buckets and fp32 softmax) | the port's first build, same settings at 262,144, TP 4 BF16, same-pod chained A/B |
| 1,024-token turn on a cached 128K context | 0.65 s (3.46 s before the long-prompt settings) | the port's first build, same settings |
| Decode at 128K of context, 1 / 4 streams | 32 / 26 to 27 tok/s per stream | the port's first build, same settings |

Decode is bound by the host: a step costs about 30 ms at 1 to 16 streams while the device is busy for 18 ms.
The [port write-up](../ports/ripple-65b.md) has the long-prompt study, the FP8 measurements (the same per-stream speed
with more KV room; FP8 is not part of this recipe because its checkpoint is not public) and the speculative decoding
study.

## Known issues

- Ripple decides per turn whether to reason, and the template has no switch: `reasoning_effort` and
  `chat_template_kwargs` thinking flags do nothing. `thinking_token_budget` is accepted by vLLM but ignored by
  vllm-gaudi `releases/v0.30.0`.
- At temperature 1.0 (the checkpoint's `generation_config.json` default, with top-p 0.95), a few percent of hard
  puzzle and math questions reason for more than 64K tokens before answering. That is long, varied reasoning, not a
  decode loop (no repeated text). Set `max_tokens` for clients that cannot wait.
- Outputs are not bit-identical across batch compositions: the batch shape changes bf16 rounding, and greedy decoding
  can take the other branch at exact ties. A request alone repeats exactly.
- A long prompt holds up the other streams on the replica: while it is read they get one token per 4,096-token
  chunk (2 to 4 s apart at 250K of context).
- Every request with tools is a grammar request (`RIPPLE_CONSTRAIN_AUTO_TOOLS=1`). vllm-gaudi applies the grammar
  bitmask on the host, which cost 14% of decode speed at 4 streams (28.4 against 32.9 tok/s per stream).
- At `--max-model-len 262144` (the production window) the same build passed the needles at 16K to 255K (11/12 at
  temperature 0, the miss at 16K as above), the stress run with a 250K prompt and the copy test at 130K and 200K, but
  the copy test failed at 250K to 255K: the continuation of the passage diverged after 29 to 174 of 251 characters
  (three placements). With fp32 softmax on every layer (`RIPPLE_FAST_SOFTMAX_MIN_KV=0`) it failed the same way, so
  the fast softmax is not the cause; whether it is the model or the engine is not settled. This recipe stays at
  131,072, where every gate passes.
- Each streamed tool-call delta re-scans the call so far (vLLM v0.30.0 has no incremental argument extender; upstream
  PR #67146). A 20,000-character call streamed without slowing down; much longer calls may.

## What not to do

- Do not set `VLLM_HPU_FSDPA_SLICE_ENABLED=true` (or leave it unset with pad buckets): the port refuses to start, since
  the slicer returns NaN on padded slices and cuts the sliding layers' window mask.
- Do not set `RIPPLE_FAST_SOFTMAX_MIN_KV` low or apply fast softmax everywhere: fast softmax on all layers failed the
  fp32 oracle comparison. Keep `VLLM_FP32_SOFTMAX=true`.
- Do not set `RIPPLE_GRAPH_BUDGET_GIB=off` on a long-running server, and do not combine the port with the
  `hpu-graph-lru` fix: both manage the same graph cache.
- Do not set `PT_HPU_SDPA_QKV_SLICE_MODE_FWD`: its sliding-window behaviour is not verified, and the model refuses to
  start with it.
- Do not use `tokenizer_config.json`'s template (drop `--chat-template`): it has no tools.
- Do not raise `--max-model-len` past 131072 for work that needs exact recall of long passages: at 262,144 the copy
  test failed past 200K tokens (Known issues). A 1M prompt was not tested; the full layers' prompt attention grows
  with the square of the length.
