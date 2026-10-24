# GLM-5.3-Flash

Z.ai's 320B-parameter mixture-of-experts model (18B active) with Kimi Delta Attention, NoPE MLA, manifold
hyper-connections and image input, served in its published block FP8 at 256K context on one 8-card Gaudi 2 node.
vllm-gaudi has no path for it, so it runs from the [GLM-5.3-Flash port](../ports/glm-5-3-flash.md), built on top of
the stock `v0.26.0` image.

| | |
| --- | --- |
| Model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) |
| Hardware | 8x Gaudi 2 (TP 8, expert parallel), one node |
| Image | port image built from vllm-gaudi `v0.26.0` (`b059f9c`), vLLM `568afb3a`, Gaudi SW 1.24.1 |
| Precision | block FP8 weights as published (328 GB), BF16 activations |
| Context | 262,144 served (KV pool 819,200 tokens) |
| Concurrency | `--max-num-seqs 32` |
| Input | text and up to 4 images per prompt (no video) |
| Parsers | tools and reasoning `glm47` |
| Recipe | [`recipes/zai-org/GLM-5.3-Flash/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/zai-org/GLM-5.3-Flash/recipe.yaml) |

## Build the image

From a checkout of this repository, on top of a stock `v0.26.0` image built as in
[Getting started](../getting-started.md):

```bash
docker build -f ports/glm-5-3-flash/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.26.0-image> -t <your-glm-5-3-flash-image> .
```

`install.sh` checks the vllm-gaudi and vLLM commits, applies the port's four source patches, builds the mHC Sinkhorn
TPC kernel, installs the `glm5_next_hpu` plugin and checks all of it again. It stops on any other base. The image
carries this repository at `/opt/gaudi-cookbook`.

## Launch

Inside the port image, with `COOKBOOK=/opt/gaudi-cookbook`:

```bash
export PT_HPU_LAZY_MODE=1
export VLLM_SKIP_WARMUP=true
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export GLM5_NEXT_VISION=1
export GLM5_NEXT_IMAGE_MAX_TOKENS=2048
export GLM5_NEXT_KDA_SCAN=1
export GLM5_NEXT_DECODE_LAYER_GROUP=45
export GLM5_NEXT_MHC_SINKHORN=kernel

# Fixes: the v0.26.0 profile plus this recipe's opt-in fixes, in the pack's order.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 glm47-streaming-text-markers || exit 1

vllm serve zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --num-gpu-blocks-override 1280 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 2048 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.82 \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort": "high"}' \
  --chat-template "$COOKBOOK/ports/glm-5-3-flash/serving/chat_template.jinja" \
  --limit-mm-per-prompt '{"image": 4}'
```

- `--enable-expert-parallel` puts 36 of the 288 routed experts on each card instead of slicing every expert across
  TP 8; with the `moe-explicit-overload` fix it is the faster layout at one stream.
- `--num-gpu-blocks-override 1280` pins the KV pool. vLLM derives a 640-token block for this hybrid model (the
  attention page is raised to the KDA state page), so 1,280 blocks hold 819,200 tokens; its autosizing over-commits
  by about one MLA layer and fails.
- `--no-enable-prefix-caching` is required: the KDA layer implements only the non-caching state contract and refuses
  to start with prefix caching.
- `--max-cudagraph-capture-size 2048` and `GLM5_NEXT_IMAGE_MAX_TOKENS=2048` bound the number of prompt-length graphs
  that images create (an image costs compile time through the prompt length it adds, not through its geometry).
- `--chat-template` is the checkpoint's template with image placeholders (one change, noted in the file's header).
  `reasoning_effort` defaults to `high` here; requests can set `low` in `chat_template_kwargs`, and the template
  treats any value other than `low` or `high` as `max`.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`: with warmup skipped, a long prefill compiles on first touch and can take
  longer than the 300 s default, which stops the engine.
- The five `GLM5_NEXT_*` switches are explained in the
  [port README](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/glm-5-3-flash/README.md#switches).
  `VISION` builds the vision tower; `KDA_SCAN`, `DECODE_LAYER_GROUP` and `MHC_SINKHORN` are the prefill and decode
  speed-ups; with them off the port runs its reference path.

### Buckets

No bucket file and no bucket variables: the image's default bucketing.

## Required fixes

The v0.26.0 profile, applied at container start on top of the port's patches. The fixes this configuration depends
on are listed in the recipe: the sampling and structured-output row fixes (`sampling-rows-refresh`,
`structured-output-rows`, `bad-words-index`, `logitsprocs-rows`), the async-scheduling fixes (`async-inputs`,
`async-discard-counter`, `resumed-prefill-logits`), `hybrid-decode-pad` (decode padding on this hybrid model), the
logprobs fixes (`gather-logprobs-int32`, `prompt-logprobs`, `logprobs-decode-guard`), `seeded-sampling`,
`moe-explicit-overload` (the FP8 MoE op without a host sync per layer), `mm-merge-idxcopy` (image embedding merge),
`rank-exception-fatal` and `xgrammar-generation-eos-stop` (`tool_choice: required` stops at GLM's end-of-turn
tokens). The opt-in `glm47-streaming-text-markers` makes the streaming tool parser catch calls the model writes as
plain-text markers. The rest of the profile applies too; the golden gate below shows it does not change the model's
output (GLM-5.3-Flash routes with sigmoid scores and one group, which `laguna-moe-sigmoid-routing` leaves unchanged).

## Verified

Run as written on one node of eight Gaudi 2 cards: a stock `v0.26.0` image with this repository copied to
`/opt/gaudi-cookbook` and `install.sh` run as the Dockerfile runs it (commits checked, four patches applied, kernel
gates passed), then the launch block above. Every gate is `ports/glm-5-3-flash/tests/hpu/run_gates.sh`:

| Gate | Result |
| --- | --- |
| golden: 12 greedy prompts, token ids against the reference outputs | 12/12 identical |
| needles: three codes at 10, 50 and 90 % depth | 3/3 at 16K, 63K, 127K and 239K tokens |
| tools: single, parallel and no call, streamed and not; multi-turn; a call after a 32K-token document | 8/8 |
| images: three solid colours, quadrants, counting, text before and after; 5 images rejected with 400 | 8/8 |
| decode: single stream at 2K and 128K, floors 15 and 10 tokens/s | pass (numbers below) |
| prefill: warm TTFT at 16K, 64K and 128K, 128K within 120 s | pass (numbers below) |
| sinkhorn: the TPC kernel against vLLM's stock loop, 1 to 1,024 rows, one card, no server | bit-identical |

**Off means stock.** The same server with `GLM5_NEXT_KDA_SCAN=0`, `GLM5_NEXT_DECODE_LAYER_GROUP=1` and
`GLM5_NEXT_MHC_SINKHORN=torch` (the reference path) passed the same six server gates, golden 12/12 identical to the
same reference. The CPU tests (`pytest ports/glm-5-3-flash/tests`, 34 tests including the patches against the
release sources and the config against the checkpoint's `config.json`) pass in the same image.

## Performance

Single stream, temperature 0, unique prompts, every shape warmed once before the measured request. Decode is 512
tokens with `ignore_eos`, timed from the first to the last streamed token. Same node and image; the server was
relaunched for each column.

| Test | Switches on (as launched above) | Switches off (reference path) |
| --- | --- | --- |
| Decode, 2K prompt | 43.5 tokens/s | 30.8 tokens/s |
| Decode, 128K prompt | 38.6 tokens/s | 28.7 tokens/s |
| TTFT, 16K prompt | 2.6 s | 5.4 s |
| TTFT, 64K prompt | 10.3 s | 22.3 s |
| TTFT, 128K prompt | 23.8 s | 42.7 s |

An earlier server with the switches on, same image and command, measured 38.4 and 36.2 tokens/s for the two decode
rows and the same TTFT within 0.3 s: treat single-stream decode as plus or minus about 10 %. Batched decode and longer
contexts are in the [write-up](../ports/glm-5-3-flash.md#performance) (different builds, labelled there).

## Known issues

- Cold start is dominated by reading 328 GB of weights: about 13 minutes from network storage (62 shards at about
  12.5 s each), then a few minutes of engine start.
- New prompt lengths compile graphs on first use (warmup is skipped). The first request at a new length can take
  several times as long as the warm one; the gates warm every shape once before measuring.
- The 11 DSA layers run as dense MLA with the indexer skipped. Up to 2,048 prompt tokens this is the same function;
  beyond it is an approximation. Needles pass to the served window (above), and to 512K in the
  [write-up](../ports/glm-5-3-flash.md#correctness).
- Video is not served, and the MTP draft head is not loaded (no speculative decoding).

## What not to do

- Do not turn on prefix caching: the model refuses to start, because the KDA layer does not store per-block states.
- Do not drop `--num-gpu-blocks-override`: KV autosizing over-commits on this hybrid model and fails.
- Do not raise `GLM5_NEXT_IMAGE_MAX_TOKENS` or `--limit-mm-per-prompt` without re-running the image gate and a long
  prompt with images: each new prompt length adds graphs.
