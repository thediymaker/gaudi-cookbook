# Mistral Large 4 (MXFP4)

A 1T-parameter mixture-of-experts model (52B active) with MLA attention, a sparse key indexer and 128 experts of
width 6144, served text only from AMD's published Quark MXFP4 checkpoint at 128K tokens per request on one 8-card
Gaudi 2 node. vllm-gaudi has no path for this model; the [Mistral Large 4 port](../ports/mistral-large-4.md) adds
one, with the checkpoint's DSpark draft head for speculative decoding and the `ml4` build of the
[MXFP4 MoE kernels](../kernels/mxfp4-moe.md).

| | |
| --- | --- |
| Model | [amd/Mistral-Large-4-1T-A52B-Quark-MXFP4](https://huggingface.co/amd/Mistral-Large-4-1T-A52B-Quark-MXFP4) at revision `d084803a`, loaded as published (no conversion) |
| Hardware | 8x Gaudi 2 (attention TP 8, experts EP 8 as 256 virtual experts of I 3072) |
| Image | vllm-gaudi `releases/v0.30.0` (`058f107`), vLLM `ced6857a`, Gaudi SW 1.24.1, plus [`ports/mistral-large-4`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/mistral-large-4) |
| Precision | MXFP4 routed experts run W4A16; the FP8 linears of the base are converted to BF16 at load; int8 latent KV cache. 578 GB |
| Context | 131,072 served (1,048,576 native) |
| Concurrency | `--max-num-seqs 8`, KV pool 243,200 tokens |
| Parsers | reasoning `mistral`, tools `mistral` (from [vllm#64534](https://github.com/vllm-project/vllm/pull/64534)) |
| Speculative decoding | DSpark, the checkpoint's draft head, 3 tokens per step |
| Licence | Mistral Ecosystem Open-Weight License (modified MIT): a Model as a Service business needs a separate commercial licence from Mistral for commercial use; internal use is exempt |
| Recipe | [`recipes/amd/Mistral-Large-4-1T-A52B-Quark-MXFP4/recipe.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/amd/Mistral-Large-4-1T-A52B-Quark-MXFP4/recipe.yaml) |

## Download the checkpoint

The plugin reads `params.json` and the `consolidated-*.safetensors` shards from a local directory:

```bash
hf download amd/Mistral-Large-4-1T-A52B-Quark-MXFP4 --revision d084803a4a2bfac7274058937f6665750741d650 \
  --local-dir /data/models/Mistral-Large-4-1T-A52B-Quark-MXFP4
```

## Build the image

From a checkout of this repository, on a stock vllm-gaudi `releases/v0.30.0` image built as in
[Getting started](../getting-started.md):

```bash
docker build -f ports/mistral-large-4/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-mistral-large-4-image> .
```

`install.sh` checks the image's commits, installs mistral-common 1.13.0 (sha256-pinned, from PyPI), the vendored
vllm#64534 files and the plugin, builds the MXFP4 MoE kernels (`kernels/mxfp4-moe`, variant `ml4`) and the port's
decode-step kernels, and runs the build gates. See
[`ports/mistral-large-4/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/mistral-large-4/README.md).

## Launch

Inside the port image (`$COOKBOOK` is `/opt/gaudi-cookbook` there), with the checkpoint at `$ML4_CHECKPOINT`. Mount
persistent storage at `/data` so the compiled graphs survive restarts, and give the container at least 200 GiB of
host memory beyond the 160 GB offload pool.

```bash
export ML4_CHECKPOINT=/data/models/Mistral-Large-4-1T-A52B-Quark-MXFP4
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_RECIPE_CACHE_CONFIG=/data/ml4-recipe-cache,False,32768
export VLLM_CONTIGUOUS_PA=false
export VLLM_SKIP_WARMUP=true
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export MAX_MODEL_LEN=131072
export ML4_KV_FP8=1
export ML4_KV_CODEC=i8-t128-f32
export ML4_INDEX_U8=1
export ML4_PLACEMENT=B
export ML4_MXK_V9G=1
export ML4_SPEC=dspark
export ML4_LAYER_GRAPHS=1
export ML4_LAYER_GROUP=8
export ML4_INDEX_DECODE_FAST=1
export ML4_INDEX_SELECT_BYTES=536870912
export ML4_ATTN_STEP_BYTES=1073741824
export ML4_INDEX_KERNEL=1
export ML4_ATTN_KERNEL=1
export ML4_NUMERICS_RW=1
export ML4_R5_FF8=x
export ML4_R5_SSL=f32
export ML4_R5_K2FOLD=1
export ML4_R5_RMS=1
export ML4_R5_ROPE=1
export ML4_R5_LAT=1
export ML4_R6_ROPEFMA=1
export ML4_R6_FFK=x
export ML4_R6_DUAL=nope
export ML4_R5_ROUTER=1
export ML4_R5_ROUTER_MODE=1
export ML4_LOAD_FAST=1

mkdir -p "${PT_HPU_RECIPE_CACHE_CONFIG%%,*}"
# Warm-up: waits for the server, compiles every shape, then writes /tmp/ml4-warm.done (point readiness at it).
python3 -m mistral_large4_hpu.warm > /tmp/ml4-warm.log 2>&1 &

# Fixes: the v0.30.0 profile plus this recipe's opt-in fix.
# Stops with a non-zero exit code if the image does not match.
python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.30.0 penalty-shapes || exit 1

vllm serve "$ML4_CHECKPOINT" \
  --served-model-name amd/Mistral-Large-4-1T-A52B-Quark-MXFP4 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --config-format mistral \
  --load-format mistral \
  --tokenizer-mode mistral \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --num-gpu-blocks-override 1900 \
  --block-size 128 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 160000000000}}' \
  --no-async-scheduling \
  --enforce-eager \
  --max-logprobs 20 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 2}' \
  --reasoning-parser mistral \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

- The `ML4_*` switches are the port's; each is described in the
  [port README](https://github.com/thediymaker/gaudi-cookbook/blob/main/ports/mistral-large-4/README.md#switches).
  `ML4_MXK=0` runs the stock MXFP4 op on the same virtual experts (the reference path; restart the server to switch).
- `--speculative-config` uses the n-gram slot only to make vLLM reserve 3 draft positions per request; with
  `ML4_SPEC=dspark` the plugin replaces the n-gram proposer with the checkpoint's DSpark head. Without
  `ML4_SPEC=dspark`, drop the flag.
- `--enforce-eager` turns off vllm-gaudi's own graph capture; the plugin replays its own decode graphs (9 groups of
  8 layers) and prompt graphs (64 to 768-row chunks). Every new shape compiles on first use, so the warm-up module
  sends each shape once before the server takes traffic.
- `--num-gpu-blocks-override 1900` with the int8 latent codec: 243,200 tokens of KV across all requests, so one
  128K request and several shorter ones fit together. 131,072 is the largest window that fits next to the weights
  (71.7 GiB per card) and the long-prompt working memory; a 160K window ran out of device memory.
- The OffloadingConnector keeps evicted prefix-cache blocks in 160 GB of host memory (20 GB per worker), so a long
  conversation that falls out of HBM reloads instead of prefilling again.
- `--no-async-scheduling`: async scheduling with DSpark is exact at 1 stream but not bit-equal to the synchronous
  path at 4 and 8 streams.

## Required fixes

All from [Fixes](../fixes/index.md). `penalty-shapes` is opt-in and named in the launch command; the rest come from
`--profile v0.30.0`.

| Fix | Why this model needs it |
| --- | --- |
| [`prompt-order`](../fixes/prompt-order.md) | prefix caching is on; a reader can run before its writer in the same step |
| [`sampling-rows-refresh`](../fixes/sampling-rows-refresh.md) | stale device sampling rows after a batch reorder |
| [`structured-output-rows`](../fixes/structured-output-rows.md) | the `mistral` tool parser gives every chat request a grammar; rows must keep their own sampling parameters |
| [`so-row-owners`](../fixes/so-row-owners.md) | grammar masks on the right rows when a long prompt is chunked |
| [`bad-words-index`](../fixes/bad-words-index.md) | `bad_words` raises `IndexError` in the engine core |
| [`logitsprocs-rows`](../fixes/logitsprocs-rows.md) | `logit_bias`, `min_p` and `min_tokens` are silently ignored |
| [`resumed-prefill-logits`](../fixes/resumed-prefill-logits.md) | a request resumed after preemption repeats its generated tail |
| [`penalty-shapes`](../fixes/penalty-shapes.md) (opt-in) | a request with a repetition, presence or frequency penalty recompiles every step otherwise |
| [`gather-logprobs-int32`](../fixes/gather-logprobs-int32.md) | `top_logprobs` ids come back corrupted in lazy mode |
| [`logprobs-decode-guard`](../fixes/logprobs-decode-guard.md) | a bad logprob id is replaced instead of stopping the API server |
| [`seeded-sampling`](../fixes/seeded-sampling.md) | a request with `seed` can segfault a TP worker in lazy mode |
| [`rank-exception-fatal`](../fixes/rank-exception-fatal.md) | an exception on a non-zero TP rank stops the engine instead of leaving ranks out of step |

## Verified

Run on one 8-card Gaudi 2 node from the stock `releases/v0.30.0` image: `install.sh` run inside a container
started from it (all 7 steps; build gates 9/9), then this page's Launch block with the checkpoint on network
storage. The kernel gates ran on the same container with the server stopped; the serving gates are
`ports/mistral-large-4/tests/hpu/run_gates.sh`, after the warm-up, on an otherwise idle server.

| Gate | Result |
| --- | --- |
| Build gates (`install.sh` step 7) | 9/9: vendored files, plugin, mistral-common 1.13.0, params.json adapter (63 KV owners, 44/28 indexers, 18 NoPE layers), quantisation claim, `ml4-v8-124da07be83e` and `k1-a711b9781071` libraries, kernel hook |
| Kernel code identity | all 6 `ml4` kernels and all 21 decode-step kernels have the same code hash as the libraries the port is served with in production |
| MoE kernel parity, checkpoint layer 5 (`run_parity.sh --variant ml4 --checkpoint`) | `PARITY PASS (paths v8, grouped)`, grouped bit-identical to v8 in 10/10 cases (1 to 64 tokens), kernel rel_l2 1.61e-3 to 1.69e-3 against fp32 (stock 3.04e-3 to 4.01e-3), `PREFILL PASS` (max abs 0 against the stock op at 256, 1024 and 4096 tokens), `SIG_DEVICE PASS` for both paths |
| Decode-step kernel gates (`run_kernel_gates.sh`, one card) | parity, K1 (fused indexer), K2 (attention gather), r5 and r6 node-cut kernels: 5/5 PASS, each bitwise equal to the native path |
| DSpark grammar check (`tests/check_spec_grammar.py`) | ALL PASS |
| Startup | fixes applied; weights loaded in 1,466 s (24 min, 101 GB read per rank); warm-up 1,178 s, 0 errors |
| Golden prompts, greedy | 12/12 correct |
| Greedy repeat (content and reasoning) | 12/12 identical |
| Tool call through the `mistral` parser | pass, non-streaming and streaming (`get_weather {"city": "Lisbon"}`) |
| Needles at 8K, 32K, 64K and 127K tokens x depths 0.1 / 0.5 / 0.9 | 12/12; the 127K prompts are 126,654 to 127,084 tokens, 165 to 178 s each end to end |
| Logprobs: 100 requests with `top_logprobs=20`, CJK and emoji output | 6,400 positions, 0 incomplete or non-finite |
| Teacher-forced scoring of fixed code text | 9,658 positions, NLL 1.1863, the same to 4 decimals on two runs |

The first run of the gates had two failures, both fixed in the gate script, not the port: the golden set asked
the model to reverse the string "gaudi", and it answered "iduga" (its reasoning spelled the right letters,
i-d-u-a-g); a character-level task is a tokenizer test, not a serving test, so it was replaced by a leap-year
question. The 127K needle at depth 0.5 asked for 4,096 answer tokens on a 127,084-token prompt, past the
131,072 window, and got HTTP 400; the gate now caps the answer budget at the window. The rerun above passed every
gate.

## Performance

Decode with DSpark on (`ml4_gates.py sweep`), 256 tokens per request with `ignore_eos` after a cached prefix of the
given length, requests asking for a long technical essay, 2 requests per stream; per stream is 1 / median time per
output token, aggregate counts all tokens over the wall time including time to first token. Measured as: one
server, this recipe as written, single run, after the gates above.

| Cached prefix | 1 stream | 4 streams, per stream | 4 streams, aggregate |
| --- | --- | --- | --- |
| 2K tokens | 52.0 tok/s | 27.3 tok/s | 82.0 tok/s |
| 32K tokens | 58.2 tok/s | 17.1 tok/s | 48.4 tok/s |

DSpark speed follows how many drafts the text accepts, so a 1-stream run on the longer prefix can come out faster;
without drafting, decode slows with context (see the write-up).

Time to first token of an uncached 32K-token prompt: 21.5 s (21.49 to 21.75 s, 3 runs, same server). A 127K
needle request took 165 to 178 s end to end.

Recorded in the [write-up](../ports/mistral-large-4.md#performance) under other conditions: DSpark chat at 1 / 4 / 8
streams 44.9 / 22.0 / 13.6 tok/s per stream through a production service, and plain decode (no drafting) 57.1 to
38.8 ms per step from the decode-step kernels.

## Known issues

- Weight loading reads 578 GB; each rank reads only its slices (about 72 GiB per card). Load time follows the
  storage: 24 minutes on the verification run, and from 6 to 106 minutes on other loads from network storage. The
  warm-up then took 20 minutes; keep the recipe cache on persistent storage.
- The first prompt of a 128K request takes minutes: prefill of a 128K prompt was 153 s cold in the
  [write-up](../ports/mistral-large-4.md#performance). Prefix-cache follow-up turns start in seconds.
- With vLLM v0.30.0, every streamed chat chunk recounts the reasoning tokens of the whole response so far, which is
  quadratic in the response length. A long reasoning stream can starve the API server's `/health` endpoint. No fix
  for it ships in this repository yet; use a generous liveness timeout.
- Greedy output is repeatable for one request at a time on one server. In a batch, output depends on which requests
  share the step (as on the stock path), and DSpark verify splits at near-ties.
- DSpark drafts only for requests whose sampling its verify step reproduces exactly: temperature, top-p, top-k and
  grammars. A request with `logprobs`, a penalty, `logit_bias`, `min_p`, `min_tokens`, `bad_words` or `seed` turns
  drafting off for the whole batch while it runs.
- Prefix-cache hits are bit-identical to a cold run only when the hit ends on a 2048-token chunk boundary; other hits
  split only at near-ties, like a moved chunk boundary without the cache.
- The checkpoint's Pixtral vision tower is not served.

## What not to do

- Do not use FP8 (e4m3) for the latent KV: it failed the perplexity bar on Gaudi 2 (KL 0.0231 against 0.02). The int8
  codec with a scale per 128 values passes at the same size.
- Do not raise `--max-model-len` above 131072 on this layout: 160K ran out of device memory in the prompt working set.
- Do not set `ML4_R5_ROUTER_MODE=4`: it is not bit-exact.
- Do not turn on async scheduling with DSpark without re-running the gates at 4 and 8 streams.
- Do not reuse a recipe cache directory across images or kernel builds.
