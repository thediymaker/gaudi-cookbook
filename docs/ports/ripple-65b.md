# Ripple-65B-Preview on Gaudi 2

Ripple-65B-Preview is a 65B dense model with 80 layers: 64 use a 1024-token sliding window with RoPE and 16 (layers
4, 9, ..., 79) use full attention without positional encoding (NoPE). Each attention output is gated by an fp32
sigmoid of a third projection, the value projection reads a learned blend of the current input and the scaled input
embeddings (value residuals), every block has four RMSNorms, and the SwiGLU is clamped at 20. vLLM added the model
after v0.30.0, and vllm-gaudi has no path for it. This port is a backport of the upstream model and parsers to the
vLLM of vllm-gaudi `releases/v0.30.0`, plus what Gaudi needs for exact windows, long contexts and long prompts.

!!! summary "At a glance"
    - **Checkpoint:** [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview),
      Apache-2.0, BF16, served as published.
    - **Hardware:** 4 Gaudi 2 cards per replica, TP 4, 16 sequences, 131,072-token context in the recipe (262,144 in
      production). Gaudi software 1.24.1, lazy mode, vllm-gaudi `releases/v0.30.0`.
    - **Status:** serving production traffic in block FP8 at 32 sequences per replica; the cookbook recipe is the
      BF16 configuration on the public checkpoint.
    - **Headline:**
        - Full-depth fp32 CPU oracle: pooled NLL 1.6000 against 1.5991 over 4,487 tokens; every confident top-1
          matches.
        - Window-sized KV for the sliding layers: 5x the tokens, 2.1M to 2.3M tokens of KV per replica.
        - A 250K-token prompt in about 3 minutes instead of 7.7 (same pod, chained A/B).

## Why it needed a port

- Upstream vLLM added `RicursiveForCausalLM` in [vllm#65745](https://github.com/vllm-project/vllm/pull/65745), merged
  after v0.30.0. vllm-gaudi `releases/v0.30.0` has no model for it.
- The model's reasoning and tool parser (`ripple`) and a change to vLLM's DelegatingParser are in the same pull
  request.
- vllm-gaudi builds exact sliding-window prefill only for Gemma 4, and its decode window is one block short for a
  model without attention sinks.
- Every layer gets a full-length KV cache on vllm-gaudi, five times what Ripple needs at TP 4.

## What was built

### Model and parser backport

`ripple_hpu/model.py` is upstream's `ricursive.py` with the vllm-gaudi idioms of the other sliding-window ports:
everything 2-D (`ricursive` is not in vllm-gaudi's flatten list), one shared rotary module registered before the
layers (the runner refreshes cos and sin for the first rotary module only), `cache_config.sliding_window` cleared so a
NoPE layer cannot inherit the window, and every layer's window checked after construction. Q, K and the output gate
share one merged projection; V reads a different input when value residuals are on, so it keeps its own.

`ripple_hpu/parser.py` is upstream's `vllm/parser/ripple.py` with two JSON helpers v0.30.0 lacks, on v0.30.0's parser
engine. `patches/parser_streams_all.py` applies the pull request's four DelegatingParser edits: Ripple may open
`<think>` after content has started, and without the edit a later turn's reasoning streamed out as content.

### Exact sliding windows

Ripple's sliding layers see keys `q - 1024 < k <= q`. Three things in the runner stand between it and that mask:

| Path | Stock v0.30.0 | Fix |
| --- | --- | --- |
| Prefill with cached context | the whole context is gathered and masked, for every model but Gemma 4 | [swa-window-context](../fixes/swa-window-context.md) builds the window block list for every interleaved model |
| Prefill lower bound | keeps `window + 1` past keys | the port drops the extra key for `ricursive` |
| Decode | `window / block` blocks: up to one block short | [swa-decode-window](../fixes/swa-decode-window.md)'s exact head mask, turned on for `ricursive`; a fully masked oldest block is dropped instead (no sinks, so it would be NaN) |

`tests/test_runner_window.py` checks every query row of the patched runner's prefill and decode masks against the
reference set at positions around block and window boundaries, with Gemma 4 and stock decode as controls.

### Window-sized KV

With `RIPPLE_SWA_KV=1` the sliding layers report vLLM's `SlidingWindowSpec`, so the hybrid KV cache manager forms one
full-attention group and four sliding groups of 16 layers over one allocation, frees blocks that leave the window,
and applies the sliding-window prefix-cache rules. KV per token drops from 81,920 to 16,384 B per card. This is the
same module and runner hooks as the [Muse Spark Open port](muse-spark-open.md), for model type `ricursive`. Tiny and
10-layer parity, decode against prefill and prefix-cache repeats were identical with it on and off.

### Tool grammar for tool_choice auto

Upstream applies the tool grammar to `tool_choice` auto only when some tool is `strict`. Agent clients send
non-strict tools, and at temperature 1.0 Ripple sometimes calls a tool it was not offered. With
`RIPPLE_CONSTRAIN_AUTO_TOOLS=1` an auto request gets the grammar as if its tools were strict: a call names an offered
tool and follows its schema, text before a call stays free, and the grammar starts after reasoning.

| Temperature 1.0, port's first build | grammar on | off |
| --- | --- | --- |
| system prompt names a `glob` tool, only `list_files` offered, 6 samples | `list_files` 6/6 | `glob` 4/6 |
| an agentic tool-calling probe, 32 samples | 32/32 | 31/32 and 32/32 (two runs) |
| decode with tools offered, 8 streams x 512 tokens | 28.1 tok/s per stream | 29.4 |

### Long prompts

A long prompt is read in 4,096-token chunks, and each chunk's 16 full layers attend to every earlier token through
FusedSDPA with an explicit bias. Two things made it slow:

- **Context buckets.** The default exponential buckets padded every context from 128K to 246K up to 258K, so every
  chunk past 128K cost the same 11.4 s. `VLLM_BUCKETING_STRATEGY=pad` pads to the next step only.
- **fp32 softmax.** A profile of one chunk at 196K: device busy 11.1 s, attention 9.7 s, almost all of it the fp32
  softmax. `RIPPLE_FAST_SOFTMAX_MIN_KV=16384` switches to the fast softmax only for the full layers' calls with 16K
  or more keys. Fast softmax everywhere moved the full-depth oracle comparison past its limit (chat and tools mean
  |Δlogprob| 0.062, limit 0.05).

One full layer's prompt attention for a 4,096-row chunk, one card, TP 4 head shapes, ms per call, all within 0.0035 of
an exact fp32 attention:

| Context | fp32 softmax | fast softmax |
| --- | --- | --- |
| 16K | 26.4 | 9.1 |
| 64K | 83.5 | 27.3 |
| 128K | 160.7 | 51.8 |
| 258K | 328.7 | 103.9 |

Time to read a new turn on a cached context (warm minimum, seconds, same pod and image):

| New tokens | Context | Exponential buckets, fp32 | Pad buckets, fast softmax (served) |
| --- | --- | --- | --- |
| 256 | 128K | 0.91 | 0.34 |
| 1,024 | 128K | 3.46 | 0.65 |
| 1,024 | 246K | 3.54 | 1.07 |
| 4,096 | 246K | 11.43 | 4.22 |

vllm-gaudi's FusedSDPA slicer (on by default with pad buckets) was tried and is refused by the port: its
online-softmax merge returned NaN when a key slice was all bucket padding (prompts of 8,192 tokens or more gave NaN
prompt logprobs), it cut the sliding layers' window mask (NLL 0.65 to 2.95 past 4,096 tokens), and with both fixed it
was slower for short turns on a long cached context.

### Decode graph memory

With `VLLM_SKIP_WARMUP` every decode shape is captured as an HPU graph the first time it runs, and vllm-gaudi's graph
cache never drops one. Measured over 38 decode shapes at TP 4, each graph keeps

    0.312 MiB x blocks + 1.251 MiB x window blocks + 0.203 MiB x rows

(the fit is exact to 2 MiB), about 20 KiB per block per layer. One graph for 32 requests at 64K of context keeps
about 5.5 GiB. Nothing bounds the total, and running out looks like a hang: one TP rank fails a small allocation in
a graph replay ("PT_DEVMEM Allocation failed"), the other ranks wait in the next all-reduce, and 10 minutes later the
device reports "No progress" on every card ([platform findings](../results/platform-findings.md#a-device-allocation-failure-on-one-rank-looks-like-a-hang)).

`graph_budget.py` predicts each new decode graph's memory from its shape and drops least recently used decode graphs
first when the total would pass the budget (`auto`: free memory at the first decode minus 8 GiB). A shape whose
single graph is over the budget runs without a graph. The prediction depends only on the shape, so every rank drops
the same graphs. Dropping freed what the formula predicted (0.44 GiB predicted, 0.43 measured), a dropped shape is
captured again in about 1 s, and even a 3 GiB budget (69 graphs dropped in one benchmark) did not slow decode. The
generic [hpu-graph-lru](../fixes/hpu-graph-lru.md) fix bounds the number of graphs instead; Ripple's graphs range
from tens of MiB to many GiB, so a count does not bound memory here, and the two must not be combined.

## Correctness

Measured on the port's first build (TP 4 BF16, the same image release and model code; `VLLM_FP32_SOFTMAX=true`,
window-sized KV on):

- **Full model against a full-depth fp32 CPU oracle** (the checkpoint's own decoder layer class, one layer in memory
  at a time):

    | Prompt | Tokens | NLL engine | NLL oracle | Mean \|Δlogprob\| | Confident top-1 |
    | --- | --- | --- | --- | --- | --- |
    | a README | 300 | 1.2353 | 1.2457 | 0.026 | 274/274 |
    | a Python module | 1,500 | 1.4648 | 1.4584 | 0.022 | 1,259/1,259 |
    | a design document | 2,500 | 1.6459 | 1.6469 | 0.022 | 2,017/2,017 |
    | chat with tools | 191 | 2.6372 | 2.6368 | 0.043 | 159/159 |
    | pooled | 4,487 | 1.6000 | 1.5991 | | |

    With the long-prompt settings the pooled NLL is 1.5995; prompt logprobs over 128K tokens moved by at most 0.013
    mean |Δlogprob| and NLL +0.002.
- **10-layer real-weight checkpoint** (layers 0 to 9, both NoPE layers 4 and 9) against the HF code in fp32, TP 1 and
  TP 4: mean |Δlogprob| 0.013 to 0.016, confident top-1 1,849/1,849, greedy identical or split where the reference
  margin is under 0.03. Without fp32 softmax the distance doubles, so the recipe sets it.
- **Decode against prefill at long positions:** a 23,435-token sampled answer re-scored as a prompt, mean |Δlogprob|
  0.0035 to 0.0065 per position band up to 24K, flat with position.
- **Needles** 9/9 at 32K, 64K and 125K and 3/3 at 250K.

On the cookbook build (the port image from the stock image, the recipe's command at 131,072 tokens) every device gate
passed: smoke, determinism, the reasoning split, decode against prefill (0.0055), needles to 124.5K, a copy test at
129K, tools, isolation and stress; see the [recipe page](../recipes/ripple-65b-preview.md#verified). At 262,144 the
same build found the needles at 255K but failed a stricter copy test past 200K tokens (a 1,500-token passage restarted
at the end of the prompt diverged after 29 to 174 of 251 characters), with fast and fp32 softmax alike. The cause,
model or engine, is not settled, so the recipe stays at 131,072.

## Performance

| Test | Result | Measured as |
| --- | --- | --- |
| Decode, 1 / 4 / 16 streams (completions, 256 tokens, greedy) | 34.1 / 34.1 / 33.1 tok/s per stream; 530 tok/s aggregate at 16 | cookbook recipe, single run |
| 250K-token prompt | about 180 s (was about 462 s with exponential buckets and fp32 softmax) | port's first build, TP 4 BF16, same pod chained A/B; 166.7 s next to 15 decodes on the cookbook build at 262,144 |
| 14 streams decoding while two 250K prompts arrive | no errors; streams get one token per 4,096-token chunk (0.4 tok/s, largest gap 4.1 s) | port's first build |

The decode step is host-bound at about 33 ms: each replay hashes the graph's 469 inputs and launches 403 recipes, each
followed by an event sync, while the device is busy for 18.4 ms (BF16).

### FP8 (measured, not in the recipe)

The production configuration serves self-quantized FP8 weights, which are not public. The method: data-free
128 x 128 block quantization with fp32 `weight_scale_inv` of q, k, v, o, the output-gate projection (vLLM merges q,
k and the gate, so the three share a format) and the MLP; embedding, LM head, norms and the value-residual lambdas
stay BF16 (640 tensors, worst block-mean relative error 0.0225, 77 GB). vllm-gaudi serves it with its FP8 linear
method, which requantizes the blocks per output channel at load, with per-token dynamic activation scales. The port
accepts block FP8 and nothing else quantized.

| Check, TP 4 | BF16 | FP8 |
| --- | --- | --- |
| top-1 against BF16 greedy, 24 coding prompts (2,993 tokens) | 0.996 (self) | 0.9906 |
| fp32 oracle, pooled NLL over 4,487 tokens (oracle 1.5991) | 1.6000 | 1.6079 |
| needles at 32K / 125K / 250K | 3/3 each | 3/3 each |
| decode per stream at 1 / 8 / 16 streams | 30.8 / 25.5 / 28.4 | 27.5 / 29.6 / 28.2 |
| KV per replica | 2.33M tokens | 2.85M tokens |

FP8 first decoded at 20 tok/s against BF16's 30: vllm-gaudi's FP8 linear returns a view of its output, and the view,
all-reduced in place after o_proj and down_proj, split the decode graph around every collective (7 recipes per layer
instead of 3). `RIPPLE_FP8_VIEW_FIX` (on by default) returns the GEMM output itself for 2-D inputs; see
[platform findings](../results/platform-findings.md#a-block-fp8-linear-view-splits-the-decode-graph-at-tp-greater-than-1).
Both formats are host-bound at the same step time, so FP8 buys KV room, not per-stream speed. On an internal
reasoning and agentic readiness suite at temperature 1.0, FP8 at 32 sequences per replica scored 87.6 against BF16's
83.8 at 16, and a two-hour soak at 8 concurrent requests ran with no restart and no allocation failure.

### DFlash2 speculative decoding (not included)

Ricursive publishes a 4B DFlash2 drafter. A Gaudi implementation was built and measured: per-stream decode 30.8 to 96.8
tok/s at one stream and 28.4 to 44.5 at 16 (TP 4 BF16), parity with a reference drafter, needles kept. Under serving
load its verify steps captured a new HPU graph per block bucket, and the TP ranks hung (the decode graph memory above).
The graph budget bounds that, but long-context verify steps then run without graphs, and the hang was not
re-soaked with drafting on. It is left out of the port.

## What it taught about Gaudi

- **An allocation failure on one rank looks like a device hang.** Read every rank's log for "PT_DEVMEM" before
  blaming the cards.
- **HPU graph memory grows with the shapes served.** With `VLLM_SKIP_WARMUP` and pad buckets a server keeps capturing;
  bound it by size, not by count, when graph sizes span orders of magnitude.
- **Fast softmax is fine where it is selective.** Full layers at long context tolerate it; applied everywhere it
  failed the oracle.
- **The FusedSDPA slicer assumes a causal mask** and must never see a sliding window's mask.
- **A view before an in-place collective splits the graph.** Look at recipes per layer, not only at device time.

## Availability

The port is in this repository: [`ports/ripple-65b/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/ripple-65b),
with the recipe [Ripple-65B-Preview](../recipes/ripple-65b-preview.md). It builds from a stock `releases/v0.30.0`
image with `ports/ripple-65b/Dockerfile` and serves the public checkpoint as published. It contains the model plugin,
the backported parser and its vLLM patch, the runner patches above, the chat template, CPU tests, build gates, and
device gates (`tests/hpu/run_gates.sh`) that run the smoke, determinism, reasoning, decode-vs-prefill, needle, copy,
tool, isolation and stress checks against a running server.

The cookbook build differs from the first build in where the generic runner fixes live: the fixes the port had
carried itself (prompt order, one logit per resumed prefill, int32 logprob ids, chunked prompt logprobs, the prefill
window gate) now come from the fixes pack, and the grammar bitmask stays on vllm-gaudi's host path.
