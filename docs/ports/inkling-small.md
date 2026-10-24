# Inkling-Small on Gaudi 2

Inkling-Small is a 276B-parameter mixture-of-experts model (12B active) from Thinking Machines. It has no RoPE:
attention adds a learned relative-position logit per head, and four short causal convolutions per layer carry
state between positions. vLLM's Inkling model runs only on CUDA and ROCm, and its layers are not attention modules
the HPU runner can give a KV cache. This port runs the published BF16 checkpoint, unconverted, text only, on one
Gaudi 2 node at 256K tokens per request.

!!! summary "At a glance"
    - **Checkpoint:** [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small),
      Apache-2.0 (the model card links an acceptable use policy). BF16, 532 GB, loaded as published. Vision, audio
      and the MTP draft head are not built.
    - **Hardware:** 8 Gaudi 2 cards, one node, TP 8. 262,144 tokens per request, 16 sequences, a KV pool of
      2,560 blocks (327,680 tokens).
    - **Stack:** Gaudi software 1.24.1, vllm-gaudi `releases/v0.30.0` (`058f107`, vLLM `ced6857a`), lazy mode.
    - **Code:** [`ports/inkling-small/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/inkling-small),
      recipe [Inkling-Small](../recipes/inkling-small.md).
    - **Status:** serving production traffic.

## Why it needed a port

- vLLM's Inkling implementation uses Triton short-conv and QKVR kernels, FA4 relative attention and CUDA events in
  the MoE. None of it runs on HPU.
- Its layers are custom modules, not `Attention`, so the HPU runner would allocate no paged KV for them.
- The relative logit `rel[i, h, i - j]` is a per-query, per-distance term. HPU attention has no hook for it except an
  explicit additive mask in prefill and the ALiBi position-bias slot in paged decode.
- The short-conv state must survive chunked prefill and prefix-cache hits, which the HPU runner only supports for
  models it recognises as Mamba-like.
- vLLM's inkling parsers wrap a Rust extension that a Gaudi build of vLLM does not compile: the server starts and
  the first chat request fails.

## What was built

42 layers, hidden 4096, 32 query and 8 KV heads of size 128. 35 layers attend within a 512-token window
(rel extent 512), 7 are global (rel extent 1024, with a log-position scale on q and the rel logits beyond 128K).
256 routed experts top 6 plus 2 shared experts that every token uses, sigmoid selection with a bias, layers 0 and 1
dense. Logits are divided by 16 (muP).

### Attention with a relative bias

Each layer computes `rel = r @ rel_logits_proj` from the R branch of a fused QKVR projection and stashes it for its
`Attention` call. A subclass of the HPU attention implementation adds it:

- **Prefill:** an additive FusedSDPA mask `[B, H, Lq, KV]` that also carries the causal, window and padding masks
  (FusedSDPA refuses `valid_seq_lengths` with an explicit mask). Later chunks and prefix-cache hits prepend the
  cached context exactly as vllm-gaudi's `_get_context` lays it out, and the query rows are processed in
  sub-blocks so the mask stays under a byte budget, with a graph break per sub-block; without the break, 42 layers
  of multi-GB masks accumulate into one lazy graph and run out of device memory near 100K tokens.
- **Decode:** a per-block bias `[blocks, H, 128]` in the ALiBi slot of paged attention. Key positions come from each
  block's ordinal in its sequence's (possibly window-truncated) block list, computed with a membership mask because
  `scatter_reduce` has no HPU kernel.
- **Local-window prefill:** local layers fetch and bias only the last 5 context blocks instead of the whole context.
  A CPU test proves the attention output identical to the full-context path over 40 random cases.

### Short-conv state without a KV group

Each layer has four depthwise convolutions (kernel 4): on raw K and V, and on the TP-summed attention and MLP
deltas. Each needs the raw inputs of the previous 3 positions. They are stored as one snapshot row per attention
block id: slot `b` holds the state after the last position processed in block `b`. A full block carries the state at
its end, which is exactly what a prefix-cache hit on it needs; requests sharing a cached prefix read the same
snapshot; a request's partial last block holds its running state.

An earlier layout declared the state as a `MambaSpec` KV layer. vLLM's hybrid KV manager then gave it its own block
per 128 tokens from the same pool, which the model never reads. Half the pool was unusable, and because admission
needs the whole sequence to fit, a prompt over half the pool waited forever with no error. Now the attention page
is padded by the conv bytes (65,536 + 50,688 B per block per layer at TP 8), so vLLM's block count still budgets the
conv memory, and the port allocates the rows right after the runner allocates the attention cache. The scheduler sees
one full-attention group: every block is usable, and a prompt is either admissible or refused with HTTP 400. The
server refuses to start if the pool cannot hold one `--max-model-len` request.

### Decode window

The vllm-gaudi runner gives interleaved sliding-window layers `window / block_size` decode blocks and adds one only
for gpt-oss, so Inkling's local layers saw 385 to 512 of their 512 keys in decode while prefill saw 512. The port's
runner patch adds the block for this model; the rel-bias mask cuts the window at 512 exactly. Measured on an earlier
build of the port: the mean gap between decode and teacher-forced prefill logprobs went from 0.089 to 0.034
(chained A/Bs: two loads of the same deployment, one variable).

### MoE and parsers

Routing is computed in the model (sigmoid plus bias for selection, weights renormalised over the 6 routed and 2
shared experts) and handed to vLLM's fused MoE through a custom routing function. With `INKLING_MOE_SINGLE_REDUCE=1`
routed and shared outputs are summed per rank and all-reduced once per layer; the model refuses to start if FusedMoE
does not actually skip its own reduce, since a second reduce would scale the output by the TP size.

The `inkling_py` parsers are pure Python. The reasoning parser finds the end of a thinking block by scanning token
ids for the newest block marker; a fixed decode window missed the opener of long thinking blocks and leaked
`<|end_message|>` into content.

## Correctness

Gates in [`tests/hpu/run_gates.sh`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/inkling-small/tests/hpu):
plugin and KV layout checks, smoke (arithmetic, tool call, JSON mode, `tool_choice` required), greedy repeats,
reasoning and streaming splits without markers, chunk-boundary needles, needles to 128K, a 200-request logprobs
soak, prefix-cache hit against cold output, 16 concurrent needles and the HTTP 400 for over-long prompts, plus a
long tier with 200K and 250K needles, three 100K prompts resident at once and the long-prefill share.

The prefix-cache gate does not ask for byte-identical output. A hit prefills only the uncached tail, so its chunk
shapes differ from the cold run, and BF16 rounding can flip a near-tie later in the output; two cold runs of the
same 8 concurrent prompts also differ, from batch composition alone. The gate requires that every hit answers
correctly and that any difference starts where the cold run's top two tokens were within 0.5 nats. A wrong conv
state or block table breaks the answer itself.

Both tiers with this repository's `install.sh` on a stock `releases/v0.30.0` image, launched as the recipe says:
every gate passed (results on the [recipe page](../recipes/inkling-small.md#verified)).

Recorded earlier on the deployed build of this port (the same model code, installed by an earlier bootstrap):

| Check | Result |
| --- | --- |
| Greedy output vs the previous release's port, 16 prompts x 192 tokens with top-5 logprobs | 16/16 identical, logprobs identical |
| GSM8K, first 50 | 47/50 |
| Needles at 128K, 200K and 250K, depths 0.1 / 0.5 / 0.9 | 9/9 |
| 32 random words copied verbatim from 250K tokens | 32/32 |
| Conv-state simulator: chunked prefill, prefix hits, shared prefixes, padding, block-boundary decode | 20/20 |

## Performance

All on one 8-card node, BF16, warm graphs.

| What | Result | Measured as |
| --- | --- | --- |
| Per-stream decode at 1 / 4 / 8 / 16 streams, port on releases/v0.30.0 | 43.9 / 38.8 / 31.2 / 21.7 tok/s | single run, production build |
| Same, port on v0.26.0 | 40.5 / 35.9 / 28.9 / 20.7 tok/s | different conditions: same harness and day, a different pod |
| `INKLING_MOE_SINGLE_REDUCE` + `INKLING_LOCAL_WINDOW_PREFILL` | 1 stream 32 to 25 ms per token; 16 concurrent 121K prompts 2,701 to 927 s | chained A/Bs on one node, v0.26.0 build |
| `INKLING_STEP_HOIST` | 1 stream 25 to 22 ms per token; 16 streams 202 to 248 tok/s aggregate | chained A/Bs on one node, v0.26.0 build |
| 16 streams decoding while a 250K prompt prefills (long-prefill share 0.5) | streams 30.8 tok/s before, 34.0 during; short requests start in 0.9 to 2.1 s | single run, production build |
| Cold TTFT at 32K / 128K / 243K | 9.7 / 46 / 138 s (first touch of a new length adds up to 14 s) | single run, production build |

The decode step is bound by op count and collectives rather than bandwidth: about 10.8K ops and 84 to 124
all-reduces per step at one stream, of which the conv streams and rel-bias index math are the largest share. That
is what the step hoist and the single reduce remove.

## What it taught about Gaudi

- An additive mask is the only way to give FusedSDPA a per-query term, and it cannot be combined with
  `valid_seq_lengths`; the mask has to carry the padding itself.
- Lazy mode accumulates everything until a graph break. Large per-layer transients need an explicit `mark_step`.
- The HPU runner keeps one block table for every layer. A second KV group (Mamba-style state) silently gets the
  wrong block ids or costs scheduler blocks the model never reads; padding the attention page is the safe way to
  budget side state.

## Availability

The port is published in [`ports/inkling-small/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/ports/inkling-small):
plugin, runner patch, CPU tests and device gates. It installs into a stock `releases/v0.30.0` image with
`install.sh`; the generic fixes it needs come from the [fixes pack](../fixes/index.md).
