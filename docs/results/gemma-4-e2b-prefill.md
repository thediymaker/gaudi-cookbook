# Gemma-4-E2B fast prefill on Gaudi 2

Gemma-4-E2B has 35 decoder layers, and the last 20 write no KV cache of their own: they read the cache of an
earlier layer. During prefill, only the position that gets sampled needs the output of those layers, so they can
run on one token per prompt instead of every token. Upstream vLLM has this as `--kv-sharing-fast-prefill`, and the
still-open [vllm#60445](https://github.com/vllm-project/vllm/pull/60445) turns it on by default for E2B and E4B.
vllm-gaudi never wired it up. This page describes an HPU port and what it does to prefill throughput on one card.

!!! summary "At a glance"
    - **Checkpoint:** [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), Apache 2.0.
    - **Hardware:** one Gaudi 2 card, BF16, vllm-gaudi releases/v0.30.0 (`058f107`), Gaudi software 1.24.1,
      lazy mode.
    - **Status:** study. Not serving traffic.
    - **Headline:** prefill-heavy throughput 72.8k to 169.7k tok/s in the matrix run, 176.5k at the sweep peak
      (same card, stock vs fast prefill, both at prompt batch cap 8). Token-exact against stock at prompt batch
      cap 1.

## Why it needed a port

In E2B, layers 15 to 34 are KV-shared: they read layer 13's cache (sliding window) or layer 14's (full attention)
and carry the double-wide MLPs. Nothing later reads their output at a prompt token except where a token is
sampled. If only the last token of each prompt runs layers 15 to 34, prompt FLOPs drop by about 3x.

vllm-gaudi v0.26.0 and v0.30.0 do not use this. `HpuGemma4Model.forward` calls upstream's
`fast_prefill_forward`, which expects GPU per-layer attention metadata and `KVSharingFastPrefillMetadata`, and
`kv_sharing_fast_prefill_eligible_layers` is never filled. With the flag on, the HPU model either crashes or
still runs all 35 layers.

## What was built

**Model.** On a prompt batch with no cached context:

1. Run layers 0 to 14 on every token.
2. Gather each row's last real token at `row * padded_len + query_len - 1`, which is the index the runner already
   uses for logits.
3. Run layers 15 to 34 on those `[batch]` tokens only, then scatter the results back.

Decode steps and prompts with context (prefix-cache hits, later chunks of a long prompt) take the normal path.

**Attention.** On a KV-shared layer during fast prefill, each sequence runs one query against its prompt K/V,
read from the target layer's cache at the prompt's slots, with an fp32 softmax. Sliding layers use the mask
`p - j < 512`, as in the Hugging Face model.

**Rotary embedding.** `HPURotaryEmbedding` prepares cos/sin for the whole batch once per forward. The gathered
tokens need tables for their own positions, so the patch re-prepares cos/sin for the gathered positions and
drops them afterwards.

**Limits** are the upstream flag's: no `prompt_logprobs`, no EAGLE, no FP8 KV cache.

## Correctness

| Gate | Result |
| --- | --- |
| Stock vs fast prefill, 64 wikitext prompts x 1024 tokens, prompt batch cap 1 | top-1 64/64; greedy 32 tokens identical 64/64 |
| Against a Hugging Face fp32 CPU reference, 16 prompts | top-1 16/16 for both; mean absolute logprob difference over the top 20: 0.20 fast, 0.23 stock |
| Served, 64 code-generation prompts, about 640 output tokens, prompt batch cap 1 | stock vs fast 64/64 character-identical, both rounds |
| Stock vs stock with prompt batch cap 8 (no patch) | 5/64 identical |
| Stock with cap 8, round to round | 60/64 identical |
| Stock with cap 8 vs fast with cap 8 | 62/64 identical |

Fast prefill itself is token-exact. Prompt batching, a stock knob, is not: putting several prompts in one forward
changes the bf16 rounding of the prompt K/V, and long greedy outputs drift mid-generation (never in the first 40
characters). Outputs then also depend on which prompts arrive together.

## Performance

Benchmark method of vllm#60445: offline `LLM.generate`, `bs` prompts of random tokens, scored as
`bs * (input + output) / batch latency`. Prefill-heavy is 1024 input tokens, 1 output token.

`VLLM_PROMPT_BS_BUCKET_MAX` sets how many prompts go into one prefill forward. It defaults to 1.

| Prefill-heavy, bs 64 | Stock | Fast prefill | Measured as |
| --- | --- | --- | --- |
| prompt batch cap 1 | 60.5k tok/s | 108.6k | same card, same harness |
| cap 2 | | 137.8k | same card |
| cap 4 | | 164.1k | same card |
| cap 8 | 72.8k | 176.5k (sweep), 169.7k (matrix run) | same card; two runs of the fast arm |
| cap 16 | | 172.1k | same card |

Matrix run at prompt batch cap 8 (same card, stock vs fast):

| Case | Stock | Fast prefill |
| --- | --- | --- |
| prefill-heavy, bs 16 | 74.8k | 157.7k |
| prefill-heavy, bs 64 | 72.8k | 169.7k |
| balanced, bs 64 | 73.8k | 149.2k |
| decode-heavy, bs 64 | 46.0k | 64.9k |
| long context, bs 16 | 32.1k | 80.3k |

Served, 64 concurrent code-generation requests with about 600-token prompts, warmed up: first-token median
1.28 to 1.34 s stock, 0.91 to 0.93 s with fast prefill, 0.66 to 0.82 s with fast prefill and cap 8 (same card,
two rounds per mode). Wall time barely moves (9.1 to 9.9 s vs 8.8 to 8.9 s) because writing code is decode-bound,
and decode is unchanged.

### Against the published A100 figures

Google published figures for this benchmark with vllm#60445 on one A100. They are cross-hardware and not
measured here, so treat any comparison as indicative.

| Case | Gaudi 2, fast prefill (measured here) | A100 (published) | Batch matched? |
| --- | --- | --- | --- |
| prefill-heavy, bs 16 | 157.7k | 81.1k | yes |
| prefill-heavy, bs 64 | 169.7k (matrix), 176.5k (sweep) | 124k headline | no |
| balanced | 149.2k at bs 64 | 62.9k at bs 16 | no |
| long context | 80.3k at bs 16 | 68.7k at bs 4 | no |

Only the bs 16 prefill-heavy row compares like with like. The 176.5k figure also depends on prompt batch cap 8,
which changes outputs (62/64 identical to stock at the same cap); at cap 1, where fast prefill is token-exact, the
figure is 108.6k.

## Serving notes

- **Warm up.** Without warm-up, decode shapes compile mid-run and erase the TTFT gain. vllm-gaudi treats any
  non-empty `VLLM_SKIP_WARMUP` as true, so to warm up, unset it; do not set it to `false`.
- **Memory.** Stock with prompt batch cap 8 runs out of device memory during warm-up at
  `--gpu-memory-utilization 0.9`. Fast prefill does not, because it skips the shared layers' prompt activations.
- **Decode** is the remaining lever. Gemma 4 MTP
  ([vllm#60689](https://github.com/vllm-project/vllm/pull/60689)) needs an HPU port: the vllm-gaudi runner only
  knows n-gram and EAGLE drafting, and the proposer's cross-model KV sharing has no HPU path.
- **Prefix caching** falls back to the full path on a hit. Extending the one-query attention to cached context is
  possible but not done.

## Availability

The integration code is not published in this repository yet. It patches the vllm-gaudi Gemma 4 model and HPU
attention backend, and depends on upstream vLLM's KV-sharing model code and on the design of vllm#60445, which
is still open.
