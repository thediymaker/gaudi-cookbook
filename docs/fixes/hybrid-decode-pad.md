# hybrid-decode-pad

Source: [`fixes/hybrid-decode-pad/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/hybrid-decode-pad)

Keep decode padding out of live KV blocks on hybrid models whose attention KV block is split into
smaller kernel blocks.

| | |
| --- | --- |
| Affects | hybrid models (GDN, linear attention or Mamba next to attention), e.g. Qwen3.5 / Qwen3-Next class |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | [vllm-gaudi#1912](https://github.com/vllm-project/vllm-gaudi/pull/1912) (open, same fix in the runner) |
| Test | `test_hybrid_decode_pad.py` (torch; the source tests need a runner source) |

## Problem

On a hybrid model vLLM sizes the attention KV block so that one attention page holds one recurrent
state, for example 512, 640, 896 or 1152 tokens. The HPU attention kernel works in smaller kernel
blocks (128 tokens), so `initialize_kv_cache` splits every KV block into `k` kernel blocks,
allocates `num_blocks * k + 1` kernel blocks (the last one is a spare), and the attention block
tables hold kernel-block ids. The pad ids are still set from the KV-block count
(`_PAD_BLOCK_ID = num_blocks`). Kernel block `num_blocks` is not the spare: it lies inside KV
block `num_blocks // k`, a live block of the pool.

## Symptom

Decode batches are padded up to bucket sizes, so every decode batch whose size is not a bucket
size writes the padding rows' K/V over a few context tokens of whichever request owns that block,
in every attention layer. That request silently loses part of its context (for example one digit
of a long number read back wrong), and with prefix caching a cached copy of the block is corrupted
for later hits too. Nothing is logged.

## Who is affected

Hybrid models whose attention KV block size is not a supported kernel block size. Models whose KV
block equals the kernel block (every non-hybrid model, and hybrids whose block size happens to be a
kernel size) are not affected: there `num_blocks` is the spare block.

## The fix

The fix wraps `initialize_kv_cache`. When an attention group's KV block is split into kernel
blocks, the pad ids move to the last kernel block of the attention tensors: the spare block after
the pool with block splitting, or the last kernel block of the spare KV block with the shared
layout. Reads of padding rows then hit the spare block as well. Unsplit layouts are left alone (one
log line). KV-cache initialisation fails with `RuntimeError` if the layout is split but the
attention tensors do not end in a spare kernel block outside the pool. The Mamba pad is not
touched. The upstream pull request makes the same change inside `initialize_kv_cache` and also
fixes the speculative decoding pad slots; when it is present, `apply.py` reports "not needed".

## Verify

```bash
python3 fixes/hybrid-decode-pad/apply.py --check   # 0 = applied or not needed, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=/path/to/site-packages pytest fixes/hybrid-decode-pad
```

At startup the server logs where the decode pad moved. End to end: on a hybrid model, keep three
(or five, seven) requests decoding together while one of them holds a long context with a
distinctive string, then ask for the string back. Without the fix the answer is occasionally wrong
in a few characters.
