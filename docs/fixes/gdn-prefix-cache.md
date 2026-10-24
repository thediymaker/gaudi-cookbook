# gdn-prefix-cache

Source: [`fixes/gdn-prefix-cache/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/gdn-prefix-cache)

Make prefix caching correct on Gated DeltaNet hybrids (Qwen3.5, Qwen3.6, Qwen3-Next class) while
keeping the full KV pool: each cached block's GDN state is stored as a checkpoint inside the
attention pages of that block. Opt-in; it does nothing unless prefix caching is on.

| | |
| --- | --- |
| Affects | GDN hybrids served with `--enable-prefix-caching` (compact GDN, the default) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py`, `vllm/v1/core/sched/scheduler.py`, `vllm_gaudi/v1/core/sched/hpu_async_scheduler.py` |
| Upstream | not filed (design discussion). Upstream's own path keeps per-block state tensors with compact GDN off: [vllm-gaudi#1827](https://github.com/vllm-project/vllm-gaudi/pull/1827) and [#1842](https://github.com/vllm-project/vllm-gaudi/pull/1842) in the v0.30.0 tag (not in `releases/v0.30.0` at `058f107`), ported to main as [#1831](https://github.com/vllm-project/vllm-gaudi/pull/1831) and [#1853](https://github.com/vllm-project/vllm-gaudi/pull/1853), with [#1828](https://github.com/vllm-project/vllm-gaudi/pull/1828), [#1890](https://github.com/vllm-project/vllm-gaudi/pull/1890) and [#1895](https://github.com/vllm-project/vllm-gaudi/pull/1895) |
| Test | `test_gdn_prefix_cache.py` (torch on CPU; the source tests need a release's sources) |

## Problem

With `--enable-prefix-caching` vLLM puts GDN layers in Mamba "align" mode. Every KV cache group
uses one block size `B` (896 tokens on the 27B models, 1,152 on Qwen3.6-35B-A3B, 640 on
Qwen3.5-397B-A17B with expert parallelism), prefill chunks end on multiples of `B`, and a full block
must keep the recurrent state after its last token, so that a hit at `k x B` tokens can resume from
block `k - 1`. The stock releases break this in four places:

1. **Compact GDN is the default.** One state slot per request, never written into a KV block, so a
   hit resumes from whatever the slot's previous owner left. The runner only logs "Compact GDN mode
   does not support prefix caching".
2. **The block size flips to 128 after start-up.** vllm-gaudi resets the live
   `cache_config.block_size` after the KV cache and the scheduler were built with `B`, so chunks end
   on 128-token multiples. On v0.30.0 the HPU async scheduler (the default) also aligns GDN chunks to
   the 128-token Mamba chunk.
3. **The prefill caching branch is Mamba2's** and asserts on a prefill batch with several requests.
4. **Nothing clears stale values.** New attention blocks are not zeroed (vLLM lists them in
   `new_block_ids_to_zero`; vllm-gaudi ignores the field), and the GDN layers clear a new request's
   state by multiplying it with 0, so a NaN left in a slot or block stays (`0 x NaN = NaN`).

With compact GDN off (`VLLM_COMPACT_GDN=0`) the stock releases also load and store the state of the
wrong block, so neither layout is correct.

## Symptom

Greedy text on a cache hit often still matches, because a GDN layer forgives a stale state
surprisingly often, but logprobs drift, two hits of the same prompt disagree, and answers that
depend on the cached part of a long prompt are occasionally wrong. Cold requests are unaffected.
That is why the cookbook's GDN recipes ran with prefix caching off on the stock releases.

## Who is affected

Anyone who turns prefix caching on for a GDN hybrid on these releases. Agent and chat traffic,
which re-sends the whole conversation every turn, gains the most from caching: on these models a
later turn otherwise prefills its whole prompt again.

## The fix

The GDN layers keep vllm-gaudi's compact per-request slots, and the state each cached block needs
is stored as a checkpoint in the attention K/V pages of that block. A KV block belongs to one cache
group at a time, so the attention rows of a block owned by a GDN group are unused, and the page is
already sized so that one GDN state fits (that is what makes `B` large). Upstream vLLM shares one
buffer for this; vllm-gaudi cannot, because torch.compile rejects writes through views that change
dtype, so the checkpoints are moved as bit-exact int32 row copies outside the compiled graphs:

- after a step whose scheduled tokens end on a block boundary, each request's slot (conv and SSM
  state, every GDN layer) is copied into that block;
- before a step that starts from a hit, or from a request resumed after preemption, the block is
  copied back into the slot;
- a preempted request hands its slot back, so preemption cannot run the slots dry;
- new attention blocks, new requests' slots, the spare kernel block and the two reserved slots are
  zeroed, so a non-finite value cannot outlive the step that produced it.

The scheduler aligns chunks to `B` taken from the KV cache config, the HPU async scheduler defers to
that split, and the Granite caching branch is skipped for GDN. The KV pool keeps its size: the
non-compact layout scales it by attention / (attention + GDN), about 0.53 on the 27B models.

It refuses to serve (`RuntimeError` at KV cache initialisation) rather than guess when the layout
does not fit: `VLLM_COMPACT_GDN=0`, speculative decoding, the KV defragmenter, KV cache groups of
different block sizes, the opt-in Mamba prefix features of vLLM (partial-block hits, prefill
checkpoint blocks, fine-grained junction caching), a resume that is not on a block boundary, or
decode padding inside the pool. It needs [hybrid-decode-pad](hybrid-decode-pad.md) applied
first (`apply_all.py` keeps the order; the script checks). Apply
[prompt-order](prompt-order.md) as well (it is in both profiles): it orders a step's
prompts so a cache hit never reads a block before its writer has computed it.

```bash
python3 fixes/apply_all.py --profile v0.26.0 gdn-prefix-cache || exit 1
export VLLM_COMPACT_GDN=1     # the default; the fix needs the compact slots
vllm serve <model> --enable-prefix-caching ...
```

Limits:

- Prompts shorter than one block never hit, and a hit is rounded down to a whole block.
- Prefill chunks end on block boundaries, so a cold prompt runs `B`-aligned chunks (for example 896
  of a 1,024-token budget). Choose `--max-num-batched-tokens` as a multiple of `B` to lose nothing.
- Caching pays only if a session comes back to the same replica.
- Upstream's non-compact path on vllm-gaudi main is the other option; it needs no runtime patch on
  an image that has it, at about half the KV pool on these models.

## Verify

```bash
python3 fixes/gdn-prefix-cache/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=/path/to/site-packages pytest fixes/gdn-prefix-cache
```

At start-up the engine logs `[gaudi-fix:gdn-prefix-cache] active: GDN groups ..., block <B> tokens`
and the checkpoint layout. End to end, send the same long prompt twice with greedy sampling and
`logprobs`, once with its own `cache_salt` (cold) and once after a request that shares its prefix
(hit): the outputs and logprobs should be identical, and `usage.prompt_tokens_details.cached_tokens`
should be a multiple of `B`. The [write-up](../results/gdn-prefix-caching.md) lists the full
gates and measurements; each recipe page lists what was run on it.
