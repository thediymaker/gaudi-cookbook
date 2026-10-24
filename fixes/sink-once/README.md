# sink-once

Count attention sinks once per sequence in paged decode, and apply them in FusedSDPA prefill on
v0.26.0.

| | |
| --- | --- |
| Affects | models with attention sinks (for example gpt-oss) |
| Releases | v0.26.0 (decode and prefill), v0.30.0 (decode) |
| Patches | `vllm_gaudi/extension/ops.py` |
| Upstream | [vllm-gaudi#1920](https://github.com/vllm-project/vllm-gaudi/pull/1920) (open, decode part) |
| Test | `test_sink_once.py` (needs a Gaudi device) |

## Problem

An attention sink is one extra logit per head that joins the softmax denominator:
`out = sum_j exp(s_j) v_j / (sum_j exp(s_j) + exp(sink))`.

Decode (`flat_pa` and `pipelined_pa`) computes the softmax per KV block and merges the blocks of a
sequence afterwards. `pipelined_pa` adds `exp(sink - block_max)` to every block's partial sum, so
after the merge a sequence that spans n blocks has n copies of the sink in its denominator instead
of one. At block size 128, a 10K-token context gets about 80 copies.

Prefill: FusedSDPA on Gaudi software 1.24.1 accepts a `sink` argument and does not apply it, so a
prompt is attended as if the model had no sinks.

## Symptom

Heads with a sink attend more weakly as the context grows. Short prompts look fine; long-context
retrieval, tool-call JSON validity and instruction following degrade with context length. Prefill
and decode also disagree with each other.

## Who is affected

Models whose attention layers have sinks (gpt-oss and others that declare `sinks`). Models without
sinks never reach the edited code. The Gaudi 3 fused block-softmax kernels are a different path
and are not changed.

## The fix

Decode: each block gets `sink - log(n)`, with n the number of blocks of the block's sequence, so
the n per-block terms add up to exactly one `exp(sink)` after the merge. Sequences that fit in one
block are unchanged. Padding blocks map to no sequence, get n = 1 and are dropped by the merge as
before. This is the change in the upstream pull request.

Prefill (v0.26.0 only): FusedSDPA gets one extra key whose logit is the sink. The head dimension
grows by one (padded to a multiple of 64), the query carries `sink / scale` there, the extra key
carries 1, and its value is zero. Real keys are untouched, so the result is exact up to the
kernel's own precision, on the same kernel. Calls that pass `window_size` keep the stock path.

On v0.30.0 the FusedSDPA call sits inside a query-tiling helper. The prefill part has not been
validated on that layout, so the fix applies only the decode part there.

## Verify

```bash
python3 fixes/sink-once/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/sink-once                     # on a Gaudi machine
```

The test applies the fix to a temporary copy of `ops.py` and compares stock and patched attention
against an fp32 reference with a sink, for decode over 1 to 40 blocks and (on v0.26.0) prefill.
