# swa-decode-window

Make the sliding window exact in paged decode for gpt-oss, instead of 129 to 256 visible tokens
for a 128-token window.

| | |
| --- | --- |
| Affects | gpt-oss (20B and 120B) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | not filed yet |
| Test | `test_swa_decode_window.py` (CPU, needs torch) |

## Problem

gpt-oss sliding layers attend to the last 128 tokens, the current one included (keys `k` with
`k > q - 128`). For those layers the HPU decode path hands the sequence's last
`window // block_size + 1` KV blocks to attention (two blocks at block size 128) and leaves every
filled slot of both visible. A sliding layer therefore sees between 129 and 256 tokens, depending
on where the current token sits in its block.

## Symptom

Decode drifts from the reference model as soon as a sequence is longer than the window: slightly
worse long-generation quality and tool-call accuracy, and prefill (which masks the window exactly)
disagrees with decode. No error is raised.

## Who is affected

gpt-oss on HPU, for every decode step past the first window. Other sliding-window models take
`window // block_size` blocks in decode and are not changed by this fix.

## The fix

When the window is a multiple of the block size and a sequence's window block list has full
length, the oldest block's usage is stored as `-lbu`, where `lbu` is the usage of the sequence's
last block. The decode bias reads a negative usage `-u` as "slots `[u, block_size)` are valid".
Those are exactly the window positions inside the oldest block, so the visible set is
`(q - window, q]`. Positive usages build the same mask as before, so every other model and path is
unchanged. When the last block is full, the oldest block is masked completely; gpt-oss always has
sinks, so its softmax stays finite.

Validated on hardware with v0.26.0. On v0.30.0 the same edit applies (the window call also passes
`force_non_contiguous=True`); the logic is identical.

## Verify

```bash
python3 fixes/swa-decode-window/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/swa-decode-window
```

The test runs the inserted usage and bias code on CPU and checks that the visible positions equal
the window for every decode position, at block sizes 64 and 128.
