# fp8-scale-lut

Source: [`fixes/fp8-scale-lut/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/fp8-scale-lut)

Halve FP8 weights for the Gaudi 2 scale adjustment with a 256-entry byte table instead of a float round
trip on the CPU. Bit-identical to the stock result; shortens the load of per-expert FP8 checkpoints. Opt-in.

| | |
| --- | --- |
| Affects | FP8 checkpoints loaded on Gaudi 2 with scale adjustment on (the default), most of all per-expert MoE checkpoints with many weight tensors (performance) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/extension/ops.py` |
| Upstream | not filed yet |
| Test | `test_fp8_scale_lut.py` (source tests need a release's sources; the bit-identity test needs torch) |

## Problem

Gaudi 2 has a narrower FP8 E4M3 range than the OCP format, so vllm-gaudi halves every FP8 weight and doubles
its scale as it loads (`gaudi_weight_wrapper`):

```python
loaded_weight = (loaded_weight.float() * 0.5).to(torch.float8_e4m3fn)
```

The tensor is still on the CPU at that point. PyTorch's CPU float8 casts are slow, and for a per-expert MoE
checkpoint the wrapper runs once per expert slice, so this line dominates the load.

## Symptom

Loading an FP8 MoE checkpoint takes many times longer than reading it, and a Python stack sample of the
loading worker sits in the `wrapper` of `vllm_gaudi/extension/ops.py` under the fused-MoE weight loader. A
long load can also run past the engine's start-up timeout.

## Who is affected

FP8 checkpoints on Gaudi 2 with scale adjustment on. Checkpoints with few, large weight tensors are less
affected than per-expert MoE checkpoints. If the load is also slow because of CPU throttling in a container,
see [load-threads-cgroup](load-threads-cgroup.md) first; the two fixes address different costs.

## The fix

An FP8 value has 256 bit patterns, so halving is a byte-to-byte map. The table is built once per device by
running the stock expression itself on all 256 bytes, and halving a weight becomes one gather through it.
The result is bit-identical to the stock line for every input, NaN and subnormal rounding included. Scales
(the non-FP8 branch) are unchanged.

## Measured

CohereLabs North-Mini-Code-1.0-fp8 (compressed-tensors per-channel FP8, 128 experts in each of 48 MoE layers)
on one Gaudi 2 card, `releases/v0.30.0`: about 15 minutes per 4.5 GB shard and 1.5 hours for the whole load
with the stock line, with every stack sample in the halving. The load time with only this fix was not recorded
separately (the same deployment also got load-threads-cgroup); the bit-identity is checked by the test on all
256 byte values.

## Verify

```bash
python3 fixes/fp8-scale-lut/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/fp8-scale-lut   # with torch for the bit-identity test
```

On a device: load the same FP8 checkpoint with and without the fix and compare the load time; greedy output
and logprobs must be identical, since the weights are bit-identical.
