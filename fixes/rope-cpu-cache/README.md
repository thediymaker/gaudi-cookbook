# rope-cpu-cache

Build rotary-embedding cos/sin tables on the CPU, because Gaudi fp32 cos/sin of large arguments
are inaccurate.

| | |
| --- | --- |
| Affects | RoPE models served with long contexts (above about 128K positions) |
| Releases | v0.26.0 (v0.30.0 already contains the upstream fix) |
| Patches | `vllm/model_executor/layers/rotary_embedding/base.py` |
| Upstream | [vllm-gaudi#1816](https://github.com/vllm-project/vllm-gaudi/pull/1816) (merged, in v0.30.0) |
| Test | `test_rope_cpu_cache.py` (CPU part needs torch; table check needs a Gaudi device) |

## Problem

vLLM computes each RoPE table (`_compute_cos_sin_cache`) on the device the model is built on. On
Gaudi, fp32 `cos` and `sin` of large arguments are inaccurate. The fastest rotary frequency has
inverse frequency 1, so its angle is the position itself in radians, and past roughly 1e5 the
device result is effectively random. Every other frequency stays accurate.

## Symptom

Long-context requests lose fine positional resolution beyond about 128K tokens. Exact copying from
far back in the context (codes, identifiers, numbers) fails, while the same prompts work on other
hardware. Shorter contexts are not affected.

## Who is affected

Models with rotary embeddings served with `--max-model-len` above about 131072.

## The fix

`_compute_cos_sin_cache` of `RotaryEmbeddingBase` and of every subclass, including subclasses
defined later (vLLM's scaling variants and vllm-gaudi's HPU classes), is wrapped so the table is
computed under a CPU default device and then moved to the device that was active. Values match
what vLLM computes on CPU or GPU. Only table construction at model load changes.

On v0.30.0, vllm-gaudi builds the tables on the CPU itself (upstream #1816); `apply.py` detects
this and reports "not needed" without writing.

## Verify

```bash
python3 fixes/rope-cpu-cache/apply.py --check   # 0 = applied or not needed, 1 = not applied, 2 = drifted
pytest fixes/rope-cpu-cache
```

End to end: place a random code at a position above 150K tokens in a long prompt and ask for it
back; without the fix the answer is often a near miss.
