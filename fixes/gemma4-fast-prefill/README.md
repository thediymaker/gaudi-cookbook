# gemma4-fast-prefill

Make `--kv-sharing-fast-prefill` work on HPU for Gemma 4 E2B and E4B: during prefill only each
prompt's last token runs the KV-shared layers, which cuts prompt compute about 3x on E2B.
Opt-in performance feature; nothing changes without the serve flag.

| | |
| --- | --- |
| Affects | Gemma 4 models with KV-shared layers (`num_kv_shared_layers` > 0: E2B, E4B) |
| Releases | v0.30.0 (the anchors also match v0.26.0, where it was not run) |
| Patches | `vllm_gaudi/models/gemma4.py`, `vllm_gaudi/attention/backends/hpu_attn.py` |
| Upstream | vLLM GPU path: [vllm#60445](https://github.com/vllm-project/vllm/pull/60445) (open, turns the flag on by default for E2B/E4B); vllm-gaudi: not filed |
| Test | `test_gemma4_fast_prefill.py` (torch on CPU; the source tests need a release's sources) |

## Problem

Gemma-4-E2B has 35 decoder layers, and the last 20 write no KV cache of their own: they read the
cache of layer 13 (sliding window) or layer 14 (full attention), and they carry the double-wide
MLPs. Nothing later reads their output at a prompt token except where a token is sampled, so only
the last token of each prompt has to run them. vLLM's `--kv-sharing-fast-prefill` does that on GPU.
vllm-gaudi's `HpuGemma4Model.forward` hands the flag to upstream's `fast_prefill_forward`, which
expects per-layer GPU attention metadata the HPU runner never builds.

## Symptom

With `--kv-sharing-fast-prefill` on the stock releases the HPU model either fails in the forward or
still runs all 35 layers on every prompt token, so prefill is no faster.

## Who is affected

Gemma 4 E2B and E4B on HPU, mostly prefill-heavy traffic (long prompts, short answers, many
concurrent requests). Models without KV-shared layers take the normal path.

## The fix

With the flag on, a prompt batch with no cached context runs:

1. the layers before the first KV-shared layer on every token;
2. then each row's last real token only (`row x padded_len + query_len - 1`, the token the runner
   samples) through the KV-shared layers, and scatters the result back. The other rows keep the
   earlier layer's states, which the runner never reads.

On a KV-shared layer in that pass, attention is one query per sequence against its prompt K/V,
read from the target layer's cache at the prompt's slots, with an fp32 softmax; sliding layers also
mask positions `p - j >= sliding_window`, as in the reference model. The HPU rotary embedding
caches cos/sin for the whole batch once per forward, so it is prepared again for the gathered
positions and dropped afterwards. Decode steps and prompts with context (prefix-cache hits, later
chunks of a long prompt) take the normal path: correct, without the speed-up.

Limits are the upstream flag's: no `prompt_logprobs` (vLLM rejects them with the flag), no EAGLE.
An FP8 KV cache is refused at runtime.

```bash
python3 fixes/apply_all.py --profile v0.30.0 gemma4-fast-prefill || exit 1
vllm serve google/gemma-4-E2B-it --kv-sharing-fast-prefill ...
```

Measured on one Gaudi 2 card, BF16, `releases/v0.30.0`, offline `LLM.generate` with the method of
vllm#60445 (1,024 random input tokens, 1 output token, 64 prompts), the same card and harness for
both columns: 60.5k tokens/s stock, 108.6k with the fix at the default prompt batch cap of 1, 176.5k
with `VLLM_PROMPT_BS_BUCKET_MAX=8`. The [write-up](../../docs/results/gemma-4-e2b-prefill.md) has
the full matrix and the parity checks.

## Verify

```bash
python3 fixes/gemma4-fast-prefill/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=/path/to/site-packages pytest fixes/gemma4-fast-prefill
```

End to end, serve the same greedy requests with and without `--kv-sharing-fast-prefill` at the
default prompt batch cap of 1: the outputs should be identical, and the time to first token lower
with the flag. Raising `VLLM_PROMPT_BS_BUCKET_MAX` batches several prompts per forward, which is
faster but changes the bf16 rounding of the prompt K/V with or without this fix, so long greedy
outputs can then differ from a run at cap 1.
