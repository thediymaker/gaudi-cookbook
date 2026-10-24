# Fixes

Runtime fixes for **stock vllm-gaudi release images**: `v0.26.0` and `releases/v0.30.0`, each with
the vLLM commit it tracks, on Gaudi software 1.24.1. Each fix is a small, generic correctness or
robustness change applied at container start, before `vllm serve`. The code lives in
[`fixes/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes).

## Using them

```bash
python3 fixes/apply_all.py --profile v0.30.0 || exit 1   # or --profile v0.26.0
exec vllm serve <model> <flags>
```

- `--profile` picks the recommended generic set for a release. Opt-in fixes are added by name,
  for example `apply_all.py --profile v0.26.0 topk-subset-sampler`.
- `--check` reports the state of each fix and writes nothing; `--list` prints every fix in order.
- `--root DIR` patches a copy (a directory holding `vllm/` and `vllm_gaudi/`) instead of the
  installed packages.
- The fixes edit the installed sources in place, so the package directories must be writable. Run
  them in the container command or in a `RUN` step of the image build, before anything imports vLLM.

Every script checks all of its anchors before writing, compiles the result, is idempotent, and
exits non-zero on drift, so a container never serves a half-patched engine.

| Exit code | Meaning |
| --- | --- |
| 0 | applied now, already applied, or not needed on this source |
| 1 | `--check` only: not applied yet, anchors match |
| 2 | source drifted, a target is missing, or the result does not compile; nothing written |

## Status

In application order. **Status** is the state of the upstream change: *open* (pull request under
review), *merged* (in upstream, but not in both releases), *not filed* (upstream report planned),
*not-upstreamable* (the defect is in model files, not in vLLM).

Two entries are not source patches. llama4-tools-lenient ships a tool-parser plugin and chat
template; its `apply.py` only validates them and prints the serve flags. gemma4-bos-tokenizer is a
startup step run on its own (`apply.py MODEL OUT_DIR`, then `--tokenizer OUT_DIR`).

| Fix | Affects | Releases | Profiles | Upstream | Status |
| --- | --- | --- | --- | --- | --- |
| [prompt-order](prompt-order.md) | prefix caching (default) | v0.26.0, v0.30.0 | both | [vllm-gaudi#1918](https://github.com/vllm-project/vllm-gaudi/pull/1918) | open |
| [sampling-rows-refresh](sampling-rows-refresh.md) | concurrent requests with different sampling parameters | v0.26.0, v0.30.0 | both | none | not filed |
| [structured-output-rows](structured-output-rows.md) | steps with a grammar request | v0.26.0, v0.30.0 | both | [vllm-gaudi#1916](https://github.com/vllm-project/vllm-gaudi/pull/1916) | open |
| [so-row-owners](so-row-owners.md) | grammar requests next to a partial prompt chunk | v0.26.0, v0.30.0 | both | none | not filed |
| [bad-words-index](bad-words-index.md) | requests with `bad_words` | v0.26.0, v0.30.0 | both | [vllm-gaudi#1917](https://github.com/vllm-project/vllm-gaudi/pull/1917) | open |
| [logitsprocs-rows](logitsprocs-rows.md) | requests with `min_p`, `logit_bias`, `min_tokens` | v0.26.0, v0.30.0 | both | none | not filed |
| [async-inputs](async-inputs.md) | async scheduling | v0.26.0, v0.30.0 | both | [vllm-gaudi#1919](https://github.com/vllm-project/vllm-gaudi/pull/1919) | open |
| [async-discard-counter](async-discard-counter.md) | async scheduling with preemption | v0.26.0 | v0.26.0 | [vllm-gaudi#1794](https://github.com/vllm-project/vllm-gaudi/pull/1794), [#1803](https://github.com/vllm-project/vllm-gaudi/pull/1803) | merged (in v0.30.0) |
| [resumed-prefill-logits](resumed-prefill-logits.md) | prefix caching with preemption | v0.26.0, v0.30.0 | both | [vllm-gaudi#1849](https://github.com/vllm-project/vllm-gaudi/pull/1849) | merged (main, after v0.30.0) |
| [hybrid-decode-pad](hybrid-decode-pad.md) | hybrid GDN / Mamba / linear-attention models | v0.26.0, v0.30.0 | both | [vllm-gaudi#1912](https://github.com/vllm-project/vllm-gaudi/pull/1912) | open |
| [gdn-prefix-cache](gdn-prefix-cache.md) | prefix caching on GDN hybrids (Qwen3.5 / Qwen3.6 class) | v0.26.0, v0.30.0 | opt-in | none (upstream has the non-compact path: [vllm-gaudi#1827](https://github.com/vllm-project/vllm-gaudi/pull/1827), [#1842](https://github.com/vllm-project/vllm-gaudi/pull/1842)) | not filed (design discussion) |
| [penalty-shapes](penalty-shapes.md) | requests with penalties (performance) | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [logits-empty-guard](logits-empty-guard.md) | tensor parallel with sync scheduling | v0.26.0, v0.30.0 | both | none | not filed |
| [gather-logprobs-int32](gather-logprobs-int32.md) | requests with `logprobs` / `top_logprobs` | v0.26.0, v0.30.0 | both | [vllm-gaudi#1823](https://github.com/vllm-project/vllm-gaudi/pull/1823) | open |
| [prompt-logprobs](prompt-logprobs.md) | requests with `prompt_logprobs` | v0.26.0, v0.30.0 | both | [vllm-gaudi#1839](https://github.com/vllm-project/vllm-gaudi/pull/1839) (crash part) | merged (main, after v0.30.0); value fixes not filed |
| [logprobs-decode-guard](logprobs-decode-guard.md) | requests with `logprobs` / `top_logprobs` | v0.26.0, v0.30.0 | both | none (vLLM) | not filed |
| [seeded-sampling](seeded-sampling.md) | requests with `seed` and temperature > 0 | v0.26.0, v0.30.0 | both | none | not filed |
| [topk-subset-sampler](topk-subset-sampler.md) | requests with `top_k` (performance) | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [sink-once](sink-once.md) | models with attention sinks (gpt-oss) | v0.26.0, v0.30.0 (decode part) | both | [vllm-gaudi#1920](https://github.com/vllm-project/vllm-gaudi/pull/1920) (decode part) | open |
| [swa-decode-window](swa-decode-window.md) | gpt-oss | v0.26.0, v0.30.0 | both | none | not filed |
| [moe-explicit-overload](moe-explicit-overload.md) | FP8 per-channel MoE models (performance) | v0.26.0, v0.30.0 | both | [vllm-gaudi#1921](https://github.com/vllm-project/vllm-gaudi/pull/1921) | open |
| [laguna-moe-sigmoid-routing](laguna-moe-sigmoid-routing.md) | MoE with sigmoid scoring, correction bias or `renormalize=False` | v0.26.0, v0.30.0 | both | [vllm-gaudi#1922](https://github.com/vllm-project/vllm-gaudi/pull/1922) | open |
| [fp8-moe-gather](fp8-moe-gather.md) | FP8 MoE decode at low batch (performance) | v0.26.0 | opt-in | [vllm-gaudi#1731](https://github.com/vllm-project/vllm-gaudi/pull/1731) | merged (in v0.30.0) |
| [moe-router-weight-on-input](moe-router-weight-on-input.md) | Llama 4 MoE | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [mm-merge-idxcopy](mm-merge-idxcopy.md) | vision-language models | v0.26.0, v0.30.0 | both | none | not filed |
| [mm-encoder-budget](mm-encoder-budget.md) | vision-language models with many large images (switch `VLLM_HPU_MM_ENCODER_MAX_PATCHES`) | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [mm-pooling-inputs](mm-pooling-inputs.md) | multimodal embedding models (`--runner pooling`) | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [glm4-moe-3d-input](glm4-moe-3d-input.md) | GLM-4.5 family MoE (`glm4_moe.py`), GLM-4.5V | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [rope-cpu-cache](rope-cpu-cache.md) | RoPE models at long context | v0.26.0 | v0.26.0 | [vllm-gaudi#1816](https://github.com/vllm-project/vllm-gaudi/pull/1816) | merged (in v0.30.0) |
| [rank-exception-fatal](rank-exception-fatal.md) | tensor parallel with the multiprocessing executor | v0.26.0, v0.30.0 | both | none (vLLM) | not filed |
| [vllm-long-prefill-share](vllm-long-prefill-share.md) | very long prompts next to other streams (scheduler feature) | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [pipeline-parallel](pipeline-parallel.md) | `--pipeline-parallel-size` > 1, stages on different nodes over the host network (feature) | v0.26.0, v0.30.0 | opt-in | none (vllm-gaudi has no pipeline path) | not filed |
| [structured-outputs-auto-whitespace](structured-outputs-auto-whitespace.md) | `disable_any_whitespace` with the `auto` backend | v0.26.0, v0.30.0 | both | [vllm#63430](https://github.com/vllm-project/vllm/pull/63430) | open |
| [xgrammar-generation-eos-stop](xgrammar-generation-eos-stop.md) | grammars on models whose turn ends with a non-tokenizer EOS | v0.26.0 | v0.26.0 | [vllm#49227](https://github.com/vllm-project/vllm/pull/49227) | merged (in the vLLM commit v0.30.0 tracks) |
| [glm47-streaming-text-markers](glm47-streaming-text-markers.md) | GLM-4.7 / GLM-5 `glm47` tool parser, streaming | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [glm45v-box-tokens](glm45v-box-tokens.md) | GLM-4.5V with the `glm45` reasoning parser | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [gptoss-harmony-parse-recovery](gptoss-harmony-parse-recovery.md) | gpt-oss harmony parsing | v0.26.0, v0.30.0 | opt-in | [vllm#59387](https://github.com/vllm-project/vllm/pull/59387) | open (partial fix on vLLM main) |
| [gemma4-required-tool-grammar](gemma4-required-tool-grammar.md) | Gemma 4 forced tool choice | v0.26.0, v0.30.0 | opt-in | issue [vllm#53363](https://github.com/vllm-project/vllm/issues/53363) | not filed |
| [gemma4-buffered-tool-args](gemma4-buffered-tool-args.md) | Gemma 4 tool-call arguments | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [gemma4-fast-prefill](gemma4-fast-prefill.md) | Gemma 4 E2B / E4B prefill with `--kv-sharing-fast-prefill` (performance) | v0.30.0 | opt-in | [vllm#60445](https://github.com/vllm-project/vllm/pull/60445) (GPU path) | not filed |
| [qwen3-strict-think-end](qwen3-strict-think-end.md) | Qwen3 reasoning parser with tools | v0.26.0, v0.30.0 | opt-in | [vllm#59821](https://github.com/vllm-project/vllm/pull/59821), issue [vllm#58147](https://github.com/vllm-project/vllm/issues/58147) | open |
| [llama4-tools-lenient](llama4-tools-lenient.md) | Llama 4 tool calling (parser plugin) | v0.26.0, v0.30.0 | opt-in | none | not filed |
| [gemma4-bos-tokenizer](gemma4-bos-tokenizer.md) | Gemma 4 instruct raw prompts (startup step) | v0.26.0, v0.30.0 | standalone | none (checkpoint tokenizer files) | not-upstreamable |

On a release that already contains the upstream change, the script reports "not needed" and
exits 0, so listing it in a container command is harmless.

## Order

Several fixes edit the same file, and a later fix's anchors are written against the text an earlier
fix leaves behind. `apply_all.py` always applies fixes in the order of the table above, whatever
the order on its command line. If you run the scripts yourself, keep that order.

| File | Fixes, in order |
| --- | --- |
| `vllm_gaudi/v1/worker/hpu_model_runner.py` | prompt-order, structured-output-rows, so-row-owners, async-inputs, hybrid-decode-pad, gdn-prefix-cache, gather-logprobs-int32, prompt-logprobs, seeded-sampling, swa-decode-window, pipeline-parallel |
| `vllm/v1/core/sched/scheduler.py` | gdn-prefix-cache, vllm-long-prefill-share, pipeline-parallel |
| `vllm_gaudi/v1/worker/hpu_input_batch.py` | sampling-rows-refresh, bad-words-index, logitsprocs-rows, resumed-prefill-logits, penalty-shapes, topk-subset-sampler |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | seeded-sampling, topk-subset-sampler |
| `vllm_gaudi/extension/ops.py` | sink-once, moe-explicit-overload |
| `vllm_gaudi/ops/hpu_fused_moe.py`, `hpu_compressed_tensors.py`, `hpu_fp8.py` | laguna-moe-sigmoid-routing, fp8-moe-gather, moe-router-weight-on-input |

so-row-owners rewrites the call that structured-output-rows edits, so it must come after it.
prompt-order adds row swaps, so apply sampling-rows-refresh with it.
gdn-prefix-cache checks that hybrid-decode-pad ran before it.

## Not included

- **GDN state masking by multiplication.** The GDN layers clear a new request's state with
  `state * mask`, so a slot that ever holds NaN stays poisoned (`0 * NaN = NaN`). vllm-gaudi main
  uses `torch.where` at both sites; a backport is small but has not been validated on the releases.
  With prefix caching on, [gdn-prefix-cache](gdn-prefix-cache.md) zeroes new slots instead.
- **Interleaved sliding window gated on one model type.** On both releases, prefill with the full
  block list still applies the window bound in the attention bias, so the gate costs compute, not
  correctness. Widening it changes shapes and performance without a validated benefit.
- **Speculative decoding fixes** (rejection sampling with temperature, verify-row layout,
  structured outputs under speculation). They need rework onto the current runner and HPU tests.
- Model ports, custom kernels and their glue, diagnostics, and fixes already contained in both
  releases.

## Tests

Fixes with a `test_*.py` run with plain `pytest`; tests skip what they cannot run (most need a CPU
build of `torch`, some an importable vLLM, two a Gaudi device). Point `GAUDI_FIXES_ROOT` at a
directory holding a release's `vllm/` and `vllm_gaudi/` to test against its sources, and add
`GAUDI_FIXES_PROFILE` to apply that release's profile to a temporary copy end to end:

```bash
GAUDI_FIXES_ROOT=/path/to/site-packages GAUDI_FIXES_PROFILE=v0.30.0 pytest fixes
```
