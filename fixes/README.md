# Fixes

Runtime fixes for **stock vllm-gaudi release images**: `v0.26.0` and `releases/v0.30.0`, each
with the vLLM commit it tracks, on Gaudi software 1.24.1. Each fix is a small, generic correctness
or robustness change that applies at container start, before `vllm serve`, and is tied to an
upstream pull request where one exists.

The status table and one page per fix are on the [documentation site](../docs/fixes/index.md).

## Quick start

Inside the container, before the server starts:

```bash
python3 fixes/apply_all.py --profile v0.30.0 || exit 1   # or --profile v0.26.0
exec vllm serve <model> <flags>
```

The fixes edit the installed `vllm` and `vllm_gaudi` sources in place, so the package directories
must be writable. Run them in the container command (as above) or in a `RUN` step of your image
build. Either way they must run before any process imports vLLM.

`--profile` picks the recommended generic set for a release. Opt-in fixes are named on the command
line and run in their place in the order:

```bash
python3 fixes/apply_all.py --profile v0.26.0 topk-subset-sampler penalty-shapes
python3 fixes/apply_all.py --check --profile v0.30.0     # report only, write nothing
python3 fixes/apply_all.py --list                        # every fix, in order, with its profiles
```

A single fix runs on its own the same way:

```bash
python3 fixes/prompt-order/apply.py            # apply
python3 fixes/prompt-order/apply.py --check    # report only
python3 fixes/prompt-order/apply.py --root /tmp/site-packages-copy   # patch a copy instead
```

## Contract

Every `apply.py` is a single stdlib-only file with the same driver:

- It finds its targets with `importlib.util.find_spec` (without importing vLLM), or under `--root`,
  a directory that holds `vllm/` and `vllm_gaudi/`.
- Every anchor must match exactly once. Any mismatch writes nothing and exits 2.
- The patched source is compiled before it is written. Files are replaced atomically and their
  `__pycache__` entries removed.
- A marker (`[gaudi-fix:<id>]`) makes it idempotent: a second run reports "applied" and exits 0.
  A fix that spans several files refuses to run on a partly applied set.
- Where the upstream fix can be recognised exactly, the script reports "not needed" and exits 0.

| Exit code | Meaning |
| --- | --- |
| 0 | applied now, already applied, or not needed on this source |
| 1 | `--check` only: not applied yet, anchors match |
| 2 | source drifted, a target is missing, or the result does not compile; nothing written |

`apply_all.py` stops at the first failure with exit code 2, so a container never serves a
half-patched engine.

## Fixes

In application order. "Profiles" says which release profiles include the fix; "opt-in" fixes are
named explicitly. Two entries are not source patches: llama4-tools-lenient ships a tool-parser
plugin and chat template (its `apply.py` only validates them and prints the serve flags), and
gemma4-bos-tokenizer is a startup step that takes `MODEL OUT_DIR` and is run on its own.

| Fix | What it fixes | Profiles |
| --- | --- | --- |
| [prompt-order](prompt-order/README.md) | prefix-cache hits read before they are written in the same step | v0.26.0, v0.30.0 |
| [sampling-rows-refresh](sampling-rows-refresh/README.md) | device sampling parameters go stale after the batch reorders rows | v0.26.0, v0.30.0 |
| [structured-output-rows](structured-output-rows/README.md) | structured-output steps sample rows with another request's parameters | v0.26.0, v0.30.0 |
| [so-row-owners](so-row-owners/README.md) | grammar masks, parameters and logprobs shifted by a partial prompt chunk | v0.26.0, v0.30.0 |
| [bad-words-index](bad-words-index/README.md) | `bad_words` stops the engine or bans words on another request | v0.26.0, v0.30.0 |
| [logitsprocs-rows](logitsprocs-rows/README.md) | `min_p`, `logit_bias` and `min_tokens` are silently ignored | v0.26.0, v0.30.0 |
| [async-inputs](async-inputs/README.md) | async scheduling decodes some requests from a stale input token | v0.26.0, v0.30.0 |
| [async-discard-counter](async-discard-counter/README.md) | a request with stale async output stops the engine (`AttributeError`) | v0.26.0 |
| [resumed-prefill-logits](resumed-prefill-logits/README.md) | a request resumed after preemption repeats its earlier output | v0.26.0, v0.30.0 |
| [hybrid-decode-pad](hybrid-decode-pad/README.md) | decode padding writes into a live KV block on hybrid models | v0.26.0, v0.30.0 |
| [gdn-prefix-cache](gdn-prefix-cache/README.md) | prefix caching on Gated DeltaNet hybrids reuses the wrong GDN state, or cannot be turned on (needs hybrid-decode-pad) | opt-in |
| [penalty-shapes](penalty-shapes/README.md) | penalties recompile the sampler graph on nearly every step (performance) | opt-in |
| [logits-empty-guard](logits-empty-guard/README.md) | tensor-parallel logits gather over an empty tensor stops the engine (sync scheduling) | v0.26.0, v0.30.0 |
| [gather-logprobs-int32](gather-logprobs-int32/README.md) | `top_logprobs` token ids read back corrupted | v0.26.0, v0.30.0 |
| [prompt-logprobs](prompt-logprobs/README.md) | `prompt_logprobs` crashes or returns wrong values | v0.26.0, v0.30.0 |
| [logprobs-decode-guard](logprobs-decode-guard/README.md) | one undecodable logprob id shuts down the server | v0.26.0, v0.30.0 |
| [seeded-sampling](seeded-sampling/README.md) | seeded sampling segfaults a tensor-parallel worker | v0.26.0, v0.30.0 |
| [topk-subset-sampler](topk-subset-sampler/README.md) | top-k / top-p sorts the whole vocabulary every step (performance) | opt-in |
| [sink-once](sink-once/README.md) | attention sinks counted once per KV block instead of once | v0.26.0, v0.30.0 |
| [swa-decode-window](swa-decode-window/README.md) | gpt-oss sliding window is not exact in paged decode | v0.26.0, v0.30.0 |
| [moe-explicit-overload](moe-explicit-overload/README.md) | FP8 per-channel MoE forces a host sync per layer | v0.26.0, v0.30.0 |
| [laguna-moe-sigmoid-routing](laguna-moe-sigmoid-routing/README.md) | MoE routing ignores `scoring_func`, correction bias and `renormalize` | v0.26.0, v0.30.0 |
| [fp8-moe-gather](fp8-moe-gather/README.md) | FP8 MoE decode reads every expert's weights at low token counts (performance) | opt-in |
| [moe-router-weight-on-input](moe-router-weight-on-input/README.md) | Llama 4 MoE applies the router weight to the expert output, not the input | opt-in |
| [mm-merge-idxcopy](mm-merge-idxcopy/README.md) | multimodal embedding merge fails to compile under concurrent images | v0.26.0, v0.30.0 |
| [mm-encoder-budget](mm-encoder-budget/README.md) | one request with many large images exhausts device memory in the vision encoder | opt-in |
| [mm-pooling-inputs](mm-pooling-inputs/README.md) | multimodal embedding models crash, or embed images as text | opt-in |
| [glm4-moe-3d-input](glm4-moe-3d-input/README.md) | GLM-4.5 family MoE stops the engine on the first token | opt-in |
| [rope-cpu-cache](rope-cpu-cache/README.md) | RoPE tables built on the device are wrong at large positions | v0.26.0 |
| [rank-exception-fatal](rank-exception-fatal/README.md) | a failed tensor-parallel worker leaves the engine serving garbage | v0.26.0, v0.30.0 |
| [vllm-long-prefill-share](vllm-long-prefill-share/README.md) | a very long prefill stalls every other stream's decode (scheduler feature) | opt-in |
| [pipeline-parallel](pipeline-parallel/README.md) | `--pipeline-parallel-size` above 1 does not run on the HPU runner (stages on separate nodes) | opt-in |
| [structured-outputs-auto-whitespace](structured-outputs-auto-whitespace/README.md) | `disable_any_whitespace` rejected with the `auto` backend | v0.26.0, v0.30.0 |
| [xgrammar-generation-eos-stop](xgrammar-generation-eos-stop/README.md) | grammar masks the model's real end-of-turn token | v0.26.0 |
| [glm47-streaming-text-markers](glm47-streaming-text-markers/README.md) | `glm47` streaming misses tool calls written as plain text | opt-in |
| [glm45v-box-tokens](glm45v-box-tokens/README.md) | GLM-4.5V answer box markers reach the client in `content` | opt-in |
| [gptoss-harmony-parse-recovery](gptoss-harmony-parse-recovery/README.md) | a malformed gpt-oss message header fails the request with HTTP 500 | opt-in |
| [gemma4-required-tool-grammar](gemma4-required-tool-grammar/README.md) | Gemma 4 forced tool choice returns prose with no tool call | opt-in |
| [gemma4-buffered-tool-args](gemma4-buffered-tool-args/README.md) | Gemma 4 tool arguments reach the client with literal escapes | opt-in |
| [gemma4-fast-prefill](gemma4-fast-prefill/README.md) | `--kv-sharing-fast-prefill` does not work on HPU for Gemma 4 E2B and E4B (performance) | opt-in |
| [qwen3-strict-think-end](qwen3-strict-think-end/README.md) | a `<tool_call>` quoted inside Qwen3 reasoning becomes a real call | opt-in |
| [llama4-tools-lenient](llama4-tools-lenient/README.md) | Llama 4 tool calls in Meta's formats are not parsed (parser plugin, writes nothing) | opt-in |
| [gemma4-bos-tokenizer](gemma4-bos-tokenizer/README.md) | Gemma 4 instruct drops `<bos>` on raw prompts (startup step, run separately) | standalone |

## Order

Several fixes edit the same file, and a later fix's anchors are written against the text an
earlier fix leaves behind. `apply_all.py` always applies fixes in the order above, whatever the
order on its command line. If you run the scripts yourself, keep that order. The shared files are:

| File | Fixes, in order |
| --- | --- |
| `vllm_gaudi/v1/worker/hpu_model_runner.py` | prompt-order, structured-output-rows, so-row-owners, async-inputs, hybrid-decode-pad, gdn-prefix-cache, gather-logprobs-int32, prompt-logprobs, seeded-sampling, swa-decode-window, mm-encoder-budget, mm-pooling-inputs, pipeline-parallel |
| `vllm_gaudi/v1/worker/hpu_input_batch.py` | sampling-rows-refresh, bad-words-index, logitsprocs-rows, resumed-prefill-logits, penalty-shapes, topk-subset-sampler |
| `vllm/v1/core/sched/scheduler.py` | gdn-prefix-cache, vllm-long-prefill-share, pipeline-parallel |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | seeded-sampling, topk-subset-sampler |
| `vllm_gaudi/extension/ops.py` | sink-once, moe-explicit-overload |
| `vllm_gaudi/ops/hpu_fused_moe.py`, `hpu_compressed_tensors.py`, `hpu_fp8.py` | laguna-moe-sigmoid-routing, fp8-moe-gather, moe-router-weight-on-input |

so-row-owners rewrites the call that structured-output-rows edits, so it must come after it.
prompt-order adds row swaps, so apply sampling-rows-refresh with it. gdn-prefix-cache checks that
hybrid-decode-pad is already in the runner.

## Not included

- **GDN state masking by multiplication.** The GDN layers clear a new request's state with
  `state * mask`, so a slot that ever holds NaN stays poisoned (`0 * NaN = NaN`). vllm-gaudi main
  uses `torch.where` at both sites; a backport is small but has not been validated on the releases.
- **interleaved-swa-model-type.** vllm-gaudi gives sliding-window layers their own window block list
  in chunked prefill only for `gemma4`. For the models tested here, prefill with the full block list
  still applies the window bound in the attention bias, so the gate costs compute, not correctness.
  Muse Glimmer on `v0.30.0` is the exception: its sliding layers returned garbage at 120K tokens
  and the `muse-glimmer-30b` port carries its own window gate. A general fix keyed on the config
  instead of `model_type` has not been validated across models.
- **Speculative decoding fixes** (rejection sampling with temperature, verify-row layout,
  structured outputs under speculation). They need rework onto the current runner and HPU tests.
- Model ports, custom kernels and their glue, diagnostics, and fixes already contained in both
  releases.

## Tests

`test_*.py` files sit next to the fixes that have them. They run with plain `pytest` and skip what
they cannot run: most need a CPU build of `torch`, some need an importable vLLM, and two need a
Gaudi device. Tests that read release sources take `GAUDI_FIXES_ROOT`, a directory holding
`vllm/` and `vllm_gaudi/` (for example a copy of a release's site-packages):

```bash
pytest fixes
GAUDI_FIXES_ROOT=/path/to/site-packages pytest fixes
```

[`test_fixes.py`](test_fixes.py) checks the contract for every fix and, with `GAUDI_FIXES_ROOT`
and `GAUDI_FIXES_PROFILE` set, applies the profile to a temporary copy twice and checks the result.

## Adding a fix

Copy an existing fix directory. Keep the driver block below `# ---- common driver` byte-identical,
write anchors that match exactly once on every release you list, add the fix to `ORDER` (and a
profile, if it is generic) in `apply_all.py`, and give it a README with the same table and
sections. When the upstream change ships in a release, add an exact `UPSTREAM_FIXED` sentinel and
drop the fix from that release's profile.
