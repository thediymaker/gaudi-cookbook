# Muse-Glimmer-30B port

Serves [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Apache 2.0,
with Meta's usage policy) on two Gaudi 2 cards. The recipe and its evidence are on the
[recipe page](../../docs/recipes/muse-glimmer-30b.md); this file says what the port changes.

vLLM `ced6857a`, the commit `releases/v0.30.0` tracks, already has the model (`muse_glimmer.py`), its
processor and the `muse_glimmer` tool and reasoning parsers. On Gaudi it still needs three model-only
source changes, so this port has no plugin package: only `patches/` and the install step.

| Patch | File | Without it |
| --- | --- | --- |
| `output_gate_rank.py` | `vllm/model_executor/models/muse_glimmer.py` | The attention output gate multiplies a 3D `[batch, query, hidden]` tensor by the HPU attention's flattened `[batch * query, hidden]` output. Batch 1 broadcasts by accident; the first two concurrent requests stop the engine with `The size of tensor a (128) must match the size of tensor b (256) at non-singleton dimension 1`. |
| `swa_window_blocks.py` | `vllm_gaudi/v1/worker/hpu_model_runner.py` | vllm-gaudi gives sliding-window layers their own window block list in chunked prefill only for `gemma4`. Muse Glimmer's 39 sliding layers then read the full context with a context-wide window mask. A 120K prompt returned a repeated word instead of the answer and every later request on that server returned garbage, then the engine failed with `PT_DEVMEM Allocation failed`. With the patch, 120K and 127K needles pass. |
| `forced_tool_grammar.py` | `vllm/tool_parsers/muse_glimmer_tool_parser.py` | `tool_choice` "required" and named functions are not constrained (the parser opts out of the JSON-schema path and installs nothing), so "required" decodes like "auto": 0/8 calls on the gate prompt. The patch installs an xgrammar structural tag in the model's own ATEM format and applies it from the first token. |

Each patch follows the [fix contract](../../fixes/README.md#contract): every anchor must match before
anything is written, `--check` reports without writing (0 applied, 1 not applied, 2 drifted), a marker
makes a second run a no-op, and `--root <dir>` patches a copy that holds `vllm/` and `vllm_gaudi/`.

## Build

From the repository root, with a stock `releases/v0.30.0` image built as in
[Getting started](../../docs/getting-started.md):

```bash
docker build -f ports/muse-glimmer-30b/Dockerfile \
  --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <your-muse-glimmer-30b-image> .
```

`install.sh` is the only build step. It checks that vllm-gaudi is `058f107` and vLLM is a build of
`ced6857a`, applies the three patches in order, and re-checks them; any mismatch stops the build.
The generic fixes are not baked in: the recipe's launch command runs `fixes/apply_all.py`.

## Serve

Use the Launch block on the [recipe page](../../docs/recipes/muse-glimmer-30b.md) inside the port
image. It applies the `v0.30.0` fix profile plus `topk-subset-sampler` and starts `vllm serve` on two
cards (TP 2, 192 sequences, 128K context, 4 API servers).

## Verify

```bash
pytest ports/muse-glimmer-30b                       # CPU; set GAUDI_FIXES_ROOT to run the apply tests
ports/muse-glimmer-30b/tests/hpu/run_gates.sh       # against the running server
```

`GAUDI_FIXES_ROOT` is a directory holding `vllm/` (ced6857a) and `vllm_gaudi/` (058f107); the tests
patch a temporary copy. `run_gates.sh` prints one PASS or FAIL line per gate: smoke, 32 concurrent
requests, needles to 127K, vision (shape count, OCR, eight image sizes, text after images), forced tool
calls (required greedy and sampled, named, auto), streaming tool deltas, logprobs, seeded traffic,
reasoning and content split, and health afterwards. `--quick` skips the 120K needles and the sampled
tool calls; `--base-url` points at another server.

## Upstream

- The output-gate rank restore is a no-op on CUDA and belongs in vLLM's `muse_glimmer.py`.
- The window gate should not be keyed on `model_type`: vllm-gaudi could use `interleaved_sliding_window`
  alone. Widening it changes graph shapes for every interleaved model, which is why the port only adds
  this model.
- The structural tag is the model's own format; upstream could ship it in the tool parser.
