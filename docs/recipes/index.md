# Recipes

Each recipe is a configuration that served a public checkpoint on Intel Gaudi 2 from a released
vllm-gaudi image, with the evidence that it works: the exact `vllm serve` command and environment,
the [fixes](../fixes/index.md) it needs and why, the checks that passed, single-run performance
numbers with their shape, and the settings that break it. The machine-readable form lives in
[`recipes/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/recipes) as one `recipe.yaml`
per model, in the shape of the [recipes.vllm.ai](https://recipes.vllm.ai) schema plus a `gaudi:`
block (format: [`recipes/README.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/recipes/README.md)).
Every recipe was verified on Gaudi 2 with Gaudi SW 1.24.1, except where the status column says
otherwise; none has been run on Gaudi 3.

| Model | Cards | Precision | Image | Features | Status |
| --- | --- | --- | --- | --- | --- |
| [Qwen3-30B-A3B-Instruct-2507](qwen3-30b-a3b-2507.md) | 1 | BF16 | v0.26.0 | tools | verified |
| [Qwen3-30B-A3B-Thinking-2507](qwen3-30b-a3b-2507.md) | 1 | BF16 | v0.26.0 | tools, reasoning | verified |
| [Qwen3-Coder-30B-A3B-Instruct](qwen3-coder-30b-a3b-instruct.md) | 1 | BF16 | v0.26.0 | tools | verified |
| [Qwen3.6-27B-FP8](qwen3-6-27b-fp8.md) | 1 | FP8 | v0.26.0 | tools, reasoning, vision, 256K | checks ran with prefix caching on; not re-run as written |
| [Qwen3.6-35B-A3B-FP8](qwen3-6-35b-a3b-fp8.md) | 1 | FP8 | v0.26.0 | tools, reasoning, vision, 256K | checks ran with prefix caching on; not re-run as written |
| [Qwen3-Embedding-8B](qwen3-embedding-8b.md) | 1 | BF16 | v0.26.0 | embeddings, Matryoshka | verified, no fixes |
| [Qwen3-235B-A22B-Instruct-2507-FP8](qwen3-235b-a22b-2507-fp8.md) | 4 (TP 4) | FP8 | v0.26.0 | tools | verified |
| [Qwen3-235B-A22B-Thinking-2507-FP8](qwen3-235b-a22b-2507-fp8.md) | 4 (TP 4) | FP8 | v0.26.0 | tools, reasoning | verified |
| [Gemma 4 31B IT](gemma-4-31b-it.md) | 4 (TP 4) | BF16 | v0.26.0 | tools, vision | verified |
| [gpt-oss-120b](gpt-oss-120b.md) | 4 (TP 4, EP) | MXFP4 | v0.26.0 | tools, reasoning | verified |
| [Llama 4 Scout 17B-16E Instruct](llama-4-scout-17b-16e-instruct.md) | 4 (TP 4) | BF16 | v0.26.0 | tools, vision | verified |
| [Laguna-S-2.1](laguna-s-2-1.md) | 4 (TP 4) | BF16 | releases/v0.30.0 | tools, reasoning | verified |
| [Qwen3.5-397B-A17B-FP8](qwen3-5-397b-a17b-fp8.md) | 8 (TP 8, EP) | FP8 | releases/v0.30.0 | tools, reasoning, 256K | checks ran with prefix caching on; not re-run as written |

## Images

| Image column | vllm-gaudi | vLLM | Gaudi SW |
| --- | --- | --- | --- |
| v0.26.0 | `v0.26.0` tag (`b059f9c`) | `568afb3a` | 1.24.1 |
| releases/v0.30.0 | `releases/v0.30.0` branch at `058f107` (before the tag) | `ced6857a` | 1.24.1 |

Build either as in [Getting started](../getting-started.md). The recipes write
`<your-vllm-gaudi-v0.26.0-image>` or `<your-vllm-gaudi-v0.30.0-image>` where an image name goes.

## Using a recipe

1. Start the container as in [Getting started](../getting-started.md), with a checkout of this
   repository available inside it. The launch blocks call it `$COOKBOOK`.
2. Export the environment from the recipe. Every variable is there for a reason given in the
   recipe; the image defaults are repeated so a different image cannot change them silently.
3. Apply the fixes with the command in the launch block:
   `python3 "$COOKBOOK/fixes/apply_all.py" --profile <release> [opt-in ids]`. The profile is the
   recommended generic set for the image's release; the opt-in fixes the recipe needs (model
   parsers, routing, performance changes) are named after it. `apply_all.py` runs them in the
   pack's own order, whatever the order on the command line, and stops with a non-zero exit code
   at the first fix that does not match the image, so the server never starts half-patched.
   `gemma4-bos-tokenizer` is the one separate startup step; its recipe shows the command.
4. Start `vllm serve` with the recipe's flags. Most recipes skip warm-up, so the first request at
   each new shape compiles an HPU graph; send a few representative requests before measuring.

Each recipe's `required_fixes` lists the fixes its checks depend on, in the pack's order. The
profile also applies the release's other generic fixes; they fix code paths a given model may not
use, and not all of them were part of every recipe's checks. `--check` reports the state of each
fix without writing anything.

## Things every recipe has in common

- **Prefix caching** is on (the vLLM default) for every generation recipe except the Gated
  DeltaNet hybrids (Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-397B-A17B). On those it is off: on these
  releases, prefix-cache hits on Gated DeltaNet layers resume from the wrong state. Upstream
  vllm-gaudi main has merged prefix caching for the non-compact layout; use a release that includes
  it before turning prefix caching on. Where prefix caching is on, `prompt-order` stops a request
  from reading KV that another request in the same step has not written yet.
- **Logprobs**: every generation recipe runs `gather-logprobs-int32` and `logprobs-decode-guard`.
  Without them one client asking for `top_logprobs` can stop the server.
- **Penalties on v0.26.0**: with async scheduling (on by default) `repetition_penalty`,
  `presence_penalty` and `frequency_penalty` are accepted and silently not applied. The v0.30
  branch applies them.
- **Lazy mode vs compile mode**: lazy mode (`PT_HPU_LAZY_MODE=1`) is the default and the fastest
  path for most models. The Qwen3.5 and Qwen3.6 hybrids need compile mode (`0`) for their Gated
  DeltaNet layers.
- **Graph-compile timeouts**: with `VLLM_SKIP_WARMUP=true` a long first prompt can compile for
  minutes; the recipes raise `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300 s) and the engine
  timeouts accordingly.
