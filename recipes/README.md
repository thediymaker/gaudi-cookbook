# Recipes

One directory per model, laid out like the Hugging Face id:

```text
recipes/<hf_org>/<hf_repo>/recipe.yaml     # the recipe
recipes/<hf_org>/<hf_repo>/<name>.buckets  # a bucket file, only when the recipe needs one
```

Each recipe has a page under [`docs/recipes/`](../docs/recipes/index.md) with the launch command,
the evidence and the pitfalls. The YAML is the machine-readable part; the page is the one to read.

## Format

The file follows the shape of the [recipes.vllm.ai](https://recipes.vllm.ai) schema
(`SCHEMA.md` and `templates/recipe.yaml` in
[vllm-project/recipes](https://github.com/vllm-project/recipes)) where it fits, and adds one
block, `gaudi:`, for what Gaudi needs. Think of it as a proposal for a Gaudi overlay: the upstream
keys keep their meaning, and a renderer that does not know `gaudi:` can ignore it.

| Key | Same as recipes.vllm.ai | Notes |
| --- | --- | --- |
| `meta` | yes | `title`, `slug`, `provider`, `description`, `date_added`, `date_updated`, `tasks`, `related_recipes`. `hardware` uses the id `gaudi2`, which is not in the upstream taxonomy yet, with the upstream states: `verified` when this exact configuration was tested end to end, `supported` when it was not (the notes say why). |
| `model` | yes | `model_id` (must match the directory), `architecture` (`dense` or `moe`), `parameter_count`, `active_parameters`, `context_length` (the checkpoint's `max_position_embeddings`; the served window is in `gaudi.max_model_len`). Optional `base_model`: the original model of a third-party quantization (the checkpoint's Hugging Face `base_model`); the catalog shows that maker's logo. |
| `features` | yes | `tool_calling`, `reasoning`, `spec_decoding`, each with `description` and `args`, using the parser names vLLM registers. |
| `opt_in_features` | yes | Features that are off by default. |
| `variants` | yes | Always a `default` variant with `precision` and `vram_minimum_gb` (`ceil(checkpoint GB x 1.2)`). |
| `gaudi` | no, new | Everything below. |

Upstream keys that describe CUDA or ROCm installs (`min_vllm_version`, `docker_image`, `install`,
`dependencies`, `compatible_strategies`, `hardware_overrides`, `guide`) are left out: the image is
named in `gaudi.image` and the guide is the docs page.

### The `gaudi:` block

```yaml
gaudi:
  hardware: gaudi2              # the only hardware verified so far
  cards: { total: 4, tp: 4, ep: false }
  image:
    vllm_gaudi: "v0.26.0"       # release tag or branch the image was built from
    vllm_gaudi_commit: "b059f9c"
    vllm_commit: "568afb3a"     # the VLLM_STABLE_COMMIT of that vllm-gaudi commit
    gaudi_sw: "1.24.1"
    placeholder: "<your-vllm-gaudi-v0.26.0-image>"
  max_model_len: 131072
  serve_args: []                # the complete verified command line after `vllm serve <model_id>`
  env: {}                       # only the variables the recipe needs; a comment says why
  setup: []                     # optional: shell lines run after env and before the fixes
  port: null                    # optional: a directory under ports/ when the model needs a port image
  serve_model: null             # optional: what `vllm serve` loads when it is not model_id (a converted checkpoint)
  buckets: null                 # bucket file name next to recipe.yaml, or null when env ranges are used
  required_fixes: []            # fix ids from fixes/ this configuration depends on, in the pack's order
  verified: []                  # concrete checks that passed on this configuration
  notes: []                     # short facts a reader needs; the docs page has the detail
```

- `serve_args` is the whole command that was verified, including the parser flags that also appear
  under `features`. `--host`, `--port`, `--served-model-name` and shutdown settings are left out.
- `setup` is optional: shell lines the launch needs before the fixes run, such as creating a
  cache directory. Most recipes have none.
- `port` is optional. When it names a directory under [`ports/`](../ports/README.md), the launch runs in the
  port image built from the stock image named in `image` (see the port's README), and the docs page shows
  that build step before the Launch block.
- `serve_model` is optional, for a model served from a converted local checkpoint (the port ships the
  conversion). The serve line becomes `vllm serve <serve_model> --served-model-name <model_id> ...`; use an
  environment variable set in `env` for the path, such as `"$CHECKPOINT_DIR"`.
- `env` holds strings only. Bucketing on these recipes is set through the `VLLM_*_BUCKET_*`
  ranges in `env`, so `buckets` is `null`. A recipe that ships a bucket file names it here and
  sets the variable that loads it in `env`.
- `required_fixes` lists the fixes the configuration depends on, in the order
  `fixes/apply_all.py` applies them. Every id is a directory under [`fixes/`](../fixes/README.md)
  (see [Fixes](../docs/fixes/index.md)). Most are in the release profile; opt-in fixes (model
  parsers, routing, performance changes) are not, and the recipe's launch command names them:

  ```bash
  python3 fixes/apply_all.py --profile v0.26.0 fp8-moe-gather qwen3-strict-think-end || exit 1
  ```

  The profile also applies the release's other generic fixes, which are not repeated in
  `required_fixes`. `apply_all.py` stops with a non-zero exit code at the first fix that does not
  match the image. `gemma4-bos-tokenizer` is a separate startup step with its own arguments
  (`apply.py MODEL OUT_DIR`, then `--tokenizer OUT_DIR`); a recipe that lists it shows the command.
- `verified` entries are measurements, not claims about other hardware or other images.

## Images

| `gaudi.image.vllm_gaudi` | vllm-gaudi commit | vLLM commit | Gaudi SW |
| --- | --- | --- | --- |
| `v0.26.0` (release tag) | `b059f9c` | `568afb3a` | 1.24.1 |
| `releases/v0.30.0` (branch, before the tag) | `058f107` | `ced6857a` | 1.24.1 |

Both are built as in [Getting started](../docs/getting-started.md). The image sets
`PT_HPU_LAZY_MODE=1` and `VLLM_SKIP_WARMUP=true` by default; recipes repeat the variables they
depend on so that a different image does not change behaviour silently.

## Adding a recipe

1. Copy an existing `recipe.yaml` with the same image and a similar layout.
2. Fill `model` from the checkpoint's `config.json` and the model card.
3. Put the exact command you ran in `serve_args`, and only the `env` it needs, with a reason.
4. List the fixes the configuration depends on and the checks you ran (`verified`). Put every
   opt-in fix in the launch command. Name the image, cards, concurrency and prompt shape for every
   performance number on the page, and say when a check ran on a different configuration.
5. Run `yamllint -c .yamllint.yaml recipes` and `mkdocs build --strict`.
