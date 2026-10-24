# Contributing

Thanks for helping. This repository collects what works for serving LLMs with
[vLLM](https://github.com/vllm-project/vllm) on Intel Gaudi through the
[vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) plugin. Everything here
should run on public images and public checkpoints.

## Ground rules

- **Stock first.** A recipe runs on a released vllm-gaudi image, optionally with
  fixes from [`fixes/`](fixes/). Anything that needs unreleased model code belongs
  in a write-up, not a recipe.
- **Upstream first.** A fix in `fixes/` has an upstream pull request or issue, or
  says why it cannot have one. When the upstream change ships, the fix is marked
  as superseded and later removed.
- **Fail closed.** Runtime patches check every anchor before writing and exit
  non-zero if anything does not match, so a container never serves a
  half-patched engine.
- **Show the evidence.** Recipes state what was checked (quality gates, accuracy
  checks, long-context needles) on which hardware and software release.
  Performance numbers name the hardware, image, batch size and prompt shape.
- **Keep it generic.** No hostnames, internal paths, registries, credentials or
  user data. Use public model IDs and placeholders such as `<registry>`.

## Layout

| Path | What lives there |
| --- | --- |
| [`recipes/`](recipes/) | One directory per model: `recipe.yaml` plus any bucket files |
| [`fixes/`](fixes/) | Runtime fixes for released vllm-gaudi images, with tests |
| [`kernels/`](kernels/) | Custom TPC kernels, their build and their Python glue |
| [`kubernetes/`](kubernetes/) | Manifests and tools for running Gaudi at fleet scale |
| [`docs/`](docs/) | The documentation site (MkDocs Material) |
| [`AGENTS.md`](AGENTS.md) | Guide for coding agents that read this repository; keep it in step with the layout |
| [`scripts/`](scripts/) | `gen_llms_txt.py` writes `docs/llms.txt`; run it after adding or renaming a recipe, fix or page |

## Style

- Python: `ruff check` and `ruff format`, line length 120.
- YAML: `yamllint -c .yamllint.yaml`.
- Docs: plain, direct sentences. Build with `mkdocs build --strict` before
  sending a pull request.

## Licence

By contributing you agree that your contribution is licensed under the
Apache License 2.0 (see [LICENSE](LICENSE)). Files copied from other projects keep
their original licence header and say where they came from.
