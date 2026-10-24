<p align="center">
  <img src="docs/assets/logo.svg" alt="Gaudi Cookbook" width="96">
</p>

<h1 align="center">Gaudi Cookbook</h1>

<p align="center">
  Tested recipes, runtime fixes, kernels and fleet practices for serving open LLMs with
  <a href="https://github.com/vllm-project/vllm">vLLM</a> on Intel Gaudi 2, through the
  <a href="https://github.com/vllm-project/vllm-gaudi">vllm-gaudi</a> plugin.
</p>

<p align="center">
  <a href="https://github.com/thediymaker/gaudi-cookbook/actions/workflows/ci.yml"><img src="https://github.com/thediymaker/gaudi-cookbook/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
</p>

Everything here was run on a large Gaudi 2 fleet serving real traffic. Everything uses public
container images and public checkpoints. Nothing depends on a private build.

## What is inside

| Section | What you get |
| --- | --- |
| [`recipes/`](recipes/) | One `recipe.yaml` per model: cards, precision, buckets, parsers, environment, the checks that passed, measured throughput, and known pitfalls. Rendered pages live under [`docs/recipes/`](docs/recipes/). |
| [`fixes/`](fixes/) | Small runtime patches for bugs in released vllm-gaudi and vLLM images. Each one is tied to an upstream pull request or issue, has a `--check` mode, and refuses to start the server if the image does not match what it expects. |
| [`kernels/`](kernels/) | Custom TPC kernels. The first is an MXFP4 MoE decode kernel for gpt-oss that reads the packed E2M1 weights and E8M0 scales directly, with build scripts, a parity test and a micro benchmark. |
| [`kubernetes/`](kubernetes/) | Running Gaudi at fleet scale: pod spec, bin-pack scheduling, start-up and warm-up, health checks, rollouts, autoscaling, routing and observability, with example manifests and small tools. |

## Quick start

1. Check the [software baseline](docs/getting-started.md): Gaudi software 1.24.1, the
   `pytorch-installer` base image, vllm-gaudi `v0.26.0` or `releases/v0.30.0`.
2. Pick a recipe, for example [`recipes/openai/gpt-oss-120b`](recipes/openai/gpt-oss-120b/).
3. Apply the fixes it lists, then start the server:

   ```bash
   python3 fixes/apply_all.py --check --profile v0.30.0   # report only, write nothing
   python3 fixes/apply_all.py --profile v0.30.0 || exit 1 # apply; exits 2 on source drift
   exec vllm serve <model> <recipe flags>
   ```

The documentation site is built from [`docs/`](docs/) with MkDocs Material:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## Principles

- **Stock first.** Recipes use released images. A fix is listed only when the stock image
  gets something wrong, and the recipe says which symptom it prevents.
- **Upstream first.** Every fix and kernel is offered upstream. When a release ships the
  change, the fix is marked superseded and recipes stop listing it.
- **Fail closed.** Patches match exact source anchors. If the code moved, they stop instead
  of guessing.
- **Show the evidence.** Claims come with the command that produced them and the numbers it
  printed.

## Hardware and scope

Intel Gaudi 2 (HL-225, 96 GB HBM per card, 8 cards per node), lazy mode. Gaudi 3 is not
covered yet. Contributions with Gaudi 3 results are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Status

This is an independent project. It is not affiliated with or endorsed by Intel or the vLLM
project. Names of models and products belong to their owners.

## License

[Apache License 2.0](LICENSE).
