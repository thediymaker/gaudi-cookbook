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

> **New: MXFP4 MoE decode kernels.** Up to 2.5x faster decode for MXFP4 mixture-of-experts models on one 8-card
> Gaudi 2 node: GLM-5.3 single stream from 23.3 to 59.1 tokens/s, Kimi-K2.7 from 17.9 to 44.5, Muse Spark Open 2.4x
> aggregate at four streams, with lower numerical error than the stock op. See [the write-up](docs/kernels/mxfp4-moe.md).
>
> **Ports.** Nine ports bring up models that released images cannot serve on Gaudi 2: Muse Spark Open,
> GLM-5.3-Flash, DeepSeek-V4.1-Flash, GLM-5.x MXFP4, Beam-1.0, MiMo-V2.6 (Flash and Pro), Kimi-K2.7-Code,
> MiniMax-M3 and Inkling-Small. See [Ports](docs/ports/index.md) and [Results](docs/results/index.md).
>
> **Using a coding agent?** Point it at this repository and have it read [`AGENTS.md`](AGENTS.md)
> first. See [Use it with your coding agent](#use-it-with-your-coding-agent).

## What is inside

| Section | What you get |
| --- | --- |
| [`recipes/`](recipes/) | One `recipe.yaml` per model: cards, precision, buckets, parsers, environment, the checks that passed, measured throughput, and known pitfalls. Rendered pages live under [`docs/recipes/`](docs/recipes/). |
| [`fixes/`](fixes/) | Small runtime patches for bugs in released vllm-gaudi and vLLM images. Each one is tied to an upstream pull request or issue, has a `--check` mode, and refuses to start the server if the image does not match what it expects. |
| [`kernels/`](kernels/) | Custom TPC kernels. The MXFP4 MoE decode kernels read the packed E2M1 weights and E8M0 scales directly and make decode up to 2.5x faster. One build variant per model family (GLM-5.x, DeepSeek-V4.1, Muse Spark Open, Kimi, gpt-oss) with build scripts, parity tests and micro benchmarks; the ports hook them in. |
| [`docs/ports/`](docs/ports/index.md) | Write-ups of models brought up on Gaudi 2 that released images cannot serve: what was hard, how it was solved, the correctness gates and measured speed. The integration code is not published yet. |
| [`docs/results/`](docs/results/index.md) | Studies and findings that are not tied to one model: platform behaviours and workarounds, prefix caching on Gated DeltaNet hybrids, MoE decode-step anatomy, two-node pipeline parallel, fast prefill, text-to-speech and embeddings. |
| [`servers/`](servers/README.md) | Small HPU servers for models vLLM does not serve: Qwen3-TTS, Chatterbox and Orpheus text-to-speech behind an OpenAI `/v1/audio/speech` API, and EmbeddingGemma 2 multimodal embeddings, each with an image, CPU tests and device gates. |
| [`kubernetes/`](kubernetes/) | Running Gaudi at fleet scale: pod spec, bin-pack scheduling, start-up and warm-up, health checks, rollouts, autoscaling, routing and observability, with example manifests and small tools. |

## Quick start

1. Check the [software baseline](docs/getting-started.md): Gaudi software 1.24.1, the
   `pytorch-installer` base image, vllm-gaudi `v0.26.0` or `releases/v0.30.0`.
2. Pick a recipe, for example [`recipes/Qwen/Qwen3.5-397B-A17B-FP8`](recipes/Qwen/Qwen3.5-397B-A17B-FP8/).
3. Apply the fixes it lists, then start the server:

   ```bash
   python3 fixes/apply_all.py --check --profile v0.30.0   # report only, write nothing
   python3 fixes/apply_all.py --profile v0.30.0 || exit 1 # apply; exits 2 on source drift
   exec vllm serve <model> <recipe flags>
   ```

## Use it with your coding agent

This repository is written to be read by coding agents as much as by people. Point Claude Code,
Codex, Cursor, Copilot, Gemini CLI or a similar agent at it, and the agent can answer Gaudi
questions from tested configurations instead of guessing. [`AGENTS.md`](AGENTS.md) tells the agent
where everything is and what it must not do. `CLAUDE.md` loads the same file for Claude Code.

**How to point your agent at it**

- **Next to your work.** Clone it beside your project or deployment repo, and tell the agent:
  "Use `../gaudi-cookbook` as the reference for vLLM on Gaudi. Read its AGENTS.md first."
- **As the workspace.** Open the agent inside the clone. Most agents read `AGENTS.md` on start.
- **By URL.** Agents that browse can start from
  `https://github.com/thediymaker/gaudi-cookbook/blob/main/AGENTS.md`. The docs site serves an
  [`llms.txt`](docs/llms.txt) index.

**What the agent gets**

- **Recipes as data.** Each `recipe.yaml` holds the full verified `vllm serve` command,
  environment, card layout, required fixes, the checks that passed, and the settings that break it.
- **Symptom lookup.** Every fix has a `## Symptom` section with the error text it causes, so the
  agent can match a log line or a wrong output to a cause, the affected releases, and the upstream
  pull request.
- **Safe patches.** `apply_all.py --check` reports first, and the fixes refuse to apply to an
  image they do not match, so the agent cannot half-patch a server.
- **Fleet knowledge.** Kubernetes pages and manifests for scheduling, warm-up, health, autoscaling
  and routing, plus read-only diagnostic tools for nodes and cards.
- **Ground rules.** It must check the image before patching, keep verified and untested apart, and
  ask before touching shared hardware.

**Things to ask**

- "Serve Qwen3-235B-A22B-Instruct-2507-FP8 on four Gaudi 2 cards with the v0.26.0 image. Give me
  the container command."
- "My engine died with `tensor_data is empty` in `hccl::allgather_out`. What is it, and how do I
  fix it?"
- "Which fixes does my v0.30.0 image need for a Gemma 4 deployment with tool calling?"
- "Turn the Qwen3.5-397B recipe into a Kubernetes Deployment for our cluster."
- "Some nodes show 8 allocatable cards but every multi-card start fails with `Device acquire
  failed`. Help me find out why."

Agents make mistakes. Review what they propose, especially anything that changes a running
deployment.

## Documentation site

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

Intel Gaudi 2 (HL-225, 96 GB HBM per card, 8 cards per node), lazy mode unless a recipe sets
compile mode. Gaudi 3 is not
covered yet. Contributions with Gaudi 3 results are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Status

This is an independent project. It is not affiliated with or endorsed by Intel or the vLLM
project. Names of models and products belong to their owners.

## License

[Apache License 2.0](LICENSE).
