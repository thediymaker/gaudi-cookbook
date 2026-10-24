---
hide:
  - navigation
---

# Gaudi Cookbook

Practical, tested ways to serve open LLMs with [vLLM](https://github.com/vllm-project/vllm)
on **Intel Gaudi 2**, through the [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) plugin.

Everything here was run on a large Gaudi 2 fleet serving real traffic, and everything uses
public images and public checkpoints.

!!! tip "New: MXFP4 MoE decode kernels"
    Up to **3x faster decode** for MXFP4 mixture-of-experts models on Gaudi 2: GLM-5.3 single stream from 23.3 to
    59.1 tokens/s, gpt-oss-120b up to 3.2x per stream, Muse Spark Open 2.1x per stream at four streams, with lower
    numerical error than the stock op. [Read the write-up](kernels/mxfp4-moe.md).

<div class="grid cards" markdown>

-   :material-chef-hat: **[Recipes](recipes/index.md)**

    ---

    Copy-paste `vllm serve` commands per model: cards, precision, buckets, parsers,
    the checks that passed and the pitfalls to avoid.

-   :material-wrench: **[Fixes](fixes/index.md)**

    ---

    Small runtime fixes for bugs in released vllm-gaudi images, each linked to its
    upstream pull request. They refuse to start if the image does not match.

-   :material-chip: **[Kernels](kernels/index.md)**

    ---

    MXFP4 MoE decode kernels that read the packed weights directly: up to 3x
    faster decode on gpt-oss, GLM and Muse Spark Open.

-   :material-rocket-launch-outline: **[Ports](ports/index.md)**

    ---

    Seven models that released images cannot serve, brought up on Gaudi 2: Muse
    Spark Open, GLM-5.3-Flash, DeepSeek-V4.1-Flash, Beam-1.0 and more.

-   :material-flask-outline: **[Results](results/index.md)**

    ---

    Platform findings, prefix caching on hybrids, MoE decode-step anatomy,
    two-node pipeline parallel, fast prefill, speech and embeddings.

-   :material-kubernetes: **[Kubernetes](kubernetes/index.md)**

    ---

    Running Gaudi at fleet scale: pod spec, bin-pack scheduling, warm-up, health,
    rollouts, autoscaling and routing.

-   :material-robot-outline: **[Use with an agent](agents.md)**

    ---

    Point Claude Code, Codex, Cursor or another coding agent at this repository.
    `AGENTS.md` tells it where everything is and what it must not do.

</div>

## Start here

1. Check the [software baseline](getting-started.md) and build or pull a matching image.
2. Pick a [recipe](recipes/index.md) and apply its required [fixes](fixes/index.md).
3. Moving to Kubernetes? Read the [pod spec](kubernetes/index.md) page before anything else.
4. Working with a coding agent? Point it at this repository; see [Use with an agent](agents.md).

## Status

This is an independent project. It is not affiliated with Intel or the vLLM project.
Fixes and kernels here are offered upstream first; when a change ships in a vllm-gaudi
release, the matching fix is marked as superseded.
