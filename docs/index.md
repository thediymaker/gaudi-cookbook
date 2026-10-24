---
hide:
  - navigation
---

# Gaudi Cookbook

Practical, tested ways to serve open LLMs with [vLLM](https://github.com/vllm-project/vllm)
on **Intel Gaudi 2**, through the [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) plugin.

Everything here was run on a large Gaudi 2 fleet serving real traffic, and everything uses
public images and public checkpoints.

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

    Custom TPC kernels, starting with MXFP4 MoE decode that reads the packed
    weights directly.

-   :material-kubernetes: **[Kubernetes](kubernetes/index.md)**

    ---

    Running Gaudi at fleet scale: pod spec, bin-pack scheduling, warm-up, health,
    rollouts, autoscaling and routing.

</div>

## Start here

1. Check the [software baseline](getting-started.md) and build or pull a matching image.
2. Pick a [recipe](recipes/index.md) and apply its required [fixes](fixes/index.md).
3. Moving to Kubernetes? Read the [pod spec](kubernetes/index.md) page before anything else.

## Status

This is an independent project. It is not affiliated with Intel or the vLLM project.
Fixes and kernels here are offered upstream first; when a change ships in a vllm-gaudi
release, the matching fix is marked as superseded.
