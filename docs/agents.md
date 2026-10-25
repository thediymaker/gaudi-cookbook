# Use with a coding agent

This cookbook is written to be read by coding agents as much as by people. Point Claude Code,
Codex, Cursor, Copilot, Gemini CLI or a similar agent at it, and the agent works from tested
configurations instead of guessing at Gaudi flags.

## Point your agent at it

=== "Next to your project"

    Clone the cookbook beside the repository you work in:

    ```bash
    git clone https://github.com/thediymaker/gaudi-cookbook.git
    ```

    Then tell the agent: *"Use `../gaudi-cookbook` as the reference for vLLM on Gaudi. Read its
    AGENTS.md first."*

=== "As the workspace"

    Open the agent inside the clone. Most agents read
    [`AGENTS.md`](https://github.com/thediymaker/gaudi-cookbook/blob/main/AGENTS.md) on start, and
    `CLAUDE.md` loads the same file for Claude Code.

=== "By URL"

    Agents that can browse can start from
    `https://github.com/thediymaker/gaudi-cookbook/blob/main/AGENTS.md`, or from this site's
    [`llms.txt`](llms.txt), an index of every page with a one-line summary and raw Markdown links.

## What the agent gets

| | |
| --- | --- |
| **Recipes as data** | Each `recipe.yaml` holds the full verified `vllm serve` command, environment, card layout, required fixes, the checks that passed and the settings that break it. |
| **Symptom lookup** | Every fix has a *Symptom* section with the error text it causes, so the agent can match a log line or a wrong output to its cause, the affected releases and the upstream pull request. |
| **Safe patches** | `apply_all.py --check` reports before anything is written, and the fixes refuse to apply to an image they do not match, so the agent cannot half-patch a server. |
| **Fleet knowledge** | The [Kubernetes](kubernetes/index.md) pages and manifests for scheduling, warm-up, health, autoscaling and routing, plus read-only diagnostic tools for nodes and cards. |
| **Ground rules** | `AGENTS.md` tells the agent to check the image before patching, keep verified and untested apart, cite the page a setting came from, and ask before touching shared hardware. |

## Things to ask

- "Serve Qwen3-235B-A22B-Instruct-2507-FP8 on four Gaudi 2 cards with the v0.26.0 image. Give me
  the container command."
- "My engine died with `tensor_data is empty` in `hccl::allgather_out`. What is it, and how do I
  fix it?"
- "Which fixes does my v0.30.0 image need for a Gemma 4 deployment with tool calling?"
- "Turn the Qwen3.5-397B recipe into a Kubernetes Deployment for our cluster."
- "Some nodes show 8 allocatable cards but every multi-card start fails with `Device acquire
  failed`. Help me find out why."

!!! note "Review what it proposes"
    Agents make mistakes. Read the commands an agent proposes before you run them, especially
    anything that changes a running deployment or takes cards on shared nodes.
