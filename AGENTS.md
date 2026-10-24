# AGENTS.md

Guide for coding agents (Claude Code, Codex, Cursor, Copilot, Gemini CLI and others) that read
this repository on someone's behalf. People: see the README section
[Use it with your coding agent](README.md#use-it-with-your-coding-agent).

You are usually here for one of two reasons:

1. **Your user runs vLLM on Intel Gaudi** and wants to serve a model, fix an error, speed
   something up or run it on Kubernetes. Use this repository as a reference: find the matching
   recipe, fix or page, and adapt it to the user's setup.
2. **You are changing this repository.** Read [Working on this repository](#working-on-this-repository)
   as well.

## Facts to anchor on

Everything here was run on these components. Treat anything outside them as untested.

| Component | Value |
| --- | --- |
| Hardware | Intel Gaudi 2 (HL-225, 96 GB HBM per card, 8 cards per node). Gaudi 3 is not covered. |
| Gaudi software | 1.24.1. The container release must match the host driver. |
| Base image | `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` |
| vllm-gaudi | `v0.26.0` tag (`b059f9c`, vLLM `568afb3a`) or `releases/v0.30.0` branch (`058f107`, vLLM `ced6857a`) |
| Execution mode | lazy mode, `PT_HPU_LAZY_MODE=1` |

## Where to look

| The user wants to | Read | Then |
| --- | --- | --- |
| Serve a listed model | `recipes/<hf_org>/<hf_repo>/recipe.yaml` and its page in `docs/recipes/` | Build the command as in [From a recipe to a command](#from-a-recipe-to-a-command). Read the page's "Known issues" and "What not to do" sections before changing any flag. |
| Serve a model that is not listed | `docs/recipes/index.md`, then the closest recipe by architecture (dense or MoE, hybrid GDN, vision, MXFP4) | Adapt it, and tell the user the result is untested. |
| Explain an error, crash or wrong output | the `## Symptom` section of every `fixes/*/README.md` | Search for the error text: `grep -rn -A8 "^## Symptom" fixes/*/README.md`. Each fix lists who is affected and the releases. |
| Know which fixes a release needs | `python3 fixes/apply_all.py --list` and `docs/fixes/index.md` | The profile for the release covers the generic fixes. Opt-in fixes are named on the command line. |
| Build a serving image | `docs/getting-started.md` | Follow the build steps. They keep the Habana PyTorch fork. |
| Speed up gpt-oss decode | `kernels/mxfp4-moe/README.md` and `docs/kernels/mxfp4-moe.md` | Opt-in and gpt-oss MXFP4 only. See the rules below. |
| Run on Kubernetes | `docs/kubernetes/*.md`, `kubernetes/manifests/`, `kubernetes/tools/` | Start with `docs/kubernetes/pod-spec.md`. Replace the placeholders `<registry>/<image>:<tag>`, `<storage-class>` and namespace `llm`. |
| Debug "Device acquire failed", missing cards or stalled engines | `docs/kubernetes/health-and-failure.md` | The matching tool is in `kubernetes/tools/`. Each prints its usage with `-h`. |

The docs site is built from `docs/`, so every page is plain Markdown you can read directly.

## From a recipe to a command

A `recipe.yaml` holds the whole verified configuration. The format is in `recipes/README.md`.

1. **Image.** `gaudi.image.vllm_gaudi` names the release. Use profile `v0.26.0` for the
   `v0.26.0` image and profile `v0.30.0` for `releases/v0.30.0`.
2. **Environment.** Export every key in `gaudi.env`. They are all strings, and each has a
   reason on the docs page.
3. **Fixes.** `gaudi.required_fixes` lists the fixes the configuration depends on. Run
   `python3 fixes/apply_all.py --profile <release> <ids that are not in the profile> || exit 1`
   before anything imports vLLM. `apply_all.py --list` shows which ids are opt-in.
   `gemma4-bos-tokenizer` is a separate startup step, and its recipe shows the command.
4. **Serve.** `vllm serve <model.model_id> <gaudi.serve_args ...>`. `serve_args` is the complete
   verified command line, except `--host`, `--port` and `--served-model-name`.

The **Launch** block on the recipe's docs page is the same command written out. Prefer copying it
over rebuilding it from the YAML.

## Rules

- **Check the image before applying fixes.** Run
  `python3 fixes/apply_all.py --check --profile <release>` first. Exit 1 means fixes are not
  applied yet and the anchors match. Exit 2 means the installed vLLM or vllm-gaudi differs from
  what the fix expects. On exit 2, stop and tell the user the image does not match. Never edit an
  anchor or an `apply.py` to make it apply.
- **Say what the fixes change.** They edit the installed `vllm` and `vllm_gaudi` sources in place.
  Run them in the container command or in an image `RUN` step, and tell the user which fixes ran.
- **Keep verified and untested apart.** `verified` in a recipe means that exact configuration
  passed the listed checks. `supported` means it was not re-run as written, and the page says why.
  Performance numbers are single runs at the stated shape. Do not present them as guarantees, and
  do not carry them over to other hardware, images or flags.
- **Respect "What not to do".** Recipes record settings that break the model, such as prefix
  caching on Gated DeltaNet models with the stock releases. Do not turn these back on to gain
  speed.
- **Ask before touching shared hardware.** Ask the user first before you apply manifests, delete
  pods, or run a tool that takes cards:
  - `hpu-acquire-check.sh` only reads without node names. Its live test takes cards on the nodes
    you name.
  - `hpu-fleet-sweep.sh` is read-only, over SSH.
  - `hpu-resource-audit.py` only reads manifest files.
- **Treat the kernel as opt-in.** The MXFP4 kernel is for gpt-oss MXFP4 checkpoints, decode calls
  of at most 32 tokens. It must be built inside the serving image. Run `tests/run_parity.sh` and
  check for `PARITY PASS` and `SIG_DEVICE PASS` before enabling `MXFP4_TPC_MOE=1`. Rollback is
  `MXFP4_TPC_MOE=0`.
- **Cite the page.** When you give the user a setting from here, name the file it came from, so
  they can read the reasoning and the evidence.

## Working on this repository

Run these before you propose a change. CI runs the same checks:

```bash
ruff check . && ruff format --check .
yamllint -c .yamllint.yaml recipes kubernetes .github
python3 scripts/gen_llms_txt.py --check   # docs/llms.txt is generated; rerun without --check after changes
mkdocs build --strict          # pip install -r requirements-docs.txt
pytest fixes                   # skips what it cannot run here (no torch, no vLLM, no device)
GAUDI_FIXES_ROOT=<site-packages of a release> GAUDI_FIXES_PROFILE=v0.30.0 pytest fixes/test_fixes.py
```

The second line applies the whole profile twice to a temporary copy of a release's sources.
Run it whenever a fix or `apply_all.py` changes.

| Change | What it needs |
| --- | --- |
| New recipe | `recipes/<hf_org>/<hf_repo>/recipe.yaml` in the format of `recipes/README.md`, a page in `docs/recipes/` with the same sections as the others, a row in `docs/recipes/index.md` and an entry in the `nav` of `mkdocs.yml`. It must run on a released image plus fixes from `fixes/`. |
| New fix | The contract and checklist in `fixes/README.md` ("Contract", "Adding a fix"): one stdlib-only `apply.py`, a README with Symptom, Problem, Who is affected, The fix and Verify, a test, a place in `ORDER` (and `PROFILES` if generic) in `fixes/apply_all.py`, a docs page and a nav entry. Link the upstream pull request or issue. |
| Kubernetes | Manifests use placeholders, never real registries, hosts or storage classes. |

General rules, from `CONTRIBUTING.md`:

- Stock images and public checkpoints only.
- Upstream first.
- Fail closed.
- Show the evidence: the hardware, the image, the command, and what it printed.
- No hostnames, internal paths, registries, credentials or user data.
