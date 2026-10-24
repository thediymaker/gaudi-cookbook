# Ports

Code for models that released vllm-gaudi images cannot serve. Each port installs into a stock image built as in
[Getting started](../docs/getting-started.md) and adds what the model needs: a vLLM plugin with the model code,
source patches with the same fail-closed contract as [`fixes/`](../fixes/README.md), and, where the model uses
them, a build of the [MXFP4 MoE kernels](../kernels/mxfp4-moe/README.md). The write-ups are under
[`docs/ports/`](../docs/ports/index.md); each served configuration is a recipe under [`recipes/`](../recipes/README.md)
that names its port.

## Layout

```text
ports/<port-id>/
  README.md        what the port contains, how to build, serve and verify it
  install.sh       the only build step; runs inside the stock image, fails closed
  Dockerfile       FROM the stock image; copies the repository to /opt/gaudi-cookbook and runs install.sh
  <package>/       the vLLM plugin (pyproject.toml with a vllm.general_plugins entry point)
  patches/         source patches the port needs that are not generic fixes
  kernels/         model-specific kernel build configuration and any model-only TPC kernels
  serving/         bucket files and other serving inputs
  tests/           CPU tests (pytest, skip cleanly without torch, vLLM or a device)
  tests/hpu/       device gates: run_gates.sh prints one PASS or FAIL line per gate
```

## Contract

- **Stock base.** The port starts from a released vllm-gaudi image at the commits in
  [`recipes/README.md`](../recipes/README.md#images). `install.sh` checks those commits first and exits non-zero on
  any other.
- **Fail closed.** `install.sh` stops at the first step that does not apply. Patches check every anchor before
  writing, exactly like fixes, and support `--check`.
- **Generic fixes stay in `fixes/`.** A bug that is not specific to the model is a fix with its own page, and the
  port's recipe lists it in `required_fixes`. The port's `patches/` holds only what exists for this model.
- **Kernels are built, never committed.** Kernel sources live once, in `kernels/mxfp4-moe/src`. The port selects a
  variant with `kernels/mxfp4-moe/build.sh --variant <name> <out-dir>` and adds model-only kernels from its own
  `kernels/` directory. Compiled libraries and recipe caches are build outputs.
- **Off means stock.** Every speed-up has a switch. With the switch off the port runs its reference path, and the
  page says which gates were run in each mode.
- **Evidence.** The recipe's `verified` list names checks that ran on an image built by this `install.sh` from the
  stock image, with this recipe's command. Anything measured on a different build is labelled as such.
- **Public inputs only.** Public checkpoints, or a conversion script in the port that produces the checkpoint from a
  public one. No private registries, hosts, paths or credentials.

## Recipes for ports

A recipe for a port adds one key to the `gaudi:` block:

```yaml
gaudi:
  port: muse-spark-open      # directory under ports/
```

The launch command then builds the port image first:

```bash
docker build -f ports/<port-id>/Dockerfile --build-arg BASE=<your-vllm-gaudi-v0.30.0-image> -t <port-image> .
```

and serves with `<port-image>` in place of the stock image. `required_fixes` and the fix step work as for any other
recipe.
