#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Install the Inkling-Small port into a stock vllm-gaudi releases/v0.30.0 image (vllm-gaudi 058f107,
# vLLM ced6857a). The only build step; the Dockerfile runs it. Fails closed: any step that does not apply
# stops the build with a non-zero exit, so an image never carries a half-installed port.
#
#   bash ports/inkling-small/install.sh
#
# 1. checks the release commits, 2. installs the inkling_hpu plugin (model, config, inkling_py parsers),
# 3. patches the runner's decode window for Inkling's local layers (patches/runner_decode_window.py).
# The generic fixes are not applied here; the recipe's fix step runs them at container start.
set -euo pipefail
PORT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 - <<'PY'
import importlib.util, pathlib, re, subprocess, sys

WANT = {"vllm_gaudi": "058f107", "vllm": "ced6857a"}


def commit(pkg):
    """HEAD of the source checkout the package is installed from (pip -e), else the +g<sha> of its version."""
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_dir = pathlib.Path(list(spec.submodule_search_locations)[0])
    try:
        out = subprocess.run(["git", "-C", str(pkg_dir.parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.CalledProcessError):
        pass
    version = pkg_dir / "_version.py"
    m = re.search(r"\+g([0-9a-f]{7,40})", version.read_text()) if version.is_file() else None
    return m.group(1) if m else None


bad = []
for pkg, want in WANT.items():
    got = commit(pkg)
    print(f"[inkling-small] {pkg}: {got}")
    if not got or not (got.startswith(want) or (len(got) >= 7 and want.startswith(got))):
        bad.append(f"{pkg} is at {got}, the port needs {want}")
if bad:
    sys.exit("[inkling-small] wrong base image: " + "; ".join(bad))
PY

pip install --no-deps --no-build-isolation --no-cache-dir "$PORT/inkling_hpu"
python3 "$PORT/patches/runner_decode_window.py"

# Read back: the plugin entry point is visible and the patch reports applied.
python3 - <<'PY'
import sys
from importlib.metadata import entry_points

names = {ep.name for ep in entry_points(group="vllm.general_plugins")}
if "inkling_hpu" not in names:
    sys.exit("[inkling-small] plugin entry point inkling_hpu not found")
print("[inkling-small] plugin entry point registered")
PY
python3 "$PORT/patches/runner_decode_window.py" --check
echo "[inkling-small] install OK"
