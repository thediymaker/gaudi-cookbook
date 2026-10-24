#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build step of the MiMo-V2.6-Flash port. Runs inside a stock vllm-gaudi v0.26.0 image (the Dockerfile runs it
# once at build time) and stops at the first step that does not apply:
#   1. check that vllm-gaudi and vLLM are the v0.26.0 release commits;
#   2. apply the release's fix profile plus the recipe's opt-in fix (the port's runner patches are written against
#      the fixed source);
#   3. apply the port's runner patches;
#   4. install the plugin package (registers the model with vLLM through its entry point).
set -euo pipefail

PORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK="$(cd "${PORT_DIR}/../.." && pwd)"
VLLM_GAUDI_COMMIT=b059f9c
VLLM_COMMIT=568afb3a
FIXES=(vllm-long-prefill-share)   # opt-in fixes of recipes/XiaomiMiMo/MiMo-V2.6-Flash-RL

say() { echo "[mimo-v2-6-flash] $*"; }
die() { echo "[mimo-v2-6-flash] $*" >&2; exit 1; }

# 1. Release commits, from the git checkouts the packages were installed from.
source_commit() {
  local dir
  dir="$(python3 -c "import importlib.util, os, sys
s = importlib.util.find_spec(sys.argv[1])
print(os.path.dirname(list(s.submodule_search_locations)[0]) if s and s.submodule_search_locations else '')" "$1")"
  [[ -n "$dir" ]] || die "package $1 not found"
  git -C "$dir" rev-parse HEAD 2>/dev/null || die "cannot read the commit of $1 (no git checkout at $dir)"
}
got="$(source_commit vllm_gaudi)"
[[ "$got" == ${VLLM_GAUDI_COMMIT}* ]] || die "vllm-gaudi is at ${got}, the port needs v0.26.0 (${VLLM_GAUDI_COMMIT})"
got="$(source_commit vllm)"
[[ "$got" == ${VLLM_COMMIT}* ]] || die "vLLM is at ${got}, the port needs ${VLLM_COMMIT}"
say "vllm-gaudi ${VLLM_GAUDI_COMMIT}, vLLM ${VLLM_COMMIT}: OK"

# 2. Fixes. The launch command runs the same line again; on this image it only reports "applied".
python3 "${COOKBOOK}/fixes/apply_all.py" --profile v0.26.0 "${FIXES[@]}" || die "fixes did not apply"

# 3. Port patches, in this order (runner_kv.py refuses to run without runner_window.py).
python3 "${PORT_DIR}/patches/runner_window.py" || die "patches/runner_window.py did not apply"
python3 "${PORT_DIR}/patches/runner_kv.py" || die "patches/runner_kv.py did not apply"

# 4. Plugin package.
pip install --no-deps --no-build-isolation --no-cache-dir "${PORT_DIR}/mimo_v2_hpu" || die "plugin install failed"
python3 - <<'PY' || die "plugin is not registered"
from importlib.metadata import entry_points

names = [e.name for e in entry_points(group="vllm.general_plugins")]
assert "mimo_v2_hpu" in names, names
import mimo_v2_hpu.ops  # noqa: F401  (imports without vLLM or a device)
PY
say "installed"
