#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Muse-Glimmer-30B port: the only build step. Runs inside the stock vllm-gaudi releases/v0.30.0 image
# (vllm-gaudi 058f107, vLLM ced6857a), checks those commits first and stops at the first step that
# does not apply. vLLM at that commit already has the model, its processor and both parsers; the port
# adds three model-only source patches. Generic fixes are not applied here: the recipe runs
# fixes/apply_all.py at launch.
set -euo pipefail

PORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_GAUDI_COMMIT=058f107
VLLM_COMMIT=ced6857a

# Commits, read without importing vLLM (the import would start the HPU platform plugin).
gaudi_repo="$(python3 -c 'import importlib.util, os; s = importlib.util.find_spec("vllm_gaudi"); print(os.path.dirname(list(s.submodule_search_locations)[0]) if s else "")')"
vllm_version="$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))')"
gaudi_head="$(git -C "$gaudi_repo" rev-parse HEAD 2>/dev/null || true)"
if [[ "$gaudi_head" != "$VLLM_GAUDI_COMMIT"* ]]; then
    echo "muse-glimmer-30b: vllm-gaudi is '${gaudi_head:-unknown}' in '${gaudi_repo:-?}', expected $VLLM_GAUDI_COMMIT" >&2
    exit 1
fi
if [[ "$vllm_version" != *"+g$VLLM_COMMIT"* ]]; then
    echo "muse-glimmer-30b: vLLM is $vllm_version, expected a build of $VLLM_COMMIT" >&2
    exit 1
fi
echo "muse-glimmer-30b: vllm-gaudi $VLLM_GAUDI_COMMIT, vLLM $vllm_version"

# Model-only patches, in order. Each checks every anchor before writing and exits non-zero on drift.
python3 "$PORT_DIR/patches/output_gate_rank.py"
python3 "$PORT_DIR/patches/swa_window_blocks.py"
python3 "$PORT_DIR/patches/forced_tool_grammar.py"

# Re-check that all three report "applied" (exit 0) so a half-patched image never ships.
python3 "$PORT_DIR/patches/output_gate_rank.py" --check
python3 "$PORT_DIR/patches/swa_window_blocks.py" --check
python3 "$PORT_DIR/patches/forced_tool_grammar.py" --check
echo "muse-glimmer-30b: port installed"
