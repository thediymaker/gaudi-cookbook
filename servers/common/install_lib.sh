# SPDX-License-Identifier: Apache-2.0
# Sourced by servers/*/install.sh. Fails closed: any check that does not hold stops the build.

# The Habana base image: Gaudi software 1.24.1 and its PyTorch fork.
check_base() {
  local plugin torch_v
  plugin=$(python3 -c "import importlib.metadata as m; print(m.version('habana-torch-plugin'))" 2>/dev/null) || {
    echo "install: habana-torch-plugin not found; build FROM the Gaudi 1.24.1 pytorch-installer image" >&2; exit 1; }
  case "$plugin" in
    1.24.1.*) ;;
    *) echo "install: Gaudi software $plugin, expected 1.24.1 (it must match the host driver)" >&2; exit 1 ;;
  esac
  torch_v=$(python3 -c "import torch; print(torch.__version__)")
  echo "install: Gaudi software $plugin, torch $torch_v"
  BASE_TORCH="$torch_v"
}

# pip must not have replaced the Habana PyTorch fork with an upstream wheel.
check_torch_kept() {
  local now
  now=$(python3 -c "import torch; print(torch.__version__)")
  if [ "$now" != "$BASE_TORCH" ]; then
    echo "install: torch changed from $BASE_TORCH to $now; a dependency pulled an upstream wheel" >&2; exit 1
  fi
  python3 -c "import habana_frameworks.torch" || { echo "install: habana_frameworks.torch no longer imports" >&2; exit 1; }
}

PIP="python3 -m pip install --no-cache-dir --root-user-action=ignore"
