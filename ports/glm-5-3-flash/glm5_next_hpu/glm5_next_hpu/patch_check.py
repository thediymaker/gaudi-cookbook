# SPDX-License-Identifier: Apache-2.0
"""Refuse to start when the port's source patches are missing.

The patches in ports/glm-5-3-flash/patches/ edit the installed vLLM and vllm-gaudi, and install.sh runs them at image
build. They cannot be applied from this plugin: by the time an entry point loads, vLLM has imported the files, so the
edit would change the file on disk and not the running process. The symptom of a missing patch is far from its cause
(the model is not treated as MLA, the KV block size derives to 384 instead of 640, and the first request dies in the
cache write), so this check names the patch instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# (package, file inside it, marker the patch writes, patch script)
PATCHES = (
    ("vllm_gaudi", "attention/backends/hpu_attn.py", "[gaudi-port:glm-5-3-flash/nope-mla]", "nope_mla.py"),
    (
        "vllm",
        "transformers_utils/model_arch_config_convertor.py",
        "[gaudi-port:glm-5-3-flash/mla-model-type]",
        "mla_model_type.py",
    ),
    (
        "vllm_gaudi",
        "v1/worker/hpu_model_runner.py",
        "[gaudi-port:glm-5-3-flash/cache-group-idx]",
        "cache_group_idx.py",
    ),
    ("vllm_gaudi", "__init__.py", "[gaudi-port:glm-5-3-flash/mhc-kernel-env]", "mhc_kernel_env.py"),
)


def _source(pkg: str, rel: str) -> str | None:
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        path = Path(loc) / rel
        if path.is_file():
            return path.read_text()
    return None


def missing_patches() -> list[str]:
    out = []
    for pkg, rel, marker, script in PATCHES:
        src = _source(pkg, rel)
        if src is None or marker not in src:
            out.append(f"{script} ({pkg}/{rel})")
    return out


def require_patches() -> None:
    missing = missing_patches()
    if missing:
        raise RuntimeError(
            "glm5_next_hpu: port patches not applied: "
            + ", ".join(missing)
            + ". Build the image with ports/glm-5-3-flash/install.sh."
        )
