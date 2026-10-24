#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mla-model-type: tell vLLM that model_type `glm5_next_text` is an MLA architecture.

vLLM decides MLA from a fixed list of model_type strings (`is_deepseek_mla`). Without the entry nothing errors: the
engine picks the non-MLA HPU attention backend, sizes the KV cache for K and V heads instead of the 512-wide latent,
derives a 384-token block instead of 640, and the first request fails in the cache write ("same slice shapes except
at dimension 0"). Additive; no other model changes. Target: vllm/transformers_utils/model_arch_config_convertor.py.

Usage: mla_model_type.py [--check] [--root SITE_PACKAGES]
"""

import sys

from patchlib import Drift, main, replace_once

MARKER = "[gaudi-port:glm-5-3-flash/mla-model-type]"

# anchored on another GLM MLA entry of the same tuple
ANCHOR = '            "glm_moe_dsa",\n'
NEW = f'            "glm5_next_text",  # {MARKER}\n' + ANCHOR


def patch(src: str) -> str:
    if src.count('"glm5_next_text"'):
        raise Drift("glm5_next_text is already listed without this patch's marker")
    return replace_once(src, ANCHOR, NEW, "glm_moe_dsa entry")


TARGETS = (("vllm", "transformers_utils/model_arch_config_convertor.py", patch),)

if __name__ == "__main__":
    sys.exit(main(__doc__, MARKER, TARGETS))
