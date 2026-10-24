#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""cache-group-idx: give GLM-5.3-Flash's KDA layers their KV cache-group index in vllm-gaudi.

In a hybrid model vllm-gaudi stacks one state-index row per KV cache group, so a linear-attention layer must know its
row. `maybe_set_mamba_kv_cache_groups_ids` hands that out, but only for an architecture list and for layers named
`.mixer` or `.linear_attn`. GLM-5.3-Flash names both its KDA and its MLA layers `.self_attn`. The patch adds the
architecture, a `.self_attn` branch that sets the index only on a module that declares `cache_group_idx` (the KDA
layer), and skips names whose layer index is not numeric (the MLA layers' "...self_attn.attn.attn").
Target: vllm_gaudi/v1/worker/hpu_model_runner.py.

Usage: cache_group_idx.py [--check] [--root SITE_PACKAGES]
"""

import sys

from patchlib import Drift, main, replace_once

MARKER = "[gaudi-port:glm-5-3-flash/cache-group-idx]"

OLD_ARCH = """        "Qwen3NextForCausalLM"
    ]
"""
NEW_ARCH = f"""        "Qwen3NextForCausalLM", "Glm5NextForConditionalGeneration"  # {MARKER}
    ]
"""
OLD_NAMES = "    mamba_like_layer = ['.mixer', '.linear_attn']\n"
NEW_NAMES = "    mamba_like_layer = ['.mixer', '.linear_attn', '.self_attn']  # .self_attn: GLM-5.3-Flash KDA\n"
OLD_IDX = """            layer_idx = int(parts[-2])  # "model.layers.5.mixer" -> 5
"""
NEW_IDX = """            if not parts[-2].isdigit():
                continue  # GLM-5.3-Flash MLA layers ("model.layers.3.self_attn.attn.attn") have no state
            layer_idx = int(parts[-2])  # "model.layers.5.mixer" -> 5
"""
OLD_BRANCH = """                if layer is not None and hasattr(layer, "linear_attn"):
                    layer.linear_attn.cache_group_idx = torch.tensor(group_idx, dtype=torch.long, device="hpu")
"""
NEW_BRANCH = (
    OLD_BRANCH
    + """            elif '.self_attn' in layer_name:
                # GLM-5.3-Flash KDA: only a module that declares cache_group_idx is touched
                layer = _get_decoder_layer_by_idx(model, layer_idx)
                attn = getattr(layer, "self_attn", None) if layer is not None else None
                if attn is not None and hasattr(attn, "cache_group_idx"):
                    attn.cache_group_idx = torch.tensor(group_idx, dtype=torch.long, device="hpu")
"""
)


def patch(src: str) -> str:
    start = src.find("def maybe_set_mamba_kv_cache_groups_ids(")
    if src.count("def maybe_set_mamba_kv_cache_groups_ids(") != 1:
        raise Drift("maybe_set_mamba_kv_cache_groups_ids is not defined exactly once")
    end = src.find("\ndef ", start + 1)
    body = src[start:end]
    body = replace_once(body, OLD_ARCH, NEW_ARCH, "mamba_like_arch list")
    body = replace_once(body, OLD_NAMES, NEW_NAMES, "mamba_like_layer list")
    body = replace_once(body, OLD_IDX, NEW_IDX, "layer index parse")
    body = replace_once(body, OLD_BRANCH, NEW_BRANCH, "linear_attn branch")
    return src[:start] + body + src[end:]


TARGETS = (("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch),)

if __name__ == "__main__":
    sys.exit(main(__doc__, MARKER, TARGETS))
