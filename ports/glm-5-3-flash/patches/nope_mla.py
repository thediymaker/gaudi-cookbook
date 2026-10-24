#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""nope-mla: let vllm-gaudi's MLA prefill run with qk_rope_head_dim == 0 (NoPE MLA, GLM-5.3-Flash).

`k_pe.view(-1, 1, self.qk_rope_head_dim)` raises "cannot reshape tensor of 0 elements into shape [-1, 1, 0]": PyTorch
cannot infer -1 when the target has zero elements. Passing the leading dimension explicitly fixes it and is the same
result for every model with a rope dimension. Every other zero-width op in the MLA prefill and decode paths already
works. Target: vllm_gaudi/attention/backends/hpu_attn.py.

Usage: nope_mla.py [--check] [--root SITE_PACKAGES]
"""

import sys

from patchlib import main, replace_once

MARKER = "[gaudi-port:glm-5-3-flash/nope-mla]"

OLD = """        k_c_normed, k_pe = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
"""
NEW = f"""        k_c_normed, k_pe = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # {MARKER} NoPE MLA: -1 cannot be inferred for a 0-element tensor
        k_pe = k_pe.reshape(k_c_normed.shape[0], 1, self.qk_rope_head_dim)
"""


def patch(src: str) -> str:
    return replace_once(src, OLD, NEW, "k_pe view")


TARGETS = (("vllm_gaudi", "attention/backends/hpu_attn.py", patch),)

if __name__ == "__main__":
    sys.exit(main(__doc__, MARKER, TARGETS))
