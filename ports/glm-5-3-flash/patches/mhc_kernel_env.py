#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mhc-kernel-env: register the mHC Sinkhorn kernel library before the device opens.

The graph compiler reads GC_KERNEL_PATH when a process opens its Gaudi device. Every vLLM process imports vllm_gaudi
(the platform plugin) before that, so a hook at the top of vllm_gaudi/__init__.py calls
`glm5_next_hpu.mhc.setup_env()`. The hook does nothing unless GLM5_NEXT_MHC_SINKHORN=kernel, and it never breaks
the plugin import (the same pattern as kernels/mxfp4-moe). Target: vllm_gaudi/__init__.py.

Usage: mhc_kernel_env.py [--check] [--root SITE_PACKAGES]
"""

import sys

from patchlib import Drift, main

MARKER = "[gaudi-port:glm-5-3-flash/mhc-kernel-env]"

HOOK = f"""# {MARKER} register the GLM-5.3-Flash mHC kernel library (inert unless GLM5_NEXT_MHC_SINKHORN=kernel)
try:
    from glm5_next_hpu.mhc import setup_env as _glm5_next_mhc_setup_env

    _glm5_next_mhc_setup_env()
except ImportError:
    pass
"""


def patch(src: str) -> str:
    if src.count("\ndef register():\n") != 1:
        raise Drift("vllm_gaudi/__init__.py has no single register() (not the platform plugin this patch expects)")
    return HOOK + src


TARGETS = (("vllm_gaudi", "__init__.py", patch),)

if __name__ == "__main__":
    sys.exit(main(__doc__, MARKER, TARGETS))
