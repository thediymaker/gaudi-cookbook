#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the glm-5-mxfp4 port patches in order.

Every patch follows the fixes/ contract (anchors, a marker, --check, --root, atomic writes, exit 0 / 1 / 2). They
run after the cookbook fixes (install.sh applies those first), in ORDER below: several patches edit the same file
and later anchors are written against the text earlier ones leave behind. The first patch that fails stops the run
with its exit code.

Usage:
  apply_all.py [--check] [--root SITE_PACKAGES]

Exit codes: 0 all applied; 1 (--check only) the first patch not applied yet has matching anchors (later patches
are not checked: they anchor on text it would write); 2 source drift or another error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Files edited by more than one patch, in the order they edit them:
#   vllm_gaudi v1/worker/hpu_model_runner.py: spec-draft-prefill, spec-verify-blocks, spec-sampling, spec-logprobs,
#     spec-structured-output, glm-dsa, decode-lean (decode-lean anchors on spec-draft-prefill's text)
#   vllm_gaudi v1/sample/hpu_rejection_sampler.py: spec-sampling, spec-penalties, decode-lean
#   vllm_gaudi v1/spec_decode/hpu_eagle.py: mtp-tuple, spec-draft-prefill
#   vllm routed_experts.py: mxfp4-fused-load, moe-layout
#   vllm_gaudi ops/hpu_mxfp4.py: mxk-hooks, moe-layout
ORDER = (
    "mxfp4_fused_load",
    "fsdpa_softmax_mode",
    "mtp_tuple",
    "spec_draft_prefill",
    "spec_verify_blocks",
    "mtp_residual_shape",
    "spec_sampling",
    "spec_logprobs",
    "spec_penalties",
    "spec_structured_output",
    "glm_dsa",
    "mxk_hooks",
    "moe_layout",
    "decode_lean",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    args = ap.parse_args()
    extra = (["--check"] if args.check else []) + (["--root", args.root] if args.root else [])
    for name in ORDER:
        rc = subprocess.call([sys.executable, str(HERE / f"{name}.py"), *extra])
        if rc != 0:
            # With --check, later patches anchor on text the unapplied one would write: stop here.
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
