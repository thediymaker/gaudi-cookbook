#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply a set of fixes from this directory in the documented order.

Fixes are named on the command line, picked by a release profile, or both. Whatever the order on
the command line, they run in ORDER below, because several fixes edit the same file and a later
fix's anchors are written against the text an earlier one leaves behind. The first fix that fails
stops the run with its exit code, so a container never starts on a half-patched engine.

Usage:
  apply_all.py --profile v0.30.0                       # the recommended set for a release
  apply_all.py --profile v0.26.0 topk-subset-sampler   # a profile plus opt-in fixes
  apply_all.py --check --profile v0.30.0               # report only, write nothing
  apply_all.py --list

Exit codes (same as each apply.py): 0 applied, already applied or not needed; 1 (--check only)
at least one fix not applied yet; 2 source drift or another error, nothing further written.

--check writes nothing, so a fix that needs an earlier fix in the source (gdn-prefix-cache needs
hybrid-decode-pad) cannot find it on an image that is not patched yet. Such a fix is checked against
a scratch copy of the files involved, with the pending fixes before it applied there.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Application order. Fixes that edit the same file must keep their relative order:
#   hpu_model_runner.py: prompt-order, structured-output-rows, so-row-owners, async-inputs,
#     hybrid-decode-pad, gdn-prefix-cache, gather-logprobs-int32, prompt-logprobs, seeded-sampling,
#     swa-decode-window, mm-encoder-budget, mm-pooling-inputs, pipeline-parallel (appends; it mirrors
#     prompt-order on non-last stages)
#   vllm scheduler.py: gdn-prefix-cache, vllm-long-prefill-share, pipeline-parallel
#   hpu_input_batch.py: sampling-rows-refresh, bad-words-index, logitsprocs-rows,
#     resumed-prefill-logits, penalty-shapes, topk-subset-sampler
#   vllm topk_topp_sampler.py: seeded-sampling, topk-subset-sampler
#   vllm_gaudi extension/ops.py: sink-once, moe-explicit-overload
#   vllm_gaudi ops/hpu_fused_moe.py, hpu_compressed_tensors.py, hpu_fp8.py: laguna-moe-sigmoid-routing,
#     fp8-moe-gather, moe-router-weight-on-input
# so-row-owners also rewrites the call structured-output-rows edits, so it must come after it.
# gdn-prefix-cache needs hybrid-decode-pad in the runner first (it checks for it).
ORDER = (
    "prompt-order",
    "sampling-rows-refresh",
    "structured-output-rows",
    "so-row-owners",
    "bad-words-index",
    "logitsprocs-rows",
    "async-inputs",
    "async-discard-counter",
    "resumed-prefill-logits",
    "hybrid-decode-pad",
    "gdn-prefix-cache",
    "penalty-shapes",
    "logits-empty-guard",
    "gather-logprobs-int32",
    "prompt-logprobs",
    "logprobs-decode-guard",
    "seeded-sampling",
    "topk-subset-sampler",
    "sink-once",
    "swa-decode-window",
    "moe-explicit-overload",
    "laguna-moe-sigmoid-routing",
    "fp8-moe-gather",
    "moe-router-weight-on-input",
    "mm-merge-idxcopy",
    "mm-encoder-budget",
    "mm-pooling-inputs",
    "glm4-moe-3d-input",
    "rope-cpu-cache",
    "rank-exception-fatal",
    "vllm-long-prefill-share",
    "pipeline-parallel",
    "structured-outputs-auto-whitespace",
    "xgrammar-generation-eos-stop",
    "glm47-streaming-text-markers",
    "glm45v-box-tokens",
    "gptoss-harmony-parse-recovery",
    "gemma4-required-tool-grammar",
    "gemma4-buffered-tool-args",
    "gemma4-fast-prefill",
    "qwen3-strict-think-end",
    "llama4-tools-lenient",
)

# Startup steps that take their own arguments; apply_all.py does not run them.
STANDALONE = {
    "gemma4-bos-tokenizer": "python3 fixes/gemma4-bos-tokenizer/apply.py MODEL OUT_DIR, then --tokenizer OUT_DIR",
}

# Recommended generic fixes per release. Everything else is opt-in and named explicitly:
# performance changes (penalty-shapes, topk-subset-sampler, fp8-moe-gather), the long-prefill
# scheduler feature, and model-specific parser, routing and tokenizer fixes.
_COMMON = (
    "prompt-order",
    "sampling-rows-refresh",
    "structured-output-rows",
    "so-row-owners",
    "bad-words-index",
    "logitsprocs-rows",
    "async-inputs",
    "resumed-prefill-logits",
    "hybrid-decode-pad",
    "logits-empty-guard",
    "gather-logprobs-int32",
    "prompt-logprobs",
    "logprobs-decode-guard",
    "seeded-sampling",
    "sink-once",
    "swa-decode-window",
    "moe-explicit-overload",
    "laguna-moe-sigmoid-routing",
    "mm-merge-idxcopy",
    "rank-exception-fatal",
    "structured-outputs-auto-whitespace",
)
PROFILES = {
    "v0.26.0": _COMMON + ("async-discard-counter", "rope-cpu-cache", "xgrammar-generation-eos-stop"),
    "v0.30.0": _COMMON,
}


def run_fix(name: str, check: bool, root: str | None, quiet: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(HERE / name / "apply.py")]
    if check:
        cmd.append("--check")
    if root:
        cmd += ["--root", root]
    return subprocess.run(cmd, check=False, capture_output=quiet, text=True)


def load_fix(name: str):
    """A fix's apply.py as a module, for its TARGETS and its locate()."""
    spec = importlib.util.spec_from_file_location(f"gaudi_fix_{name.replace('-', '_')}", HERE / name / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def applies_after(name: str, pending: list[str], root: str | None) -> bool:
    """True if `name` reports "not applied, anchors match" once the fixes in `pending` are in.

    Copies the files those fixes and `name` touch to a scratch directory, applies the pending fixes there in
    order and checks `name` against the result. Nothing under `root` or in the installed packages is written.
    """
    with tempfile.TemporaryDirectory(prefix="gaudi-fix-check-") as scratch:
        for fix in (*pending, name):
            mod = load_fix(fix)
            for pkg, rel, _fn, _key in getattr(mod, "TARGETS", ()):
                src, dst = mod.locate(pkg, rel, root), Path(scratch) / pkg / rel
                if src is None:
                    return False
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
        if any(run_fix(fix, False, scratch, quiet=True).returncode != 0 for fix in pending):
            return False
        return run_fix(name, True, scratch, quiet=True).returncode == 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fixes", nargs="*", help="fix ids to apply in addition to the profile")
    ap.add_argument("--profile", choices=sorted(PROFILES), help="recommended set for a release")
    ap.add_argument("--check", action="store_true", help="report the state of each fix, write nothing")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    ap.add_argument("--list", action="store_true", help="list the fixes in order with their profiles")
    args = ap.parse_args()

    if args.list:
        for name in ORDER:
            profiles = ", ".join(p for p, members in PROFILES.items() if name in members) or "opt-in"
            print(f"{name:38} {profiles}")
        for name, how in STANDALONE.items():
            print(f"{name:38} standalone: {how}")
        return 0

    wanted = set(args.fixes)
    for name in sorted(wanted & set(STANDALONE)):
        print(f"[apply_all] {name} is a startup step, run it separately: {STANDALONE[name]}", file=sys.stderr)
    wanted -= set(STANDALONE)
    unknown = sorted(wanted - set(ORDER))
    if unknown:
        print(f"unknown fix id(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.profile:
        wanted |= set(PROFILES[args.profile])
    if not wanted and not args.fixes:
        ap.error("name at least one fix or a --profile")

    worst = 0
    pending: list[str] = []  # --check: fixes that are not applied yet and whose anchors match
    for name in ORDER:
        if name not in wanted:
            continue
        proc = run_fix(name, args.check, args.root, quiet=args.check)
        rc = proc.returncode
        if not args.check:
            if rc != 0:
                print(f"[apply_all] {name} failed with exit code {rc}; stopping", file=sys.stderr)
                return 2
            continue
        if rc == 2 and pending and applies_after(name, pending, args.root):
            print(f"[gaudi-fix:{name}] not applied (anchors match after the pending fixes before it)")
            rc = 1
        else:
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
        if rc == 1:
            pending.append(name)
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    sys.exit(main())
