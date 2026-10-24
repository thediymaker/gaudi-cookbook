#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""spec-sampling: speculative decoding with temperature / top-p sampling on HPU.

Generic (speculative decoding on vllm-gaudi); a candidate for fixes/.

  1. vllm-gaudi's rejection sampler asserts sampling_metadata.all_greedy, so the first request with temperature
     above 0 stops the engine once MTP or EAGLE is on. Added: the standard rejection sampler (Leviathan et al.
     2023) for drafts without draft probabilities (MTP and EAGLE drafts are the draft model's argmax): keep draft
     d with probability p_target(d), at the first rejection emit a token sampled from p_target with d removed,
     append the bonus token when every draft is kept. The output distribution is the target's. Greedy requests
     keep the argmax-match rule. Rows use the HPU padded layout (request i owns rows i*k .. i*k+k-1). Per-request
     seeds are not honoured on this path.
  2. vLLM's expand_batch_to_tokens is a Triton kernel; replaced with a torch version for the padded layout.
  3. The runner's speculative branch built sampling metadata padded to the verify rows and sampled the bonus token
     twice. Metadata is now sized to the bonus rows and the redundant pass is skipped. Sampled ids stay on the
     logits' device (a CPU tensor there broke every request that asked for logprobs).
  4. The HPU input batch files greedy requests with temperature -1.0 (vLLM's GPU batch uses 0.0). In a batch mixing
     greedy and sampled requests a greedy request's logits were divided by -1 and it was treated as sampled. Any
     temperature below 1e-5 now counts as greedy, as in vLLM's Sampler.

Usage: spec_sampling.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:spec-sampling"
MARKER = f"[{FIX_ID}]"

SAMPLER_BLOCK = """\


# ---- [glm-5-mxfp4:spec-sampling] greedy + random rejection sampling ----
# The HPU input batch stores greedy requests with temperature -1.0 (vLLM GPU: 0.0), so
# "greedy" here means temperature < _HPU_SPEC_EPS (same rule as vLLM's Sampler).
_HPU_SPEC_EPS = 1e-5


def _hpu_spec_expand_batch_to_tokens(x, cu_num_tokens, num_tokens, replace_from=0, replace_to=0):
    # HPU layout: request i owns token rows i*k .. i*k+k-1, k = num_tokens // num_requests.
    b = cu_num_tokens.shape[0]
    xs = x[:b]
    if xs.shape[0] < b:
        xs = torch.cat([xs, xs[-1:].expand(b - xs.shape[0])])
    mask = xs == replace_from
    if replace_from == GREEDY_TEMPERATURE and replace_from != replace_to and xs.is_floating_point():
        mask = mask | (xs < _HPU_SPEC_EPS)  # temperature call: HPU greedy rows are -1.0
    xs = torch.where(mask, torch.full_like(xs, replace_to), xs)
    k = max(1, num_tokens // b)
    out = xs.repeat_interleave(k)
    if out.shape[0] < num_tokens:
        out = torch.cat([out, out[-1:].expand(num_tokens - out.shape[0])])
    return out[:num_tokens]


def _hpu_spec_rejection_sample(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
                               draft_probs, target_probs, bonus_token_ids, sampling_metadata,
                               synthetic_mode=False, synthetic_conditional_rates=None, use_fp64_gumbel=False):
    # target_probs holds target LOGITS after temperature / top-k / top-p (upstream naming).
    if sampling_metadata.all_greedy:
        target_argmax = target_probs.argmax(dim=-1)
        # Return on the logits' device: RejectionSampler._get_logprobs_tensors indexes HPU logits with
        # these ids, and a CPU tensor there kills the engine on any request that asks for logprobs.
        return rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, num_draft_tokens,
                                        cu_num_draft_tokens).to(target_probs.device)
    assert draft_probs is None, "draft probabilities are not supported on HPU"
    logits = target_probs.float()
    n_rows = logits.shape[0]
    n_req = len(num_draft_tokens)
    k = max(1, n_rows // n_req)
    draft = draft_token_ids.reshape(-1).to(torch.long)[:n_rows]
    probs = logits.softmax(dim=-1)
    p_draft = probs.gather(1, draft.unsqueeze(1)).squeeze(1)
    argmax = logits.argmax(dim=-1)
    # Gumbel-max over p_target with the draft token removed: argmax(p / E), E ~ Exp(1),
    # one noise draw per request (only its first rejection is ever used).
    e = torch.rand((n_req, probs.shape[1]), device=probs.device, dtype=torch.float32)
    e = e.clamp_min_(1e-20).log_().neg_()
    masked = probs.scatter(1, draft.unsqueeze(1), 0.0)
    recovered = (masked / e.repeat_interleave(k, dim=0)[:n_rows]).argmax(dim=-1)
    packed = torch.stack([p_draft, recovered.float(), argmax.float(), draft.float()]).cpu()
    p_draft_c, rec_c, arg_c, draft_c = packed[0], packed[1].long(), packed[2].long(), packed[3].long()
    bonus = bonus_token_ids.reshape(-1).cpu().to(torch.int32)
    temp = sampling_metadata.temperature
    if temp is not None and not sampling_metadata.all_random:
        greedy = (temp[:n_req] < _HPU_SPEC_EPS).cpu().tolist()
    else:
        greedy = [False] * n_req
    greedy = greedy + [False] * (n_req - len(greedy))
    u = torch.rand(n_rows, dtype=torch.float64)
    out = torch.full((n_req, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    for i, n in enumerate(num_draft_tokens):
        rejected = False
        for j in range(n):
            r = i * k + j
            if greedy[i]:
                tok = int(arg_c[r])
                rejected = tok != int(draft_c[r])
            elif float(p_draft_c[r]) >= float(u[r]) and float(p_draft_c[r]) > 0.0:
                tok = int(draft_c[r])
            else:
                tok = int(rec_c[r])
                rejected = True
            out[i, j] = tok
            if rejected:
                break
        if not rejected:
            out[i, n] = int(bonus[i])
    return out.to(target_probs.device)


rejection_sampler.rejection_sample = _hpu_spec_rejection_sample
rejection_sampler.expand_batch_to_tokens = _hpu_spec_expand_batch_to_tokens
"""

RUNNER_OLD = """\
                    spec_decode_metadata = decode_data.spec_decode_metadata
                    sampler_output, sampling_metadata = self._run_sampling(
                        batch_changed, logits_device
                        if spec_decode_metadata is None else logits_device[spec_decode_metadata.bonus_logits_indices],
                        pd_info.decode_req_ids, logits_device.shape[0])
"""

RUNNER_NEW = """\
                    spec_decode_metadata = decode_data.spec_decode_metadata
                    if spec_decode_metadata is None:
                        sampler_output, sampling_metadata = self._run_sampling(
                            batch_changed, logits_device, pd_info.decode_req_ids, logits_device.shape[0])
                    else:  # [glm-5-mxfp4:spec-sampling] metadata sized to the bonus rows; the
                        # RejectionSampler samples the bonus token itself (no second pass).
                        htorch.core.mark_step()
                        sampling_metadata = self._prepare_sampling(
                            batch_changed, pd_info.decode_req_ids,
                            int(spec_decode_metadata.bonus_logits_indices.shape[0]))
                        sampler_output = None
"""


class Drift(Exception):
    """The installed source does not match what this patch expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


# module-level names of hpu_rejection_sampler.py the appended block uses
SAMPLER_NEEDS = (
    "def rejection_sample_pytorch(",
    "\nrejection_sampler.rejection_sample = rejection_sample\n",
    "\nPLACEHOLDER_TOKEN_ID = ",
    "\nGREEDY_TEMPERATURE = ",
    "\nimport torch\n",
)


def patch_sampler(src: str, ctx: dict[str, str]) -> str:
    for need in SAMPLER_NEEDS:
        require_once(src, need, "vllm_gaudi/v1/sample/hpu_rejection_sampler.py")
    return src + SAMPLER_BLOCK


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "\nimport habana_frameworks.torch as htorch\n", "vllm_gaudi/v1/worker/hpu_model_runner.py")
    require_once(src, "    def _prepare_sampling(self,", "vllm_gaudi/v1/worker/hpu_model_runner.py")
    return replace_once(src, RUNNER_OLD, RUNNER_NEW, "decode speculative sampling")


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "v1/sample/hpu_rejection_sampler.py", patch_sampler, "sampler"),
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
)


# ---- common driver (identical in every fix) --------------------------------------------------


def locate(pkg: str, rel: str, root: str | None) -> Path | None:
    """Find <pkg>/<rel> under --root, or in the installed package without importing it."""
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".gaudi-fix.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def run(check: bool, root: str | None) -> int:
    ctx: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for pkg, rel, _, key in TARGETS:
        path = locate(pkg, rel, root)
        if path is None:
            print(f"{MARKER} {pkg}/{rel} not found", file=sys.stderr)
            return 2
        paths[key] = path
        ctx[key] = path.read_text()
    if UPSTREAM_FIXED and all(MARKER not in ctx[k] and needle in ctx[k] for k, needle in UPSTREAM_FIXED):
        print(f"{MARKER} not needed: the upstream fix is present")
        return 0
    done, todo = [], []
    try:
        for _pkg, _rel, fn, key in TARGETS:
            if fn is None:
                continue
            src = ctx[key]
            if MARKER in src:
                done.append(paths[key])
                continue
            new = fn(src, ctx)
            if MARKER not in new:
                raise Drift("patched text lacks the marker")
            compile(new, str(paths[key]), "exec", dont_inherit=True)
            todo.append((paths[key], new))
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if done and todo:
        print(f"{MARKER} partly applied ({', '.join(map(str, done))}); restore the files and re-run", file=sys.stderr)
        return 2
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for path, new in todo:
        write_atomic(path, new)
        print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
