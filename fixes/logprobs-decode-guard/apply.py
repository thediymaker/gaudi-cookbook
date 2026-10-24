#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""logprobs-decode-guard: an undecodable logprob token id no longer stops the API server.

When a platform returns a token id outside the vocabulary in logprobs, detokenizing it raises
OverflowError (or a similar error) inside AsyncLLM.output_handler, the loop that serves every
request. There is no request boundary there, so the engine shuts down and every in-flight request
fails. One request that asks for top_logprobs is enough.

This fix does not correct the ids (see gather-logprobs-int32 for the HPU cause). It contains the
failure to the response that asked for logprobs: an id that cannot be decoded becomes the string
"<bad_id>", the other ids of the same call still decode normally, and the first occurrence is
logged so the defect stays visible. Two sites, both reached only for logprobs requests:
convert_ids_list_to_tokens (wrapped at module scope, so later imports pick up the wrapper) and
LogprobsProcessor._correct_decoded_token (returns "", its documented value for an incomplete
byte sequence).

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "logprobs-decode-guard"
MARKER = f"[gaudi-fix:{FIX_ID}]"

DETOK_ANCHOR = """# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
"""
DETOK_NEW = (
    f"""# ---- {MARKER} undecodable ids become "<bad_id>" instead of raising ----
_gaudi_fix_orig_convert_ids_list_to_tokens = convert_ids_list_to_tokens
_gaudi_fix_bad_id_logged = False


def convert_ids_list_to_tokens(tokenizer, token_ids):
    global _gaudi_fix_bad_id_logged
    try:
        return _gaudi_fix_orig_convert_ids_list_to_tokens(tokenizer, token_ids)
    except (OverflowError, ValueError, TypeError, IndexError, KeyError):
        pass
    try:
        vocab_n = len(tokenizer)
    except Exception:
        vocab_n = getattr(tokenizer, "vocab_size", None)
    ok = []
    for tid in token_ids:
        try:
            ok.append(vocab_n is not None and 0 <= int(tid) < int(vocab_n))
        except (OverflowError, ValueError, TypeError):
            ok.append(False)
    if not _gaudi_fix_bad_id_logged:
        _gaudi_fix_bad_id_logged = True
        import logging

        bad = [t for t, good in zip(token_ids, ok) if not good]
        logging.getLogger(__name__).warning(
            "%d/%d logprob token ids are outside the vocabulary (size=%r), e.g. %r; "
            "returning <bad_id> for them. Logprobs of this request are not reliable.",
            len(bad), len(token_ids), vocab_n, bad[:8])
    safe = [tid if good else 0 for tid, good in zip(token_ids, ok)]
    try:
        out = _gaudi_fix_orig_convert_ids_list_to_tokens(tokenizer, safe)
    except Exception:
        return ["<bad_id>"] * len(token_ids)
    return [s if good else "<bad_id>" for s, good in zip(out, ok)]


"""
    + DETOK_ANCHOR
)

LP_OLD = """            context = context_token_ids[-num_ctx:]
            full_decoded = self.tokenizer.decode(context + [token_id])
"""
LP_NEW = f"""            context = context_token_ids[-num_ctx:]
            try:  # {MARKER}
                full_decoded = self.tokenizer.decode(context + [token_id])
            except (OverflowError, ValueError, TypeError, IndexError):
                return ""
"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()!r} found {n} times, expected 1")


def patch_detok(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "\ndef convert_ids_list_to_tokens(\n", "detokenizer_utils.py")
    return replace_once(src, DETOK_ANCHOR, DETOK_NEW, "detokenize_incrementally header")


def patch_logprobs(src: str, ctx: dict[str, str]) -> str:
    require_once(src, "    def _correct_decoded_token(", "logprobs.py")
    return replace_once(src, LP_OLD, LP_NEW, "_correct_decoded_token decode")


# Not fixed upstream.
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm", "tokenizers/detokenizer_utils.py", patch_detok, "detok"),
    ("vllm", "v1/engine/logprobs.py", patch_logprobs, "logprobs"),
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
