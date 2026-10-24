#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""spec-draft-prefill: MTP / EAGLE drafting with chunked prefill on the HPU runner.

Generic (speculative decoding on vllm-gaudi); a candidate for fixes/.

With speculative decoding on, async scheduling is off, so a prefill chunk that does not finish its prompt produces
no logits and no sampled token. The runner's draft path assumes every prefill batch has both:
  1. a step whose only work is a partial chunk leaves sampling_metadata unbound (UnboundLocalError) and the draft
     token list empty (torch.cat([]) raises): the engine dies;
  2. sampled tokens are indexed by prefill batch, but only batches with logits have one, so a partial chunk ahead
     of a finished one shifts every later batch onto the wrong token;
  3. the draft model never runs over partial chunks, so its KV holds nothing for every position except the last
     chunk, and acceptance collapses on long prompts;
  4. the drafter's prefill-sized calls are captured as new HPU graphs per context bucket (the target model keeps
     prompts above --max-cudagraph-capture-size out of graphs); a 64K prompt died in the graph replay.
The patch maps sampled tokens to the batches that have logits; for a batch without logits it runs the draft model
once over the chunk to write the draft-layer KV (the next prompt token at the last position, as the GPU runner
does) and emits a placeholder row the scheduler ignores; and it keeps draft calls above 2048 tokens out of HPU
graphs. Single-request prefill batches only (the default).

Usage: spec_draft_prefill.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied yet, 2 source drifted (nothing written).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "glm-5-mxfp4:spec-draft-prefill"
MARKER = f"[{FIX_ID}]"

OLD_BIND = """\
        warmup_mode = self.warmup_mode
        self.scheduler_output = None
        self.warmup_mode = False
"""

NEW_BIND = """\
        warmup_mode = self.warmup_mode
        self.scheduler_output = None
        self.warmup_mode = False
        sampling_metadata = None  # [glm-5-mxfp4:spec-draft-prefill] partial-chunk-only steps sample nothing
"""

OLD_LOOP = """\
                for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata, logits_indices,
                          logits_requests) in enumerate(zip(*shallow_tuple(prefill_data))):
                    if idx >= len(prefill_sampled_token_ids_tensor):
                        continue
                    _draft_token_ids = self.propose_eagle_prefill(
                        prefill_sampled_token_ids_tensor,
                        hidden_states_prefills,
                        aux_hidden_states_prefills,
                        idx,
                        token_ids,
                        position_ids,
                        attn_metadata,
                        logits_indices,
                        prefill_batch_start_idx,
                    )
                    draft_token_ids_prefill.append(_draft_token_ids)
                    prefill_batch_start_idx += len(req_id)
"""

NEW_LOOP = """\
                # [glm-5-mxfp4:spec-draft-prefill] sampled tokens exist only for batches with logits; partial
                # chunks get a draft-KV fill and a placeholder row instead.
                _hpu_sampled = list(prefill_sampled_token_ids_tensor)
                _hpu_num_spec = self.speculative_config.num_speculative_tokens
                for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata, logits_indices,
                          logits_requests) in enumerate(zip(*shallow_tuple(prefill_data))):
                    _hpu_has_logits = logits_indices is not None and logits_indices.numel() > 0
                    if not _hpu_has_logits or not _hpu_sampled:
                        self._hpu_spec_fill_draft_kv(scheduler_output, hidden_states_prefills,
                                                     aux_hidden_states_prefills, idx, req_id, token_ids,
                                                     position_ids, attn_metadata)
                        draft_token_ids_prefill.append(
                            torch.zeros((len(req_id), _hpu_num_spec), dtype=torch.int64, device=self.device))
                        prefill_batch_start_idx += len(req_id)
                        continue
                    _hpu_next = [None] * (idx + 1)
                    _hpu_next[idx] = _hpu_sampled.pop(0)
                    _draft_token_ids = self.propose_eagle_prefill(
                        _hpu_next,
                        hidden_states_prefills,
                        aux_hidden_states_prefills,
                        idx,
                        token_ids,
                        position_ids,
                        attn_metadata,
                        logits_indices,
                        prefill_batch_start_idx,
                    )
                    draft_token_ids_prefill.append(_draft_token_ids)
                    prefill_batch_start_idx += len(req_id)
"""

OLD_METHOD = """\
    def propose_ngram_draft_token_ids(
"""

NEW_METHOD = '''\
    def _hpu_spec_fill_draft_kv(self, scheduler_output, hidden_states_prefills, aux_hidden_states_prefills, idx,
                                req_ids, token_ids, position_ids, attn_metadata):
        """[glm-5-mxfp4:spec-draft-prefill] Run the draft model once over a prefill chunk that produced
        no logits, so its draft-layer KV exists when later chunks and decode steps attend to
        it. The draft output is discarded. Single-request prefill batches only."""
        if hidden_states_prefills is None or idx >= len(hidden_states_prefills):
            return
        if token_ids.dim() != 2 or token_ids.size(0) != 1 or len(req_ids) != 1:
            logger.warning_once(
                "[glm-5-mxfp4:spec-draft-prefill] draft KV fill skipped for a multi-request prefill batch")
            return
        rid = req_ids[0]
        req_index = self.input_batch.req_id_to_index.get(rid)
        num_sched = scheduler_output.num_scheduled_tokens.get(rid, 0)
        if req_index is None or num_sched <= 0 or num_sched > token_ids.size(1):
            return
        if self.use_aux_hidden_state_outputs:
            if aux_hidden_states_prefills is None or idx >= len(aux_hidden_states_prefills):
                return
            target_hidden_states = torch.cat(aux_hidden_states_prefills[idx], dim=-1)
        else:
            target_hidden_states = hidden_states_prefills[idx]
        if target_hidden_states.dim() == 2:
            target_hidden_states = target_hidden_states.unsqueeze(0)
        if self.drafter.method == "eagle3":
            target_hidden_states = self.drafter.model.model.combine_hidden_states(target_hidden_states)
        flat = token_ids.reshape(-1)
        shifted = flat.clone()
        shifted[:-1].copy_(flat[1:])
        next_pos = int(self.input_batch.num_computed_tokens_cpu[req_index]) + num_sched
        if next_pos < int(self.input_batch.num_tokens_no_spec[req_index]):
            shifted[num_sched - 1] = int(self.input_batch.token_ids_cpu[req_index, next_pos])
        htorch.core.mark_step()  # materialise the target hidden states before the draft reads them
        self.drafter.model(
            input_ids=shifted.view(1, -1),
            positions=position_ids,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=attn_metadata,
            bypass_hpu_graphs=shifted.numel() > 2048,  # prefill-sized: no HPU graph (as the target model)
        )
        htorch.core.mark_step()

    def propose_ngram_draft_token_ids(
'''

OLD_EAGLE = """\
        ret_hidden_states = self.model(
            input_ids=target_token_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=common_attn_metadata,
        )
"""

NEW_EAGLE = """\
        ret_hidden_states = self.model(
            input_ids=target_token_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=common_attn_metadata,
            # [glm-5-mxfp4:spec-draft-prefill] prefill-sized draft calls run outside HPU graphs, as the target's do
            bypass_hpu_graphs=target_token_ids.numel() > 2048,
        )
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


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for need in ("\nimport habana_frameworks.torch as htorch\n", "\nlogger = init_logger(", "\nimport torch\n"):
        require_once(src, need, "vllm_gaudi/v1/worker/hpu_model_runner.py")
    src = replace_once(src, OLD_BIND, NEW_BIND, "bind sampling_metadata")
    src = replace_once(src, OLD_LOOP, NEW_LOOP, "eagle prefill loop")
    return replace_once(src, OLD_METHOD, NEW_METHOD, "draft KV fill helper")


def patch_eagle(src: str, ctx: dict[str, str]) -> str:
    return replace_once(src, OLD_EAGLE, NEW_EAGLE, "draft forward")


UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
    ("vllm_gaudi", "v1/spec_decode/hpu_eagle.py", patch_eagle, "eagle"),
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
