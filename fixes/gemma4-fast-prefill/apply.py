#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gemma4-fast-prefill: --kv-sharing-fast-prefill for Gemma 4 E2B / E4B on HPU.

Gemma-4-E2B has 35 decoder layers, and the last 20 (num_kv_shared_layers) write no KV cache of
their own: they read the cache of layer 13 (sliding window) or 14 (full attention). During
prefill nothing later reads their output at a prompt token except where a token is sampled, so
only the last token of each prompt has to run them. vLLM's --kv-sharing-fast-prefill does this
on GPU (vllm#60445 turns it on by default for E2B and E4B); it cuts prompt FLOPs about 3x on E2B,
because the shared layers carry the double-wide MLPs. vllm-gaudi forwards the flag to upstream's
fast_prefill_forward, which needs GPU attention metadata the HPU runner never builds, so with
the flag on the HPU model crashes or still runs every layer on every token.

With the flag on, this fix makes HpuGemma4Model.forward run a prompt batch with no cached context
(block_list is None) as:
  1. layers before the first KV-shared layer on every token;
  2. each row's last real token gathered (row * padded_len + query_len - 1, the token the runner
     samples), the KV-shared layers run on those [batch] tokens only, then scattered back.
On a KV-shared layer in that pass, HPU attention is one query per sequence against its prompt
K/V, read from the target layer's cache at the prompt slots, with an fp32 softmax; sliding
layers also mask p - j >= sliding_window. The HPU rotary embedding caches cos/sin for the whole
batch once per forward, so it is prepared again for the gathered positions and dropped after.
Decode steps and prompts with context (prefix-cache hits, later chunks) take the normal path.

Without --kv-sharing-fast-prefill nothing changes. Limits are the upstream flag's: no
prompt_logprobs, no EAGLE; an FP8 KV cache is refused at runtime.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "gemma4-fast-prefill"
MARKER = f"[gaudi-fix:{FIX_ID}]"

MODEL_OLD = """        # Fast prefill path uses _run_decoder_layers which we patch separately
        if self.fast_prefill_enabled:
            hidden_states = self.fast_prefill_forward(
                input_ids,
                positions,
                inputs_embeds,
                per_layer_inputs,
                **kwargs,
            )
            hidden_states = self.norm(hidden_states)
            return hidden_states
"""
MODEL_NEW = f"""        # {MARKER} prompt tokens stop after the last non-shared layer; only each
        # row's sampled token runs the KV-shared layers. Decode and prompts with context fall through.
        if self.fast_prefill_enabled:
            hidden_states = _gemma4_fp_forward(self, input_ids, positions, inputs_embeds, per_layer_inputs, **kwargs)
            if hidden_states is not None:
                return hidden_states
"""
MODEL_REQUIRED = (
    "class HpuGemma4Model(UpstreamGemma4Model):",
    "from vllm.distributed import get_pp_group\n",
    '            layer.register_buffer(\n                "layer_idx_tensor",',
)
MODEL_HELPERS = f'''

# ---- {MARKER} KV-sharing fast prefill on HPU -------------------------------------------
import collections as _fp_collections  # noqa: E402

from vllm.forward_context import get_forward_context as _fp_get_forward_context  # noqa: E402

_FP_METADATA_TYPES: dict = {{}}


def _fp_metadata(md, full_bias, swa_bias):
    """The step's attention metadata plus the two prompt masks the shared layers read."""
    fields = tuple(md._fields) + ("fp_full_bias", "fp_swa_bias")
    md_type = _FP_METADATA_TYPES.get(fields)
    if md_type is None:
        md_type = _FP_METADATA_TYPES[fields] = _fp_collections.namedtuple("FastPrefillAttentionMetadata", fields)
    return md_type(*md, full_bias, swa_bias)


def _fp_shared_ropes(model, first_shared):
    ropes = model.__dict__.get("_fp_ropes")
    if ropes is None:
        ropes = []
        for layer in model.layers[first_shared:]:
            rope = layer.self_attn.rotary_emb
            if hasattr(rope, "prepare_cos_sin") and all(rope is not r for r in ropes):
                ropes.append(rope)
        model.__dict__["_fp_ropes"] = ropes
    return ropes


def _gemma4_fp_forward(model, input_ids, positions, inputs_embeds, per_layer_inputs, **kwargs):
    """Final-normed hidden states [num_tokens, hidden], or None to take the normal path."""
    ctx = _fp_get_forward_context()
    md = ctx.attn_metadata
    if (md is None or not hasattr(md, "_fields") or not md.is_prompt or md.block_list is not None
            or md.seq_lens_tensor is None or get_pp_group().world_size > 1):
        return None
    config = model.config
    first_shared = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
    if not 0 < first_shared < config.num_hidden_layers:
        return None

    if inputs_embeds is not None:
        hidden_states = inputs_embeds
        per_layer_inputs = model.project_per_layer_inputs(hidden_states, per_layer_inputs)
    else:
        hidden_states = model.embed_input_ids(input_ids)
        per_layer_inputs = model.project_per_layer_inputs(hidden_states, model.get_per_layer_inputs(input_ids))
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    if per_layer_inputs is not None:
        per_layer_inputs = per_layer_inputs.reshape(num_tokens, *per_layer_inputs.shape[-2:])

    for layer in model.layers[:first_shared]:
        layer_input = (torch.index_select(per_layer_inputs, 1, layer.layer_idx_tensor).squeeze(1)
                       if per_layer_inputs is not None else None)
        hidden_states, _ = layer(positions, hidden_states, None, per_layer_input=layer_input, **kwargs)

    # Prompt rows are [batch, padded_len] flattened; seq_lens_tensor holds each row's query
    # length (0 for padding rows). A row's last real token is the one the runner samples.
    device = hidden_states.device
    lens = md.seq_lens_tensor.to(torch.long).clamp(min=1)
    batch = lens.shape[0]
    padded_len = num_tokens // batch
    last = torch.arange(batch, dtype=torch.long, device=device) * padded_len + lens - 1
    shared_hidden = hidden_states.index_select(0, last)
    shared_positions = positions.reshape(-1).index_select(0, last)
    shared_per_layer = per_layer_inputs.index_select(0, last) if per_layer_inputs is not None else None

    cols = torch.arange(padded_len, dtype=torch.long, device=device).unsqueeze(0)
    real = cols < lens.unsqueeze(1)
    in_window = real & (cols > lens.unsqueeze(1) - 1 - config.sliding_window)
    zeros = torch.zeros((batch, padded_len), dtype=torch.float32, device=device)
    full_bias = zeros.masked_fill(~real, float("-inf")).view(batch, 1, 1, padded_len)
    swa_bias = zeros.masked_fill(~in_window, float("-inf")).view(batch, 1, 1, padded_len)

    ropes = _fp_shared_ropes(model, first_shared)
    for rope in ropes:
        rope.prepare_cos_sin(shared_positions)
    ctx.attn_metadata = _fp_metadata(md, full_bias, swa_bias)
    try:
        for layer in model.layers[first_shared:]:
            layer_input = (torch.index_select(shared_per_layer, 1, layer.layer_idx_tensor).squeeze(1)
                           if shared_per_layer is not None else None)
            shared_hidden, _ = layer(shared_positions, shared_hidden, None, per_layer_input=layer_input, **kwargs)
    finally:
        ctx.attn_metadata = md
        for rope in ropes:  # the next forward prepares cos/sin for its own positions
            for name in ("cos", "sin"):
                if hasattr(rope, name):
                    delattr(rope, name)
    return hidden_states.index_copy(0, last, model.norm(shared_hidden))
'''

ATTN_OLD = """            if self.kv_sharing_target_layer_name is None:
                # Reshape the input keys and values and store them in the cache.
"""
ATTN_NEW = f"""            if (self.kv_sharing_target_layer_name is not None
                    and getattr(attn_metadata, "fp_full_bias", None) is not None):
                # {MARKER} one query per sequence (its sampled token) against the
                # prompt K/V the target layer wrote in this pass.
                return _gemma4_fp_attention(self, query, key_cache, value_cache, k_scales, attn_metadata,
                                            slot_mapping, output_shape)
            if self.kv_sharing_target_layer_name is None:
                # Reshape the input keys and values and store them in the cache.
"""
ATTN_REQUIRED = (
    "class HPUAttentionImpl(AttentionImpl, torch.nn.Module):",
    "        output_shape = query.shape\n",
    "            query = query.view(batch_size, seq_len, -1)\n",
)
ATTN_HELPERS = f'''

# ---- {MARKER} one-query attention for KV-shared layers in fast prefill ----------------
def _gemma4_fp_attention(impl, query, key_cache, value_cache, k_scales, attn_metadata, slot_mapping, output_shape):
    if k_scales is not None:
        raise NotImplementedError("{MARKER} Gemma 4 fast prefill does not support an FP8 KV cache")
    batch = query.shape[0]
    padded_len = slot_mapping.numel() // batch
    kv_heads, head_size = impl.num_kv_heads, impl.head_size
    group = impl.num_heads // kv_heads
    key = key_cache.index_select(0, slot_mapping).view(batch, padded_len, kv_heads, head_size)
    value = value_cache.index_select(0, slot_mapping).view(batch, padded_len, kv_heads, head_size)
    q = query.reshape(batch, kv_heads, group, head_size)
    scores = torch.matmul(q, key.permute(0, 2, 3, 1)).float() * impl.scale
    bias = attn_metadata.fp_swa_bias if impl.sliding_window else attn_metadata.fp_full_bias
    probs = torch.softmax(scores + bias, dim=-1).to(value.dtype)
    return torch.matmul(probs, value.transpose(1, 2)).reshape(output_shape)
'''


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


def patch_model(src: str, ctx: dict[str, str]) -> str:
    for needle in MODEL_REQUIRED:
        require_once(src, needle, "gemma4.py")
    src = replace_once(src, MODEL_OLD, MODEL_NEW, "HpuGemma4Model.forward fast-prefill branch")
    return src.rstrip("\n") + "\n" + MODEL_HELPERS


def patch_attn(src: str, ctx: dict[str, str]) -> str:
    for needle in ATTN_REQUIRED:
        require_once(src, needle, "hpu_attn.py")
    src = replace_once(src, ATTN_OLD, ATTN_NEW, "HPUAttentionImpl.forward shared-layer branch")
    return src.rstrip("\n") + "\n" + ATTN_HELPERS


# No upstream sentinel: vllm-gaudi has no HPU fast-prefill path yet.
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "models/gemma4.py", patch_model, "model"),
    ("vllm_gaudi", "attention/backends/hpu_attn.py", patch_attn, "attn"),
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
