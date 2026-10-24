# SPDX-License-Identifier: Apache-2.0
"""CPU tests for gemma4-fast-prefill.

1. Source patch on a copy of a release (GAUDI_FIXES_ROOT): applies, is idempotent, refuses drift.
2. The one-query attention against a reference: the last token of each padded prompt row, full and
   sliding-window masks, grouped KV heads.
3. The fast-prefill forward on a toy model: the sampled token of every row gets exactly what the
   full path computes, padding rows included, the attention metadata is restored, and decode or
   prompts with cached context return None (normal path).

Run with: pytest fixes/gemma4-fast-prefill
"""

import collections
import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "gemma4_fast_prefill_apply")


@pytest.fixture
def site(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("set GAUDI_FIXES_ROOT to a directory holding vllm/ and vllm_gaudi/ of a release")
    for pkg, rel, _, _ in fix_mod.TARGETS:
        dst = tmp_path / pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(root) / pkg / rel, dst)
    return tmp_path


def _run(root, *extra):
    cmd = [sys.executable, str(HERE / "apply.py"), "--root", str(root), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=False).returncode


def test_source_patch(site):
    assert _run(site, "--check") == 1
    assert _run(site) == 0
    texts = [(site / pkg / rel).read_text() for pkg, rel, _, _ in fix_mod.TARGETS]
    for text in texts:
        assert fix_mod.MARKER in text
        compile(text, "patched", "exec")
    assert _run(site) == 0 and [(site / p / r).read_text() for p, r, _, _ in fix_mod.TARGETS] == texts
    assert _run(site, "--check") == 0


def test_source_patch_refuses_drift(site):
    model = site / "vllm_gaudi" / "models/gemma4.py"
    with pytest.raises(fix_mod.Drift):
        fix_mod.patch_model(model.read_text().replace(fix_mod.MODEL_OLD, "", 1), {})
    attn = site / "vllm_gaudi" / "attention/backends/hpu_attn.py"
    with pytest.raises(fix_mod.Drift):
        fix_mod.patch_attn(attn.read_text().replace(fix_mod.ATTN_OLD, "", 1), {})


torch = pytest.importorskip("torch")


def _helpers(text, **names):
    ns = dict(torch=torch, **names)
    exec(
        compile(
            text.replace("from vllm.forward_context import get_forward_context as _fp_get_forward_context", "pass"),
            "helpers",
            "exec",
        ),
        ns,
    )
    return ns


@pytest.mark.parametrize("window", [None, 5])
def test_one_query_attention(window):
    ns = _helpers(fix_mod.ATTN_HELPERS)
    batch, padded, kvh, group, hd = 3, 12, 2, 4, 16
    lens = torch.tensor([12, 7, 1])
    g = torch.Generator().manual_seed(0)
    slots = torch.arange(batch * padded)
    key_cache = torch.randn(batch * padded + 8, kvh, hd, generator=g)
    value_cache = torch.randn(batch * padded + 8, kvh, hd, generator=g)
    query = torch.randn(batch, kvh * group * hd, generator=g)
    cols = torch.arange(padded).unsqueeze(0)
    real = cols < lens.unsqueeze(1)
    in_win = real & (cols > lens.unsqueeze(1) - 1 - (window or 10**6))
    zeros = torch.zeros(batch, padded)
    md = types.SimpleNamespace(
        fp_full_bias=zeros.masked_fill(~real, float("-inf")).view(batch, 1, 1, padded),
        fp_swa_bias=zeros.masked_fill(~in_win, float("-inf")).view(batch, 1, 1, padded),
    )
    impl = types.SimpleNamespace(
        num_kv_heads=kvh, head_size=hd, num_heads=kvh * group, scale=hd**-0.5, sliding_window=window
    )
    out = ns["_gemma4_fp_attention"](impl, query, key_cache, value_cache, None, md, slots, query.shape)
    for b in range(batch):
        n = int(lens[b])
        lo = max(0, n - window) if window else 0
        k = key_cache[b * padded + lo : b * padded + n]
        v = value_cache[b * padded + lo : b * padded + n]
        q = query[b].view(kvh, group, hd)
        ref = torch.stack([torch.softmax(q[h] @ k[:, h].T * impl.scale, -1) @ v[:, h] for h in range(kvh)])
        torch.testing.assert_close(out[b].view(kvh, group, hd), ref, rtol=1e-5, atol=1e-5)
    with pytest.raises(NotImplementedError):
        ns["_gemma4_fp_attention"](impl, query, key_cache, value_cache, object(), md, slots, query.shape)


MD = collections.namedtuple("MD", "is_prompt block_list seq_lens_tensor")


class Rope:
    def __init__(self):
        self.prepared = []

    def prepare_cos_sin(self, positions):
        self.prepared.append(positions.clone())
        self.cos = self.sin = positions


class ToyLayer(torch.nn.Module):
    """Non-shared layers mix tokens within a row (prefix sum, like attention); shared layers act
    per token and check they see the fast-prefill metadata and their own positions."""

    def __init__(self, i, shared, padded, ctx, rope):
        super().__init__()
        self.i, self.shared, self.padded, self.ctx = i, shared, padded, ctx
        self.layer_idx_tensor = torch.tensor([i])
        self.self_attn = types.SimpleNamespace(rotary_emb=rope)

    def forward(self, positions, hidden, residual, per_layer_input=None, **kw):
        out = hidden * 1.5 + per_layer_input + positions.reshape(-1, 1).to(hidden.dtype) * 0.01
        if self.shared:
            md = self.ctx.attn_metadata
            assert hasattr(md, "fp_full_bias") == self.ctx.fast
        else:
            rows = out.view(-1, self.padded, out.shape[-1])
            out = (rows.cumsum(1) / 8).reshape(out.shape)
        return out, None


class ToyModel:
    def __init__(self, layers, first_shared, padded, ctx):
        self.config = types.SimpleNamespace(
            num_hidden_layers=layers, num_kv_shared_layers=layers - first_shared, sliding_window=4
        )
        rope = Rope()
        self.rope = rope
        self.layers = [ToyLayer(i, i >= first_shared, padded, ctx, rope) for i in range(layers)]

    def embed_input_ids(self, ids):
        return ids.reshape(-1, 1).to(torch.float32).repeat(1, 4) / 100

    def get_per_layer_inputs(self, ids):
        return ids.reshape(-1, 1, 1).to(torch.float32).repeat(1, len(self.layers), 4) / 1000

    def project_per_layer_inputs(self, hidden, per_layer):
        return per_layer + 0.001

    def norm(self, x):
        return x / (x.abs().amax(-1, keepdim=True) + 1)

    def full(self, ids, positions):
        h = self.embed_input_ids(ids)
        pl = self.project_per_layer_inputs(h, self.get_per_layer_inputs(ids))
        for layer in self.layers:
            h, _ = layer(positions, h, None, per_layer_input=pl[:, layer.i])
        return self.norm(h)


def _forward_ns(ctx):
    pp = types.SimpleNamespace(world_size=1)
    ns = _helpers(fix_mod.MODEL_HELPERS, get_pp_group=lambda: pp)
    ns["_fp_get_forward_context"] = lambda: ctx
    return ns


def test_fast_forward_matches_full_on_sampled_tokens():
    batch, padded, layers, first = 3, 8, 6, 3
    lens = torch.tensor([8, 5, 0])  # a full row, a short row, a padding row
    ctx = types.SimpleNamespace(attn_metadata=MD(True, None, lens), fast=True)
    model = ToyModel(layers, first, padded, ctx)
    ids = torch.randint(1, 100, (batch * padded,), generator=torch.Generator().manual_seed(1))
    positions = torch.arange(padded).repeat(batch)
    ns = _forward_ns(ctx)
    md = ctx.attn_metadata
    out = ns["_gemma4_fp_forward"](model, ids, positions, None, None)
    assert ctx.attn_metadata is md and not hasattr(model.rope, "cos")
    ctx.fast = False
    ref = model.full(ids, positions)
    last = torch.arange(batch) * padded + lens.clamp(min=1) - 1
    torch.testing.assert_close(out[last], ref[last])
    torch.testing.assert_close(model.rope.prepared[0], positions[last])


@pytest.mark.parametrize("md", [MD(False, None, torch.tensor([1])), MD(True, torch.tensor([1]), torch.tensor([4]))])
def test_normal_path_for_decode_and_context(md):
    ctx = types.SimpleNamespace(attn_metadata=md, fast=False)
    ns = _forward_ns(ctx)
    model = ToyModel(4, 2, 4, ctx)
    assert ns["_gemma4_fp_forward"](model, torch.ones(4, dtype=torch.long), torch.arange(4), None, None) is None
