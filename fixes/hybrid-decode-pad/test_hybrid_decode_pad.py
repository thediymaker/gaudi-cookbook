# SPDX-License-Identifier: Apache-2.0
"""CPU tests for hybrid-decode-pad.

1. Source patch on a copy of the runner: applies, only appends, is idempotent, and refuses
   runners with a missing or duplicated anchor.
2. The runner's own decode padding code, cut out of the runner file and run on CPU against a
   split hybrid layout (num_blocks * k kernel blocks + 1 spare): with the stock pad ids the padding
   slots land inside a live KV block; after the fix they land in the spare kernel block.
3. The helper on stub runners: split layouts (block splitting and naive sharing, MHA and MLA
   tensors, list-wrapped kv_cache), inert on unsplit layouts, idempotent, and RuntimeError when
   there is no spare block or the attention tensors disagree.

Needs torch (CPU is enough). Tests 1 and 2 need a runner source: set GAUDI_FIXES_ROOT to a
directory holding vllm_gaudi/, or run where vllm_gaudi is installed. vLLM's KV cache spec classes
are used when importable, otherwise minimal stand-ins with the same constructors.

Run with: pytest fixes/hybrid-decode-pad
"""

import dataclasses
import importlib.util
import itertools
import os
import shutil
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
RUNNER_REL = "v1/worker/hpu_model_runner.py"


def _load_apply():
    spec = importlib.util.spec_from_file_location("hybrid_decode_pad_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load_apply()


@pytest.fixture
def runner_src():
    path = fix_mod.locate("vllm_gaudi", RUNNER_REL, os.environ.get("GAUDI_FIXES_ROOT"))
    if path is None:
        pytest.skip("no vllm_gaudi runner source (set GAUDI_FIXES_ROOT)")
    src = path.read_text()
    cut = src.find("\n\n# ---- " + fix_mod.MARKER)
    return src[:cut] + "\n" if cut >= 0 else src


# ---------------------------------------------------------------- KV cache spec classes


def _stub_kv_interface():
    m = types.ModuleType("vllm.v1.kv_cache_interface")

    @dataclasses.dataclass(frozen=True)
    class KVCacheSpec:
        block_size: int

    @dataclasses.dataclass(frozen=True)
    class AttentionSpec(KVCacheSpec):
        num_kv_heads: int = 1
        head_size: int = 1
        dtype: object = None

        @property
        def page_size_bytes(self):
            return 2 * self.block_size * self.num_kv_heads * self.head_size * 2

    class FullAttentionSpec(AttentionSpec):
        pass

    class EncoderOnlyAttentionSpec(AttentionSpec):
        pass

    @dataclasses.dataclass(frozen=True)
    class MambaSpec(KVCacheSpec):
        shapes: tuple = ()
        dtypes: tuple = ()
        page_size_padded: int | None = None

    @dataclasses.dataclass
    class KVCacheGroupSpec:
        layer_names: list
        kv_cache_spec: KVCacheSpec

    @dataclasses.dataclass
    class KVCacheConfig:
        num_blocks: int
        kv_cache_tensors: list
        kv_cache_groups: list

    for c in (
        KVCacheSpec,
        AttentionSpec,
        FullAttentionSpec,
        EncoderOnlyAttentionSpec,
        MambaSpec,
        KVCacheGroupSpec,
        KVCacheConfig,
    ):
        setattr(m, c.__name__, c)
    return m


@pytest.fixture
def kvi(monkeypatch):
    try:
        import vllm.v1.kv_cache_interface as real

        return real
    except Exception:
        stub = _stub_kv_interface()
        monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
        monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
        monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", stub)
        return stub


@pytest.fixture
def fix(kvi):
    class _StubRunnerCls:
        initialize_kv_cache = staticmethod(lambda self, cfg: None)

    log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    ns = {"HPUModelRunner": _StubRunnerCls, "logger": log}
    exec(fix_mod.HELPERS, ns)
    return ns["_gaudi_fix_hybrid_pad"]


def make_runner(
    kvi,
    num_blocks,
    block,
    kb,
    rows,
    mla=False,
    wrap_list=False,
    n_attn=2,
    n_mamba=2,
    extra_rows=None,
    encoder_only=False,
):
    attn = kvi.FullAttentionSpec(block_size=block, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
    groups = [kvi.KVCacheGroupSpec([f"layers.{i}.attn" for i in range(n_attn)], attn)]
    if n_mamba:
        mamba = kvi.MambaSpec(
            block_size=block, shapes=((4, 8),), dtypes=(torch.float32,), page_size_padded=attn.page_size_bytes
        )
        groups.append(kvi.KVCacheGroupSpec([f"layers.{n_attn + i}.linear_attn" for i in range(n_mamba)], mamba))
    if encoder_only:
        enc = kvi.EncoderOnlyAttentionSpec(block_size=block, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
        groups.append(kvi.KVCacheGroupSpec(["vision.0.attn"], enc))
    cfg = kvi.KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    fctx = {}
    for i, ln in enumerate(groups[0].layer_names):
        r = extra_rows if (extra_rows is not None and i == len(groups[0].layer_names) - 1) else rows
        k = torch.zeros((r, 16)) if mla else torch.zeros((r, 2, 8))
        v = None if mla else torch.zeros((r, 2, 8))
        kv = (k, v, None, None)
        fctx[ln] = types.SimpleNamespace(kv_cache=[kv] if wrap_list else kv)
    for g in groups[1:]:
        for ln in g.layer_names:
            fctx[ln] = types.SimpleNamespace(kv_cache=(torch.zeros(num_blocks + 1, 4, 8),))
    return types.SimpleNamespace(
        kv_cache_config=cfg,
        attn_block_size=kb,
        runner_only_attn_layers={"vision.0.attn"},
        _PAD_BLOCK_ID=num_blocks,
        _PAD_SLOT_ID=num_blocks * kb,
        _MAMBA_PAD_BLOCK_ID=num_blocks,
        _resolve_block=lambda b: b,
        vllm_config=types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(static_forward_context=fctx),
        ),
    )


# ------------------------------------------------------------------------ 1. source patch


def test_source_patch(runner_src, tmp_path):
    target = tmp_path / "vllm_gaudi" / RUNNER_REL
    target.parent.mkdir(parents=True)
    target.write_text(runner_src)
    assert fix_mod.run(True, str(tmp_path)) == 1
    assert fix_mod.run(False, str(tmp_path)) == 0
    patched = target.read_text()
    assert patched.startswith(runner_src.rstrip("\n")) and fix_mod.MARKER in patched
    assert fix_mod.run(False, str(tmp_path)) == 0 and target.read_text() == patched
    assert fix_mod.run(True, str(tmp_path)) == 0


@pytest.mark.parametrize("idx", range(len(fix_mod.ANCHORS)))
def test_source_patch_refuses_drift(runner_src, idx):
    anchor = fix_mod.ANCHORS[idx]
    with pytest.raises(fix_mod.Drift):
        fix_mod.patch_runner(runner_src.replace(anchor, "", 1), {})
    with pytest.raises(fix_mod.Drift):
        fix_mod.patch_runner(runner_src + "\n" + anchor, {})


# ------------------------------------------------------- 2. the runner's decode padding code


def decode_pad_snippet(src):
    start = src.index(
        "        block_number = torch.ones((padded_batch_size, num_tokens), dtype=torch.int32) * self._PAD_BLOCK_ID"
    )
    end_marker = "        slot_mapping[num_decodes:].apply_(lambda _, ds=dummy_slots: next(ds))\n"
    end = src.index(end_marker, start) + len(end_marker)
    return textwrap.dedent(src[start:end])


def run_decode_pad(snippet, runner, block_table, ctx_lens, padded_bs, kb):
    num_decodes = len(ctx_lens)
    num_tokens = 1
    positions = torch.zeros((padded_bs, num_tokens), dtype=torch.int32)
    positions[:num_decodes, 0] = torch.tensor(ctx_lens, dtype=torch.int32)
    padded_index = torch.zeros((padded_bs, num_tokens), dtype=torch.int64)
    padded_index[:num_decodes] = positions.to(torch.int64)[:num_decodes]
    ns = dict(
        self=runner,
        torch=torch,
        itertools=itertools,
        padded_batch_size=padded_bs,
        num_tokens=num_tokens,
        num_decodes=num_decodes,
        block_table_cpu_tensor=block_table,
        index=padded_index[:num_decodes],
        decode_block_size=kb,
        padded_index=padded_index,
    )
    exec(snippet, ns)
    return ns["slot_mapping"].view(-1).tolist()


@pytest.mark.parametrize(
    "num_blocks,block,kb",
    [(1412, 1152, 128), (972, 896, 128), (1280, 640, 128), (592, 512, 128), (9988, 384, 128), (37, 1152, 128)],
)
def test_decode_padding(runner_src, kvi, fix, num_blocks, block, kb):
    snippet = decode_pad_snippet(runner_src)
    k = block // kb
    rows = (num_blocks * k + 1) * kb  # split allocation: num_kernel_blocks + 1 spare
    runner = make_runner(kvi, num_blocks, block, kb, rows)
    pool_end = num_blocks * block
    victim = num_blocks // k
    # one request holds the victim KV block; three decodes padded to a batch of four
    bt = torch.zeros((3, 64), dtype=torch.int32)
    bt[0, :8] = torch.tensor([victim * k + j for j in range(8)], dtype=torch.int32)
    bt[1, :2] = torch.tensor([k, k + 1], dtype=torch.int32)
    bt[2, :2] = torch.tensor([2 * k, 2 * k + 1], dtype=torch.int32)
    ctx = [8 * kb - 5, 2 * kb - 1, kb + 3]
    before = run_decode_pad(snippet, runner, bt, ctx, 4, kb)[3:]
    hit = [s for s in before if s < pool_end]
    assert len(hit) == 1 and hit[0] // block == victim, "stock pad slot should be inside a live KV block"
    fix(runner)
    after = run_decode_pad(snippet, runner, bt, ctx, 4, kb)[3:]
    assert all(pool_end <= s < rows for s in after)
    pad_block = runner._PAD_BLOCK_ID
    assert pad_block == num_blocks * k
    many = run_decode_pad(snippet, runner, bt[:1], ctx[:1], 8, kb)[1:]
    assert all(pool_end <= s < rows for s in many)
    mamba_pad = runner._MAMBA_PAD_BLOCK_ID
    assert mamba_pad == num_blocks


# ----------------------------------------------------------------------- 3. stub runners


def test_naive_sharing_layout(kvi, fix):
    r = make_runner(kvi, 100, 1152, 128, 101 * 1152)
    fix(r)
    assert r._PAD_BLOCK_ID == 101 * 9 - 1 and r._PAD_SLOT_ID == (101 * 9 - 1) * 128


def test_mla_and_list_wrapped_kv_cache(kvi, fix):
    r = make_runner(kvi, 1280, 640, 128, (1280 * 5 + 1) * 128, mla=True, wrap_list=True)
    fix(r)
    assert r._PAD_BLOCK_ID == 1280 * 5


def test_encoder_only_group_ignored(kvi, fix):
    r = make_runner(kvi, 64, 896, 128, (64 * 7 + 1) * 128, encoder_only=True)
    fix(r)
    assert r._PAD_BLOCK_ID == 64 * 7


@pytest.mark.parametrize("block,kb,n_mamba", [(128, 128, 0), (128, 128, 2), (768, 768, 2), (528, 528, 2)])
def test_inert_when_not_split(kvi, fix, block, kb, n_mamba):
    r = make_runner(kvi, 500, block, kb, 501 * block, n_mamba=n_mamba)
    fix(r)
    pad = (r._PAD_BLOCK_ID, r._PAD_SLOT_ID)
    assert pad == (500, 500 * kb)


def test_idempotent(kvi, fix):
    r = make_runner(kvi, 37, 1152, 128, (37 * 9 + 1) * 128)
    fix(r)
    first = (r._PAD_BLOCK_ID, r._PAD_SLOT_ID)
    fix(r)
    second = (r._PAD_BLOCK_ID, r._PAD_SLOT_ID)
    assert second == first


def test_fails_closed(kvi, fix):
    with pytest.raises(RuntimeError):
        fix(make_runner(kvi, 37, 1152, 128, 37 * 1152))
    with pytest.raises(RuntimeError):
        fix(make_runner(kvi, 37, 1152, 128, (37 * 9 + 1) * 128, extra_rows=(37 * 9 + 2) * 128))
    with pytest.raises(RuntimeError):
        fix(make_runner(kvi, 37, 1152, 128, (37 * 9 + 1) * 128 + 5))


def test_release_copy_roundtrip(tmp_path):
    """Optional: apply to a full copy of GAUDI_FIXES_ROOT and re-check."""
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("GAUDI_FIXES_ROOT not set")
    dst = tmp_path / "site"
    shutil.copytree(Path(root) / "vllm_gaudi", dst / "vllm_gaudi")
    assert fix_mod.run(False, str(dst)) == 0
    assert fix_mod.run(True, str(dst)) == 0
