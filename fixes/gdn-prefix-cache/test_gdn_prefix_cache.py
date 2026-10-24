# SPDX-License-Identifier: Apache-2.0
"""CPU tests for gdn-prefix-cache.

1. Source patch on a copy of a release (GAUDI_FIXES_ROOT): applies after hybrid-decode-pad, is
   idempotent, refuses to run without hybrid-decode-pad, and refuses a runner with a missing anchor.
2. The appended runner helpers on a stub runner with a small in-page layout (torch on CPU):
   - checkpoint save and restore are bit-exact, NaN payloads included, and touch nothing else;
   - a toy recurrence served through the wrapped _update_states / sample_tokens: a prefix-cache
     hit resumes from exactly the state a cold run reaches, in chunks and across preemption;
   - preempted requests hand their slot back; new slots, new attention blocks and the pad
     memory are zeroed;
   - setup refuses layouts and options the checkpoints do not support.

Run with: pytest fixes/gdn-prefix-cache
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIXES = HERE.parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fix_mod = _load(HERE / "apply.py", "gdn_prefix_cache_apply")
PKG_FILES = [(pkg, rel) for pkg, rel, _, _ in fix_mod.TARGETS]


# ------------------------------------------------------------------ 1. source patch


@pytest.fixture
def site(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("set GAUDI_FIXES_ROOT to a directory holding vllm/ and vllm_gaudi/ of a release")
    for pkg, rel in PKG_FILES:
        src = Path(root) / pkg / rel
        if not src.is_file():
            pytest.skip(f"{src} not found")
        dst = tmp_path / pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    return tmp_path


def _run(name, root, *extra):
    cmd = [sys.executable, str(FIXES / name / "apply.py"), "--root", str(root), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _texts(root):
    return {rel: (root / pkg / rel).read_text() for pkg, rel in PKG_FILES}


def test_needs_hybrid_decode_pad(site):
    before = _texts(site)
    proc = _run("gdn-prefix-cache", site)
    assert proc.returncode == 2 and "hybrid-decode-pad" in proc.stderr
    assert _texts(site) == before


def test_source_patch(site):
    assert _run("hybrid-decode-pad", site).returncode == 0
    assert _run("gdn-prefix-cache", site, "--check").returncode == 1
    proc = _run("gdn-prefix-cache", site)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    patched = _texts(site)
    for rel, text in patched.items():
        assert fix_mod.MARKER in text, rel
        compile(text, rel, "exec")
    assert _run("gdn-prefix-cache", site).returncode == 0 and _texts(site) == patched
    assert _run("gdn-prefix-cache", site, "--check").returncode == 0


@pytest.mark.parametrize("idx", range(len(fix_mod.RUNNER_EDITS)))
def test_source_patch_refuses_drift(site, idx):
    assert _run("hybrid-decode-pad", site).returncode == 0
    runner = site / "vllm_gaudi" / "v1/worker/hpu_model_runner.py"
    src = runner.read_text()
    with pytest.raises(fix_mod.Drift):
        fix_mod.patch_runner(src.replace(fix_mod.RUNNER_EDITS[idx][1], "", 1), {})


# ------------------------------------------------------------------ 2. runner helpers

torch = pytest.importorskip("torch")

KB = 4  # kernel block (tokens)
B = 16  # KV block (tokens): 4 kernel blocks
NB = 16  # KV blocks in the pool
LAYERS = 2  # layer positions
G = 2  # GDN groups
MAX_REQS = 4
NSLOTS = MAX_REQS * G + 2


class FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class MambaSpec:
    def __init__(self, block_size, mamba_type="gdn_attention", mode="align"):
        self.block_size, self.mamba_type, self.mamba_cache_mode = block_size, mamba_type, mode


class BlockTable:
    def __init__(self, rows, cols):
        self.t = torch.zeros((rows, cols), dtype=torch.int32)

    def get_cpu_tensor(self):
        return self.t


class StubRunner:
    """Mirrors the runner bookkeeping the helpers rely on. The 'forward' is a toy recurrence on the
    compact slots: state = state * 31 + token, per layer position and GDN group, as int32 bits."""

    def __init__(self, kv_ctx, kv_cfg, block_size=B):
        self.num_gdn = 1
        self.attn_block_size = KB
        self.kv_cache_config = kv_cfg
        self.vllm_config = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(enable_prefix_caching=True),
            speculative_config=None,
            compilation_config=types.SimpleNamespace(static_forward_context=kv_ctx),
        )
        self._compact_gdn_group_ids = {1, 2}
        self._num_gdn_groups = G
        self._PAD_BLOCK_ID = NB * B // KB  # the spare kernel block, as hybrid-decode-pad leaves it
        self.defragmenter = None
        self.warmup_mode = False
        self._gdn_req_to_base_slot = {}
        self._gdn_slot_free_list = list(range(MAX_REQS))
        self.input_batch = types.SimpleNamespace(req_ids=[], block_table=[BlockTable(MAX_REQS, 32) for _ in range(3)])
        self.step = None

    def initialize_kv_cache(self, kv_cache_config):
        return None

    def prepare_mamba_state_idxs(self, req_indices, offsets, target_bs):
        out = torch.full((3, target_bs), -1, dtype=torch.int32)
        for i, r in enumerate(req_indices):
            base = self._gdn_req_to_base_slot[self.input_batch.req_ids[r]]
            for g in range(G):
                out[1 + g, i] = base * G + g + 1
        return out

    def _update_states(self, so):
        for rid in so.finished_req_ids:
            slot = self._gdn_req_to_base_slot.pop(rid, None)
            if slot is not None:
                self._gdn_slot_free_list.append(slot)
        for rid in so.num_scheduled_tokens:
            if rid not in self._gdn_req_to_base_slot:
                self._gdn_req_to_base_slot[rid] = self._gdn_slot_free_list.pop()

    def sample_tokens(self, grammar_output=None):
        # self.step: [(row, req_id, ctx, tokens)]
        rows = [r for r, _, _, _ in self.step]
        idx = _ns["_gdn_pc_indices"](
            self, rows, [c for _, _, c, _ in self.step], [len(t) for _, _, _, t in self.step], len(rows)
        )
        for i, (_, _, _, toks) in enumerate(self.step):
            for g in range(G):
                slot = int(idx[1 + g, i])
                for _, _, conv, ssm in _entries(self):
                    s = ssm.view(torch.int32)[slot].reshape(-1)
                    c = conv.view(torch.int32)[slot].reshape(-1)
                    for t in toks:
                        s.mul_(31).add_(t + g)
                        c.mul_(7).add_(t)
        return "out"


def _entries(runner):
    seen, out = set(), []
    for pos in range(LAYERS):
        kc, vc = _ns["_gdn_pc_kv"](runner, f"attn.{pos}")[:2]
        conv, ssm = _ns["_gdn_pc_kv"](runner, f"gdn0.{pos}")[:2]
        if ssm.data_ptr() not in seen:
            seen.add(ssm.data_ptr())
            out.append((kc, vc, conv, ssm))
    return out


def _build(ssm_shape=(4, 8, 8), conv_shape=(8, 32), attn_block=B, share=True):
    rows = (NB * B // KB + 1) * KB
    ctx, attn_names, gdn_names = {}, [], [[] for _ in range(G)]
    for pos in range(LAYERS):
        kc = torch.zeros((rows, 1, 64), dtype=torch.bfloat16)
        vc = torch.zeros((rows, 1, 64), dtype=torch.bfloat16)
        ctx[f"attn.{pos}"] = types.SimpleNamespace(kv_cache=[(kc, vc, None, None)])
        attn_names.append(f"attn.{pos}")
        conv = torch.zeros((NSLOTS, *conv_shape), dtype=torch.bfloat16)
        ssm = torch.zeros((NSLOTS, *ssm_shape), dtype=torch.float32)
        for g in range(G):
            mine = (conv, ssm) if share or g == 0 else (conv.clone(), ssm.clone())
            ctx[f"gdn{g}.{pos}"] = types.SimpleNamespace(kv_cache=mine)
            gdn_names[g].append(f"gdn{g}.{pos}")
    groups = [types.SimpleNamespace(kv_cache_spec=FullAttentionSpec(attn_block), layer_names=attn_names)]
    groups += [types.SimpleNamespace(kv_cache_spec=MambaSpec(B), layer_names=gdn_names[g]) for g in range(G)]
    cfg = types.SimpleNamespace(kv_cache_groups=groups, num_blocks=NB)
    return ctx, cfg


_ns = {}


@pytest.fixture
def helpers():
    _ns.clear()
    logger = types.SimpleNamespace(info=lambda *a, **k: None)
    _ns.update(
        torch=torch,
        logger=logger,
        MambaSpec=MambaSpec,
        FullAttentionSpec=FullAttentionSpec,
        _GDN_MAMBA_TYPES=("gdn_attention", "linear_attention"),
        HPUModelRunner=StubRunner,
    )
    orig = {n: getattr(StubRunner, n) for n in ("initialize_kv_cache", "_update_states", "sample_tokens")}
    exec(compile(fix_mod._m(fix_mod.RUNNER_HELPERS), "helpers", "exec"), _ns)
    yield _ns
    for n, f in orig.items():
        setattr(StubRunner, n, f)


def _runner(**kw):
    ctx, cfg = _build(**kw)
    r = StubRunner(ctx, cfg)
    r.initialize_kv_cache(cfg)
    return r


def _so(sched, finished=(), preempted=(), zero=None):
    return types.SimpleNamespace(
        num_scheduled_tokens=sched,
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        new_block_ids_to_zero=zero,
        kv_cache_block_copies=None,
    )


def test_setup_layout(helpers):
    r = _runner()
    st = r._gdn_pc
    assert st["B"] == B and st["groups"] == [1, 2]
    # page 16 x 128 B = 2048 B, SSM 1024 B, conv 512 B: copy rows of 512 B
    assert (st["P"], st["S"], st["V"], st["a"], st["C"]) == (4, 2, 1, 2, 128)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: setattr(r, "_compact_gdn_group_ids", set()), "VLLM_COMPACT_GDN=1"),
        (lambda r: setattr(r.vllm_config, "speculative_config", object()), "speculative"),
        (lambda r: setattr(r, "_PAD_BLOCK_ID", NB * B // KB - 1), "hybrid-decode-pad"),
        (lambda r: setattr(r, "defragmenter", types.SimpleNamespace(enabled=True)), "defragmenter"),
        (lambda r: setattr(r.kv_cache_config.kv_cache_groups[1].kv_cache_spec, "mamba_cache_mode", "all"), "align"),
    ],
)
def test_setup_refuses(helpers, change, match):
    ctx, cfg = _build()
    r = StubRunner(ctx, cfg)
    change(r)
    with pytest.raises(RuntimeError, match=match):
        r.initialize_kv_cache(cfg)


def test_setup_refuses_layouts(helpers):
    with pytest.raises(RuntimeError, match="block sizes"):
        _runner(attn_block=2 * B)
    with pytest.raises(RuntimeError, match="share compact"):
        _runner(share=False)
    with pytest.raises(RuntimeError, match="does not fit"):
        _runner(ssm_shape=(16, 8, 8))


def test_inert_without_prefix_caching(helpers):
    ctx, cfg = _build()
    r = StubRunner(ctx, cfg)
    r.vllm_config.cache_config.enable_prefix_caching = False
    r.initialize_kv_cache(cfg)
    assert r._gdn_pc is None


def test_save_restore_bit_exact(helpers):
    r = _runner()
    g = torch.Generator().manual_seed(0)
    ents = _entries(r)
    for kc, vc, conv, ssm in ents:  # random bit patterns everywhere, NaN encodings included
        for t in (kc, vc, conv, ssm):
            t.view(torch.int16 if t.element_size() == 2 else torch.int32).copy_(
                torch.randint(-(2**15), 2**15, t.shape, generator=g, dtype=torch.int32).to(
                    torch.int16 if t.element_size() == 2 else torch.int32
                )
            )
    snap = [[t.clone() for t in e] for e in ents]
    slot, blk = 5, 7
    helpers["_gdn_pc_copy"](r, [(slot, blk)], False)
    for (kc, vc, conv, ssm), (kc0, vc0, conv0, ssm0) in zip(ents, snap, strict=False):
        assert torch.equal(conv.view(torch.int16), conv0.view(torch.int16))
        assert torch.equal(ssm.view(torch.int32), ssm0.view(torch.int32))
        outside = torch.ones(kc.shape[0], dtype=torch.bool)
        outside[blk * B : (blk + 1) * B] = False
        assert torch.equal(kc.view(torch.int16)[outside], kc0.view(torch.int16)[outside])
        assert torch.equal(vc.view(torch.int16)[outside], vc0.view(torch.int16)[outside])
        ssm[slot].zero_()
        conv[slot].zero_()
    helpers["_gdn_pc_copy"](r, [(slot, blk)], True)
    for (_, _, conv, ssm), (_, _, conv0, ssm0) in zip(ents, snap, strict=False):
        assert torch.equal(ssm.view(torch.int32), ssm0.view(torch.int32))
        assert torch.equal(conv.view(torch.int16), conv0.view(torch.int16))


def _serve(r, steps):
    """steps: list of (sched [(req_id, ctx, ntokens)], finished, preempted, block tables {req: [blocks 1-5]})."""
    for sched, finished, preempted, tables in steps:
        r._update_states(_so({rid: n for rid, _, n in sched}, finished, preempted))
        r.input_batch.req_ids = [rid for rid, _, _ in sched]
        for row, (rid, _, _) in enumerate(sched):
            for g in range(3):  # each cache group owns its own blocks: group g uses 1-5 shifted by 5 * g
                bt = r.input_batch.block_table[g].t
                bt[row].zero_()
                bt[row, : len(tables[rid])] = torch.tensor([b + 5 * g for b in tables[rid]], dtype=torch.int32)
        r.step = [(row, rid, ctx, [((ctx + j) * 13) % 97 for j in range(n)]) for row, (rid, ctx, n) in enumerate(sched)]
        r.sample_tokens()


def _slot_states(r, rid):
    base = r._gdn_req_to_base_slot[rid]
    return [ssm.view(torch.int32)[base * G + g + 1].clone() for _, _, _, ssm in _entries(r) for g in range(G)]


def test_hit_matches_cold(helpers):
    cold = _runner()
    # cold: 3 blocks in chunks of one block, then 5 decode tokens
    plan = [
        ([("a", 0, B)], (), (), {"a": [1]}),
        ([("a", B, B)], (), (), {"a": [1, 2]}),
        ([("a", 2 * B, B)], (), (), {"a": [1, 2, 3]}),
    ]
    plan += [([("a", 3 * B + i, 1)], (), (), {"a": [1, 2, 3, 4]}) for i in range(5)]
    _serve(cold, plan)
    want = _slot_states(cold, "a")

    hot = _runner()
    # request x fills blocks 1 and 2 and finishes; request b hits 2 blocks and computes the rest
    warm = [([("x", 0, 2 * B)], (), (), {"x": [1, 2]}), ([], ["x"], (), {})]
    hit = [([("b", 2 * B, B)], (), (), {"b": [1, 2, 3]})]
    hit += [([("b", 3 * B + i, 1)], (), (), {"b": [1, 2, 3, 4]}) for i in range(5)]
    _serve(hot, warm + hit)
    got = _slot_states(hot, "b")
    assert all(torch.equal(w, g) for w, g in zip(want, got, strict=False))
    assert helpers["_GDN_PC_STATS"]["restores"] == G


def test_preemption_resumes_from_checkpoint(helpers):
    cold = _runner()
    _serve(cold, [([("a", 0, 2 * B)], (), (), {"a": [1, 2]}), ([("a", 2 * B, 3)], (), (), {"a": [1, 2, 3]})])
    want = _slot_states(cold, "a")

    r = _runner()
    _serve(r, [([("a", 0, 2 * B)], (), (), {"a": [1, 2]}), ([("a", 2 * B, 1)], (), (), {"a": [1, 2, 3]})])
    free = len(r._gdn_slot_free_list)
    r._update_states(_so({}, preempted=["a"]))  # preempted, not rescheduled: the slot goes back
    assert "a" not in r._gdn_req_to_base_slot and len(r._gdn_slot_free_list) == free + 1
    _serve(r, [([("a", 2 * B, 3)], (), (), {"a": [1, 2, 3]})])  # resumes from the block-2 checkpoint
    assert all(torch.equal(w, g) for w, g in zip(want, _slot_states(r, "a"), strict=False))


def test_resume_off_boundary_refused(helpers):
    r = _runner()
    _serve(r, [([("a", 0, B)], (), (), {"a": [1]})])
    r._update_states(_so({}, preempted=["a"]))
    with pytest.raises(RuntimeError, match="not a multiple"):
        _serve(r, [([("a", B + 3, 1)], (), (), {"a": [1, 2]})])


def test_zeroing(helpers):
    r = _runner()
    ents = _entries(r)
    nan = torch.tensor(float("nan"))
    for kc, vc, conv, ssm in ents:
        ssm.fill_(nan)
        conv.fill_(nan)
        kc.fill_(nan)
        vc.fill_(nan)
    r._update_states(_so({"a": 1}, zero=[3]))
    base = r._gdn_req_to_base_slot["a"]
    for kc, _vc, conv, ssm in ents:
        for g in range(G):
            assert not ssm[base * G + g + 1].isnan().any() and not conv[base * G + g + 1].isnan().any()
        assert not kc[3 * B : 4 * B].isnan().any() and kc[4 * B : 5 * B].isnan().all()
    r.input_batch.req_ids = ["a"]
    r.input_batch.block_table[1].t[0, 0] = 1
    r.step = [(0, "a", 0, [5])]
    r.sample_tokens()
    for kc, vc, conv, ssm in ents:
        assert not kc[NB * B :].isnan().any() and not vc[NB * B :].isnan().any()
        assert not ssm[0].isnan().any() and not ssm[-1].isnan().any()
        assert not conv[0].isnan().any() and not conv[-1].isnan().any()


def test_scheduler_block_size():
    """The scheduler finds the GDN groups by their backend enum (vLLM's MambaAttentionBackendEnum)."""
    import enum

    class Backend(enum.Enum):
        MAMBA2 = "m2"
        GDN_ATTN = "gdn"
        LINEAR = "lin"

    edit = next(new for label, _, new in fix_mod.SCHED_EDITS if label == "module helper")
    ns = {"init_logger": lambda name: None, "__name__": "sched"}
    exec(compile(fix_mod._m(edit), "sched", "exec"), ns)

    def cfg(*specs):
        return types.SimpleNamespace(kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=s) for s in specs])

    attn = types.SimpleNamespace(block_size=896)
    gdn = types.SimpleNamespace(block_size=896, mamba_type=Backend.GDN_ATTN)
    assert ns["_gdn_pc_block_size"](cfg(attn, gdn, gdn)) == 896
    assert ns["_gdn_pc_block_size"](cfg(attn, types.SimpleNamespace(block_size=640, mamba_type=Backend.LINEAR))) == 640
    assert ns["_gdn_pc_block_size"](cfg(attn, types.SimpleNamespace(block_size=896, mamba_type=Backend.MAMBA2))) is None
    assert ns["_gdn_pc_block_size"](cfg(attn)) is None
