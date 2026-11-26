"""HPU-free unit test for ML4_INDEX_KERNEL (mistral_large4_hpu.k1ix). No device is opened: torch CPU only.

  1. The kernels' selection algorithm, emulated on CPU step for step (k1_sc's order key + unseen flag, the four 4-bit
     radix passes over NCH chunks, k1_fin's per-chunk offsets / tie quotas, k1_cm1's take masks, k1_cm2's ascending
     compaction and dense rows), equals the plugin's decode path (select_topk_key + where(see) + dense_topk) on
     random, tied, all-equal, -inf-heavy, short and mixed rows, 4K-128K keys, 1-8 rows.
  2. eligible() gates: off by default; only decode rows on the DECODE_FAST + key-select path, U8 cache, no DCP.
  3. setup_env(): the kernel-library chain with and without the mxk MoE library, idempotent, PT_HPU_LAZY_ACC_PAR_MODE=0,
     missing libraries turn the flag off; mxk.setup_env() calls it.
Run: TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<pkg> python -m pytest -q tests/test_k1_cpu.py   (or python <this file>)
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
for k, v in {"ML4_INDEX_DECODE_FAST": "1", "ML4_INDEX_U8": "1"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml4_hpu"))

import torch  # noqa: E402

from mistral_large4_hpu import indexer as IX  # noqa: E402

K_OUT = 2048


# ----------------------------------------------------------------------------------------------- 1. algorithm
def ref_select(score_bf16: torch.Tensor, see: torch.Tensor, pos: torch.Tensor, k: int) -> torch.Tensor:
    """The plugin's decode select (indexer.ML4Indexer.select, DECODE_FAST + key) from the bf16 score rows."""
    score = score_bf16.masked_fill(~see, float("-inf"))
    idx = IX.select_topk_key(score, k)
    t = torch.where(torch.gather(see, 2, idx), idx, -1)
    return torch.where((pos <= k - 1).unsqueeze(-1), IX.dense_topk(pos, k), t)


def emu_select(score_bf16: torch.Tensor, see: torch.Tensor, pos: torch.Tensor, k: int, nch: int = 24) -> torch.Tensor:
    """The k1 kernels' algorithm (k1_sc .. k1_cm2) on CPU."""
    B, S, Lk = score_bf16.shape
    r16 = score_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
    r16 = torch.where(r16 == 0x8000, 0, r16)  # -0 -> +0
    u = torch.where(r16 >= 0x8000, 0xFFFF - r16, r16 + 0x8000)  # order-preserving u16
    u = torch.where(see, u, 0x80)  # unseen: -inf's key (lowest finite)
    flag = (~see).to(torch.int32)
    nv = Lk // 64
    per = -(-nv // nch)
    out = torch.full((B, S, k), -7, dtype=torch.int64)
    for b in range(B):
        for t in range(S):
            p = int(pos[b, t])
            if p <= k - 1:
                j = torch.arange(k)
                out[b, t] = torch.where(j <= p, j, -1)
                continue
            uu = u[b, t]
            base = 0
            for shift in (12, 8, 4, 0):  # k1_sc pass 1, k1_p2..p4: 4 bits each
                nn = (uu - base) >> shift  # arithmetic shift (keys below the prefix: < 0)
                tot = torch.stack([(nn >= j).sum() for j in range(17)])
                base += (int((tot >= k).sum()) - 1) << shift  # k1_digit
            v = base
            need = k - int((uu > v).sum())
            gtb = eqb = 0
            o_rows = []
            for c in range(nch):
                v0 = c * per
                v1 = min(v0 + per, nv)
                if v0 >= v1:
                    continue
                seg = uu[v0 * 64 : v1 * 64]
                gt, eq = seg > v, seg == v
                off = gtb + min(eqb, need)
                q = max(min(need - eqb, int(eq.sum())), 0)
                take = gt | (eq & (torch.cumsum(eq.int(), 0) <= q))
                sidx = torch.arange(v0 * 64, v1 * 64)[take]
                vals = torch.where(flag[b, t, v0 * 64 : v1 * 64][take] == 0, sidx, -1)
                out[b, t, off : off + len(vals)] = vals
                o_rows.append((off, len(vals)))
                gtb += int(gt.sum())
                eqb += int(eq.sum())
            assert sum(n for _, n in o_rows) == k, (b, t, o_rows)
    return out


def make_case(Lk: int, B: int, S: int, kind: str, seed: int):
    g = torch.Generator().manual_seed(seed * 7919 + Lk + 13 * S + B)
    lens = [Lk - int(torch.randint(0, 128, (1,), generator=g)) for _ in range(B)]
    if kind == "short":
        lens = [int(torch.randint(S + 1, K_OUT, (1,), generator=g)) for _ in range(B)]
    if kind == "mixed" and B > 1:
        lens[0] = int(torch.randint(S + 1, K_OUT, (1,), generator=g))
    pos = torch.stack([torch.arange(L - S, L) for L in lens]).long()
    if kind == "ties":
        sc = torch.randint(-3, 4, (B, S, Lk), generator=g).float() * 0.5
    elif kind == "zeros":
        sc = torch.zeros(B, S, Lk)
    else:
        sc = torch.randn(B, S, Lk, generator=g) * torch.exp(torch.randn(B, S, 1, generator=g))
    see = torch.arange(Lk).view(1, 1, Lk) <= pos.unsqueeze(-1)
    if kind == "invalid":  # tag-invalid keys: -inf, unseen
        see = see & (torch.rand(B, 1, Lk, generator=g) > 0.03)
    if kind == "neg0":
        sc = torch.where(torch.rand(B, S, Lk, generator=g) < 0.5, -0.0, 0.0) * torch.ones(B, S, Lk)
    return sc.to(torch.bfloat16), see, pos


def test_algorithm_matches_plugin():
    n = 0
    for Lk in (4096, 16384, 65536, 131072):
        for S in (1, 4, 8):
            for kind in ("rand", "ties", "zeros", "invalid", "short", "mixed", "neg0"):
                B = 2 if kind == "mixed" else 1
                if Lk > 16384 and S > 1 and kind not in ("ties", "invalid"):
                    continue  # keep the CPU run short
                sc, see, pos = make_case(Lk, B, S, kind, 0)
                ref = ref_select(sc, see, pos, K_OUT)
                got = emu_select(sc, see, pos, K_OUT, nch=max(1, 24 // S))  # k1ix.select's auto chunk count
                assert torch.equal(ref, got), (Lk, S, kind, int((ref != got).sum()))
                n += 1
    assert n > 30
    sc, see, pos = make_case(16384, 1, 2, "ties", 1)  # other chunk counts
    for nch in (1, 5, 24, 64):
        assert torch.equal(ref_select(sc, see, pos, K_OUT), emu_select(sc, see, pos, K_OUT, nch=nch)), nch


# ----------------------------------------------------------------------------------------------- 2. gating
class _Cache:
    def __init__(self, fake=False, dcp=None):
        self.fake, self.dcp = fake, dcp


def _k1(env: dict):
    for k in ("ML4_INDEX_KERNEL", "ML4_INDEX_KERNEL_LIB"):
        os.environ.pop(k, None)
    os.environ.update(env)
    from mistral_large4_hpu import k1ix

    return importlib.reload(k1ix)


def test_default_off():
    k1 = _k1({})
    assert not k1.ENABLED
    assert not k1.eligible(1, 4096, 2048, _Cache())
    assert not k1.load_ops()


def test_eligible_gates():
    k1 = _k1({"ML4_INDEX_KERNEL": "1"})
    k1._ops_loaded = True  # as after load_ops() in the model build
    try:
        c = _Cache()
        assert k1.eligible(1, 4096, 2048, c)
        assert k1.eligible(8, 131072, 2048, c)
        assert not k1.eligible(9, 4096, 2048, c)  # prompt chunk
        assert not k1.eligible(1, 2048, 2048, c)  # k_out == Lk (short-table rows)
        assert not k1.eligible(1, 4096 + 32, 2048, c)  # Lk not a multiple of 64
        assert not k1.eligible(1, 4096, 2048, _Cache(fake=True))
        assert not k1.eligible(1, 4096, 2048, _Cache(dcp=(8, 0)))
        fast = IX.DECODE_FAST
        IX.DECODE_FAST = False
        try:
            assert not k1.eligible(1, 4096, 2048, c)
        finally:
            IX.DECODE_FAST = fast
        k1._ops_loaded = False
        assert not k1.eligible(1, 4096, 2048, c)  # ops not loaded: native
    finally:
        k1._ops_loaded = False


# ----------------------------------------------------------------------------------------------- 3. env chain
def _fake_libs(d: str, name: str, tag: bytes, op: str):
    open(os.path.join(d, name), "wb").write(b"\x7fELF...." + tag + b" vtest\0rest")
    open(os.path.join(d, op), "wb").write(b"\x7fELF")


def _clean_env():
    for k in (
        "GC_KERNEL_PATH",
        "MXK_TPC_FWD_LIB",
        "K1_TPC_FWD_LIB",
        "PT_HPU_LAZY_ACC_PAR_MODE",
        "PT_HPU_RECIPE_CACHE_CONFIG",
    ):
        os.environ.pop(k, None)


def test_env_chain_alone_and_missing():
    _clean_env()
    with tempfile.TemporaryDirectory() as d:
        k1 = _k1({"ML4_INDEX_KERNEL": "1", "ML4_INDEX_KERNEL_LIB": d})
        assert k1.setup_env(None) is None and not k1.ENABLED  # no libraries: off
        _fake_libs(d, "libk1_tpc.so", b"k1_tpc", "libk1_op_lazy.so")
        k1 = _k1({"ML4_INDEX_KERNEL": "1", "ML4_INDEX_KERNEL_LIB": d})
        os.environ["GC_KERNEL_PATH"] = k1.STOCK
        assert k1.setup_env(None) == "k1-vtest"
        assert os.environ["GC_KERNEL_PATH"] == os.path.join(d, "libk1_tpc.so")
        assert os.environ["K1_TPC_FWD_LIB"] == k1.STOCK
        assert os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] == "0"
        assert k1.env_ok()
        k1.setup_env(None)  # idempotent
        assert os.environ["GC_KERNEL_PATH"] == os.path.join(d, "libk1_tpc.so")
        assert os.environ["K1_TPC_FWD_LIB"] == k1.STOCK
    _clean_env()


def test_env_chain_behind_mxk():
    _clean_env()
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as m, tempfile.TemporaryDirectory() as rc:
        _fake_libs(d, "libk1_tpc.so", b"k1_tpc", "libk1_op_lazy.so")
        _fake_libs(m, "libmxk_tpc.so", b"mxk_tpc", "libmxk_op_lazy.so")
        os.environ.update(
            {
                "ML4_MXK": "1",
                "ML4_MXK_LIB": m,
                "GC_KERNEL_PATH": "/usr/lib/habanalabs/libtpc_kernels.so",
                "PT_HPU_RECIPE_CACHE_CONFIG": rc + "/cache,false,8192",
            }
        )
        _k1({"ML4_INDEX_KERNEL": "1", "ML4_INDEX_KERNEL_LIB": d})
        from mistral_large4_hpu import mxk

        mxk = importlib.reload(mxk)
        for _ in range(2):  # idempotent
            mxk.setup_env()
            assert os.environ["GC_KERNEL_PATH"] == os.path.join(m, "libmxk_tpc.so")
            assert os.environ["MXK_TPC_FWD_LIB"] == os.path.join(d, "libk1_tpc.so")
            assert os.environ["K1_TPC_FWD_LIB"] == "/usr/lib/habanalabs/libtpc_kernels.so"
            assert os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] == "0"
        assert "mxk-vtest_k1-vtest" in os.environ["PT_HPU_RECIPE_CACHE_CONFIG"]
        from mistral_large4_hpu import k1ix

        assert mxk.env_ok() and k1ix.env_ok()
    for k in ("ML4_MXK", "ML4_MXK_LIB"):
        os.environ.pop(k, None)
    _clean_env()


def test_no_device_opened():
    assert "habana_frameworks" not in sys.modules and "habana_frameworks.torch" not in sys.modules


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
    print("ALL PASS")
