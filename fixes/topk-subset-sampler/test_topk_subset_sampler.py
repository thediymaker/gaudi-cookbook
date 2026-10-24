# SPDX-License-Identifier: Apache-2.0
"""CPU test for topk-subset-sampler.

1. The text patch: both files patch and compile; drifted sources are refused.
2. The patched sampler code, cut out of the patched topk_topp_sampler.py and run on CPU:
   * hpu_topk_subset_probs equals the stock path (apply_top_k_top_p_pytorch, then softmax) to 1e-5
     for mixed per-row top_k (1..1024) and top_p, with and without top_p, at every bucket size;
   * _hpu_topk_subset_sample draws with the right frequencies (chi-square);
   * it returns None (stock path) for no top_k, per-request generators, a row without top_k, and
     processed-logprobs mode;
   * the hpu_max_top_k tag picks the bucket without k.max().
3. The input-batch helper tags the max top_k from the CPU copy.

Needs torch and numpy (CPU) and unpatched vllm and vllm_gaudi source trees: the installed packages,
or GAUDI_FIXES_ROOT=<dir holding vllm/ and vllm_gaudi/>. Run with: pytest fixes/topk-subset-sampler
"""

import ast
import importlib.util
import os
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
SAMPLER_REL = Path("vllm/v1/sample/ops/topk_topp_sampler.py")
BATCH_REL = Path("vllm_gaudi/v1/worker/hpu_input_batch.py")


def _load_apply():
    spec = importlib.util.spec_from_file_location("topk_subset_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load_apply()


def _source(rel: Path) -> str:
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if root:
        path = Path(root) / rel
    else:
        spec = importlib.util.find_spec(rel.parts[0])
        if spec is None or not spec.submodule_search_locations:
            pytest.skip(f"{rel.parts[0]} not installed; set GAUDI_FIXES_ROOT")
        path = Path(list(spec.submodule_search_locations)[0]).parent / rel
    if not path.is_file():
        pytest.skip(f"{path} not found")
    text = path.read_text()
    if A.MARKER in text:
        pytest.skip(f"{path} is already patched")
    return text


@pytest.fixture(scope="module")
def srcs():
    return {"sampler": _source(SAMPLER_REL), "batch": _source(BATCH_REL)}


@pytest.fixture(scope="module")
def m(srcs):
    patched = A.patch_sampler(srcs["sampler"], srcs)
    tree = ast.parse(patched)
    funcs = {n.name: ast.get_source_segment(patched, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    code = "\n\n".join(funcs[f] for f in ("apply_top_k_only", "apply_top_k_top_p_pytorch"))
    ns = {"torch": torch}
    exec(patched[patched.index(f"# ---- {A.MARKER}") :] + "\n\n" + code, ns)
    return types.SimpleNamespace(**ns)


def test_text_patch_and_refusals(srcs):
    s = A.patch_sampler(srcs["sampler"], srcs)
    b = A.patch_batch(srcs["batch"], srcs)
    compile(s, "sampler", "exec")
    compile(b, "batch", "exec")
    assert s.count(A.SAMPLER_NEW) == 1 and b.count(A.BATCH_NEW) == 1
    for fn, key, old in (
        (A.patch_sampler, "sampler", A.SAMPLER_OLD),
        (A.patch_batch, "batch", A.BATCH_OLD),
    ):
        for mutated in (srcs[key].replace(old, "", 1), srcs[key] + "\n" + old):
            with pytest.raises(A.Drift):
                fn(mutated, srcs)


def test_subset_equals_stock(m):
    g = torch.Generator().manual_seed(0)
    V, B = 5000, 48
    for K, p_on in ((64, True), (64, False), (128, True), (256, True), (1024, True)):
        logits = torch.randn(B, V, generator=g) * 3
        k = torch.randint(1, K + 1, (B,), generator=g, dtype=torch.int32)
        k[0], k[1] = 1, K
        p = (torch.rand(B, generator=g) * 0.95 + 0.05) if p_on else None
        if p is not None:
            p[2] = 1.0
        ref = m.apply_top_k_top_p_pytorch(logits.clone(), k.clone(), None if p is None else p.clone())
        ref = ref.softmax(dim=-1, dtype=torch.float32)
        probs, ids = m.hpu_topk_subset_probs(logits.clone(), k, p, K)
        full = torch.zeros(B, V).scatter_(1, ids, probs)
        assert (full - ref).abs().max().item() < 1e-5, f"K={K} top_p={p_on}"


def test_sampling_frequencies_and_fallbacks(m):
    sampler = types.SimpleNamespace(logprobs_mode="raw_logprobs", use_fp64_gumbel=False)
    logits = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0, -1.0] + [-5.0] * 200])
    k = torch.tensor([4], dtype=torch.int32)
    p = torch.tensor([0.9])
    ref = m.apply_top_k_top_p_pytorch(logits.clone(), k.clone(), p.clone()).softmax(-1)[0]
    n = 20000
    draws = torch.cat(
        [
            m._hpu_topk_subset_sample(
                sampler, logits.expand(1000, -1).clone(), {}, k.expand(1000).clone(), p.expand(1000).clone()
            )[0]
            for _ in range(n // 1000)
        ]
    )
    counts = torch.bincount(draws, minlength=logits.shape[1]).float()
    support = ref > 0
    exp = ref[support] * n
    assert counts[~support].sum().item() == 0
    assert (((counts[support] - exp) ** 2) / exp).sum().item() < 20.0

    V = 5000
    lg = torch.randn(4, V)
    kk = torch.full((4,), 64, dtype=torch.int32)
    assert m._hpu_topk_subset_sample(sampler, lg, {}, None, torch.full((4,), 0.9)) is None
    assert m._hpu_topk_subset_sample(sampler, lg, {0: object()}, kk, None) is None
    kv = kk.clone()
    kv[3] = V
    assert m._hpu_topk_subset_sample(sampler, lg, {}, kv, None) is None
    s2 = types.SimpleNamespace(logprobs_mode="processed_logprobs", use_fp64_gumbel=False)
    assert m._hpu_topk_subset_sample(s2, lg, {}, kk, None) is None

    class NoMax(torch.Tensor):
        def max(self, *a, **kw):
            raise AssertionError("k.max() called despite the tag")

    kt = kk.clone().as_subclass(NoMax)
    kt.hpu_max_top_k = 64
    out = m._hpu_topk_subset_sample(sampler, lg, {}, kt, None)
    assert out is not None and out[0].shape == (4,)
    assert bool((lg.topk(64, dim=-1).indices == out[0].unsqueeze(1)).any(dim=1).all())


def test_batch_helper_tags_max_from_cpu_copy(srcs):
    patched = A.patch_batch(srcs["batch"], srcs)
    ns: dict = {}
    exec(patched[patched.index("def _hpu_tag_max_top_k(") :], ns)
    cpu = np.array([64, 20, 202048, 40], dtype=np.int32)
    t = ns["_hpu_tag_max_top_k"](torch.tensor([64, 20, 40]), cpu, [0, 1, 3])
    assert getattr(t, "hpu_max_top_k", None) == 64
    t = ns["_hpu_tag_max_top_k"](torch.tensor([64, 202048]), cpu, [0, 2])
    assert getattr(t, "hpu_max_top_k", None) == 202048
