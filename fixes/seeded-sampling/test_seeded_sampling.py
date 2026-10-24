# SPDX-License-Identifier: Apache-2.0
"""CPU test for seeded-sampling.

1. The text patch: both files patch and compile, and drifted sources are refused (an anchor
   removed or duplicated, the rejection-sampler replacement gone).
2. The patched sampler code, cut out of the patched topk_topp_sampler.py and run on CPU (the HPU
   branch is switched on for CPU tensors):
   * the counter noise matches a pure-Python integer reference and is an Exp(1) sample;
   * Gumbel-max with this noise samples a fixed distribution with the right frequencies;
   * a seeded row's token depends only on (seed, step), not on its row or its neighbours;
   * padded copies of a seeded row share its token and advance the generator once per call;
   * unseeded rows still draw fresh noise and plain generators keep the stock loop;
   * the same seed replays the same token sequence.

Needs torch (CPU) and unpatched vllm and vllm_gaudi source trees: the installed packages, or
GAUDI_FIXES_ROOT=<dir holding vllm/ and vllm_gaudi/>. Run with: pytest fixes/seeded-sampling
"""

import ast
import importlib.util
import math
import os
import random
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
SAMPLER_REL = Path("vllm/v1/sample/ops/topk_topp_sampler.py")
RUNNER_REL = Path("vllm_gaudi/v1/worker/hpu_model_runner.py")
REJECTION_REL = Path("vllm_gaudi/v1/sample/hpu_rejection_sampler.py")
SAMPLER_FUNCS = ("empty_exponential_noise_like", "sample_with_exponential_noise", "random_sample")


def _load_apply():
    spec = importlib.util.spec_from_file_location("seeded_sampling_apply", HERE / "apply.py")
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
    return {"sampler": _source(SAMPLER_REL), "runner": _source(RUNNER_REL), "rejection": _source(REJECTION_REL)}


@pytest.fixture(scope="module")
def m(srcs):
    """The patched sampler functions and the appended block, with the HPU branch enabled on CPU."""
    patched = A.patch_sampler(srcs["sampler"], srcs)
    tree = ast.parse(patched)
    funcs = {n.name: ast.get_source_segment(patched, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    code = "\n\n".join(funcs[f] for f in SAMPLER_FUNCS)
    code = code.replace('if q.device.type == "hpu":', 'if q.device.type in ("hpu", "cpu"):', 1)
    block = patched[patched.index(f"# ---- {A.MARKER}") :]
    ns = {"torch": torch}
    exec(block + "\n\n" + code, ns)
    return types.SimpleNamespace(**ns)


def test_text_patch_and_refusals(srcs):
    s = A.patch_sampler(srcs["sampler"], srcs)
    r = A.patch_runner(srcs["runner"], srcs)
    compile(s, "sampler", "exec")
    compile(r, "runner", "exec")
    assert "def _hpu_seeded_noise_(" in s and "torch.Generator(device=self.device)" not in r
    cases = [
        ("sampler", A.SAMPLER_OLD, ""),
        ("sampler", A.SAMPLER_OLD, A.SAMPLER_OLD + "\n" + A.SAMPLER_OLD),
        ("runner", A.RUNNER_GEN_OLD, ""),
        ("runner", A.RUNNER_DISCARD_OLD, ""),
        ("rejection", A.REJECTION_REBIND, "pass"),
    ]
    for key, old, new in cases:
        ctx = dict(srcs, **{key: srcs[key].replace(old, new, 1)})
        with pytest.raises(A.Drift):
            A.patch_sampler(ctx["sampler"], ctx)
            A.patch_runner(ctx["runner"], ctx)


def _py_mix32(x):
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return x ^ (x >> 16)


def _py_noise(k0, k1, j):
    x = _py_mix32((j + k0) & 0xFFFFFFFF)
    x = _py_mix32(x ^ (k1 & 0xFFFFFFFF))
    return -math.log(((x >> 9) + 0.5) * 2.0**-23)


def test_noise_matches_reference_and_is_exponential(m):
    V = 50000
    keys = [m.hpu_seeded_key(s, t) for s, t in ((0, 0), (1, 0), (1, 1), (-5, 3), (2**63 + 7, 10**9))]
    noise = m.hpu_counter_exponential(torch.tensor(keys, dtype=torch.int32), V)
    worst = max(
        abs(float(noise[r, j]) - _py_noise(k0, k1, j)) / _py_noise(k0, k1, j)
        for r, (k0, k1) in enumerate(keys)
        for j in (0, 1, 2, 777, V - 1)
    )
    assert worst < 1e-6
    assert bool((noise > 0).all()) and bool(torch.isfinite(noise).all())
    flat = noise.flatten().double()
    assert abs(float(flat.mean()) - 1) < 0.01 and abs(float(flat.var()) - 1) < 0.02
    xs = flat[:100000].sort().values
    n = xs.numel()
    ks = float((1 - torch.exp(-xs) - torch.arange(1, n + 1, dtype=torch.float64) / n).abs().max())
    assert ks < 1.63 / math.sqrt(n)
    corr = float(torch.corrcoef(torch.stack([noise[1, :20000], noise[2, :20000]]))[0, 1])
    assert abs(corr) < 0.03


def test_gumbel_frequencies(m):
    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05] + [0.0] * 11)
    counts = torch.zeros(16)
    rnd = random.Random(0)
    for _ in range(400):
        probs = p.repeat(50, 1)
        q = torch.empty_like(probs)
        assert not m._hpu_seeded_noise_(q, {i: m.HpuSeededGenerator.create(rnd.randrange(1 << 40)) for i in range(50)})
        counts += torch.bincount((probs / q).argmax(dim=-1), minlength=16).float()
    exp = p * counts.sum()
    nz = exp > 0
    assert counts[~nz].sum() == 0
    assert float(((counts[nz] - exp[nz]) ** 2 / exp[nz]).sum()) < 13.28  # chi2, df=4, 1%


def test_random_sample(m):
    V = 4096
    torch.manual_seed(0)
    P = torch.randn(8, V).softmax(-1)
    G = m.HpuSeededGenerator.create

    def sample(rows, gens):
        return m.random_sample(P[rows].clone(), gens).tolist()

    a = sample([3], {0: G(11)})
    b = sample([0, 1, 2, 4, 5, 3, 6], {5: G(11), 1: G(99)})
    c = sample([3] * 8, {7: G(11)})
    assert a[0] == b[5] == c[7]

    g = G(5)
    out = sample([2, 6, 2, 2, 2], {0: g, 2: g, 3: g, 4: g})
    assert len({out[0], out[2], out[3], out[4]}) == 1 and g.hpu_step == 1

    o3 = sample([0, 1, 2], {0: G(1), 1: G(2), 2: G(3)})
    assert o3 == [sample([r], {0: G(s)})[0] for r, s in zip((0, 1, 2), (1, 2, 3), strict=True)]

    assert len({tuple(sample([0, 1, 2, 3], {})) for _ in range(8)}) > 1
    assert len({sample([0, 1], {0: G(8)})[1] for _ in range(40)}) > 1

    q = torch.empty(2, V)
    plain = torch.Generator().manual_seed(3)
    rest = m._hpu_seeded_noise_(q, {0: plain, 1: G(3)})
    assert list(rest) == [0] and rest[0] is plain
    assert sample([4], {0: torch.Generator().manual_seed(77)}) == sample([4], {0: torch.Generator().manual_seed(77)})

    def run(seed, steps=30):
        g = G(seed)
        return [sample([6], {0: g})[0] for _ in range(steps)], g.hpu_step

    s1, n1 = run(1234)
    s2, _ = run(1234)
    s3, _ = run(1235)
    assert s1 == s2 and n1 == 30
    assert len(set(s1)) > 1 and s1 != s3
    assert m._hpu_seeded_noise_(torch.empty(2, V), {5: G(1)}) == {}
