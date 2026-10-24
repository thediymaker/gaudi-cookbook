# SPDX-License-Identifier: Apache-2.0
"""Conv-state slot simulator: the real InklingConvState, fed runner-shaped prefill and decode metadata, must
match a whole-sequence causal conv exactly through chunked prefill, prefix-cache hits (exact repeats,
extensions, a shared cached prefix under concurrency), batch padding and block-boundary decode. Runs with
INKLING_STEP_HOIST on and off. Needs torch; skips without it."""

import random
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from inkling_hpu import conv_state as cs  # noqa: E402
from inkling_hpu import ops  # noqa: E402

BS, TAPS, CH, W = 8, 3, 5, 4
NUM_BLOCKS = 400


class Pool:
    """Block allocator with a prefix cache of full blocks, like vLLM's (block 0 = null block)."""

    def __init__(self):
        self.free = list(range(1, NUM_BLOCKS))
        random.shuffle(self.free)
        self.cache = {}

    def alloc(self):
        if not self.free:  # evict a cached block
            h, b = next(iter(self.cache.items()))
            del self.cache[h]
            return b
        return self.free.pop()


class Req:
    def __init__(self, pool, toks, x_full, plen):
        self.pool, self.toks, self.x, self.plen = pool, toks, x_full, plen
        self.blocks = []
        n_hit = 0
        for i in range((len(toks) - 1) // BS):  # hits capped at n - 1 tokens, as get_computed_blocks
            h = tuple(toks[: (i + 1) * BS])
            if h not in pool.cache:
                break
            self.blocks.append(pool.cache[h])
            n_hit += 1
        self.done = self.hit = n_hit * BS
        self.out = [None] * len(toks)

    def ensure(self, upto):
        while len(self.blocks) * BS < upto:
            self.blocks.append(self.pool.alloc())

    def cache_full(self):
        for i in range(self.done // BS):
            self.pool.cache.setdefault(tuple(self.toks[: (i + 1) * BS]), self.blocks[i])


class Sim:
    def __init__(self):
        self.weight = torch.randn(CH, W)
        self.m = cs.InklingConvState(NS(compilation_config=NS(static_forward_context={})), 1, 1, W, "L0")
        self.m.kv_cache = (torch.full((NUM_BLOCKS + 1, TAPS, CH), float("nan")),)  # stale garbage
        self.pool = Pool()

    def reference(self, x_full):
        xin = torch.cat([torch.zeros(TAPS, CH), x_full], 0)
        return x_full + F.conv1d(xin.t().unsqueeze(0), self.weight.unsqueeze(1), groups=CH)[0].t()

    def prefill_step(self, batch, pad_bs, pad_seq):
        ops.STEP_CACHE.clear()  # one step == one model forward
        ctx_rows, sm_rows, xs, ctxs, qls = [], [], [], [], []
        for r, n in batch:
            s = r.done
            r.ensure(s + n)
            ctxs.append(s)
            qls.append(n)
            ctx_rows.append(r.blocks[: -(-s // BS)])
            sm_rows.append([r.blocks[p // BS] * BS + p % BS for p in range(s, s + n)])
            xs.append(r.x[s : s + n])
        tb = max([len(c) for c in ctx_rows] + [0])
        tb = tb + random.randint(0, 2) if tb else 0  # bucket padding
        bsz, length = len(batch) + pad_bs, max(qls) + pad_seq
        x = torch.randn(bsz, length, CH) * 100  # pad garbage
        sm = torch.full((bsz, length), -1, dtype=torch.long)
        for i, (xx, row) in enumerate(zip(xs, sm_rows, strict=True)):
            x[i, : len(row)] = xx
            sm[i, : len(row)] = torch.tensor(row)
        bl = None
        if tb:
            bl = torch.full((bsz, tb), -1, dtype=torch.long)
            for i, c in enumerate(ctx_rows):
                if c:
                    bl[i, : len(c)] = torch.tensor(c)
            bl = bl.flatten()
        md = NS(
            is_prompt=True,
            slot_mapping=sm.flatten(),
            block_list=bl,
            context_lens_tensor=torch.tensor(ctxs + [0] * pad_bs),
            seq_lens_tensor=torch.tensor(qls + [0] * pad_bs),
            block_size=BS,
        )
        y = self.m.conv(0, x, self.weight, True, md)
        for i, (r, n) in enumerate(batch):
            for j in range(n):
                r.out[r.done + j] = y[i, j]
            r.done += n
            r.cache_full()

    def decode_step(self, reqs, pad_bs, pad_blocks):
        ops.STEP_CACHE.clear()
        bl, bg, bu, xs = [], [], [], []
        for i, r in enumerate(reqs):
            p = r.done
            r.ensure(p + 1)
            nb = p // BS + 1
            bl += r.blocks[:nb]
            bg += [i] * nb
            bu += [BS] * (nb - 1) + [p % BS + 1]
            xs.append(r.x[p])
        bl += [NUM_BLOCKS] * pad_blocks
        bg += [-1] * pad_blocks
        bu += [1] * pad_blocks
        x = torch.randn(len(reqs) + pad_bs, CH) * 100
        for i, xx in enumerate(xs):
            x[i] = xx
        md = NS(
            is_prompt=False,
            block_list=torch.tensor(bl),
            block_groups=torch.tensor(bg),
            block_usage=torch.tensor(bu, dtype=torch.bfloat16),
            block_size=BS,
        )
        y = self.m.conv(0, x, self.weight, False, md)
        for i, r in enumerate(reqs):
            r.out[r.done] = y[i]
            r.done += 1
            r.cache_full()

    def run(self, reqs, chunk_sizes, decode_len):
        while any(r.done < r.plen for r in reqs):
            batch = [(r, min(random.choice(chunk_sizes), r.plen - r.done)) for r in reqs if r.done < r.plen]
            random.shuffle(batch)
            self.prefill_step(batch, random.randint(0, 2), random.randint(0, 5))
        for _ in range(decode_len):
            dec = [r for r in reqs if r.done < len(r.toks)]
            if not dec:
                break
            random.shuffle(dec)
            self.decode_step(dec, random.randint(0, 3), random.randint(0, 6))

    def errors(self, r):
        """Positions this request computed (cache hits excluded) that differ from the reference."""
        ref = self.reference(r.x)
        bad = []
        for p in range(r.hit, len(r.toks)):
            if r.out[p] is None:
                continue
            if torch.isnan(r.out[p]).any() or (r.out[p] - ref[p]).abs().max() > 1e-4:
                bad.append(p)
        return bad


def _xfor(toks):
    """Inputs as a function of (token, position), so equal prefixes give equal inputs (a deterministic
    model)."""
    table = torch.randn(10, 200, CH, generator=torch.Generator().manual_seed(1234))
    return torch.stack([table[t, p] for p, t in enumerate(toks)])


@pytest.mark.parametrize("hoist", [False, True])
def test_conv_state_matches_whole_sequence(monkeypatch, hoist):
    monkeypatch.setattr(ops, "STEP_HOIST", hoist)
    torch.manual_seed(0)
    random.seed(0)
    sim = Sim()
    cases = []

    def new(toks, plen, x=None):
        r = Req(sim.pool, toks, _xfor(toks) if x is None else x, plen)
        return r

    # 1) cold requests, odd chunk sizes, batched decode
    reqs = []
    for _ in range(5):
        n = random.randint(20, 90)
        toks = [random.randint(0, 9) for _ in range(n)]
        reqs.append(new(toks, n - 10, torch.randn(n, CH)))
    sim.run(reqs, [3, 5, 8, 13, 16, 30], 10)
    cases += [(f"cold/{i}", r) for i, r in enumerate(reqs)]

    # 2) exact repeat and extensions of a cached prompt
    base = [random.randint(0, 9) for _ in range(61)]
    seed = new(base, 61)
    sim.run([seed], [16], 0)
    cases.append(("prefix/seed", seed))
    for tag, toks in (
        ("repeat", base),
        ("extend", base + [3, 1, 4, 1, 5, 9, 2, 6]),
        ("extend-long", base + [random.randint(0, 9) for _ in range(40)]),
    ):
        r = new(toks, len(toks) - 4)
        sim.run([r], [7, 16], 4)
        cases.append((f"prefix/{tag}", r))

    # 3) concurrent requests sharing a cached system prefix
    sysp = [random.randint(0, 9) for _ in range(40)]
    sim.run([new(sysp + [0], 41)], [16], 0)
    shared = []
    for _ in range(6):
        toks = sysp + [random.randint(0, 9) for _ in range(random.randint(5, 30))]
        shared.append(new(toks, len(toks) - 3))
    sim.run(shared, [4, 9, 16], 3)
    cases += [(f"shared/{i}", r) for i, r in enumerate(shared)]

    # 4) prompt lengths around block multiples, decode across several block boundaries
    for i, n in enumerate([BS - 1, BS, BS + 1, 2 * BS, 3 * BS - 1]):
        toks = [random.randint(0, 9) for _ in range(n + 3 * BS)]
        r = new(toks, n)
        sim.run([r], [BS, 5], 3 * BS)
        cases.append((f"boundary/{i}", r))

    assert len(cases) == 20
    failures = {name: bad[:3] for name, r in cases if (bad := sim.errors(r))}
    assert not failures, failures
    assert any(r.hit for _, r in cases), "no prefix-cache hit was exercised"


def test_state_bytes_per_block_tp8():
    """At TP 8 one block's conv rows are 3 x (2 x 128 + 2 x 4096) bf16 values = 50,688 bytes per layer."""
    m = cs.InklingConvState(NS(compilation_config=NS(static_forward_context={})), 128, 4096, 4, "L0")
    assert m.state_bytes_per_block() == 50688


def test_register_refuses_kv_layers():
    cs.CONV_STATES.clear()
    ctx = {"sconv_layers.1000": object()}
    m = cs.InklingConvState(NS(compilation_config=NS(static_forward_context=ctx)), 1, 1, 4, "sconv_layers.1000")
    with pytest.raises(ValueError):
        m.register()
    ok = cs.InklingConvState(NS(compilation_config=NS(static_forward_context={})), 1, 1, 4, "sconv_layers.1001")
    ok.register()
    with pytest.raises(ValueError):
        ok.register()
    cs.CONV_STATES.clear()
