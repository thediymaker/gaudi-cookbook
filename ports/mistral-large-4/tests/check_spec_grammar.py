# Grammar-request drafting (ML4_SPEC_GRAMMAR) CPU unit test, no cards: PYTHONPATH=<plugin>:/workspace/vllm python3 test_spec_grammar_cpu.py
#  1. _patch_sample_tokens(grammar=True) applies every anchor on the installed vllm-gaudi (and grammar=False as before)
#  2. mask_rows == xgrammar apply_token_bitmask_inplace_cpu (bitwise, bf16 and fp32, random masks, V = W*32 and V > W*32)
#  3. so_verify puts request b's 1 + n rows on b*T + i (uniform) and on the ragged offsets; non-structured rows untouched
#  4. so_prompts: first mask row of each prompt request, skipping decode requests' 1 + n rows
import os, sys, types

os.environ.setdefault("ML4_SPEC", "dspark")
import numpy as np, torch
from mistral_large4_hpu import spec as S

fails = 0


def ok(c, m):
    global fails
    print(("PASS " if c else "FAIL ") + m)
    fails += not c


import vllm_gaudi.v1.worker.hpu_model_runner as hmr
import inspect

cls = hmr.HPUModelRunner
raw = cls.sample_tokens
S._patch_sample_tokens(cls, True)
ok(cls.sample_tokens is not raw and cls.sample_tokens._ml4_spec, "anchors grammar=True")


# grammar=False on a fresh copy of the class
class C2(hmr.HPUModelRunner):
    pass


C2.sample_tokens = raw
S._patch_sample_tokens(C2, False)
ok(C2.sample_tokens is not raw, "anchors grammar=False")
xgr_cpu = hmr.xgr_cpu
g = torch.Generator().manual_seed(0)
for dt in (torch.bfloat16, torch.float32):
    for V, W in ((4096, 128), (4100, 128)):
        R = 16
        lg = torch.randn(R, V, generator=g).to(dt)
        bm = torch.randint(-(2**31), 2**31 - 1, (24, W), generator=g, dtype=torch.int64).to(torch.int32).numpy()
        bm[3] = -1
        rows = [0, 1, 2, 5, 9, 15]
        src = [4, 5, 6, 3, 0, 23]
        a = lg.clone()
        S.mask_rows(a, bm, rows, src)
        full = np.full((R, W), -1, np.int32)
        full[rows] = bm[src]
        b = lg.clone().float()
        xgr_cpu.apply_token_bitmask_inplace_cpu(
            b, torch.from_numpy(full), indices=torch.tensor(rows, dtype=torch.int32)
        )
        b = b.to(dt)
        ok(torch.equal(a, b), f"mask_rows == xgr cpu dtype {dt} V {V} W {W} (-inf {int(torch.isinf(a).sum())})")


# so_verify layouts
def md_of(nd, T):
    Bp = len(nd)
    return types.SimpleNamespace(num_draft_tokens=nd, draft_token_ids=torch.zeros(Bp * (T - 1), dtype=torch.long))


V, W = 64, 2
rng = np.random.default_rng(1)


def run_verify(nd, T, so_ids, spec, r2i, num_decodes, R=None, F=None):
    md = md_of(nd, T)
    if F is not None:
        md.draft_token_ids = torch.zeros(F, dtype=torch.long)
    R = R or len(nd) * T
    nrows = sum(1 + len(spec.get(r, ())) for r in so_ids)
    bm = rng.integers(-(2**31), 2**31 - 1, (nrows, W), dtype=np.int64).astype(np.int32)
    go = types.SimpleNamespace(structured_output_request_ids=so_ids, grammar_bitmask=bm)
    so = types.SimpleNamespace(scheduled_spec_decode_tokens=spec)
    runner = types.SimpleNamespace(input_batch=types.SimpleNamespace(req_id_to_index=r2i))
    lg = torch.zeros(R, V)
    dd = types.SimpleNamespace(spec_decode_metadata=md)
    r = S.so_verify(runner, so, go, dd, lg, num_decodes)
    allow = np.unpackbits(bm.view(np.uint8), axis=1, bitorder="little").astype(bool)
    return r, lg, allow


os.environ["ML4_CTL"] = "/nonexistent"
# uniform: 4 decodes, T=4; a (b0, 3 drafts, structured), b (b1, 1 draft, plain), c (b2, 0 drafts, structured),
# d (b3, 2 drafts, structured); prompt p structured (b4, not a decode)
spec = {"a": [1, 2, 3], "b": [5], "d": [7, 8]}
r2i = {"a": 0, "b": 1, "c": 2, "d": 3, "p": 4}
r, lg, al = run_verify([3, 1, 0, 2], 4, ["c", "a", "p", "d"], spec, r2i, 4)
exp = torch.zeros_like(lg)
plan = {"c": (8, 0, 1), "a": (0, 1, 4), "d": (12, 6, 3)}  # row0, bitmask row0, rows
for rid, (r0, s0, n) in plan.items():
    for i in range(n):
        exp[r0 + i] = torch.where(torch.from_numpy(al[s0 + i][:V]), 0.0, float("-inf"))
ok(r and torch.equal(lg, exp), "so_verify uniform rows (c rows 8, a rows 0-3, p skipped (bitmask 5), d rows 12-14)")
# ragged: nd [2, 0] with R = 3 + ... use ragged_layout example: Tm=2: rows max(n+1,2): [3,2] -> R=5, F=max(n,1): [2,1]=3
r, lg, al = run_verify([2, 0], 2, ["x", "y"], {"x": [1, 2]}, {"x": 0, "y": 1}, 2, R=5, F=3)
exp = torch.zeros_like(lg)
for i in range(3):
    exp[0 + i] = torch.where(torch.from_numpy(al[i][:V]), 0.0, float("-inf"))
exp[3] = torch.where(torch.from_numpy(al[3][:V]), 0.0, float("-inf"))
ok(r and torch.equal(lg, exp), "so_verify ragged offsets")
# mismatch -> RuntimeError
try:
    run_verify([1, 0], 2, ["x"], {"x": [1, 2]}, {"x": 0}, 2)
    ok(False, "mismatch raises")
except RuntimeError as e:
    ok(True, f"mismatch raises: {e}")
# not a verify step -> False
go = types.SimpleNamespace(structured_output_request_ids=["x"], grammar_bitmask=np.zeros((1, W), np.int32))
ok(
    S.so_verify(None, None, go, types.SimpleNamespace(spec_decode_metadata=None), torch.zeros(2, V), 2) is False,
    "no spec metadata -> today's path",
)


# so_prompts
class FakeRunner:
    def _run_sampling(self, bc, logits, reqs, pad):
        self.seen = (logits.clone(), list(reqs), pad)
        return types.SimpleNamespace(sampled_token_ids=logits.argmax(-1).view(-1, 1), logprobs_tensors=None), None


fr = FakeRunner()
bm = rng.integers(-(2**31), 2**31 - 1, (6, W), dtype=np.int64).astype(np.int32)
go = types.SimpleNamespace(structured_output_request_ids=["a", "p", "q"], grammar_bitmask=bm)
so = types.SimpleNamespace(scheduled_spec_decode_tokens={"a": [1, 2, 3]})
pt, seg = [], []
S.so_prompts(fr, True, so, go, [torch.zeros(1, V), torch.zeros(2, V)], ["z", "q", "p"], pt, seg)
al = np.unpackbits(bm.view(np.uint8), axis=1, bitorder="little").astype(bool)
exp = torch.zeros(3, V)
exp[2] = torch.where(torch.from_numpy(al[4][:V]), 0.0, float("-inf"))
exp[1] = torch.where(torch.from_numpy(al[5][:V]), 0.0, float("-inf"))
ok(
    torch.equal(fr.seen[0], exp) and fr.seen[1] == ["z", "q", "p"] and len(pt) == 3 and seg[0][0] == ["z", "q", "p"],
    "so_prompts rows (p <- bitmask 4, q <- 5, z plain)",
)
# so_prompts with only partial prompt chunks (0 logits rows, no requests): no sampling call
fr2 = FakeRunner()
fr2.seen = None
pt2, seg2 = [], []
S.so_prompts(fr2, True, so, go, [torch.zeros(0, V)], [], pt2, seg2)
ok(fr2.seen is None and pt2 == [] and seg2 == [], "so_prompts: partial chunks only -> no sampler call")
# _plain
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

sp = SamplingParams(temperature=0.7, structured_outputs=StructuredOutputsParams(grammar='root ::= "a"'))
ok(not S._plain(sp) and S._plain(sp, True), "_plain strict / grammar")
sp2 = SamplingParams(temperature=0.7, logprobs=2, structured_outputs=StructuredOutputsParams(grammar='root ::= "a"'))
ok(not S._plain(sp2, True), "grammar + logprobs stays non-plain")
print("ALL PASS" if not fails else f"{fails} FAILED")
