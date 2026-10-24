# SPDX-License-Identifier: Apache-2.0
"""CPU test for prompt-logprobs.

Cuts the prompt-logprobs code out of the stock and the patched runner (the collection lines in
sample_tokens and _get_prompt_logprobs_dict) and runs it on CPU with a stub model (logits = h @ W.T)
over chunked prefills, against a reference computed from the whole prompt at once, for every output
layout: flat [T, H] with one prompt, flat [B*T, H], [B, T, H] and merged prefill. The stock code
raises on the flat and merged layouts and matches on [B, T, H]; the patched code matches on all.

Needs torch (CPU) and an unpatched vllm_gaudi source tree: the installed package, or
GAUDI_FIXES_ROOT=<dir holding vllm_gaudi/>. Run with: pytest fixes/prompt-logprobs
"""

import ast
import importlib.util
import os
import textwrap
import types
from pathlib import Path
from typing import NamedTuple, Optional

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
RUNNER_REL = Path("vllm_gaudi/v1/worker/hpu_model_runner.py")


def _load_apply():
    spec = importlib.util.spec_from_file_location("prompt_logprobs_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load_apply()


def _runner_source() -> str:
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if root:
        path = Path(root) / RUNNER_REL
    else:
        spec = importlib.util.find_spec("vllm_gaudi")
        if spec is None or not spec.submodule_search_locations:
            pytest.skip("vllm_gaudi not installed; set GAUDI_FIXES_ROOT")
        path = Path(list(spec.submodule_search_locations)[0]).parent / RUNNER_REL
    if not path.is_file():
        pytest.skip(f"{path} not found")
    src = path.read_text()
    if A.MARKER in src:
        pytest.skip(f"{path} is already patched")
    return src


class LogprobsTensors(NamedTuple):
    """Stand-in for vllm.v1.outputs.LogprobsTensors (same fields and empty_cpu)."""

    logprob_token_ids: object
    logprobs: object
    selected_token_ranks: object

    @staticmethod
    def empty_cpu(num_positions, num_tokens_per_position):
        ids = torch.empty((num_positions, num_tokens_per_position), dtype=torch.int32)
        return LogprobsTensors(
            ids, torch.empty_like(ids, dtype=torch.float32), torch.empty(num_positions, dtype=torch.int32)
        )


V, H = 997, 24
GEN = torch.Generator().manual_seed(0)
W = torch.randn(V, H, generator=GEN)
WARN: list[str] = []


class StubModel:
    def compute_logits(self, h):
        if h.dim() != 2 or h.shape[1] != H:  # the device matmul raises here too
            raise RuntimeError(f"compute_logits got {tuple(h.shape)}, expected [n, {H}]")
        return h.float() @ W.T


def _method_source(src, name):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "HPUModelRunner":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return textwrap.dedent("    " + ast.get_source_segment(src, f))
    raise KeyError(name)


def _collect_snippet(src):
    start = src.index("                # Collect prefill hidden states for prompt logprobs.\n")
    end = src.index("                if self.use_aux_hidden_state_outputs:\n", start)
    return textwrap.dedent(src[start:end])


def _build(src):
    if not hasattr(torch, "hpu"):
        torch.hpu = types.SimpleNamespace()
    torch.hpu.synchronize = lambda *a, **k: None
    logger = types.SimpleNamespace(warning=lambda *a: WARN.append(a[0] % a[1:] if len(a) > 1 else a[0]))
    ns = {"torch": torch, "LogprobsTensors": LogprobsTensors, "os": os, "logger": logger, "Optional": Optional}
    if A.MARKER in src:
        exec(src[src.index(f"# ---- {A.MARKER} prompt-logprobs helpers") :], ns)
    for m in ("_get_prompt_logprobs_dict", "_gather_prompt_logprobs_hpu"):
        exec(_method_source(src, m), ns)
    ns["__collect__"] = compile(_collect_snippet(src), "collect", "exec")
    return ns


def _reference(prompt, hs_full, k):
    lp = (hs_full[: len(prompt) - 1].float() @ W.T).log_softmax(-1)
    tgt = torch.tensor(prompt[1:])
    tok = lp.gather(-1, tgt[:, None])
    ids, lps = [tgt[:, None]], [tok]
    if k:
        tl, ti = torch.topk(lp, k, dim=-1)
        ids.append(ti)
        lps.append(tl)
    return torch.cat(ids, 1).int(), torch.cat(lps, 1), (lp >= tok).sum(-1).int()


def _run_layout(ns, layout, P, C, k, B=3, plp_slot=1):
    rid = "plp"
    prompt = torch.randint(0, V, (P,), generator=GEN).tolist()
    hs_full = torch.randn(P, H, generator=GEN)
    req = types.SimpleNamespace(prompt_token_ids=prompt, num_computed_tokens=0)
    sampler = types.SimpleNamespace(compute_logprobs=lambda x: x.log_softmax(dim=-1, dtype=torch.float32))
    runner = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(num_prompt_logprobs={rid: k}, in_progress_prompt_logprobs_cpu={}),
        requests={rid: req},
        model=StubModel(),
        sampler=sampler,
        device="cpu",
        use_merged_prefill=(layout == "merged"),
    )
    runner._gather_prompt_logprobs_hpu = lambda *a: ns["_gather_prompt_logprobs_hpu"](runner, *a)
    computed, out = 0, None
    while computed < P:
        n = min(C, P - computed)
        req.num_computed_tokens = computed
        nb = 1 if layout == "flat1" else B
        slot = 0 if nb == 1 else plp_slot
        req_ids = [rid if i == slot else f"other{i}" for i in range(nb)]
        lens = [n if i == slot else 5 + 3 * i for i in range(nb)]
        sched = types.SimpleNamespace(num_scheduled_tokens=dict(zip(req_ids, lens, strict=True)))
        if layout == "merged":
            T = -(-sum(lens) // 32) * 32
            token_ids = torch.zeros(1, T, dtype=torch.long)
            hidden = torch.randn(T, H, generator=GEN)
            off = sum(lens[:slot])
            hidden[off : off + n] = hs_full[computed : computed + n]
        else:
            T = -(-max(lens) // 32) * 32
            token_ids = torch.zeros(nb, T, dtype=torch.long)
            h3 = torch.randn(nb, T, H, generator=GEN)
            h3[slot, :n] = hs_full[computed : computed + n]
            hidden = h3 if layout == "bth" else h3.reshape(nb * T, H)
        store: dict = {}
        scope = dict(
            ns,
            self=runner,
            prefill_hidden_states_for_logprobs=store,
            non_flattened_hidden_states=hidden,
            token_ids=token_ids,
            req_id=req_ids,
            scheduler_output=sched,
        )
        exec(ns["__collect__"], scope)
        res = ns["_get_prompt_logprobs_dict"](runner, store, sched)
        computed += n
        if rid in res:
            out = res[rid]
    return out, _reference(prompt, hs_full, k)


def _matches(out, ref):
    ids, lps, rk = ref
    return (
        out is not None
        and torch.equal(out.logprob_token_ids, ids)
        and torch.equal(out.selected_token_ranks, rk)
        and torch.allclose(out.logprobs, lps, atol=1e-5, rtol=0)
    )


@pytest.fixture(scope="module")
def sources():
    os.environ["VLLM_HPU_PROMPT_LOGPROBS_ROWS"] = "128"  # padding and several logits chunks
    src = _runner_source()
    return src, A.patch_runner(src, {})


def test_patch_text_and_refusals(sources):
    src, patched = sources
    assert A.MARKER in patched and "def _gaudi_plp_score(" in patched
    compile(patched, "hpu_model_runner.py", "exec")
    for old in (A.COLLECT_OLD, A.H2D_OLD, A.SCORE_OLD):
        with pytest.raises(A.Drift):
            A.patch_runner(src.replace(old, "", 1), {})
        with pytest.raises(A.Drift):
            A.patch_runner(src + "\n" + old, {})


@pytest.mark.parametrize("layout", ["flat1", "flatB", "bth", "merged"])
def test_runner_code_matches_reference(sources, layout):
    src, patched = sources
    stock, fixed = _build(src), _build(patched)
    C = 96
    for P in (2, 17, C - 1, C, C + 1, 2 * C + 1, 300):
        for k in (0, 1, 5):
            torch.manual_seed(P * 10 + k)
            try:
                o, r = _run_layout(stock, layout, P, C, k)
                stock_res = "match" if _matches(o, r) else "wrong"
            except Exception as e:  # noqa: BLE001
                stock_res = "raises " + type(e).__name__
            o, r = _run_layout(fixed, layout, P, C, k)
            assert _matches(o, r), f"patched {layout} P={P} k={k} (stock: {stock_res})"
            if layout == "bth":
                assert stock_res == "match", f"control: stock bth P={P} k={k}"
            else:
                assert stock_res.startswith("raises"), f"stock {layout} P={P} k={k}: {stock_res}"


def test_layout_mismatch_is_skipped_with_one_warning(sources):
    ns = _build(sources[1])
    WARN.clear()
    runner = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(num_prompt_logprobs={"a": 1}), use_merged_prefill=False
    )
    store: dict = {}
    sched = types.SimpleNamespace(num_scheduled_tokens={"a": 10})
    for _ in range(2):
        ns["_gaudi_plp_collect"](runner, store, torch.zeros(37, H), torch.zeros(1, 64, dtype=torch.long), ["a"], sched)
    assert store == {} and len(WARN) == 1
