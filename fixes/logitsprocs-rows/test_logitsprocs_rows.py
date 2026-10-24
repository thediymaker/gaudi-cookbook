# SPDX-License-Identifier: Apache-2.0
"""CPU test for logitsprocs-rows (no HPU needed).

1. The source patch, against an unpatched hpu_input_batch.py: applies alone and after
   bad-words-index, is idempotent, and refuses a drifted source without writing.
2. Behaviour, on CPU InputBatch objects built from the stock and the patched file: stock sampling
   metadata carries no state for logit_bias, min_p or min_tokens; patched metadata applies each
   option to its own request's row only, in any row order.

Needs an unpatched vllm_gaudi source: the installed package, or GAUDI_FIXES_ROOT=<dir> holding
vllm/ and vllm_gaudi/. Part 2 also needs torch and an importable vllm. Run with:
  pytest fixes/logitsprocs-rows
"""

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
APPLY = HERE / "apply.py"
REL = "v1/worker/hpu_input_batch.py"
ROOT = os.environ.get("GAUDI_FIXES_ROOT")
if ROOT:
    sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("logitsprocs_rows_apply", APPLY)
FIX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIX)


def stock_source() -> Path:
    path = FIX.locate("vllm_gaudi", REL, ROOT)
    if path is None:
        pytest.skip("no vllm_gaudi source (install it or set GAUDI_FIXES_ROOT)")
    if FIX.MARKER in path.read_text():
        pytest.skip(f"{path} is already patched; point GAUDI_FIXES_ROOT at an unpatched copy")
    return path


def stage(tmp_path: Path, name: str, text: str) -> Path:
    root = tmp_path / name
    dst = root / "vllm_gaudi" / REL
    dst.parent.mkdir(parents=True)
    dst.write_text(text)
    return root


def run_apply(root: Path, *args: str) -> int:
    return subprocess.run([sys.executable, str(APPLY), "--root", str(root), *args], capture_output=True).returncode


BAD_WORDS_APPLY = HERE.parent / "bad-words-index" / "apply.py"


def test_applies_alone_and_after_bad_words_index(tmp_path):
    stock = stock_source().read_text()
    for name, first in (("alone", None), ("after-bad-words", BAD_WORDS_APPLY)):
        root = stage(tmp_path, name, stock)
        if first is not None:
            if not first.is_file():
                continue
            assert subprocess.run([sys.executable, str(first), "--root", str(root)]).returncode == 0
        assert run_apply(root, "--check") == 1
        assert run_apply(root) == 0
        text = (root / "vllm_gaudi" / REL).read_text()
        compile(text, "hpu_input_batch.py", "exec")
        assert "logitsprocs=_lp_rows(self, req_indices)" in text
        assert run_apply(root) == 0 and (root / "vllm_gaudi" / REL).read_text() == text
        assert run_apply(root, "--check") == 0


def test_refuses_drift(tmp_path):
    stock = stock_source().read_text()
    text = stock.replace(FIX.EDITS[2][1], "        req_index = self.req_id_to_index.pop(req_id)\n", 1)
    assert text != stock
    root = stage(tmp_path, "drift", text)
    assert run_apply(root) == 2
    assert (root / "vllm_gaudi" / REL).read_text() == text


# ---- behaviour ----------------------------------------------------------------------------------


def load_batch(path: Path, name: str, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm.sampling_params")
    try:
        import vllm_gaudi.utils  # noqa: F401
    except Exception:  # no HPU stack: give the module the two helpers it imports
        pkg = types.ModuleType("vllm_gaudi")
        pkg.__path__ = []
        utils = types.ModuleType("vllm_gaudi.utils")
        utils.async_h2d_copy = utils.async_h2d_update = None
        monkeypatch.setitem(sys.modules, "vllm_gaudi", pkg)
        monkeypatch.setitem(sys.modules, "vllm_gaudi.utils", utils)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def h2d_update(source, dest, indices, device=None):
        idx = torch.tensor(indices, dtype=torch.long, device=dest.device)
        dest.index_copy_(0, idx, source[indices].to(dest.device))

    def h2d_copy(source, dest_tensor=None, dtype=None, device=None):
        return dest_tensor.copy_(source)

    mod.async_h2d_update = h2d_update
    mod.async_h2d_copy = h2d_copy
    return mod


def build(mod):
    import torch
    from vllm.sampling_params import SamplingParams

    batch = mod.InputBatch(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=100,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    params = {
        "plain": SamplingParams(temperature=0.7, max_tokens=8),
        "bias": SamplingParams(temperature=0.7, max_tokens=8, logit_bias={5: 100.0}),
        "minp": SamplingParams(temperature=0.7, max_tokens=8, min_p=0.5),
        "mintok": SamplingParams(temperature=0.7, max_tokens=8, min_tokens=3, stop_token_ids=[9]),
    }
    for rid, sp in params.items():
        batch.add_request(
            mod.CachedRequestState(
                req_id=rid,
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=sp,
                generator=None,
                block_ids=[[0]],
                num_computed_tokens=0,
                output_token_ids=[],
            )
        )
    batch.refresh_sampling_metadata()
    return batch


def run_processors(md, rows):
    """Apply the metadata's processors the way vLLM's sampler does, to a known logits tensor."""
    import torch

    logits = torch.zeros(len(rows), 100)
    logits[:, 3] = 5.0  # a dominant token, so min_p 0.5 masks everything else
    for proc in md.logitsprocs.non_argmax_invariant:
        logits = proc.apply(logits)
    for proc in md.logitsprocs.argmax_invariant:
        logits = proc.apply(logits)
    return logits


def test_behaviour_stock_vs_patched(tmp_path, monkeypatch):
    src = stock_source()
    root = stage(tmp_path, "patched", src.read_text())
    assert run_apply(root) == 0
    stock = load_batch(src, "lp_rows_stock", monkeypatch)
    fixed = load_batch(root / "vllm_gaudi" / REL, "lp_rows_fixed", monkeypatch)
    neg = float("-inf")
    order = ["mintok", "plain", "minp", "bias"]

    md = build(stock).make_selective_sampling_metadata([(r, []) for r in order], skip_copy=False)
    logits = run_processors(md, order)
    assert logits[3, 5].item() == 0.0 and logits[0, 9].item() == 0.0 and logits[2, 0].item() == 0.0

    md = build(fixed).make_selective_sampling_metadata([(r, []) for r in order], skip_copy=False)
    logits = run_processors(md, order)
    row = {r: i for i, r in enumerate(order)}
    assert logits[row["bias"], 5].item() == 100.0
    assert logits[row["mintok"], 9].item() == neg
    assert logits[row["minp"], 0].item() == neg and logits[row["minp"], 3].item() == 5.0
    plain = logits[row["plain"]]
    assert plain[5].item() == 0.0 and plain[9].item() == 0.0 and plain[0].item() == 0.0

    # A request sampled alone, as for a prefill.
    md = build(fixed).make_selective_sampling_metadata([("bias", [])], skip_copy=False)
    logits = run_processors(md, ["bias"])
    assert logits[0, 5].item() == 100.0
