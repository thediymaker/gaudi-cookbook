# SPDX-License-Identifier: Apache-2.0
"""CPU test for sampling-rows-refresh (no HPU needed).

1. The source patch, against an unpatched hpu_input_batch.py: applies, is idempotent, and refuses
   drifted sources (an anchor removed or duplicated, swap_states already moving the device rows).
2. Behaviour, on CPU InputBatch objects built from the stock and the patched file side by side:
   after a join step, two swap_states calls and a skip_copy=True sampling call, stock returns the
   previous owners' temperature, top_k, top_p and repetition penalty and the old row order of the
   prompt tokens; patched returns every request's own values. No swap means no extra copy, and
   without penalties the penalty tensors are not copied.

Needs an unpatched vllm_gaudi source: the installed package, or GAUDI_FIXES_ROOT=<dir> holding
vllm/ and vllm_gaudi/. Part 2 also needs torch and an importable vllm. Run with:
  pytest fixes/sampling-rows-refresh
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

_spec = importlib.util.spec_from_file_location("sampling_rows_refresh_apply", APPLY)
FIX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIX)
RIDS = ("g", "r", "p")


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


def test_applies_and_is_idempotent(tmp_path):
    root = stage(tmp_path, "ok", stock_source().read_text())
    assert run_apply(root, "--check") == 1
    assert run_apply(root) == 0
    text = (root / "vllm_gaudi" / REL).read_text()
    assert text.count(FIX.MARKER) == 1
    compile(text, "hpu_input_batch.py", "exec")
    assert run_apply(root) == 0 and (root / "vllm_gaudi" / REL).read_text() == text
    assert run_apply(root, "--check") == 0


def test_refuses_drift(tmp_path):
    stock = stock_source().read_text()
    mutants = {
        "anchor removed": stock.replace(FIX.ANCHORS[1], "    def swap_rows(self, i1: int, i2: int) -> None:"),
        "anchor duplicated": stock + "\n" + FIX.ANCHORS[2] + "\n        pass\n",
        "device rows already swapped": stock.replace(
            FIX.CPU_ONLY_SWAPS[0],
            "        self.temperature[i1], self.temperature[i2] = self.temperature[i2], self.temperature[i1]\n"
            + FIX.CPU_ONLY_SWAPS[0],
        ),
    }
    for name, text in mutants.items():
        root = stage(tmp_path, name.replace(" ", "-"), text)
        assert run_apply(root) == 2, name
        assert (root / "vllm_gaudi" / REL).read_text() == text, name


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
    copies = []

    def h2d_update(source, dest, indices, device=None):
        idx = torch.tensor(indices, dtype=torch.long, device=dest.device)
        dest.index_copy_(0, idx, source[indices].to(dest.device))

    def h2d_copy(source, dest_tensor=None, dtype=None, device=None):
        copies.append(dest_tensor)
        return dest_tensor.copy_(source)

    mod.async_h2d_update = h2d_update
    mod.async_h2d_copy = h2d_copy
    mod.copies = copies
    return mod


def build(mod, penalties=True):
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
        "g": SamplingParams(temperature=0.0, max_tokens=8),
        "r": SamplingParams(temperature=0.9, top_k=5, top_p=0.95, max_tokens=8),
        "p": SamplingParams(temperature=0.5, top_p=0.8, max_tokens=8, repetition_penalty=1.3 if penalties else 1.0),
    }
    prompts = {"g": [1, 2, 3], "r": [10, 11, 12, 13], "p": [20, 21]}
    for rid in RIDS:
        batch.add_request(
            mod.CachedRequestState(
                req_id=rid,
                prompt_token_ids=prompts[rid],
                mm_features=[],
                sampling_params=params[rid],
                generator=None,
                block_ids=[[0]],
                num_computed_tokens=0,
                output_token_ids=[],
            )
        )
    batch.refresh_sampling_metadata()  # the join step's copy
    want = {
        "temperature": [-1.0, 0.9, 0.5],
        "top_k": [100, 5, 100],
        "top_p": [1.0, 0.95, 0.8],
        "rep": [1.0, 1.0, 1.3 if penalties else 1.0],
        "prompts": prompts,
    }
    return batch, want


def sample(batch, skip_copy):
    return batch.make_selective_sampling_metadata([(rid, []) for rid in RIDS], skip_copy=skip_copy)


def matches(md, want, penalties=True):
    import torch

    ok = (
        torch.allclose(md.temperature.float(), torch.tensor(want["temperature"]))
        and md.top_k.tolist() == want["top_k"]
        and torch.allclose(md.top_p.float(), torch.tensor(want["top_p"]))
    )
    if penalties:
        ok = ok and torch.allclose(md.repetition_penalties.float(), torch.tensor(want["rep"]))
        rows = md.prompt_token_ids.tolist()
        for i, rid in enumerate(RIDS):
            ok = ok and rows[i][: len(want["prompts"][rid])] == want["prompts"][rid]
    return ok


def test_behaviour_stock_vs_patched(tmp_path, monkeypatch):
    src = stock_source()
    root = stage(tmp_path, "patched", src.read_text())
    assert run_apply(root) == 0
    stock = load_batch(src, "sampling_rows_stock", monkeypatch)
    fixed = load_batch(root / "vllm_gaudi" / REL, "sampling_rows_fixed", monkeypatch)

    for name, mod in (("stock", stock), ("patched", fixed)):
        batch, want = build(mod)
        assert matches(sample(batch, skip_copy=False), want), name
        sample(batch, skip_copy=True)  # a quiet step: builds the cached prompt-token tensor
        batch.swap_states(0, 1)  # reordering on a step where nobody joined or left
        batch.swap_states(1, 2)
        md = sample(batch, skip_copy=True)
        if name == "stock":
            assert not matches(md, want), "the stock file should show the stale device rows"
            continue
        assert matches(md, want)
        assert not batch._sampling_rows_moved
        n = len(mod.copies)
        assert matches(sample(batch, skip_copy=True), want) and len(mod.copies) == n  # no swap, no copy
        batch.swap_states(2, 2)
        assert not batch._sampling_rows_moved
        batch.swap_states(0, 2)
        batch.refresh_sampling_metadata()
        assert not batch._sampling_rows_moved


def test_no_penalties_skips_penalty_copies(tmp_path, monkeypatch):
    src = stock_source()
    root = stage(tmp_path, "patched", src.read_text())
    assert run_apply(root) == 0
    fixed = load_batch(root / "vllm_gaudi" / REL, "sampling_rows_fixed_np", monkeypatch)
    batch, want = build(fixed, penalties=False)
    sample(batch, skip_copy=True)
    del fixed.copies[:]
    batch.swap_states(0, 2)
    md = sample(batch, skip_copy=True)
    assert matches(md, want, penalties=False)
    assert len(fixed.copies) == len(FIX.DEVICE_PAIRS)
