# SPDX-License-Identifier: Apache-2.0
"""CPU test for bad-words-index (no HPU needed).

1. The source patch, against an unpatched hpu_input_batch.py: applies, is idempotent, and refuses
   drifted sources without writing.
2. Behaviour, on CPU InputBatch objects built from the stock and the patched file side by side,
   with vLLM's own apply_bad_words: stock raises IndexError for a one-row prefill of a request at
   batch index 1 and bans another request's row in a decode batch; patched bans the requester's
   row only.

Needs an unpatched vllm_gaudi source: the installed package, or GAUDI_FIXES_ROOT=<dir> holding
vllm/ and vllm_gaudi/. Part 2 also needs torch and an importable vllm. Run with:
  pytest fixes/bad-words-index
"""

import dataclasses
import importlib.util
import inspect
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

_spec = importlib.util.spec_from_file_location("bad_words_index_apply", APPLY)
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
    i = stock.find("    def make_selective_sampling_metadata(\n")
    j = stock.find(FIX.OLD, i)
    mutants = {
        "re-keying gone": stock.replace("for i, req_idx in enumerate(req_indices)", "for i, req_idx in []"),
        "method duplicated": stock + "\n\nclass _X:\n\n    def make_selective_sampling_metadata(\n        self):\n"
        "        pass\n",
        "line removed": stock[:j] + "            logitsprocs=" + stock[j + len(FIX.OLD) :],
        "line duplicated": stock[:j] + FIX.OLD.replace("logitsprocs=", "") + stock[j:],
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

    def h2d_update(source, dest, indices, device=None):
        idx = torch.tensor(indices, dtype=torch.long, device=dest.device)
        dest.index_copy_(0, idx, source[indices].to(dest.device))

    def h2d_copy(source, dest_tensor=None, dtype=None, device=None):
        return dest_tensor.copy_(source)

    mod.async_h2d_update = h2d_update
    mod.async_h2d_copy = h2d_copy
    return mod


def build(mod, words):
    import torch
    from vllm.sampling_params import SamplingParams

    init = inspect.signature(mod.InputBatch.__init__).parameters
    kwargs = dict(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=100,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    batch = mod.InputBatch(**{k: v for k, v in kwargs.items() if k in init})
    fields = {f.name for f in dataclasses.fields(mod.CachedRequestState)}
    for rid, w in words.items():
        sp = SamplingParams(temperature=0.0, max_tokens=8)
        sp._bad_words_token_ids = w  # what update_from_tokenizer fills from bad_words=[...]
        cand = dict(
            req_id=rid,
            prompt_token_ids=[1, 2, 3],
            mm_features=[],
            sampling_params=sp,
            pooling_params=None,
            generator=None,
            block_ids=([0],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        batch.add_request(mod.CachedRequestState(**{k: v for k, v in cand.items() if k in fields}))
    batch.refresh_sampling_metadata()
    return batch


def select(batch, rids):
    fn = batch.make_selective_sampling_metadata
    kw = {"skip_copy": False} if "skip_copy" in inspect.signature(fn).parameters else {}
    return fn([(r, []) for r in rids], **kw)


def test_behaviour_stock_vs_patched(tmp_path, monkeypatch):
    src = stock_source()
    root = stage(tmp_path, "patched", src.read_text())
    assert run_apply(root) == 0
    stock = load_batch(src, "bad_words_stock", monkeypatch)
    fixed = load_batch(root / "vllm_gaudi" / REL, "bad_words_fixed", monkeypatch)
    import torch
    from vllm.v1.sample.ops.bad_words import apply_bad_words

    words = {"a": None, "b": [[7]], "c": [[8]]}
    neg = float("-inf")

    # A request sampled alone, as for a prefill: stock keys it by batch index 1.
    md = select(build(stock, words), ["b"])
    with pytest.raises(IndexError):
        apply_bad_words(torch.zeros(1, 100), md.bad_words_token_ids, [[]])
    md = select(build(fixed, words), ["b"])
    assert md.bad_words_token_ids == {0: [[7]]}
    logits = torch.zeros(1, 100)
    apply_bad_words(logits, md.bad_words_token_ids, [[]])
    assert logits[0, 7].item() == neg

    # A decode batch in an order that differs from the batch order.
    order = ["c", "a", "b"]
    md = select(build(stock, words), order)
    logits = torch.zeros(3, 100)
    apply_bad_words(logits, md.bad_words_token_ids, [[], [], []])
    assert logits[1, 7].item() == neg and logits[2, 7].item() != neg  # stock: b's word on a's row
    md = select(build(fixed, words), order)
    assert md.bad_words_token_ids == {0: [[8]], 2: [[7]]}
    logits = torch.zeros(3, 100)
    apply_bad_words(logits, md.bad_words_token_ids, [[], [], []])
    assert logits[2, 7].item() == neg and logits[0, 8].item() == neg
    assert logits[1, 7].item() != neg and logits[1, 8].item() != neg
