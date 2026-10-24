# SPDX-License-Identifier: Apache-2.0
"""The port's source patches: the vendored files match their manifest, the inserted code does what it
says (on CPU), and with GAUDI_FIXES_ROOT (a copy of a releases/v0.30.0 site-packages holding vllm/ and
vllm_gaudi/) the whole install order applies, is idempotent, and refuses an unfixed runner."""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parent.parent
COOKBOOK = PORT.parent.parent
PATCHES = ("vllm_pr_62052.py", "beam_sliding_window.py", "grammar_device_mask.py")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendored_files_match_manifest():
    entries = [ln.split() for ln in (PORT / "vendor" / "MANIFEST").read_text().splitlines() if ln and ln[0] != "#"]
    assert len(entries) == 6
    for digest, rel in entries:
        assert hashlib.sha256((PORT / "vendor" / rel).read_bytes()).hexdigest() == digest, rel


# ----------------------------------------------------------------------------- sliding window, decode
def _decode_visible(pos: int, window: int, block: int, torch) -> set[int]:
    """Key positions a sliding layer sees in decode at query position `pos`, built with the inserted code:
    the beam block drop (this port), the -lbu usage of swa-decode-window and its decode bias."""
    swa = _load(COOKBOOK / "fixes" / "swa-decode-window" / "apply.py", "swa_fix")
    sw = _load(PORT / "patches" / "beam_sliding_window.py", "beam_sw")
    blocks = list(range(pos // block + 1))  # block ids = block index of the sequence
    head = window // block + 1
    block_tables = [blocks[-head:]]
    slot_mapping = [[pos]]
    block_size = block
    head_mask_full_len = head
    ns = {"block_tables": block_tables, "slot_mapping": slot_mapping, "block_size": block_size}
    ns["head_mask_full_len"] = head_mask_full_len
    drop = sw.DROP_NEW.replace('if head_mask_full_len and self._get_model_type() == "beam":', "if head_mask_full_len:")
    code = drop + swa.USAGE_NEW  # USAGE_NEW starts with the stock block_usage line it extends
    exec(compile("if True:\n" + code, "decode", "exec"), {}, ns)  # noqa: S102
    table, usage = ns["block_tables"][0], ns["block_usage"][0]
    mask_ns = {"torch": torch, "device": "cpu", "block_size": block_size, "block_usage": torch.tensor(usage)}
    exec(compile("if True:\n" + swa.MASK_NEW, "mask", "exec"), {}, mask_ns)  # noqa: S102
    masked = mask_ns["mask"]
    seen = set()
    for b, row in zip(table, masked, strict=True):
        assert not bool(row.all()), "a fully masked block would make the blockwise softmax NaN"
        seen |= {b * block + s for s in range(block) if not bool(row[s]) and b * block + s <= pos}
    return seen


def test_decode_window_is_exact_without_fully_masked_blocks():
    torch = pytest.importorskip("torch")
    window, block = 512, 128
    for pos in range(window, window + 3 * block + 1):
        assert _decode_visible(pos, window, block, torch) == set(range(pos - window + 1, pos + 1)), pos


# ----------------------------------------------------------------------------- grammar device mask
def _device_mask_fn(torch):
    gm = _load(PORT / "patches" / "grammar_device_mask.py", "beam_gm")
    ns = {"torch": torch, "os": os}
    exec(compile(gm.HELPER, "helper", "exec"), ns)  # noqa: S102
    return ns["_beam_apply_bitmask_device"]


def _reference_mask(logits, bitmask):
    out = logits.clone()
    rows, vocab = out.shape
    for i in range(rows):
        for j in range(min(vocab, 32 * bitmask.shape[1])):
            if not (int(bitmask[i, j // 32]) >> (j % 32)) & 1:
                out[i, j] = float("-inf")
    return out


@pytest.mark.parametrize("vocab,words", [(96, 3), (100, 4), (130, 4), (64, 3)])
def test_device_mask_matches_reference(vocab, words):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    fn = _device_mask_fn(torch)
    g = torch.Generator().manual_seed(vocab)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(5, vocab, generator=g).to(dtype)
        bm = torch.randint(-(2**31), 2**31 - 1, (5, words), generator=g, dtype=torch.int64).to(torch.int32).numpy()
        bm[2] = -1  # a row without a grammar
        want = _reference_mask(logits, bm)
        got = logits.clone()
        fn(got, np.ascontiguousarray(bm))
        assert torch.equal(got, want)
        try:
            import xgrammar.kernels.apply_token_bitmask_inplace_cpu as xgr_cpu
        except ImportError:
            continue
        ref = logits.float().clone()
        xgr_cpu.apply_token_bitmask_inplace_cpu(ref, torch.from_numpy(bm))
        assert torch.equal(got.float(), ref)


# ----------------------------------------------------------------------------- whole install order
@pytest.fixture
def release(tmp_path):
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root:
        pytest.skip("GAUDI_FIXES_ROOT not set")
    dst = tmp_path / "site"
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(root) / pkg, dst / pkg, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _run(script: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def test_runner_patches_refuse_an_unfixed_runner(release):
    for name in ("beam_sliding_window.py", "grammar_device_mask.py"):
        r = _run(PORT / "patches" / name, "--root", str(release))
        assert r.returncode == 2 and "first" in r.stderr, r.stderr


def test_install_order_applies_and_is_idempotent(release):
    r = _run(COOKBOOK / "fixes" / "apply_all.py", "--profile", "v0.30.0", "--root", str(release))
    assert r.returncode == 0, r.stderr
    for name in PATCHES:
        assert _run(PORT / "patches" / name, "--root", str(release), "--check").returncode == 1
        r = _run(PORT / "patches" / name, "--root", str(release))
        assert r.returncode == 0, r.stderr
    for name in PATCHES:
        r = _run(PORT / "patches" / name, "--root", str(release))
        assert r.returncode == 0 and "applied" in r.stdout, r.stdout + r.stderr
    r = _run(COOKBOOK / "fixes" / "apply_all.py", "--profile", "v0.30.0", "--root", str(release), "--check")
    assert r.returncode == 0, r.stdout + r.stderr
