# SPDX-License-Identifier: Apache-2.0
"""CPU tests for patches/: anchors, idempotence and order against v0.26.0 sources, and the patched logic.

Tests that read release sources take GAUDI_FIXES_ROOT, a directory holding vllm/ and vllm_gaudi/ from the
v0.26.0 image (as for fixes/test_fixes.py), and skip without it. The w13 interleave test needs torch.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parent.parent
REPO = PORT.parent.parent
PATCHES = [PORT / "patches" / "swigluoai_fp8_moe.py", PORT / "patches" / "reasoning_prompt_scope.py"]


def _root():
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if not root or not (Path(root) / "vllm_gaudi").is_dir() or not (Path(root) / "vllm").is_dir():
        pytest.skip("set GAUDI_FIXES_ROOT to a directory holding v0.26.0's vllm/ and vllm_gaudi/")
    return Path(root)


@pytest.fixture
def tree(tmp_path):
    src = _root()
    shutil.copytree(src / "vllm", tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    shutil.copytree(src / "vllm_gaudi", tmp_path / "vllm_gaudi", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def _run(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True).returncode


def test_patches_apply_once_then_fix_profile(tree):
    for p in PATCHES:
        assert _run(p, "--check", "--root", str(tree)) == 1
        assert _run(p, "--root", str(tree)) == 0
        assert _run(p, "--root", str(tree)) == 0
        assert _run(p, "--check", "--root", str(tree)) == 0
    # The serving container applies the release profile on top of the port image.
    assert _run(REPO / "fixes" / "apply_all.py", "--profile", "v0.26.0", "--root", str(tree)) == 0


def test_drifted_source_writes_nothing(tree):
    target = tree / "vllm_gaudi" / "ops" / "hpu_compressed_tensors.py"
    text = target.read_text().replace("permuted_weights=True,", "permuted_weights=True,  # drift", 1)
    target.write_text(text)
    assert _run(PATCHES[0], "--root", str(tree)) == 2
    assert target.read_text() == text


def _scoped_parser(source: str):
    """The patched marker-scan methods of MiniMaxM3ReasoningParser on a stub with M3's token ids."""
    cls = next(
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3ReasoningParser"
    )
    keep = {"_rfind_token_sequence", "_current_turn_token_ids", "is_reasoning_end"}
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    stub = ast.ClassDef(name="Stub", bases=[], keywords=[], body=body, decorator_list=[], type_params=[])
    ns = {"Sequence": list}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), "<parser>", "exec"), ns)
    obj = ns["Stub"]()
    obj._start_token_ids, obj._end_token_ids, obj._role_open_token_ids = (200059,), (200060,), (200019,)
    return obj


THINK, END, ROLE = 200059, 200060, 200019
# system turn quoting both markers, a user turn, then the generation prompt "]~b]ai\n"
PROMPT = [ROLE, 1, 2, THINK, END, 3, END, 4, ROLE, 5, 6, ROLE, 7, 8]


def test_reasoning_scope_logic(tree):
    path = tree / "vllm" / "reasoning" / "minimax_m3_reasoning_parser.py"
    stock = _scoped_parser(path.read_text())
    stock._role_open_token_ids = ()  # scoping off == stock behaviour
    assert stock.is_reasoning_end(PROMPT) is True  # the bug: the prompt alone "ends" reasoning
    assert _run(PATCHES[1], "--root", str(tree)) == 0
    p = _scoped_parser(path.read_text())
    assert p.is_reasoning_end(PROMPT) is False
    assert p.is_reasoning_end(PROMPT + [THINK]) is False  # thinking_mode "enabled" prefill
    assert p.is_reasoning_end(PROMPT + [THINK, 9, 10]) is False  # still reasoning
    assert p.is_reasoning_end(PROMPT + [THINK, 9, END]) is True  # closed in this turn
    assert p.is_reasoning_end([THINK, 9, END, 11]) is True  # generated ids only: full scan as before


def test_w13_interleave_matches_gpt_swiglu():
    """The load-time interleave on the uint8 view gives rows (g0, u0, g1, u1, ...); SwiGLU-OAI read that way
    equals the concatenated reference, and reading concatenated rows as interleaved does not."""
    torch = pytest.importorskip("torch")
    E, N, H = 2, 4, 3
    w = torch.randint(0, 256, (E, 2 * N, H), dtype=torch.uint8)
    inter = w.view(E, 2, N, H).transpose(1, 2).reshape(E, 2 * N, H)
    assert torch.equal(inter[:, 0::2], w[:, :N]) and torch.equal(inter[:, 1::2], w[:, N:])

    def swiglu(g, u, alpha=1.702, limit=7.0):
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        return g * torch.sigmoid(alpha * g) * (u + 1)

    h = torch.randn(5, 2 * N) * 4
    ref = swiglu(h[:, :N], h[:, N:])
    hi = h.view(5, 2, N).transpose(1, 2).reshape(5, 2 * N)
    assert torch.allclose(swiglu(hi[:, 0::2], hi[:, 1::2]), ref)
    assert not torch.allclose(swiglu(h[:, 0::2], h[:, 1::2]), ref)
