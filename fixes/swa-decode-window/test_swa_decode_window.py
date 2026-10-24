# SPDX-License-Identifier: Apache-2.0
"""CPU test for swa-decode-window: the visible decode set equals the sliding window.

Runs the block-usage and bias code the fix inserts on a CPU torch build and compares the visible
key positions of a sliding layer with (q - window, q]. Run with: pytest fixes/swa-decode-window
"""

import importlib.util
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent


def _apply_module():
    spec = importlib.util.spec_from_file_location("swa_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _usage_fn(mod, patched):
    body = mod.USAGE_NEW if patched else mod.USAGE_OLD
    code = (
        "def usage(block_tables, slot_mapping, block_size, head_mask_full_len):\n"
        "        last_block_usage = [slot[0] % block_size + 1 for slot in slot_mapping]\n"
        + body
        + "        return [x for u in block_usage for x in u]\n"
    )
    ns: dict = {}
    exec(code, ns)
    return ns["usage"]


def _mask(mod, patched, block_usage, block_size):
    ns = {"torch": torch, "block_usage": block_usage, "block_size": block_size, "device": "cpu"}
    exec(textwrap.dedent(mod.MASK_NEW if patched else mod.MASK_OLD), ns)
    return ns["mask"]  # True = masked


def _visible(mod, patched, q, window, bs):
    n_win = window // bs + 1  # what the runner takes for gpt-oss
    table = list(range(q // bs + 1))[-n_win:]
    usage = _usage_fn(mod, patched)([table], [[(q // bs) * bs + q % bs]], bs, n_win if patched else None)
    masked = _mask(mod, patched, torch.tensor(usage, dtype=torch.bfloat16), bs)
    return {blk * bs + s for row, blk in enumerate(table) for s in range(bs) if not masked[row, s]}


@pytest.mark.parametrize("window, bs", [(128, 128), (128, 64), (256, 128)])
def test_patched_window_is_exact(window, bs):
    mod = _apply_module()
    for q in range(0, 5 * window):
        assert _visible(mod, True, q, window, bs) == set(range(max(0, q - window + 1), q + 1)), q


def test_stock_window_is_too_wide():
    mod = _apply_module()
    widths = {len(_visible(mod, False, q, 128, 128)) for q in range(256, 512)}
    assert max(widths) == 256 and min(widths) == 129


def test_positive_usage_mask_unchanged():
    mod = _apply_module()
    usage = torch.tensor([1, 5, 64, 128], dtype=torch.bfloat16)
    assert torch.equal(_mask(mod, True, usage, 128), _mask(mod, False, usage, 128))
