# SPDX-License-Identifier: Apache-2.0
"""serving/warmup.py reaches every prefill cell it lists, with and without decoder replay (stdlib only, no server).

The server is modelled as the recipe runs it: a prefix-cache hit on the cached base is cut to end at least KEEP tokens
before the request's end (replay.install_hit_cap), the rest is prefilled in chunks of CHUNK tokens, and the last
chunk compiles the (query bucket, block-table width) cell of its length and the request's highest position.
"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("dsv41_warmup", Path(__file__).parents[1] / "serving" / "warmup.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


def hit(cached: int, total: int, keep: int) -> int:
    """Tokens served from the prefix cache: install_hit_cap's cut, block-aligned."""
    if keep <= 0 or cached <= total - keep:
        return cached
    return max(total - keep, 0) // W.BLOCK * W.BLOCK


def compiled_cell(s: int, k: int, fresh: bool, keep: int, max_len: int) -> tuple[int, int]:
    total = k * W.BLOCK + s
    left = total - (0 if fresh else hit(k * W.BLOCK, total, keep))
    last = left - (left - 1) // W.CHUNK * W.CHUNK
    bucket = next(b for b in W.QB if b >= last)
    return bucket, W.width(total, max_len)


@pytest.mark.parametrize("keep", [0, 140])
@pytest.mark.parametrize("max_len", [131072, 65536])
def test_every_prefill_cell_is_compiled_by_its_request(keep, max_len):
    cells = W.prefill_cells(max_len, keep)
    assert len({key for _, _, key, _ in cells}) == len(cells)
    for s, k, key, fresh in cells:
        assert compiled_cell(s, k, fresh, keep, max_len) == key, (key, k, fresh)


def test_fresh_prompts_only_where_a_hit_cannot_reach():
    cells = W.prefill_cells(131072, 140)
    assert sorted(key for _, _, key, fresh in cells if fresh) == [(128, 128), (128, 256), (128, 512), (128, 1024)]
    assert not any(fresh for *_, fresh in W.prefill_cells(131072, 0))


def test_the_old_ladder_left_four_cells_cold():
    # every cell through a hit, as before: the S = 128 cells past the narrowest width compile (256, W) instead
    cells = W.prefill_cells(131072, 140)
    reached = {compiled_cell(s, k, False, 140, 131072) for s, k, _, _ in cells}
    assert {key for *_, key, _ in cells} - reached == {(128, 128), (128, 256), (128, 512), (128, 1024)}
