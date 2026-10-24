# SPDX-License-Identifier: Apache-2.0
"""CPU checks of mimo_v2_hpu.ops: the exact attention, the tiled mask, the expert re-encode and the weight layout."""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimo_v2_hpu"))
from mimo_v2_hpu import ops  # noqa: E402

W, D, BS = 128, 192, 128


@pytest.mark.parametrize("kvh", [1, 2])
@pytest.mark.parametrize(
    "S,ctx", [(1, 0), (100, 0), (128, 0), (129, 0), (300, 0), (37, 50), (200, 128), (256, 200), (64, 1000)]
)
def test_swa_band_matches_reference(S, ctx, kvh):
    torch.manual_seed(S * 7 + ctx)
    nh = 16
    q = torch.randn(1, S, nh, D) * 0.5
    k = torch.randn(1, S, kvh, D) * 0.5
    v = torch.randn(1, S, kvh, D)
    sinks = torch.linspace(-1, 4, nh)
    kc = vc = ctxt = frame = None
    if ctx:
        nblk = -(-ctx // BS)
        wc = 2 * BS  # the runner's window list: window // block + 1 blocks
        frame = torch.tensor([max(nblk * BS - wc, 0)])
        kc = torch.randn(1, wc, kvh, D) * 0.5
        vc = torch.randn(1, wc, kvh, D)
        ctxt = torch.tensor([ctx])
    ref = ops.swa_reference(q, k, v, sinks, D**-0.5, W, kc, vc, ctxt, frame)
    out = ops.swa_band_attention(q, k, v, sinks, D**-0.5, W, kc, vc, ctxt, frame)
    assert ((out - ref).norm() / ref.norm()).item() < 1e-5


def test_sink_carries_the_mass():
    """With logits near -25 and sinks near +1 (MiMo's regime) the output is tiny; dropping the sink is not."""
    torch.manual_seed(0)
    q = torch.full((1, 4, 2, 8), 1.0)
    k = -torch.full((1, 4, 1, 8), 25.0 / 8 / 8**-0.5)
    v = torch.randn(1, 4, 1, 8)
    with_sink = ops.swa_reference(q, k, v, torch.ones(2), 8**-0.5, W)
    no_sink = ops.swa_reference(q, k, v, torch.full((2,), -1e9), 8**-0.5, W)
    assert with_sink.abs().max() < 1e-6 < no_sink.abs().max()


def _runner_mask(B, S, P, ctx, seq):
    """vllm-gaudi v0.26.0 _set_attn_bias, as a bool mask (True = masked)."""
    past = torch.arange(P).view(1, -1).expand(B, -1).ge(ctx.view(-1, 1)).view(B, 1, -1).expand(B, S, -1)
    len_mask = torch.arange(S).view(1, S).ge(seq.unsqueeze(-1)).view(B, 1, 1, S)
    causal = torch.triu(torch.ones((B, 1, S, S), dtype=torch.bool), diagonal=1)
    return torch.cat((past.view(B, 1, S, -1), causal.logical_or(len_mask)), dim=-1)


@pytest.mark.parametrize("B,S,P", [(1, 256, 0), (1, 384, 640), (2, 256, 512), (3, 512, 1024)])
def test_ga_tiles_equal_untiled(B, S, P):
    torch.manual_seed(B + S + P)
    H, Dh = 4, 16
    for _ in range(3):
        ctx = torch.randint(0, P + 1, (B,))
        seq = torch.randint(1, S + 1, (B,))
        full = _runner_mask(B, S, P, ctx, seq)
        q, k, v = torch.randn(B, H, S, Dh), torch.randn(B, H, P + S, Dh), torch.randn(B, H, P + S, Dh)

        def attend(qs, m, k=k, v=v):
            s = (qs @ k.transpose(-1, -2)) / math.sqrt(Dh)
            return torch.softmax(s.masked_fill(m, float("-inf")), -1).nan_to_num(0.0) @ v

        for budget in (1 << 29, 128 * (P + S), 256 * (P + S)):
            rows = ops.ga_tile_rows(S, P + S, budget=budget)
            starts = range(0, S, rows)
            tiles = [ops.ga_tile_mask(torch.arange(t, min(S, t + rows)), P, S, ctx, seq) for t in starts]
            assert torch.equal(torch.cat(tiles, dim=2), full)
            out = torch.cat([attend(q[:, :, t : t + rows], m) for t, m in zip(starts, tiles, strict=True)], 2)
            assert (out - attend(q, full)).abs().max() < 1e-5


def test_ga_tile_budget_counts_heads():
    # an 8192-token chunk at the 4032-block context bucket (keys = 4032 * 128 + 8192 = 524288), 8 heads per rank
    # at TP8: 256-row tiles, 2 GiB of bf16 scores. Rows x keys alone would have allowed 2048 rows.
    keys = 4032 * 128 + 8192
    assert ops.ga_tile_rows(8192, keys, heads=8) == 256
    assert ops.ga_tile_rows(8192, keys, heads=1) == 2048


def _random_mxfp4(rows, cols, gen, lo=124, hi=131):
    w = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=gen)
    s = torch.randint(lo, hi, (rows, cols // 32), dtype=torch.uint8, generator=gen)
    return w, s


def test_mxfp4_to_fp8_is_exact():
    gen = torch.Generator().manual_seed(0)
    w, s = _random_mxfp4(64, 512, gen)
    q, scale = ops.mxfp4_to_fp8(w, s)
    assert q.dtype == torch.float8_e4m3fn and scale.shape == (64,)
    ref = ops.mxfp4_reference(w, s)
    got = q.double() * scale.double()[:, None]
    assert torch.equal(got, ref)
    # power-of-two row scales
    assert torch.equal(torch.log2(scale), torch.log2(scale).round())


def test_mxfp4_to_fp8_refuses_lossy_rows():
    gen = torch.Generator().manual_seed(1)
    w, s = _random_mxfp4(4, 64, gen)
    s[0, 0] = 100  # one block 2^-27 below the row maximum: outside e4m3's exponent range
    s[0, 1] = 135
    w[0, :] = 0x77  # every nibble nonzero (6.0)
    with pytest.raises(RuntimeError, match="not exact"):
        ops.mxfp4_to_fp8(w, s)


def test_qkv_rank_rows_cover_each_head_once():
    nh, kvh, hd, vhd, ckpt = 64, 4, 192, 128, 4
    total = nh * hd + kvh * hd + kvh * vhd
    for tp in (4, 8):
        q_rows = []
        for r in range(tp):
            idx = ops.qkv_rank_rows(nh, kvh, hd, vhd, ckpt, r, tp)
            nq = nh // tp * hd
            q_rows.append(idx[:nq])
            assert idx.max() < total and len(set(idx.tolist())) == idx.numel()
        assert sorted(torch.cat(q_rows).tolist()) == sorted(
            c * (total // ckpt) + i for c in range(ckpt) for i in range(nh // ckpt * hd)
        )


def test_qkv_scale_index_layouts():
    rows = 4 * (16 * 192 + 192 + 128)  # four chunks, not block aligned
    per_chunk = ops.qkv_scale_index(rows, 4 * -(-(rows // 4) // 128), 4)
    assert per_chunk[rows // 4] == -(-(rows // 4) // 128)  # chunk 1 starts on a fresh scale row
    assert torch.equal(ops.qkv_scale_index(rows, -(-rows // 128), 4), torch.arange(rows) // 128)
    with pytest.raises(ValueError):
        ops.qkv_scale_index(rows, 7, 4)


def test_exact_cos_sin_at_long_positions():
    table = ops.exact_cos_sin(200_001, 64, 1e7)
    assert table.shape == (200_001, 64)
    # the fastest frequency's angle is the position itself
    assert abs(table[200_000, 0].item() - math.cos(200_000.0)) < 1e-12
    assert abs(table[200_000, 32].item() - math.sin(200_000.0)) < 1e-12


def test_checkpoint_experts_round_trip():
    """Set MIMO_CHECKPOINT_DIR to a MiMo-V2.6-Flash checkpoint to re-encode a sample of real expert tensors."""
    import json
    import os

    ckpt = os.environ.get("MIMO_CHECKPOINT_DIR")
    if not ckpt:
        pytest.skip("MIMO_CHECKPOINT_DIR not set")
    safe_open = pytest.importorskip("safetensors").safe_open
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    for layer, e in ((1, 0), (5, 77), (24, 200), (47, 255)):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
            with safe_open(os.path.join(ckpt, wm[name]), framework="pt") as f:
                w, s = f.get_tensor(name), f.get_tensor(name + "_scale")
            q, scale = ops.mxfp4_to_fp8(w, s)
            assert torch.equal(q.double() * scale.double()[:, None], ops.mxfp4_reference(w, s)), name
