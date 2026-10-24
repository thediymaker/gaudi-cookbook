#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sliding-window masks of the vllm-gaudi v0.30.0 runner for model type muse_spark (CPU, no device).

Reference (the transformers model): a query at absolute position q sees keys k with q - W < k <= q, exactly
min(W, q + 1) keys, the oldest at max(0, q - W + 1). Checked on the runner patched by patches/runner_window.py:

  prefill  HPUAttentionMetadataProcessor._set_attn_bias_for_sliding_window for every ctx x seq_len below: the
           gated branch (per-sequence window block list), the ungated branch (whole context) and ctx = 0. Every
           query row's finite columns, mapped back to absolute positions, must equal the reference set.
  decode   get_habana_paged_attn_buffers + _set_block_mapping with the window list the patched decode path
           selects (W // block + 1 blocks, exact head mask, fully masked head block dropped), at positions around
           block and window boundaries; no window block may be fully masked (NaN in the blockwise softmax).
  wiring   the patched source sites, and Gemma 4's image-mask gate left alone.
Controls (each must differ from the reference where stated):
  gemma4 prefill   the stock bound: rows with ctx >= 1, q >= W and in-chunk index i < W see W + 1 keys.
  decode, stock    W // block blocks, no head mask: q >= W sees W - block + lbu keys.
  decode, no mask  W // block + 1 blocks, no head mask: q >= W sees W + lbu keys.

  python3 test_runner_window.py [--runner PATH]     (or through pytest: skipped without vllm_gaudi)
"""

import argparse
import importlib.util
import sys
import types

try:
    import torch
except ImportError:  # collected by pytest on machines without torch: the test skips
    torch = None

W = 2048
BLOCK = 128
CTXS = [1, 127, 128, 2047, 2048, 4096, 5000]
SEQ_LENS = [1, 61, 4096]
DECODE_POS = [
    0,
    1,
    126,
    127,
    128,
    129,
    2046,
    2047,
    2048,
    2049,
    2050,
    2175,
    2176,
    2177,
    2303,
    2304,
    4095,
    4096,
    4097,
    5000,
    131071,
]
DTYPE = torch.bfloat16 if torch is not None else None


def load_runner(path):
    name = "vllm_gaudi.v1.worker.hpu_model_runner"
    if path:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    else:
        import vllm_gaudi.v1.worker.hpu_model_runner as mod
    return mod


def vllm_config(model_type):
    text = types.SimpleNamespace(layer_types=["sliding_attention", "full_attention"])
    text.get_text_config = lambda: text
    model = types.SimpleNamespace(
        dtype=DTYPE,
        get_sliding_window=lambda: W,
        hf_text_config=text,
        hf_config=types.SimpleNamespace(model_type=model_type),
    )
    return types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=BLOCK), model_config=model)


def processor(M, model_type):
    p = M.HPUAttentionMetadataProcessor(vllm_config(model_type))
    assert p.interleaved_sliding_window and p.sliding_window == W
    # the served prefill path; the CPU default may differ, the bias code is the same
    p.prefill_use_fusedsdpa = True
    p.use_window_sdpa = False
    return p


def prefill_bias(M, proc, ctx, seq_len, gated):
    """(bias [seq_len, cols], absolute position of every column, column valid flag).

    Batch 1: the runner never merges a prompt that has cached context with another
    (`_can_merge_prefill_contents`), and the bias code reshapes its causal part to it."""
    nblk = -(-ctx // BLOCK)
    blocks = list(range(1000, 1000 + nblk))  # physical ids are irrelevant to the mask
    if ctx == 0:
        block_list = window_block_list = None
    else:
        # whole-context list, padded past the sequence as a bucket would be
        block_list = torch.tensor(blocks + [-1] * 3, dtype=torch.int32)
        window_block_list = None
        if gated:
            sb = W // BLOCK + 1  # the runner's prefill sliding_block_size
            wb = blocks[-sb:]
            window_block_list = torch.tensor(wb + [-1] * (sb - len(wb)), dtype=torch.int32)
    md = M.HPUAttentionMetadataV1.make_prefill_metadata(
        attn_bias=None,
        block_list=block_list,
        context_lens_tensor=torch.tensor([ctx], dtype=torch.int32),
        seq_lens_tensor=torch.tensor([seq_len], dtype=torch.int32),
        slot_mapping=None,
        block_size=BLOCK,
        window_block_list=window_block_list,
    )
    md = M.trim_attn_metadata(md)
    out = proc._set_attn_bias_for_sliding_window(md, 1, seq_len, W, torch.device("cpu"), DTYPE)
    bias = out.window_attn_bias.reshape(seq_len, -1)
    n_past = bias.shape[1] - seq_len
    if ctx == 0:
        frame_base = 0
    elif gated:
        frame_base = max(0, nblk * BLOCK - n_past)
    else:
        frame_base = 0
    past_abs = torch.arange(n_past) + frame_base
    col_abs = torch.cat([past_abs, ctx + torch.arange(seq_len)])
    col_ok = torch.cat([past_abs < ctx, torch.ones(seq_len, dtype=torch.bool)])
    return bias, col_abs, col_ok


def ref_mask(q, col_abs, col_ok):
    return col_ok.unsqueeze(0) & (col_abs.unsqueeze(0) > q.unsqueeze(1) - W) & (col_abs.unsqueeze(0) <= q.unsqueeze(1))


def check_prefill(M, fails, notes):
    muse, gemma = processor(M, "muse_spark"), processor(M, "gemma4")
    if muse._swa_lb_off != 0 or gemma._swa_lb_off != 1:
        fails.append(f"_swa_lb_off muse={muse._swa_lb_off} gemma4={gemma._swa_lb_off}, want 0 / 1")
    cases = [(0, s, False) for s in SEQ_LENS] + [(c, s, g) for c in CTXS for s in SEQ_LENS for g in (True, False)]
    n_rows = n_ctrl = 0
    for ctx, seq_len, gated in cases:
        tag = f"prefill ctx={ctx} seq={seq_len} {'gated' if gated else 'ungated'}"
        q = ctx + torch.arange(seq_len)
        bias, col_abs, col_ok = prefill_bias(M, muse, ctx, seq_len, gated)
        finite = torch.isfinite(bias.float())
        ref = ref_mask(q, col_abs, col_ok)
        if not torch.equal(finite & col_ok, ref):
            bad = (finite & col_ok) != ref
            r = int(bad.any(1).nonzero()[0])
            fails.append(
                f"{tag}: {int(bad.any(1).sum())} rows differ; row {r} (q={int(q[r])}) "
                f"sees {int((finite[r] & col_ok[r]).sum())}, want {int(ref[r].sum())}"
            )
        else:
            cnt = finite.logical_and(col_ok).sum(1)
            want = torch.clamp(q + 1, max=W)
            oldest = torch.where(finite & col_ok, col_abs.unsqueeze(0), torch.iinfo(torch.int64).max).min(1).values
            if not torch.equal(cnt, want) or not torch.equal(oldest, torch.clamp(q - W + 1, min=0)):
                fails.append(f"{tag}: count/oldest mismatch")
        if (finite & ~col_ok).any():
            # columns past the context (bucket padding) must never be visible
            fails.append(f"{tag}: {int((finite & ~col_ok).sum())} padding columns visible")
        n_rows += seq_len
        # control: the stock bound (gemma4 keeps it)
        if ctx == 0:
            continue
        gbias, gcol, gok = prefill_bias(M, gemma, ctx, seq_len, gated)
        gfin = torch.isfinite(gbias.float()) & gok
        i = torch.arange(seq_len)
        extra_rows = (q >= W) & (i < W)
        gref = ref_mask(q, gcol, gok)
        # expected stock mask: the reference plus key q - W on extra_rows
        gwant = gref | (extra_rows.unsqueeze(1) & (gcol.unsqueeze(0) == (q - W).unsqueeze(1)) & gok)
        if not torch.equal(gfin, gwant):
            fails.append(f"control gemma4 {tag}: stock mask is not reference + key q-W on rows q>=W, i<W")
        elif extra_rows.any():
            cnt = gfin[extra_rows].sum(1)
            if not bool((cnt == W + 1).all()):
                fails.append(f"control gemma4 {tag}: extra rows see {cnt.unique().tolist()}, want {W + 1}")
            n_ctrl += int(extra_rows.sum())
    notes.append(f"prefill: {len(cases)} cases, {n_rows} query rows checked; gemma4 control: {n_ctrl} rows at W+1")
    if n_ctrl == 0:
        fails.append("control gemma4: no row exercised the extra key")


class _Bucketing:
    def find_decode_bucket(self, batch_size, num_blocks):
        return (batch_size, 1, num_blocks)  # no padding: every group is a real sequence


def decode_visible(M, proc, positions, n_window, head_mask):
    """Visible absolute positions per sequence, as the decode window bias encodes them."""
    g = torch.Generator().manual_seed(0)
    tables, slots, nblks = [], [], []
    next_id = 1
    for q in positions:
        nblk = q // BLOCK + 1
        ids = (torch.randperm(nblk, generator=g) + next_id).tolist()
        next_id += nblk
        tables.append(ids)
        slots.append([ids[q // BLOCK] * BLOCK + q % BLOCK])
        nblks.append(nblk)
    window_tables = [t[-n_window:] for t in tables]
    stub = types.SimpleNamespace(
        attn_block_size=BLOCK,
        use_contiguous_pa=True,
        bucketing_manager=_Bucketing(),
        get_dp_padding=lambda n: 0,
        _PAD_BLOCK_ID=0,
        model_config=types.SimpleNamespace(dtype=DTYPE),
    )
    bl, bg, bu = M.HPUModelRunner.get_habana_paged_attn_buffers(
        stub,
        window_tables,
        slots,
        len(positions),
        block_size=BLOCK,
        force_non_contiguous=True,
        head_mask_full_len=n_window if head_mask else None,
        drop_masked_head=head_mask,
    )
    md = M.HPUAttentionMetadataV1.make_decode_metadata(
        block_list=bl,
        block_usage=bu,
        block_groups=bg,
        input_positions=None,
        slot_mapping=None,
        block_size=BLOCK,
        window_block_list=bl,
        window_block_usage=bu,
        window_block_groups=bg,
        chunked_block_list=None,
        chunked_block_usage=None,
        chunked_block_groups=None,
    )
    md = M.trim_attn_metadata(md)
    out = proc._set_block_mapping(md, len(positions), torch.device("cpu"), DTYPE, True)
    vis = torch.isfinite(out.window_attn_bias.float())  # [n_blocks, BLOCK]
    groups = bg.tolist()
    # a block with no visible slot makes the flat-PA blockwise softmax NaN on HPU
    empty = int((~vis.any(-1)).sum())
    seen = [[] for _ in positions]
    k_of_seq = [0] * len(positions)
    for row, s in enumerate(groups):
        k = k_of_seq[s]
        k_of_seq[s] += 1
        base = (nblks[s] - groups.count(s) + k) * BLOCK  # the list holds this sequence's last blocks
        seen[s].extend(base + j for j in vis[row].nonzero().flatten().tolist())
    decode_visible.empty_blocks = empty
    return seen


def check_decode(M, fails, notes):
    proc = processor(M, "muse_spark")
    model_type = "muse_spark"
    # the patched decode path: +1 block for ("gpt_oss", "muse_spark"), exact head mask for muse_spark
    n_window = W // BLOCK + (1 if model_type in ("gpt_oss", "muse_spark") else 0)
    exact = model_type in ("muse_spark",) and W % BLOCK == 0
    seen = decode_visible(M, proc, DECODE_POS, n_window, exact)
    for q, s in zip(DECODE_POS, seen, strict=False):
        want = list(range(max(0, q - W + 1), q + 1))
        if sorted(s) != want:
            fails.append(
                f"decode q={q}: sees {len(s)} keys [{min(s)}..{max(s)}], want {len(want)} [{want[0]}..{want[-1]}]"
            )
    if decode_visible.empty_blocks:
        fails.append(f"decode: {decode_visible.empty_blocks} window blocks with every slot masked (NaN on HPU)")
    notes.append(
        f"decode: {len(DECODE_POS)} positions checked (window list {n_window} blocks, head mask on, "
        f"no fully masked block)"
    )
    # controls
    for label, nw, hm, want_n in (
        ("stock", W // BLOCK, False, lambda q: W - BLOCK + q % BLOCK + 1),
        ("no head mask", W // BLOCK + 1, False, lambda q: W + q % BLOCK + 1),
    ):
        seen = decode_visible(M, proc, DECODE_POS, nw, hm)
        pairs = list(zip(DECODE_POS, seen, strict=True))
        wrong = [q for q, s in pairs if sorted(s) != list(range(max(0, q - W + 1), q + 1))]
        counts_ok = all(len(s) == want_n(q) for q, s in pairs if q >= W and q % BLOCK != BLOCK - 1)
        if not wrong or not counts_ok:
            fails.append(f"control decode {label}: wrong at {wrong}, counts as predicted={counts_ok}")
        else:
            notes.append(
                f"control decode {label}: differs at {len(wrong)}/{len(DECODE_POS)} positions "
                f"(e.g. q=2048 sees {len(seen[DECODE_POS.index(2048)])})"
            )


def check_wiring(M, fails, notes):
    with open(M.__file__) as f:
        src = f.read()
    want = {
        'self.interleaved_sliding_window and self._get_model_type() in ("gemma4", "muse_spark")': 2,
        'self.interleaved_sliding_window and self._get_model_type() == "gemma4"': 0,
        'model_type in ["gpt_oss", "muse_spark"]': 1,
        '_exact_window = model_type in ("gpt_oss", "muse_spark")': 1,
        "head_mask_full_len=sliding_block_size if _exact_window else None,": 1,
        'drop_masked_head=model_type == "muse_spark"': 1,
        "mask = torch.where(usage >= 0, mask >= usage, mask < -usage)": 1,
        "self._swa_lb_off": 3,
        "window_size + torch.arange(seq_len, device=device) - 1": 0,
        'self._gemma4_mm_prefix = (self._get_model_type() == "gemma4"': 1,
    }
    for s, n in want.items():
        if src.count(s) != n:
            fails.append(f"source: {s!r} found {src.count(s)}x, want {n}")
    notes.append(f"wiring: {len(want)} source counts checked in {M.__file__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=None, help="patched hpu_model_runner.py to load instead of the installed one")
    args = ap.parse_args()
    from vllm_gaudi.extension.runtime import get_config

    # hw is overridden so the config never probes a device (CPU pod, or an HPU in use elsewhere)
    get_config(hw="gaudi2")
    M = load_runner(args.runner)
    if not hasattr(M.HPUAttentionMetadataProcessor(vllm_config("muse_spark")), "_swa_lb_off"):
        print(f"FAIL: {M.__file__} is not patched by patches/runner_window.py")
        return 1
    fails, notes = run_checks(M)
    for n in notes:
        print(n)
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else f"FAILED ({len(fails)})")
    return 1 if fails else 0


def run_checks(M):
    fails, notes = [], []
    check_wiring(M, fails, notes)
    check_prefill(M, fails, notes)
    check_decode(M, fails, notes)
    return fails, notes


def test_runner_window():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("vllm_gaudi")
    from vllm_gaudi.extension.runtime import get_config

    get_config(hw="gaudi2")
    M = load_runner(None)
    if not hasattr(M.HPUAttentionMetadataProcessor(vllm_config("muse_spark")), "_swa_lb_off"):
        pytest.skip("the installed runner is not patched by patches/runner_window.py")
    fails, _ = run_checks(M)
    assert not fails, fails


if __name__ == "__main__":
    sys.exit(main())
