# SPDX-License-Identifier: Apache-2.0
"""Load format `beam` planning and staging on CPU, on a small synthetic safetensors checkpoint.

The loader must put exactly what the stock path puts into each parameter: the rows (gate/up) or the
column strip (down) of the rank's MoE TP slice, rescaled as vllm-gaudi's Gaudi 2 wrapper rescales them,
and only the rank's own experts under EP.
"""

import json
import struct

import pytest

torch = pytest.importorskip("torch")

from beam_hpu import load_plan as LP  # noqa: E402

E, INTER, H = 4, 8, 16  # experts, intermediate, hidden
LAYER = 3
SAFE = {torch.float8_e4m3fn: "F8_E4M3", torch.bfloat16: "BF16", torch.float32: "F32"}


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": SAFE[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


@pytest.fixture
def ckpt(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {"model.layers.0.mlp.down_proj.weight": torch.randn(H, 2 * INTER, generator=g).to(torch.bfloat16)}
    for e in range(E):
        p = f"model.layers.{LAYER}.mlp.experts.{e}"
        for proj, shape in (("gate_proj", (INTER, H)), ("up_proj", (INTER, H)), ("down_proj", (H, INTER))):
            tensors[f"{p}.{proj}.weight"] = (torch.randn(*shape, generator=g) * 50).to(torch.float8_e4m3fn)
            tensors[f"{p}.{proj}.weight_scale"] = torch.rand(shape[0], 1, generator=g).to(torch.bfloat16)
    path = tmp_path / "model-00001-of-00001.safetensors"
    write_safetensors(path, tensors)
    return str(path), tensors


def test_plan_tp_rank_reads_only_its_rows(ckpt):
    path, _ = ckpt
    jobs, skipped = LP.plan_jobs([path], {LAYER: (2, 1, lambda e: e)}, rescale=True)
    assert skipped == 0
    by = {j.name: j for j in jobs}
    gate = by[f"model.layers.{LAYER}.mlp.experts.0.gate_proj.weight"]
    down = by[f"model.layers.{LAYER}.mlp.experts.0.down_proj.weight"]
    scale = by[f"model.layers.{LAYER}.mlp.experts.0.gate_proj.weight_scale"]
    assert (
        gate.rows == (INTER // 2, INTER)
        and gate.rescale == ("rows", INTER // 2, INTER)
        and gate.read_bytes == gate.nbytes // 2
    )
    assert down.rows is None and down.rescale == ("cols", INTER // 2, INTER)
    assert scale.rows is None and scale.rescale is None
    assert by["model.layers.0.mlp.down_proj.weight"].exp is None


def test_plan_ep_skips_other_experts(ckpt):
    path, _ = ckpt
    jobs, skipped = LP.plan_jobs([path], {LAYER: (1, 0, lambda e: e - 2 if e >= 2 else -1)}, rescale=True)
    assert skipped == 2 * 6
    experts = {j.exp[1] for j in jobs if j.exp}
    assert experts == {0, 1}
    assert all(j.rescale in (None, ("all",)) for j in jobs)


class _Cfg:
    def __init__(self, tp, rank):
        self.is_act_and_mul = True
        self.tp_rank = rank
        self.moe_parallel_config = type("P", (), {"tp_size": tp})()


class _Routed:
    """The attributes of a FusedMoE layer the stage reads, with CPU parameters."""

    def __init__(self, local, tp, rank):
        i_tp = INTER // tp
        self.local_num_experts, self.intermediate_size_per_partition, self.hidden_size = local, i_tp, H
        self.moe_config = _Cfg(tp, rank)
        self.w13_weight = torch.nn.Parameter(torch.zeros(local, 2 * i_tp, H, dtype=torch.float8_e4m3fn), False)
        self.w2_weight = torch.nn.Parameter(torch.zeros(local, H, i_tp, dtype=torch.float8_e4m3fn), False)
        self.w13_weight_scale = torch.nn.Parameter(torch.zeros(local, 2 * i_tp, 1), False)
        self.w2_weight_scale = torch.nn.Parameter(torch.zeros(local, H, 1), False)


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
def test_reader_and_stage_match_the_stock_layout(ckpt, tp, rank):
    path, tensors = ckpt
    routed = _Routed(E, tp, rank)
    assert LP.ExpertStage.check(routed) is None
    stage = LP.ExpertStage(routed, scale_x2=True)
    jobs, _ = LP.plan_jobs([path], {LAYER: (tp, rank, lambda e: e)}, rescale=True, file_group=8)
    for j, t in LP.Reader(jobs, threads=4, window=1 << 20, stages={LAYER: stage}):
        if t is None:
            stage.done(j)
    assert stage.complete()
    n = INTER // tp
    rows = slice(rank * n, (rank + 1) * n)
    for e in range(E):
        p = f"model.layers.{LAYER}.mlp.experts.{e}"
        gate, up, down = (LP.gaudi2_rescale(tensors[f"{p}.{x}.weight"]) for x in ("gate_proj", "up_proj", "down_proj"))
        w13 = torch.cat([gate[rows], up[rows]]).view(torch.uint8)
        assert torch.equal(routed.w13_weight.data[e].view(torch.uint8), w13)
        assert torch.equal(routed.w2_weight.data[e].view(torch.uint8), down[:, rows].contiguous().view(torch.uint8))
        s13 = torch.cat([tensors[f"{p}.gate_proj.weight_scale"][rows], tensors[f"{p}.up_proj.weight_scale"][rows]])
        assert torch.equal(routed.w13_weight_scale.data[e], (s13 * 2.0).float())
        assert torch.equal(routed.w2_weight_scale.data[e], (tensors[f"{p}.down_proj.weight_scale"] * 2.0).float())


def test_rescale_is_the_wrapper_expression():
    w = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert torch.equal(
        LP.gaudi2_rescale(w).view(torch.uint8), (w.float() * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    )
