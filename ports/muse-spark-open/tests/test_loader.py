# SPDX-License-Identifier: Apache-2.0
"""muse_spark_hpu.loader on a small synthetic checkpoint in the AMD Quark layout (CPU, no vLLM).

The checkpoint has two layers: layer 0 with BF16 experts (as layers 0 and 61 of the real one), layer 1 with MXFP4
experts (U8 weights + U8 scales), an untied lm_head, and one audio and one vision tensor. Checks:
  * the headers-only dry run is clean at TP1, TP2 / EP2 and TP2 / EP1 (every parameter tiled once per rank,
    every tensor read whole or partitioned, audio and vision skipped, no tensor data read);
  * load_into fills every parameter, and the ranks reassemble to the source tensors byte for byte;
  * controls: a missing tensor, an unknown tensor and an inexact dtype copy are refused.
"""

from __future__ import annotations

import json
import os
import struct

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from muse_spark_hpu import loader as LD  # noqa: E402

CFG = {
    "architectures": ["MuseSparkForConditionalGeneration"],
    "text_config": {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_hidden_layers": 2,
        "vocab_size": 200,
        "num_local_experts": 4,
        "moe_hidden_size": 64,
        "moe_intermediate_size": 64,
    },
}
H, NQ, NK, L, V, E, HM, INTER = 64, 64, 32, 2, 200, 4, 64, 64
NAME = {torch.bfloat16: "BF16", torch.float32: "F32", torch.uint8: "U8"}
T = LD.TEXT


def tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def bf(*s):
        return torch.randn(*s, generator=g).to(torch.bfloat16)

    def f32(*s):
        return torch.randn(*s, generator=g)

    def u8(*s):
        return torch.randint(0, 256, s, generator=g, dtype=torch.uint8)

    out = {T + "embed_tokens.weight": bf(V, H), "lm_head.weight": bf(V, H), T + "norm.weight": f32(H)}
    for layer in range(L):
        p = f"{T}layers.{layer}."
        out.update(
            {
                p + "self_attn.q_proj.weight": bf(NQ, H),
                p + "self_attn.k_proj.weight": bf(NK, H),
                p + "self_attn.v_proj.weight": bf(NK, H),
                p + "self_attn.gate_proj.weight": bf(NQ, H),
                p + "self_attn.o_proj.weight": bf(H, NQ),
                p + "mlp.gate.weight": f32(E, H),
                p + "mlp.gate.e_score_correction_bias": f32(E),
                p + "mlp.pre_expert_proj.weight": bf(HM, H),
                p + "mlp.post_expert_proj.weight": bf(H, HM),
                p + "mlp.pre_expert_norm.weight": f32(HM),
                p + "mlp.experts.post_expert_norm.weight": f32(HM),
            }
        )
        for nm in ("input_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            out[p + nm + ".weight"] = f32(H)
        for nm in ("post_attention_residual_gate", "post_feedforward_residual_gate"):
            out[p + nm + ".gate"] = f32(H)
        for e in range(E):
            x = f"{p}mlp.experts.{e}."
            if layer == 0:
                out.update(
                    {
                        x + "gate_proj.weight": bf(INTER, HM),
                        x + "up_proj.weight": bf(INTER, HM),
                        x + "down_proj.weight": bf(HM, INTER),
                    }
                )
            else:
                for pj, (rows, cols) in (("gate", (INTER, HM)), ("up", (INTER, HM)), ("down", (HM, INTER))):
                    out[x + pj + "_proj.weight"] = u8(rows, cols // 2)
                    out[x + pj + "_proj.weight_scale"] = u8(rows, cols // 32)
    out["model.audio_tower.proj.weight"] = bf(4, 4)
    out["model.vision_tower.proj.weight"] = bf(4, 4)
    return out


def write_checkpoint(path, ts: dict[str, torch.Tensor], cfg=CFG) -> None:
    os.makedirs(path, exist_ok=True)
    names = sorted(ts)
    halves = {0: names[: len(names) // 2], 1: names[len(names) // 2 :]}
    weight_map = {}
    for i, part in halves.items():
        fn = f"model-0000{i + 1}-of-00002.safetensors"
        hdr, blobs, off = {}, [], 0
        for n in part:
            b = ts[n].contiguous().view(torch.uint8).numpy().tobytes()
            hdr[n] = {"dtype": NAME[ts[n].dtype], "shape": list(ts[n].shape), "data_offsets": [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
            weight_map[n] = fn
        h = json.dumps(hdr).encode()
        h += b" " * (-len(h) % 8)
        with open(os.path.join(path, fn), "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    ts = tensors()
    path = str(tmp_path_factory.mktemp("muse-syn"))
    write_checkpoint(path, ts)
    return path, ts


LAYOUTS = [(1, None), (2, 2), (2, 1)]


@pytest.mark.parametrize("tp,ep", LAYOUTS)
def test_dry_run(ckpt, tp, ep):
    rep = LD.dry_run(ckpt[0], tp, ep)
    assert rep["ok"], rep
    assert rep["expert_kind"] == {"bf16": 1, "mxfp4": 1}
    assert rep["skipped"] == {"audio": 1, "vision": 1}
    assert rep["data_bytes_read"] == 0


def load_rank(path, tp, rank, ep):
    lay = LD.Layout(tp, rank, ep)
    specs, _ = LD.specs_for(path, lay)
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}
    params = {n: torch.full(s.shape, 7, dtype=dt[s.dtype]) for n, s in specs.items()}
    rep = LD.load_into(params, path, lay, workers=2)
    assert rep["loaded"] == set(specs)
    return params


@pytest.mark.parametrize("tp,ep", LAYOUTS)
def test_load_reassembles(ckpt, tp, ep):
    path, ts = ckpt
    ranks = [load_rank(path, tp, r, ep) for r in range(tp)]
    ep_n = tp if ep is None else ep
    s = tp // ep_n
    n = E // ep_n
    q, k = NQ // tp, NK // tp
    for layer in range(L):
        src, dst = f"{T}layers.{layer}.", f"model.layers.{layer}."
        qkv = [p[dst + "self_attn.qkv_proj.weight"] for p in ranks]
        assert torch.equal(torch.cat([x[:q] for x in qkv]), ts[src + "self_attn.q_proj.weight"])
        assert torch.equal(torch.cat([x[q : q + k] for x in qkv]), ts[src + "self_attn.k_proj.weight"])
        assert torch.equal(torch.cat([x[q + k :] for x in qkv]), ts[src + "self_attn.v_proj.weight"])
        o = torch.cat([p[dst + "self_attn.o_proj.weight"] for p in ranks], dim=1)
        assert torch.equal(o, ts[src + "self_attn.o_proj.weight"])
        assert torch.equal(
            ranks[-1][dst + "mlp.post_expert_norm.weight"], ts[src + "mlp.experts.post_expert_norm.weight"]
        )
        tails = ("", "_scale") if layer == 1 else ("",)
        for tail in tails:
            sfx = "weight_scale" if tail else "weight"
            for e in range(E):
                g, j0 = e // n, e % n  # expert group and local index
                parts = [ranks[g * s + j] for j in range(s)]
                w13 = torch.cat([p[dst + "mlp.w13_weight" + tail][j0] for p in parts], dim=0)
                Is = w13.shape[0] // (2 * s)
                gate = torch.cat([w13[2 * Is * j : 2 * Is * j + Is] for j in range(s)])
                up = torch.cat([w13[2 * Is * j + Is : 2 * Is * (j + 1)] for j in range(s)])
                x = f"{src}mlp.experts.{e}."
                assert torch.equal(gate, ts[x + "gate_proj." + sfx]), (layer, e, tail)
                assert torch.equal(up, ts[x + "up_proj." + sfx]), (layer, e, tail)
                w2 = torch.cat([p[dst + "mlp.w2_weight" + tail][j0] for p in parts], dim=1)
                assert torch.equal(w2, ts[x + "down_proj." + sfx]), (layer, e, tail)
    per = ranks[0]["model.embed_tokens.weight"].shape[0]
    emb = torch.cat([p["model.embed_tokens.weight"] for p in ranks])
    assert torch.equal(emb[:V], ts[T + "embed_tokens.weight"])
    assert per * tp >= V and not emb[V:].any()  # vocab padding rows are zeroed


def test_control_missing_tensor(ckpt, tmp_path):
    ts = dict(ckpt[1])
    del ts[f"{T}layers.1.self_attn.k_proj.weight"]
    write_checkpoint(str(tmp_path), ts)
    rep = LD.dry_run(str(tmp_path), 2, 2)
    assert not rep["ok"] and rep["missing"]


def test_control_unknown_tensor(ckpt, tmp_path):
    ts = dict(ckpt[1])
    ts[f"{T}layers.0.mlp.shared_expert.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    write_checkpoint(str(tmp_path), ts)
    rep = LD.dry_run(str(tmp_path), 2, 2)
    assert not rep["ok"] and rep["unknown"] == [f"{T}layers.0.mlp.shared_expert.weight"]


def test_control_inexact_copy():
    x = LD.Read("a", ((0, 4),), "b", ((0, 4),), False, 16)
    with pytest.raises(TypeError):
        LD.copy_into(torch.zeros(4, dtype=torch.bfloat16), x, torch.zeros(4, dtype=torch.float32))
    with pytest.raises(TypeError):
        LD.copy_into(torch.zeros(4, dtype=torch.int8), x, torch.zeros(4, dtype=torch.uint8))
