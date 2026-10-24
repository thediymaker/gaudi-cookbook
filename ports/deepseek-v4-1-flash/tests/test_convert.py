# SPDX-License-Identifier: Apache-2.0
"""convert/ end to end on a small synthetic source checkpoint: convert, validate, check_checkpoint, and the values.

The source has the real expert shapes (the converter checks them) but only two experts in one layer, a few small
FP8 and bf16 tensors, and one small Engram table, so it runs in seconds on a CPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PORT / "convert"))
import convert_to_hpu_mxfp4 as conv  # noqa: E402

TEXT_CONFIG = {
    "num_hidden_layers": 1,
    "engram_layer_ids": [1],
    "engram_max_ngram_size": 3,
    "engram_n_heads": 2,
    "engram_vocab_size": 50,
}


def make_source(root: Path, g: torch.Generator) -> dict:
    sizes = sum(conv.engram_bucket_sizes(TEXT_CONFIG)[1])
    tc = {**TEXT_CONFIG, "engram_num_embeddings": [sizes]}
    (root / "config.json").write_text(json.dumps({"architectures": ["DeepseekV41ForCausalLM"], "text_config": tc}))

    def u8(shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, dtype=torch.int32).to(torch.uint8)

    t: dict[str, tuple[str, torch.Tensor]] = {}
    for e in range(2):
        b = f"layers.0.ffn.experts.{e}"
        t[f"{b}.w1.weight"] = ("I8", u8((conv.INTER, conv.H // 2)))
        t[f"{b}.w3.weight"] = ("I8", u8((conv.INTER, conv.H // 2)))
        t[f"{b}.w2.weight"] = ("I8", u8((conv.H, conv.INTER // 2)))
        t[f"{b}.w1.scale"] = ("F8_E8M0", u8((conv.INTER, conv.H // 32), 110, 140))
        t[f"{b}.w3.scale"] = ("F8_E8M0", u8((conv.INTER, conv.H // 32), 110, 140))
        t[f"{b}.w2.scale"] = ("F8_E8M0", u8((conv.H, conv.INTER // 32), 110, 140))
    w = u8((40, 70))
    w = torch.where((w & 0x7F) == 0x7F, w - 1, w)  # no NaN codes
    t["layers.0.attn.wq.weight"] = ("F8_E4M3", w)
    t["layers.0.attn.wq.scale"] = ("F8_E8M0", u8((2, 3), 120, 130))
    t["embed.weight"] = ("BF16", torch.randn(64, 16, generator=g).to(torch.bfloat16))
    t["head.weight"] = ("BF16", torch.randn(64, 16, generator=g).to(torch.bfloat16))
    t["layers.0.attn_norm.weight"] = ("F32", torch.randn(16, generator=g))
    ew = u8((sizes, 8))
    t["layers.1.engram.embed.weight"] = ("F8_E4M3", torch.where((ew & 0x7F) == 0x7F, ew - 1, ew))
    t["layers.1.engram.embed.scale"] = ("F8_E8M0", u8((sizes, 1), 120, 130))

    shards = {"model-00001-of-00002.safetensors": [], "model-00002-of-00002.safetensors": []}
    for i, name in enumerate(sorted(t)):
        shards[list(shards)[i % 2]].append(name)
    wm = {}
    for fn, names in shards.items():
        entries = []
        for n in names:
            dt, x = t[n]
            entries.append((n, dt, list(x.shape), [x.contiguous().view(torch.uint8).numpy().tobytes()]))
            wm[n] = fn
        conv.write_raw_safetensors(str(root / fn), entries)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    return t


def run(script: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PORT / script), *map(str, args)], capture_output=True, text=True, timeout=600
    )


def test_convert_validate_check(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    t = make_source(src, torch.Generator().manual_seed(0))
    r = run("convert/convert_to_hpu_mxfp4.py", "--src", src, "--dst", dst, "--part", "all", "--layers", "0")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "structural gate OK" in r.stdout
    r = run("convert/validate_conversion.py", "--src", src, "--dst", dst, "--part", "all", "--layers", "0")
    assert r.returncode == 0 and "VALIDATE_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    r = run("serving/check_checkpoint.py", dst)
    assert r.returncode == 0, r.stderr
    assert run("serving/check_checkpoint.py", src).returncode == 1  # the source is not servable

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "dsv41_hpu_mxfp4"
    assert cfg["quantization_config"]["routed_experts"]["w13_layout"] == "interleaved_gate_up"

    out = conv.Source(str(dst))
    w13 = out.bytes_u8("layers.0.ffn.experts.routed_experts.w13_weight")
    s13 = out.bytes_u8("layers.0.ffn.experts.routed_experts.w13_weight_scale")
    assert w13.shape == (2, 2 * conv.INTER, conv.H // 2)
    # interleaved: even rows are gate (w1), odd rows up (w3); bytes equal after -0 -> +0
    for e in range(2):
        gate = conv.canon_neg_zero(t[f"layers.0.ffn.experts.{e}.w1.weight"][1])
        up = conv.canon_neg_zero(t[f"layers.0.ffn.experts.{e}.w3.weight"][1])
        assert torch.equal(w13[e, 0::2], gate) and torch.equal(w13[e, 1::2], up)
        assert torch.equal(s13[e, 0::2], t[f"layers.0.ffn.experts.{e}.w1.scale"][1])
    assert not bool(((w13 & 0x0F) == 8).any() or ((w13 >> 4) == 8).any())
    # FP8 block-scaled -> bf16, exact
    wq = out.tensor("layers.0.attn.wq.weight")
    w, s = t["layers.0.attn.wq.weight"][1], t["layers.0.attn.wq.scale"][1]
    want = (
        w.view(torch.float8_e4m3fn).float()
        * torch.pow(2.0, s.float() - 127).repeat_interleave(32, 0)[:40].repeat_interleave(32, 1)[:, :70]
    )
    assert wq.dtype == torch.bfloat16 and torch.equal(wq.float(), want)
    # passthrough by bytes
    assert torch.equal(out.tensor("embed.weight"), t["embed.weight"][1])
    assert torch.equal(out.tensor("layers.0.attn_norm.weight"), t["layers.0.attn_norm.weight"][1])
    # engram buckets concatenate back to the source rows
    heads = sorted(n for n in out.weight_map if n.startswith("layers.1.engram.embed.head") and n.endswith("weight"))
    assert len(heads) == 4
    rows = torch.cat([out.bytes_u8(n) for n in heads])
    assert torch.equal(rows, t["layers.1.engram.embed.weight"][1])


def test_validator_catches_a_flipped_expert_byte(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    make_source(src, torch.Generator().manual_seed(1))
    assert (
        run("convert/convert_to_hpu_mxfp4.py", "--src", src, "--dst", dst, "--part", "all", "--layers", "0").returncode
        == 0
    )
    fn = dst / "experts-layers-0.safetensors"
    data = bytearray(fn.read_bytes())
    data[-1] ^= 0x11  # last byte: inside the last tensor's data, a value change in both nibbles
    fn.write_bytes(bytes(data))
    r = run("convert/validate_conversion.py", "--src", src, "--dst", dst, "--part", "experts", "--layers", "0")
    assert r.returncode == 1 and "VALIDATE_FAILED" in r.stdout
