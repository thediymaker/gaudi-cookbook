# SPDX-License-Identifier: Apache-2.0
"""convert/stage_checkpoint.py on a small fake checkpoint (stdlib only)."""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "convert" / "stage_checkpoint.py"


def test_stage(tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    cfg = {"model_type": "mimo_v2", "hybrid_layer_pattern": [0, 1, 1, 0], "quantization_config": {"fmt": "e4m3"}}
    (src / "config.json").write_text(json.dumps(cfg))
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s0.safetensors"}}))
    (src / "s0.safetensors").write_bytes(b"x")
    out = tmp_path / "out"
    for _ in range(2):  # a second run refreshes in place
        subprocess.run([sys.executable, str(SCRIPT), str(src), str(out)], check=True)
    staged = json.loads((out / "config.json").read_text())
    assert "quantization_config" not in staged
    assert staged["layer_types"] == ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    assert json.loads((out / "config.orig.json").read_text()) == cfg
    assert os.path.islink(out / "s0.safetensors")


def test_refuses_other_models(tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
    r = subprocess.run([sys.executable, str(SCRIPT), str(src), str(tmp_path / "out")], capture_output=True)
    assert r.returncode != 0
