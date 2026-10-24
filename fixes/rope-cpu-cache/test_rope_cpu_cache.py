# SPDX-License-Identifier: Apache-2.0
"""Tests for rope-cpu-cache.

test_wrapper_*: CPU only (torch). The block the fix appends runs against a stand-in
RotaryEmbeddingBase; tables must be computed under a CPU default device and land on the device
that was active, for existing and later subclasses.

test_hpu_table_accuracy: needs a Gaudi device and vLLM; compares a 256K-position table built
with the installed vLLM against float64. Run with: pytest fixes/rope-cpu-cache
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent


def _apply_module():
    spec = importlib.util.spec_from_file_location("rope_apply", HERE / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _namespace():
    seen = []

    class RotaryEmbeddingBase:
        def _compute_cos_sin_cache(self):
            seen.append(torch.get_default_device().type)
            return torch.arange(4.0)

    class Early(RotaryEmbeddingBase):
        def _compute_cos_sin_cache(self):
            seen.append(torch.get_default_device().type)
            return torch.ones(2)

    ns = {"torch": torch, "RotaryEmbeddingBase": RotaryEmbeddingBase}
    exec(_apply_module().PATCH, ns)
    return ns, seen, Early


def test_wrapper_computes_on_cpu_and_moves_to_active_device():
    ns, seen, early = _namespace()
    base = ns["RotaryEmbeddingBase"]

    class Late(base):
        def _compute_cos_sin_cache(self):
            seen.append(torch.get_default_device().type)
            return torch.zeros(3)

    with torch.device("meta"):
        outs = [cls()._compute_cos_sin_cache() for cls in (base, early, Late)]
    assert seen == ["cpu", "cpu", "cpu"]
    assert all(o.device.type == "meta" for o in outs)


def test_wrapper_is_idempotent():
    ns, _, early = _namespace()
    fn = early.__dict__["_compute_cos_sin_cache"]
    ns["_rope_cpu_wrap"](ns["RotaryEmbeddingBase"])
    assert early.__dict__["_compute_cos_sin_cache"] is fn


def test_hpu_table_accuracy():
    pytest.importorskip("habana_frameworks.torch")
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    n, theta, rot = 262144, 1e7, 64
    with set_current_vllm_config(VllmConfig()), torch.device("hpu"):
        rope = get_rope(
            head_size=192,
            max_position=n,
            is_neox_style=True,
            dtype=torch.float32,
            rope_parameters={"rope_type": "default", "rope_theta": theta, "partial_rotary_factor": 64 / 192},
        )
    got = rope.cos_sin_cache.float().cpu().double()
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    ang = torch.outer(torch.arange(n, dtype=torch.float64), inv)
    ref = torch.cat([ang.cos(), ang.sin()], dim=-1)
    assert float((got - ref).abs()[131072:].max()) < 0.05
