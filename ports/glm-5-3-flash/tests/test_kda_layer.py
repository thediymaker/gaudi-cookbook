# SPDX-License-Identifier: Apache-2.0
"""CPU check of the KDA layer's state indexing (needs torch; vLLM and vllm-gaudi are stubbed).

vllm-gaudi pads a decode batch with state index -1. Its own GDN layer reads `ssm_state[state_indices]`, where -1 wraps
to the last row; this layer uses `index_select` / `index_copy_`, which do not wrap, so every load and the save must
take the remapped indices. Here the decode path runs on CPU with the real KDA kernels and identity convolutions: a -1
row must read and write the last row, not raise. What HPU does with an unmapped -1 is not exercised here.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_PKG_DIR = Path(__file__).resolve().parents[1] / "glm5_next_hpu" / "glm5_next_hpu"
_PKG = "glm5_next_kda_layer_pkg"


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture(scope="module")
def layer():
    stubs = {
        "vllm": _stub("vllm"),
        "vllm.distributed": _stub("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0),
        "vllm.forward_context": _stub("vllm.forward_context", get_forward_context=None),
        "vllm.model_executor": _stub("vllm.model_executor"),
        "vllm.model_executor.layers": _stub("vllm.model_executor.layers"),
        "vllm.model_executor.layers.mamba": _stub("vllm.model_executor.layers.mamba"),
        "vllm.model_executor.layers.mamba.gdn": _stub("vllm.model_executor.layers.mamba.gdn"),
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": _stub(
            "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn", KimiGatedDeltaNetAttention=torch.nn.Module
        ),
        "vllm.model_executor.model_loader": _stub("vllm.model_executor.model_loader"),
        "vllm.model_executor.model_loader.weight_utils": _stub(
            "vllm.model_executor.model_loader.weight_utils", default_weight_loader=None
        ),
        "vllm.model_executor.utils": _stub("vllm.model_executor.utils", set_weight_attrs=None),
        "vllm_gaudi": _stub("vllm_gaudi"),
        "vllm_gaudi.ops": _stub("vllm_gaudi.ops"),
        "vllm_gaudi.ops.causal_conv1d_pytorch": _stub(
            "vllm_gaudi.ops.causal_conv1d_pytorch",
            hpu_causal_conv1d_fn=None,
            hpu_causal_conv1d_update=lambda x, **_: x,
        ),
    }
    pkg = _stub(_PKG)
    pkg.__path__ = [str(_PKG_DIR)]
    with pytest.MonkeyPatch.context() as mp:
        for name, mod in {**stubs, _PKG: pkg}.items():
            mp.setitem(sys.modules, name, mod)
        spec = importlib.util.spec_from_file_location(f"{_PKG}.kda_layer", _PKG_DIR / "kda_layer.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        yield mod


def test_state_slots_wrap_padding(layer):
    idx = torch.tensor([2, -1, 0, -1], dtype=torch.int32)
    assert layer._state_slots(idx, 5).tolist() == [2, 4, 0, 4]
    assert layer._state_slots(idx, 5).dtype == torch.long


def test_decode_padding_row_uses_last_slot(layer, monkeypatch):
    heads, d, slots = 2, 4, 4
    gen = torch.Generator().manual_seed(0)
    state_indices = torch.tensor([1, -1])
    md = types.SimpleNamespace(is_prompt=False, load_indices_tensor=state_indices, query_start_loc=None)
    monkeypatch.setattr(layer, "get_forward_context", lambda: types.SimpleNamespace(attn_metadata=md))

    def conv():
        return types.SimpleNamespace(weight=torch.zeros(heads * d, 1, 4), bias=None)

    ssm_state = torch.randn(slots, heads, d, d, generator=gen)
    before = ssm_state.clone()
    fake = types.SimpleNamespace(
        prefix="layer",
        kv_cache=(torch.zeros(slots, 3, 3 * heads * d), ssm_state),
        _resolve_state_indices=lambda m: m.load_indices_tensor,
        head_dim=d,
        A_log=torch.zeros(1, 1, heads, 1),
        dt_bias=torch.zeros(heads * d),
        gate_lower_bound=-5.0,
        activation="silu",
        q_conv1d=conv(),
        k_conv1d=conv(),
        v_conv1d=conv(),
    )
    q, k, v, g1 = (torch.randn(2, heads * d, generator=gen) for _ in range(4))
    beta = torch.rand(2, heads, generator=gen)
    out = torch.zeros(1, 2, heads, d)

    layer.Glm5NextHPUKDAAttention._forward(fake, q, k, v, g1, beta, out)

    # the same step written out: the padding row reads the last slot (3), as ssm_state[-1] would
    seq = [t.reshape(2, 1, heads, d) for t in (q, k, v)]
    g = layer.hpu_kda_gate(g1.reshape(2, 1, -1), fake.A_log, d, g_bias=fake.dt_bias, lower_bound=-5.0)
    ref_out, ref_final = layer.hpu_recurrent_kda(*seq, g, beta.unsqueeze(1), initial_state=before[[1, 3]])
    assert torch.allclose(out, ref_out.reshape(1, 2, heads, d))
    assert torch.allclose(ssm_state[[1, 3]], ref_final)
    assert torch.equal(ssm_state[[0, 2]], before[[0, 2]])
