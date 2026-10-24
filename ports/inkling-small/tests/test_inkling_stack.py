# SPDX-License-Identifier: Apache-2.0
"""Checks that need vLLM (run them in the port image; they skip elsewhere). With INKLING_MODEL_DIR set to the
checkpoint they also resolve its config. run_gates.sh runs this file first, before any request.

- The plugin registers the config, the model for both architectures and the inkling_py parsers.
- KV layout: 42 attention specs padded by the conv bytes form ONE FullAttention group in vLLM's own
  grouping; the old layout (a MambaSpec per layer next to attention) does not (negative control).
- In the port image (INKLING_PORT_IMAGE=1): the plugin is installed as an entry point and the runner patch
  is present.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_gaudi")

import inkling_hpu  # noqa: E402

MODEL_DIR = os.environ.get("INKLING_MODEL_DIR")
TP = int(os.environ.get("INKLING_TP", "8"))


@pytest.fixture(scope="module")
def registered():
    inkling_hpu.register()


def test_config_resolves_to_port_config(registered):
    if not MODEL_DIR:
        pytest.skip("INKLING_MODEL_DIR not set")
    from vllm.transformers_utils.config import get_config
    from vllm_gaudi.v1.worker.hpu_model_runner import is_interleaved

    from inkling_hpu.config import InklingHPUConfig

    cfg = get_config(MODEL_DIR, trust_remote_code=True)
    assert isinstance(cfg, InklingHPUConfig)
    assert cfg.layer_types.count("sliding_attention") == 35 and len(cfg.layer_types) == 42
    assert cfg.sliding_window == 512 and is_interleaved(cfg)
    assert cfg.padded_vocab_size >= cfg.vocab_size


def test_registry_and_parsers(registered):
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    for arch in inkling_hpu.ARCHITECTURES:
        entry = ModelRegistry.models.get(arch)
        assert getattr(entry, "module_name", None) == "inkling_hpu.model", arch
    assert "inkling_py" in ToolParserManager.list_registered()
    assert "inkling_py" in ReasoningParserManager.list_registered()


def test_kv_layout_is_one_attention_group():
    from types import SimpleNamespace as NS

    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs

    kv_heads, head_dim, hidden, bs = 8 // TP, 128, 4096, 128
    kv_dim = kv_heads * head_dim
    conv_bytes = 3 * (2 * kv_dim + 2 * hidden) * 2
    attn = FullAttentionSpec(block_size=bs, num_kv_heads=kv_heads, head_size=head_dim, dtype=torch.bfloat16)
    padded = FullAttentionSpec(
        block_size=bs,
        num_kv_heads=kv_heads,
        head_size=head_dim,
        dtype=torch.bfloat16,
        page_size_padded=attn.real_page_size_bytes + conv_bytes,
    )
    vllm_config = NS(
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        cache_config=NS(get_resolved_kv_cache_layout=lambda: NS(is_block_outermost=False)),
    )
    new = {f"model.layers.{i}.attn.attn": padded for i in range(42)}
    groups = kv_cache_utils.get_kv_cache_groups(vllm_config, new)
    assert len(groups) == 1 and isinstance(groups[0].kv_cache_spec, FullAttentionSpec)
    assert len(groups[0].layer_names) == 42
    assert groups[0].kv_cache_spec.page_size_bytes == attn.real_page_size_bytes + conv_bytes

    conv = MambaSpec(
        shapes=((3, kv_dim), (3, kv_dim), (3, hidden), (3, hidden)),
        dtypes=(torch.bfloat16,) * 4,
        block_size=bs,
        page_size_padded=attn.real_page_size_bytes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="none",
        num_speculative_blocks=0,
    )
    old = {}
    for i in range(42):
        old[f"model.layers.{i}.attn.attn"] = attn
        old[f"sconv_layers.{1000 + i}"] = conv
    assert UniformTypeKVCacheSpecs.is_uniform_type(new)
    assert not UniformTypeKVCacheSpecs.is_uniform_type(old)


@pytest.mark.skipif(os.environ.get("INKLING_PORT_IMAGE") != "1", reason="only in the port image")
def test_port_image_installed():
    from importlib.metadata import entry_points
    from pathlib import Path

    import vllm_gaudi

    names = {ep.name for ep in entry_points(group="vllm.general_plugins")}
    assert "inkling_hpu" in names
    runner = Path(vllm_gaudi.__file__).parent / "v1" / "worker" / "hpu_model_runner.py"
    assert runner.read_text().count("[inkling-small:decode-window]") == 2
