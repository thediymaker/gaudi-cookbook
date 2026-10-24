# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the port's source patches (stdlib and pytest only).

Each patch runs against a temporary tree of minimal files that hold its anchors: apply, idempotent second run,
--check before and after, and a drifted anchor that must exit 2 and write nothing. With GAUDI_FIXES_ROOT pointing at
a directory that holds vllm/ and vllm_gaudi/ of the v0.26.0 release (the variable fixes/test_fixes.py uses), the
patches also run against a copy of the real sources.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PATCHES = Path(__file__).resolve().parents[1] / "patches"
NAMES = ["nope_mla", "mla_model_type", "cache_group_idx", "mhc_kernel_env"]

# Minimal stand-ins for the target files: the anchors each patch needs, in valid Python.
FAKE = {
    "vllm_gaudi/attention/backends/hpu_attn.py": (
        "def f(self, latent_vec_k):\n"
        "        k_c_normed, k_pe = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)\n"
        "        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)\n"
        "        return k_pe\n"
    ),
    "vllm/transformers_utils/model_arch_config_convertor.py": (
        'MLA = (\n            "deepseek_v3",\n            "glm_moe_dsa",\n)\n'
    ),
    "vllm_gaudi/v1/worker/hpu_model_runner.py": (
        "import torch\n\n\n"
        "def maybe_set_mamba_kv_cache_groups_ids(model, kv_cache_config):\n"
        "    mamba_like_arch = [\n"
        '        "GraniteMoeHybridForCausalLM",\n'
        '        "Qwen3NextForCausalLM"\n'
        "    ]\n"
        "    mamba_like_layer = ['.mixer', '.linear_attn']\n\n"
        "    def _get_decoder_layer_by_idx(model_obj, idx):\n"
        "        return None\n\n"
        "    for group_idx, kv_group in enumerate(kv_cache_config.kv_cache_groups):\n"
        "        for layer_name in kv_group.layer_names:\n"
        "            parts = layer_name.split('.')\n"
        '            layer_idx = int(parts[-2])  # "model.layers.5.mixer" -> 5\n'
        "            if '.mixer' in layer_name:\n"
        "                pass\n"
        "            elif 'linear_attn' in layer_name:\n"
        "                layer = _get_decoder_layer_by_idx(model, layer_idx)\n"
        '                if layer is not None and hasattr(layer, "linear_attn"):\n'
        '                    layer.linear_attn.cache_group_idx = torch.tensor(group_idx, dtype=torch.long, device="hpu")\n'  # noqa: E501
        "\n\n"
        "def other():\n"
        "    pass\n"
    ),
    "vllm_gaudi/__init__.py": "import os\n\n\ndef register():\n    return None\n",
}


def run(name, root, *args):
    return subprocess.run(
        [sys.executable, str(PATCHES / f"{name}.py"), "--root", str(root), *args], capture_output=True, text=True
    )


@pytest.fixture
def fake_root(tmp_path):
    for rel, text in FAKE.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


@pytest.mark.parametrize("name", NAMES)
def test_apply_idempotent_check(name, fake_root):
    assert run(name, fake_root, "--check").returncode == 1
    before = {rel: (fake_root / rel).read_text() for rel in FAKE}
    r = run(name, fake_root)
    assert r.returncode == 0, r.stderr
    changed = [rel for rel in FAKE if (fake_root / rel).read_text() != before[rel]]
    assert len(changed) == 1
    text = (fake_root / changed[0]).read_text()
    assert "[gaudi-port:glm-5-3-flash/" in text
    compile(text, changed[0], "exec")
    r2 = run(name, fake_root)
    assert r2.returncode == 0 and "applied" in r2.stdout
    assert (fake_root / changed[0]).read_text() == text
    assert run(name, fake_root, "--check").returncode == 0


@pytest.mark.parametrize("name", NAMES)
def test_drift_writes_nothing(name, fake_root):
    # break every anchor: each target loses the text its patch keys on
    for rel in FAKE:
        p = fake_root / rel
        p.write_text(
            p.read_text()
            .replace("k_pe.view", "k_pe.viewx")
            .replace('"glm_moe_dsa"', '"glm_moe_dsax"')
            .replace("mamba_like_layer =", "mamba_layers =")
            .replace("def register():", "def register(x):")
        )
    before = {rel: (fake_root / rel).read_text() for rel in FAKE}
    r = run(name, fake_root)
    assert r.returncode == 2
    assert all((fake_root / rel).read_text() == before[rel] for rel in FAKE)


def test_missing_target(tmp_path):
    assert run("nope_mla", tmp_path).returncode == 2


def test_cache_group_idx_logic(fake_root):
    """The patched function assigns the index to a KDA `self_attn` and skips MLA names."""
    assert run("cache_group_idx", fake_root).returncode == 0
    src = (fake_root / "vllm_gaudi/v1/worker/hpu_model_runner.py").read_text()
    assert '"Glm5NextForConditionalGeneration"' in src
    assert "'.self_attn'" in src and "parts[-2].isdigit()" in src


REAL = os.environ.get("GAUDI_FIXES_ROOT")


@pytest.mark.skipif(not REAL, reason="set GAUDI_FIXES_ROOT to a release's site-packages to run on real sources")
def test_real_sources(tmp_path):
    for pkg in ("vllm", "vllm_gaudi"):
        shutil.copytree(Path(REAL) / pkg, tmp_path / pkg, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    for name in NAMES:
        assert run(name, tmp_path, "--check").returncode == 1, name
        r = run(name, tmp_path)
        assert r.returncode == 0, r.stderr
        assert run(name, tmp_path, "--check").returncode == 0, name
        assert run(name, tmp_path).returncode == 0, name
