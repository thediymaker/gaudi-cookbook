# SPDX-License-Identifier: Apache-2.0
"""The MiMo-V2.6-Pro kernel switch (mimo_v2_hpu/mxk.py) and the MXFP4 expert layout, on the CPU."""

import importlib
import os
import re
import sys
from pathlib import Path

import pytest

PORT = Path(__file__).resolve().parents[1]
REPO = PORT.parents[1]
sys.path.insert(0, str(PORT / "mimo_v2_hpu"))
sys.path.insert(0, str(REPO / "kernels" / "mxfp4-moe" / "python"))

# MiMo-V2.6-Pro (config.json of XiaomiMiMo/MiMo-V2.6-Pro-RL) at TP8, and Flash for contrast
PRO = dict(n_routed_experts=384, hidden_size=6144, moe_intermediate_size=2048, num_experts_per_tok=8, tp=8)
FLASH = dict(n_routed_experts=256, hidden_size=4096, moe_intermediate_size=2048, num_experts_per_tok=8, tp=8)


def fresh_mxk(monkeypatch, **env):
    for k in ("MIMO_MXK", "MIMO_MXK_LIB", "MIMO_MXK_MAX_TOKENS", "GC_KERNEL_PATH", "MXK_TPC_FWD_LIB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("PT_HPU_RECIPE_CACHE_CONFIG", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import mimo_v2_hpu.mxk as m

    return importlib.reload(m)


def fake_lib(tmp_path, version="glm-v8-0123456789ab", pt2=True):
    d = tmp_path / "mxk"
    d.mkdir()
    (d / "libmxk_tpc.so").write_bytes(b"\x7fELF...mxk_tpc " + version.encode() + b"\0...")
    if pt2:
        (d / "libmxk_op_pt2.so").write_bytes(b"\x7fELF")
    return d


def test_pro_fits_the_glm_variant(monkeypatch):
    m = fresh_mxk(monkeypatch)
    n_local = PRO["n_routed_experts"] // PRO["tp"]
    assert n_local == 48
    assert m.layer_reason(n_local, PRO["hidden_size"], PRO["moe_intermediate_size"], "mxfp4") is None
    # decode at --max-num-seqs 8: 8 tokens x top-8 = 64 routes; the cap of 16 tokens is exactly 128 routes
    assert m.covers(8, PRO["num_experts_per_tok"]) and m.covers(16, 8)
    assert not m.covers(17, 8) and not m.covers(0, 8) and not m.covers(16, 9)


def test_layer_reason_refuses_what_the_kernels_cannot_run(monkeypatch):
    m = fresh_mxk(monkeypatch)
    assert "fp8" in m.layer_reason(32, FLASH["hidden_size"], FLASH["moe_intermediate_size"], "fp8")
    assert "local experts" in m.layer_reason(129, 6144, 2048, "mxfp4")
    assert "hidden" in m.layer_reason(48, 7168, 2048, "mxfp4")  # longer rows need mxk_w13xt (kimi variant)
    assert "hidden" in m.layer_reason(48, 6000, 2048, "mxfp4")
    assert "intermediate" in m.layer_reason(48, 6144, 3584, "mxfp4")
    assert "intermediate" in m.layer_reason(48, 6144, 2304, "mxfp4")


def test_caps_match_the_kernel_doc():
    """The caps in mxk.py are the glm row of the "Supported shapes" table in docs/kernels/mxfp4-moe.md."""
    import mimo_v2_hpu.mxk as m

    doc = (REPO / "docs" / "kernels" / "mxfp4-moe.md").read_text()
    row = next(ln for ln in doc.splitlines() if ln.strip().startswith("| `glm`, `muse` |"))
    assert f"at most {m.MAX_K}" in row and f"at most {m.MAX_I}" in row and "at most 16" in row
    assert "at most 128 routes per call and at most 128 local experts" in doc
    assert m.MAX_ROUTES == 128 and m.MAX_LOCAL_EXPERTS == 128


def test_off_by_default(monkeypatch, tmp_path):
    lib = fake_lib(tmp_path)
    m = fresh_mxk(monkeypatch, MIMO_MXK_LIB=str(lib))
    m.setup_env()
    assert "GC_KERNEL_PATH" not in os.environ and not m.ENABLED and not m.load()


def test_setup_env_points_the_compiler_at_the_build(monkeypatch, tmp_path):
    lib = fake_lib(tmp_path)
    cache = tmp_path / "cache"
    m = fresh_mxk(
        monkeypatch,
        MIMO_MXK="1",
        MIMO_MXK_LIB=str(lib),
        GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so",
        PT_HPU_RECIPE_CACHE_CONFIG=f"{cache},False,65536",
    )
    m.setup_env()
    assert os.environ["GC_KERNEL_PATH"] == str(lib / "libmxk_tpc.so")
    assert os.environ["MXK_TPC_FWD_LIB"] == "/usr/lib/habanalabs/libtpc_kernels.so"
    want = f"{cache}_mxk-glm-v8-0123456789ab/,False,65536"
    assert os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] == want
    assert os.path.isdir(want.split(",")[0])
    m.setup_env()  # idempotent
    assert os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] == want and m.env_ok()


@pytest.mark.parametrize(
    "version, pt2, why",
    [("kimi-v8-0123456789ab", True, "other variant"), ("glm-v8-0123456789ab", False, "no PT2 op library")],
)
def test_setup_env_refuses_a_wrong_build(monkeypatch, tmp_path, version, pt2, why):
    lib = fake_lib(tmp_path, version, pt2)
    m = fresh_mxk(monkeypatch, MIMO_MXK="1", MIMO_MXK_LIB=str(lib))
    m.setup_env()
    assert not m.ENABLED, why
    assert "GC_KERNEL_PATH" not in os.environ


def test_load_needs_compile_mode(monkeypatch, tmp_path):
    lib = fake_lib(tmp_path)
    m = fresh_mxk(monkeypatch, MIMO_MXK="1", MIMO_MXK_LIB=str(lib), PT_HPU_LAZY_MODE="1")
    m.setup_env()
    assert m.ENABLED and not m.load()  # lazy mode keeps the stock op (the PT2 ops do not run in lazy graphs)


def test_permute_is_the_kernel_layout():
    torch = pytest.importorskip("torch")
    import mimo_v2_hpu.mxk as m
    import mxk_harness

    x = torch.arange(2 * 6144, dtype=torch.float32).reshape(2, 6144)
    xp = m.permute(x)
    assert torch.equal(xp, mxk_harness.permute_x(x))
    # README "Op reference": xp[kc + 128 p + i] = x[kc + 4 i + p] within each 512-value chunk
    kc, p, i = 512, 3, 17
    assert xp[1, kc + 128 * p + i] == x[1, kc + 4 * i + p]


def test_expert_slots():
    pytest.importorskip("torch")
    from mimo_v2_hpu.ops import mxfp4_expert_slot

    assert mxfp4_expert_slot("gate_proj", "weight", 2048) == ("w13_weight", slice(0, 2048))
    assert mxfp4_expert_slot("up_proj", "weight_scale", 2048) == ("w13_weight_scale", slice(2048, 4096))
    assert mxfp4_expert_slot("down_proj", "weight", 2048) == ("w2_weight", None)
    with pytest.raises(KeyError):
        mxfp4_expert_slot("w1", "weight", 2048)
    with pytest.raises(KeyError):
        mxfp4_expert_slot("gate_proj", "bias", 2048)


def test_pro_recipe_turns_the_mxfp4_path_on():
    """The Pro recipe serves MXFP4 experts with the kernels on, and the Flash recipe keeps its FP8 default."""
    recipes = REPO / "recipes" / "XiaomiMiMo"
    pro = (recipes / "MiMo-V2.6-Pro-RL" / "recipe.yaml").read_text()
    flash = (recipes / "MiMo-V2.6-Flash-RL" / "recipe.yaml").read_text()

    def env(text, key):
        hit = re.search(rf'^    {key}: "([^"]*)"', text, re.M)
        return hit.group(1) if hit else None

    assert env(pro, "MIMO_EXPERT_DTYPE") == "mxfp4" and env(pro, "MIMO_MXK") == "1"
    assert env(pro, "PT_HPU_LAZY_MODE") == "0"
    assert re.search(r"^  port: mimo-v2-6-flash$", pro, re.M)
    assert env(flash, "MIMO_EXPERT_DTYPE") in (None, "fp8") and env(flash, "MIMO_MXK") in (None, "0")
    install = (PORT / "install.sh").read_text()
    assert 'build.sh" --variant glm --pt2' in install
