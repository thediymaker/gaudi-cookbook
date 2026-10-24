# SPDX-License-Identifier: Apache-2.0
"""The vendored config reproduces the checkpoint's config.json (needs transformers; skips without it).

Without a checkpoint the defaults are compared with the published values below. With GLM5_NEXT_CHECKPOINT pointing at
a local snapshot of zai-org/GLM-5.3-Flash, every key of its config.json is checked after parsing.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

_CFG = Path(__file__).resolve().parents[1] / "glm5_next_hpu" / "glm5_next_hpu" / "config.py"
_spec = importlib.util.spec_from_file_location("glm5_next_config", _CFG)
cfg = importlib.util.module_from_spec(_spec)
sys.modules["glm5_next_config"] = cfg
_spec.loader.exec_module(cfg)

# Published values (config.json of zai-org/GLM-5.3-Flash) the model code depends on.
PUBLISHED_TEXT = {
    "vocab_size": 154880,
    "hidden_size": 4096,
    "num_hidden_layers": 45,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "swiglu_limit": 10.0,
    "max_position_embeddings": 1048576,
}


def test_defaults_match_published():
    t = cfg.Glm5NextTextConfig()
    for key, value in PUBLISHED_TEXT.items():
        assert getattr(t, key) == value, key
    assert (t.swiglu_alpha, t.swiglu_beta) == (1.0, 0.0)


def test_linear_attention_aliases():
    lac = {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0}
    t = cfg.Glm5NextTextConfig(linear_attn_config=lac)
    assert (t.linear_num_heads, t.linear_head_dim, t.linear_conv_kernel_dim, t.linear_lower_bound) == (
        64,
        128,
        4,
        -5.0,
    )


def test_top_level_mirrors_text():
    top = cfg.Glm5NextConfig(text_config={"hidden_size": 4096, "num_hidden_layers": 45})
    assert top.hidden_size == 4096 and top.num_hidden_layers == 45
    assert top.vision_config.depth == 24


CKPT = os.environ.get("GLM5_NEXT_CHECKPOINT")


@pytest.mark.skipif(not CKPT, reason="set GLM5_NEXT_CHECKPOINT to a local snapshot of zai-org/GLM-5.3-Flash")
def test_checkpoint_config_round_trip():
    raw = json.loads((Path(CKPT) / "config.json").read_text())
    top = cfg.Glm5NextConfig(**{k: v for k, v in raw.items() if k != "model_type"})
    for section, obj in (("text_config", top.text_config), ("vision_config", top.vision_config)):
        for key, value in raw[section].items():
            got = getattr(obj, key)
            if key == "dtype":  # transformers turns the string into a torch dtype
                got = str(got).removeprefix("torch.")
            assert got == value, f"{section}.{key}"
    for key in ("image_token_id", "video_token_id", "image_start_token_id", "image_end_token_id"):
        assert getattr(top, key) == raw[key], key
