"""Model config and per-layer architecture table (PR models/mistral/common/layer_spec.py).

`RefConfig.from_hf(path)` reads an HF `config.json` (`text_config`), which the tiny model, the AMD
checkpoint and syn8l all ship. `RefConfig.from_params(path)` reads a Mistral-native `params.json`
(the FP8/AMD checkpoints and syn8l). Both give the same tables for the real model.
"""

from __future__ import annotations

import dataclasses
import json
import os


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    idx: int
    use_rope: bool  # False for NoPE layers (i % 4 == 3 in the real model)
    kv_source: int  # layer whose latent cache this layer reads (itself if it owns one)
    has_indexer: bool  # "full" indexer layer
    index_source: int  # layer whose top-k this layer uses (itself if full)
    indexer_use_rope: bool  # match_nope: same as use_rope
    is_moe: bool

    @property
    def shares_kv(self) -> bool:
        return self.kv_source != self.idx


@dataclasses.dataclass
class RefConfig:
    vocab_size: int = 204800
    hidden_size: int = 6144
    num_layers: int = 72
    num_heads: int = 64
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    intermediate_size: int = 24576
    moe_intermediate_size: int = 6144
    n_routed_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    rms_norm_eps: float = 1e-5
    indexer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_factor: float = 2048.0
    rope_original_max: int = 512
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    softcap: float = 30.0
    nope_layers: tuple = ()  # bool per layer
    kv_source_layers: tuple = ()  # int per layer
    indexer_types: tuple = ()  # "full" | "shared" per layer
    fp8_indexer: bool = True

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_hf(cls, path: str) -> "RefConfig":
        p = os.path.join(path, "config.json") if os.path.isdir(path) else path
        c = json.load(open(p))
        t = c.get("text_config", c)
        n = t["num_hidden_layers"]
        rp = t.get("rope_parameters") or {}
        nrl = t.get("no_rope_layers") or [int((i + 1) % t.get("no_rope_layer_interval", 4) != 0) for i in range(n)]
        return cls(
            vocab_size=t["vocab_size"],
            hidden_size=t["hidden_size"],
            num_layers=n,
            num_heads=t["num_attention_heads"],
            kv_lora_rank=t["kv_lora_rank"],
            qk_nope_head_dim=t["qk_nope_head_dim"],
            qk_rope_head_dim=t["qk_rope_head_dim"],
            v_head_dim=t["v_head_dim"],
            index_n_heads=t["index_n_heads"],
            index_head_dim=t["index_head_dim"],
            index_topk=t["index_topk"],
            intermediate_size=t["intermediate_size"],
            moe_intermediate_size=t["moe_intermediate_size"],
            n_routed_experts=t["n_routed_experts"],
            num_experts_per_tok=t["num_experts_per_tok"],
            n_shared_experts=t.get("n_shared_experts", 1),
            routed_scaling_factor=t.get("routed_scaling_factor", 1.0),
            first_k_dense_replace=t.get("first_k_dense_replace", 1),
            rms_norm_eps=t.get("rms_norm_eps", 1e-5),
            rope_theta=rp.get("rope_theta", 10000.0),
            rope_factor=float(rp.get("factor", 2048)),
            rope_original_max=rp.get("original_max_position_embeddings", 512),
            rope_beta_fast=float(rp.get("beta_fast", 32)),
            rope_beta_slow=float(rp.get("beta_slow", 1)),
            softcap=t.get("final_logit_softcapping", 30.0),
            nope_layers=tuple(not bool(v) for v in nrl[:n]),
            kv_source_layers=tuple(t.get("kv_source_layers") or range(n)),
            indexer_types=tuple(t["indexer_types"]),
        )

    @classmethod
    def from_params(cls, path: str) -> "RefConfig":
        p = os.path.join(path, "params.json") if os.path.isdir(path) else path
        c = json.load(open(p))
        n = c["n_layers"]
        pat = c["nope"].split(",")
        dsa, moe, yarn = c["dsa"], c["moe"], c["yarn"]
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["dim"],
            num_layers=n,
            num_heads=c["n_heads"],
            kv_lora_rank=c["kv_lora_rank"],
            qk_nope_head_dim=c["qk_nope_head_dim"],
            qk_rope_head_dim=c["qk_rope_head_dim"],
            v_head_dim=c["v_head_dim"],
            index_n_heads=dsa["n_heads"],
            index_head_dim=dsa["head_dim"],
            index_topk=dsa["topk"],
            intermediate_size=c["hidden_dim"],
            moe_intermediate_size=moe["expert_hidden_dim"],
            n_routed_experts=moe["num_experts"],
            num_experts_per_tok=moe["num_experts_per_tok"],
            n_shared_experts=moe["num_shared_experts"],
            routed_scaling_factor=moe["routed_scale"],
            first_k_dense_replace=moe["first_k_dense_replace"],
            rms_norm_eps=c["norm_eps"],
            indexer_norm_eps=dsa.get("norm_eps", 1e-6),
            rope_theta=c["rope_theta"],
            rope_factor=float(yarn["factor"]),
            rope_original_max=yarn["original_max_position_embeddings"],
            rope_beta_fast=float(yarn["beta"]),
            rope_beta_slow=float(yarn["alpha"]),
            softcap=c["softmax_tanh_gating"],
            nope_layers=tuple(pat[i % len(pat)] == "nope" for i in range(n)),
            kv_source_layers=tuple(int(x) for x in c["kv_sharing"]["source_layers_str"].split(",")),
            indexer_types=tuple(dsa["sharing"].split(",")),
            fp8_indexer=dsa.get("fp8_indexer", True),
        )

    @classmethod
    def load(cls, path: str) -> "RefConfig":
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return cls.from_hf(path)
        return cls.from_params(path)


def get_layer_specs(cfg: RefConfig) -> list[LayerSpec]:
    specs, last_full = [], None
    for i in range(cfg.num_layers):
        full = cfg.indexer_types[i] == "full"
        if full:
            last_full = i
        assert last_full is not None, "layer 0 must have a full indexer"
        src = cfg.kv_source_layers[i]
        assert src <= i and cfg.kv_source_layers[src] == src
        assert cfg.nope_layers[src] == cfg.nope_layers[i], "KV source and reader must share the NoPE flag"
        specs.append(
            LayerSpec(
                idx=i,
                use_rope=not cfg.nope_layers[i],
                kv_source=src,
                has_indexer=full,
                index_source=i if full else last_full,
                indexer_use_rope=not cfg.nope_layers[i],
                is_moe=bool(cfg.n_routed_experts) and i >= cfg.first_k_dense_replace,
            )
        )
    return specs
