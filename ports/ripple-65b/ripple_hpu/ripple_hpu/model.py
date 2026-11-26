# SPDX-License-Identifier: Apache-2.0
"""Ricursive Ripple-65B as a vLLM model on Gaudi (vllm-gaudi v0.30, lazy mode).

Registered for architecture RicursiveForCausalLM by ripple_hpu.register(). A backport of
upstream vLLM's vllm/model_executor/models/ricursive.py (PR #65745, merge 03bd3c6) with the
vllm-gaudi idioms of beam_hpu / muse_spark_hpu (same attention family):

  h = E[id] * embed_scale                      bf16; value residual vr = h (use_valres)
  per layer (PeriNorm, 4 RMSNorms):
    a = rms(h) * w_in
    q, k, g = qkg_proj(a); v = v_proj(lerp(vr, a, sigmoid(lambda)))
    q, k = rms_head(q) * wq, rms_head(k) * wk
    sliding layers (window 1024): q, k = rope(pos, q, k)    NeoX, theta 1e4
    full layers: NoPE
    o = attn(q, k, v) * sigmoid(float(g))      scale head_dim^-0.5
    h = h + rms(o_proj(o)) * w_post_attn
    m = down(silu(clamp(gate, max=c)) * clamp(up, -c, c))   c = ffn_clamp_threshold
    h = h + rms(m) * w_post_ffn
  logits = lm_head(rms(h) * w_final)

vllm-gaudi specifics (shared with the beam-1-0 and muse-spark-open ports):
  * Everything is 2-D [T, ...]. ricursive is not in vllm-gaudi's flatten_input list, so the
    model flattens at entry, HPUAttentionImpl recovers [B, S] from its metadata, and the output
    goes back to the input's leading shape (the runner indexes prompt rows by batch row).
  * ONE rotary module, registered on the model before the layers: the runner refreshes cos/sin
    per step only for the first *RotaryEmbedding module it finds. Its table is sized for
    max_model_len, not the config's 2M positions.
  * cache_config.sliding_window is cleared, so a full layer's per_layer_sliding_window=None
    cannot fall back to the model's window; every layer's window is checked after construction.
  * Exact sliding windows need patches/runner_window.py (model type "ricursive"); window-sized KV
    for the sliding layers is RIPPLE_SWA_KV=1 with patches/swa_kv_hooks.py (swa_kv.py).
  * The decoder layer class name ends in "DecoderLayer": the runner's per-layer mark_step hook.
  * runner_hooks.install(): FusedSDPA query tiles, fast softmax for long full-layer prompt
    attention, and the decode graph memory budget (graph_budget.py).

Env:
  RIPPLE_PREFILL_INFLIGHT=K  (default 1; 0 = off) in a forward of at least
                             RIPPLE_PREFILL_INFLIGHT_MIN_T tokens (default 256) the host waits
                             for the HPU event K layers back before enqueuing more, so in-flight
                             lazy graphs cannot pile up activations past free HBM.
  RIPPLE_FP8_VIEW_FIX=0      block-FP8 checkpoints only: restore vllm-gaudi's linear output view
                             (see _install_fp8_view_fix).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from . import runner_hooks, swa_kv

logger = logging.getLogger("vllm.ripple_hpu")  # under "vllm": its handlers print

PREFILL_INFLIGHT = int(os.environ.get("RIPPLE_PREFILL_INFLIGHT", "1"))
PREFILL_INFLIGHT_MIN_T = int(os.environ.get("RIPPLE_PREFILL_INFLIGHT_MIN_T", "256"))


_FP8_VIEW_FIX = {"installed": False}


def _install_fp8_view_fix() -> None:
    """Block FP8 with VLLM_HPU_FORCE_CHANNEL_FP8 (vllm-gaudi's default): apply_block_fp8_linear_hpu returns
    `out.to(input.dtype).view(*input.shape[:-1], -1)`. For Ripple's 2-D bf16 inputs both are no-ops, but the
    view, all-reduced in place after o_proj and down_proj, splits the decode HPU graph around every collective:
    7 recipes per layer instead of BF16's 3, each with a host event sync: at TP 4 the decode step took 50 ms
    against BF16's 33 ms while the device was busy 12.7 ms against 18.4 ms. For a 2-D input in the output dtype,
    return the GEMM output itself. RIPPLE_FP8_VIEW_FIX=0 restores vllm-gaudi's path."""
    if _FP8_VIEW_FIX["installed"] or os.environ.get("RIPPLE_FP8_VIEW_FIX", "1") == "0":
        return
    from vllm_gaudi.extension import ops as hpu_ops

    orig = hpu_ops.apply_block_fp8_linear_hpu

    def apply_block_fp8_linear_hpu(input, layer, block_size, bias=None, do_unpad=False, force_channel_fp8=False):
        if force_channel_fp8 and input.dim() == 2:
            # bias by keyword: vllm-gaudi passes it 4th, into apply_fp8_linear_hpu's input_scale (Ripple has none)
            out = hpu_ops.apply_fp8_linear_hpu(input, layer.weight, layer.weight_scale_inv, bias=bias)
            return out if out.dtype == input.dtype else out.to(input.dtype)
        return orig(input, layer, block_size, bias=bias, do_unpad=do_unpad, force_channel_fp8=force_channel_fp8)

    hpu_ops.apply_block_fp8_linear_hpu = apply_block_fp8_linear_hpu
    _FP8_VIEW_FIX["installed"] = True
    logger.warning(
        "ripple_hpu: block FP8 linears return the GEMM output for 2-D inputs (no view before the "
        "all-reduce; RIPPLE_FP8_VIEW_FIX=0 to restore vllm-gaudi's)"
    )


def _rope_theta(cfg) -> float:
    rp = getattr(cfg, "rope_parameters", None) or {}
    if rp.get("rope_type", "default") != "default":
        raise ValueError(f"ripple_hpu: rope_type {rp.get('rope_type')!r} is not supported (default only)")
    return float(rp.get("rope_theta", getattr(cfg, "rope_theta", 10000.0)))


class RippleMLP(nn.Module):
    """SwiGLU with Ripple's activation clamp: silu(min(gate, c)) * clamp(up, -c, c)."""

    def __init__(self, hidden: int, inter: int, clamp: float | None, quant_config, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden, [inter] * 2, bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(
            inter, hidden, bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj"
        )
        self.clamp = float(clamp) if clamp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x)[0].chunk(2, dim=-1)
        if self.clamp is not None:
            gate = gate.clamp(max=self.clamp)
            up = up.clamp(min=-self.clamp, max=self.clamp)
        return self.down_proj(F.silu(gate) * up)[0]


class RippleAttention(nn.Module):
    def __init__(self, cfg, layer_id: int, rope, cache_config, quant_config, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        H, nq, nkv, d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        if nq % tp or nkv % tp:
            raise ValueError(f"ripple_hpu: {nq} q heads and {nkv} kv heads must divide over tp={tp}")
        self.layer_id = layer_id
        self.nh, self.nkv, self.d = nq // tp, nkv // tp, d
        self.q_size, self.kv_size = self.nh * d, self.nkv * d
        self.use_gate = bool(cfg.use_gated_attention)
        self.gate_fp32 = bool(cfg.gate_full_precision)
        self.use_valres = bool(cfg.use_valres)
        sizes = [nq * d, nkv * d] + ([nq * d] if self.use_gate else [])
        # V reads a different input than Q/K/G when value residuals are on, so only Q, K and
        # the output gate share a projection (as upstream).
        self.qkg_proj = MergedColumnParallelLinear(
            H, sizes, bias=False, quant_config=quant_config, prefix=f"{prefix}.qkg_proj"
        )
        self.v_proj = ColumnParallelLinear(H, nkv * d, bias=False, quant_config=quant_config, prefix=f"{prefix}.v_proj")
        self.o_proj = RowParallelLinear(nq * d, H, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        if self.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.q_norm = RMSNorm(d, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(d, eps=cfg.rms_norm_eps)
        self.sliding = cfg.layer_types[layer_id] == "sliding_attention"
        if self.sliding and rope is None:
            raise ValueError(f"ripple_hpu: sliding layer {layer_id} needs the rotary module")
        self._rope = (rope if self.sliding else None,)  # a tuple: registered once, on the model
        self.window = int(cfg.sliding_window) if self.sliding else None
        self.attn = Attention(
            self.nh,
            d,
            d**-0.5,
            num_kv_heads=self.nkv,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=self.window,
            prefix=f"{prefix}.attn",
        )
        if self.attn.impl.sliding_window != self.window:
            raise RuntimeError(
                f"ripple_hpu: layer {layer_id} attention window {self.attn.impl.sliding_window}, expected {self.window}"
            )
        # vllm-gaudi's FusedSDPA slicing (VLLM_HPU_FSDPA_SLICE_ENABLED) assumes a causal + padding mask: in
        # the chunk's own region it masks only the diagonal slice. A sliding layer's window arrives as a
        # mask too (no window_size argument), so sliced it would see keys outside its window (NLL 0.65 ->
        # 2.95 past 4096 tokens). Slicing is never allowed on a sliding layer.
        fsdpa = getattr(self.attn.impl, "fused_scaled_dot_product_attention", None)
        self.attn.impl._ripple_full_attention = not self.sliding
        if fsdpa is not None:
            fsdpa._ripple_full_attention = not self.sliding
            if self.sliding and getattr(getattr(fsdpa, "_sliced_module", None), "enable_slicing", False):
                fsdpa._sliced_module.enable_slicing = False

    def forward(self, a: torch.Tensor, pos: torch.Tensor, vr: torch.Tensor | None) -> torch.Tensor:
        """a [T, H] (input-normed), pos [T], vr [T, H] value residual -> o_proj output [T, H]."""
        T = a.shape[0]
        qkg = self.qkg_proj(a)[0]
        q, k = qkg[:, : self.q_size], qkg[:, self.q_size : self.q_size + self.kv_size]
        v_in = a
        if self.use_valres:
            blend = torch.sigmoid(self.value_res_lambda).to(a.dtype)
            v_in = torch.lerp(vr, a, blend)
        v = self.v_proj(v_in)[0]
        q = self.q_norm(q.reshape(T, self.nh, self.d)).reshape(T, self.q_size)
        k = self.k_norm(k.reshape(T, self.nkv, self.d)).reshape(T, self.kv_size)
        if self.sliding:
            q, k = self._rope[0](pos, q, k)
        o = self.attn(q, k, v)
        if self.use_gate:
            g = qkg[:, self.q_size + self.kv_size :]
            if self.gate_fp32:
                g = g.float()
            o = o * torch.sigmoid(g).to(o.dtype)
        return self.o_proj(o)[0]


class RippleDecoderLayer(nn.Module):
    """PeriNorm block (input / post-attention / pre-FFN / post-FFN RMSNorms), residual fused
    into the next norm as upstream: returns (h, residual)."""

    def __init__(self, vllm_config: VllmConfig, layer_id: int, rope, prefix: str):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.layer_id = layer_id
        self.self_attn = RippleAttention(
            cfg, layer_id, rope, vllm_config.cache_config, vllm_config.quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = RippleMLP(
            H, cfg.intermediate_size, cfg.ffn_clamp_threshold, vllm_config.quant_config, prefix=f"{prefix}.mlp"
        )
        self.input_layernorm = RMSNorm(H, eps=eps)
        self.post_attention_layernorm = RMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(H, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(H, eps=eps)

    def forward(
        self, h: torch.Tensor, residual: torch.Tensor | None, pos: torch.Tensor, vr: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = h
            h = self.input_layernorm(h)
        else:
            h, residual = self.input_layernorm(h, residual)
        h = self.post_attention_layernorm(self.self_attn(h, pos, vr))
        h, residual = self.pre_feedforward_layernorm(h, residual)
        h = self.post_feedforward_layernorm(self.mlp(h))
        return h, residual


class RippleHPUModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        cfg = vllm_config.model_config.hf_config
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, prefix=f"{prefix}.embed_tokens")
        # the ONE rotary module, registered before the layers so the runner's search finds it
        self.rotary_emb = None
        if "sliding_attention" in cfg.layer_types:
            self.rotary_emb = get_rope(
                cfg.head_dim,
                max_position=vllm_config.model_config.max_model_len,
                is_neox_style=True,
                rope_parameters={"rope_type": "default", "rope_theta": _rope_theta(cfg)},
            )
        self.layers = nn.ModuleList(
            [
                RippleDecoderLayer(vllm_config, i, self.rotary_emb, prefix=f"{prefix}.layers.{i}")
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.embed_scale = cfg.embed_scale
        self.use_valres = bool(cfg.use_valres)
        self._kv_gid = None  # RIPPLE_SWA_KV=1: KV group id per layer (set by swa_kv.after_bind)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.embed_tokens(ids)
        return e * self.embed_scale if self.embed_scale is not None else e

    def forward(
        self, ids: torch.Tensor | None, pos: torch.Tensor, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.embed(ids) if inputs_embeds is None else inputs_embeds
        # cloned: the fused add-norm may update the residual in place
        vr = h.clone() if self.use_valres else None
        residual = None
        bound = PREFILL_INFLIGHT if (PREFILL_INFLIGHT > 0 and h.shape[0] >= PREFILL_INFLIGHT_MIN_T) else 0
        events = []
        # RIPPLE_SWA_KV=1 (swa_kv.py): each sliding layer reads and writes its own KV group's blocks,
        # through a metadata view with that group's slot mapping and window block list
        fc = views = None
        if swa_kv.active():
            from vllm.forward_context import get_forward_context

            fc = get_forward_context()
            views = swa_kv.layer_views(fc.attn_metadata)
            if views is not None:
                base_md = fc.attn_metadata
                if self._kv_gid is None:
                    self._kv_gid = [int(layer.self_attn.attn._ripple_kv_gid) for layer in self.layers]
        try:
            for i, layer in enumerate(self.layers):
                if views is not None:
                    fc.attn_metadata = views[self._kv_gid[i]]
                h, residual = layer(h, residual, pos, vr)  # the runner's per-layer hook marks the step
                if bound:
                    import habana_frameworks.torch as htorch

                    ev = htorch.hpu.Event()
                    ev.record()
                    events.append(ev)
                    if len(events) > bound:
                        events.pop(0).synchronize()
        finally:
            if views is not None:
                fc.attn_metadata = base_md
        h, _ = self.norm(h, residual)
        return h


class RippleHPUForCausalLM(nn.Module):
    """Ripple causal LM on HPU (text only; BF16, or block FP8 weights)."""

    packed_modules_mapping = {
        "qkg_proj": ["q_proj", "k_proj", "g_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        runner_hooks.install()
        mc = vllm_config.model_config
        cfg = mc.hf_config
        if cfg.tie_word_embeddings:
            raise NotImplementedError("ripple_hpu: tie_word_embeddings=True is not supported")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("ripple_hpu: pipeline parallelism is not supported")
        qc = vllm_config.quant_config
        if qc is not None and not (qc.get_name() == "fp8" and getattr(qc, "weight_block_size", None)):
            # block FP8 (128 x 128 blocks on q/k/g/v/o and the MLP; embedding, lm_head and norms BF16) is the only
            # quantized format checked so far
            raise NotImplementedError(
                f"ripple_hpu: quantization {qc.get_name()} is not supported (BF16 or block FP8 only)"
            )
        if qc is not None:
            _install_fp8_view_fix()
        if any(t not in ("sliding_attention", "full_attention") for t in cfg.layer_types):
            raise ValueError(f"ripple_hpu: unknown layer types {sorted(set(cfg.layer_types))}")
        if os.environ.get("PT_HPU_SDPA_QKV_SLICE_MODE_FWD"):
            raise RuntimeError(
                "ripple_hpu: PT_HPU_SDPA_QKV_SLICE_MODE_FWD must be unset (its window semantics are unverified)"
            )
        # Every layer passes its own per_layer_sliding_window (None on full layers); without this,
        # Attention.__init__ falls back to cache_config.sliding_window and the full layers get a
        # window. The runner reads the window from model_config (hf_config.sliding_window), which
        # stays as configured.
        if vllm_config.cache_config is not None:
            vllm_config.cache_config.sliding_window = None
        self.config = cfg
        self.vllm_config = vllm_config
        self.model = RippleHPUModel(vllm_config, prefix="model")
        self.model._rank = get_tensor_model_parallel_rank()
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(cfg.vocab_size)
        self._check_rope()
        n_full = sum(t == "full_attention" for t in cfg.layer_types)
        logger.warning(
            "ripple_hpu: model built; %d layers (%d sliding w%d RoPE, %d full NoPE), tp %d, "
            "embed_scale %s, valres %s, gate %s (fp32 %s), ffn clamp %s, max_model_len %d, "
            "prefill in-flight layers %s (from %d tokens), swa_kv requested %s",
            cfg.num_hidden_layers,
            cfg.num_hidden_layers - n_full,
            cfg.sliding_window,
            n_full,
            get_tensor_model_parallel_world_size(),
            cfg.embed_scale,
            cfg.use_valres,
            cfg.use_gated_attention,
            cfg.gate_full_precision,
            cfg.ffn_clamp_threshold,
            mc.max_model_len,
            PREFILL_INFLIGHT if PREFILL_INFLIGHT > 0 else "unbounded (DISABLED)",
            PREFILL_INFLIGHT_MIN_T,
            swa_kv.requested(),
        )

    def _check_rope(self) -> None:
        """Exactly one *RotaryEmbedding module (the runner refreshes cos/sin per step only for
        the first one it finds)."""
        n = len({id(m) for m in self.modules() if type(m).__name__.endswith("RotaryEmbedding")})
        want = 1 if self.model.rotary_emb is not None else 0
        if n != want:
            raise RuntimeError(f"ripple_hpu: {n} rotary modules, expected {want}")

    # ------------------------------------------------------------- forward
    def embed_input_ids(
        self, input_ids: torch.Tensor, multimodal_embeddings=None, *, is_multimodal=None
    ) -> torch.Tensor:
        return self.model.embed(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        H = self.config.hidden_size
        lead = tuple(input_ids.shape) if input_ids is not None else tuple(inputs_embeds.shape[:-1])
        ids = input_ids.reshape(-1) if input_ids is not None else None
        pos = positions.reshape(-1)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(-1, H)
        h = self.model(ids, pos, inputs_embeds)
        return h.reshape(*lead, H)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # ------------------------------------------------------------- load
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = load_tensors(params, weights)
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(
                f"ripple_hpu: parameters not loaded from the checkpoint: {missing[:20]}"
                f"{' ...' if len(missing) > 20 else ''} ({len(missing)})"
            )
        logger.warning("ripple_hpu: loaded %d parameters", len(loaded))
        return set(params)


# checkpoint name piece -> (param piece, shard id)
_STACKED = (
    (".self_attn.q_proj.", ".self_attn.qkg_proj.", 0),
    (".self_attn.k_proj.", ".self_attn.qkg_proj.", 1),
    (".self_attn.g_proj.", ".self_attn.qkg_proj.", 2),
    (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
    (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
)


def load_tensors(params: dict, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Route checkpoint tensors (HF names) into `params` (vLLM names). Raises on a tensor
    without a parameter (the reference SGLang loader refuses unused tensors too)."""
    loaded: set[str] = set()
    for name, w in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        for old, new, shard in _STACKED:
            if old in name:
                pname = name.replace(old, new)
                param = params[pname]
                param.weight_loader(param, w, shard)
                loaded.add(pname)
                break
        else:
            if name not in params:
                raise KeyError(f"ripple_hpu: checkpoint tensor {name} {tuple(w.shape)} has no parameter")
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, w)
            loaded.add(name)
    return loaded


__all__ = ["RippleHPUForCausalLM", "RippleHPUModel", "RippleDecoderLayer", "load_tensors"]
