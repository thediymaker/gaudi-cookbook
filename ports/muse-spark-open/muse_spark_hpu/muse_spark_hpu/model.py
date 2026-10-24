# SPDX-License-Identifier: Apache-2.0
"""Muse Spark Open text model for vLLM on Gaudi (vllm-gaudi v0.30.0, lazy mode).

Forward, all tensors 2-D [T, ...], residual stream in fp32:

    x = rms(E[id], 1e-5)                                weightless embedding norm, once
    per layer:
      a = rms(x) * (w_in + 1)         -> attention (attention.py) -> y [T, H] bf16
      x = alpha_a * x + beta_a * rms(y, 1e-8)           gated residual
      f = rms(x) * (w_preffn + 1)     -> latent MoE (moe.py)       -> m [T, H] bf16
      x = alpha_f * x + beta_f * rms(m * (w_postffn + 1), 1e-8)
    h = rms(x) * w_final                                plain final gain (centre 0), bf16
    logits = 20 * tanh(lm_head(h) * 0.1767767 / 20)     fp32 softcap in compute_logits

Gains are built once after loading as w.float() + 1 and the residual-gate coefficients on the CPU in fp32. The
checkpoint is read by loader.py with ranged per-rank reads. The model returns h with the input's leading shape
([B, S, H] for vllm-gaudi's padded batches). Text only: serve with --language-model-only.

Prefill run-ahead bound. In lazy mode the host launches a layer's prefill graph in about 7 ms while the device needs
20 to 80 ms, so without a bound the host queues all 62 layers of a chunk, each holding its intermediates until it
runs, and HBM fills while decodes share the step. In forwards of at least PREFILL_INFLIGHT_MIN_T tokens an HPU
event is recorded after each layer's launch and the host waits for the event of the previous layer. Device work is
unchanged (the device is the bottleneck in prefill); decode graphs are untouched.

MUSE_SPARK_DECODE_LAYER_GROUP=G (default 1, off; the recipe uses 62). At one stream decode is bound by host graph
launches: the step replays about 5 cached recipes per layer at 25 to 170 us of host time each, while the device is
busy for half the step. Each layer's two all-reduces cut its graph anyway, so the runner's per-layer mark_step hook
only adds a third, separate tail recipe (post-expert norm, sum, post_expert_proj, residual). In a pure decode
forward ([B, 1] inputs, the whole-model HPU graph capture) the layers run through forward() instead of __call__,
which skips that hook, and the model marks a step after every G layers. G = 62 drops 62 of the 313 launches per
step. Same ops in the same order. Prompts and mixed steps keep the hook (the run-ahead bound relies on it).
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.muse_spark.common.omni import MuseSpark1xOmniMixin, register_omni_processor

from . import runner_hooks, swa_kv
from .attention import MuseSparkHPUAttention, make_rope
from .loader import Layout, checkpoint_dir, load_into, specs_for
from .moe import MuseSparkMoEHPU
from .numerics import gain, gate_coeffs, rms, softcap

logger = logging.getLogger("vllm.muse_spark_hpu")

PREFILL_INFLIGHT_MIN_T = 256
DECODE_LAYER_GROUP = max(1, int(os.environ.get("MUSE_SPARK_DECODE_LAYER_GROUP", "1")))
_TORCH = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}


class _NormWeight(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, dtype=torch.float32), requires_grad=False)


class _ResidualGate(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(n, dtype=torch.float32), requires_grad=False)


class MuseSparkHPUDecoderLayer(nn.Module):
    def __init__(self, tcfg, layer_id: int, kind: str, layout, rope, cache_config, quant_config, prefix: str):
        super().__init__()
        H = tcfg.hidden_size
        self.eps, self.post_eps, self.temperature = (
            tcfg.rms_norm_eps,
            tcfg.post_norm_eps,
            tcfg.residual_gate_temperature,
        )
        self.input_layernorm = _NormWeight(H)
        self.self_attn = MuseSparkHPUAttention(tcfg, layer_id, rope, cache_config, quant_config, f"{prefix}.self_attn")
        self.post_attention_residual_gate = _ResidualGate(H)
        self.pre_feedforward_layernorm = _NormWeight(H)
        self.mlp = MuseSparkMoEHPU(tcfg, layer_id, kind, layout, quant_config, prefix=f"{prefix}.mlp")
        self.post_feedforward_layernorm = _NormWeight(H)
        self.post_feedforward_residual_gate = _ResidualGate(H)

    def finalize(self) -> None:
        """Gains and gate coefficients from the raw loaded values, on the CPU in fp32."""
        dev = self.input_layernorm.weight.device

        def cpu(p):
            return p.data.cpu()

        self.g_in = gain(cpu(self.input_layernorm.weight), 1.0).to(dev)
        self.g_preffn = gain(cpu(self.pre_feedforward_layernorm.weight), 1.0).to(dev)
        self.g_postffn = gain(cpu(self.post_feedforward_layernorm.weight), 1.0).to(dev)
        a, b = gate_coeffs(cpu(self.post_attention_residual_gate.gate), self.temperature)
        self.alpha_a, self.beta_a = a.to(dev), b.to(dev)
        a, b = gate_coeffs(cpu(self.post_feedforward_residual_gate.gate), self.temperature)
        self.alpha_f, self.beta_f = a.to(dev), b.to(dev)
        self.mlp.finalize()

    def forward(self, x: torch.Tensor, pos: torch.Tensor, decode: bool = False) -> torch.Tensor:
        """x [T, H] fp32 residual stream, pos [T] -> x."""
        assert x.dim() == 2 and pos.dim() == 1, (tuple(x.shape), tuple(pos.shape))
        bf16 = torch.bfloat16
        y = self.self_attn((rms(x, self.eps) * self.g_in).to(bf16), pos)
        x = self.alpha_a * x + self.beta_a * rms(y, self.post_eps)
        m = self.mlp((rms(x, self.eps) * self.g_preffn).to(bf16), decode)
        return self.alpha_f * x + self.beta_f * rms(m.float() * self.g_postffn, self.post_eps)


class MuseSparkHPUTextModel(nn.Module):
    def __init__(self, tcfg, kinds, layout, cache_config, quant_config, max_position: int, prefix: str = "model"):
        super().__init__()
        self.eps = tcfg.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(
            tcfg.vocab_size, tcfg.hidden_size, params_dtype=torch.bfloat16, prefix=f"{prefix}.embed_tokens"
        )
        # the one rotary module, registered before the layers: the runner refreshes cos/sin per step only for the
        # first *RotaryEmbedding it finds
        self.rotary_emb = make_rope(tcfg, max_position)
        self.layers = nn.ModuleList(
            [
                MuseSparkHPUDecoderLayer(
                    tcfg,
                    i,
                    kinds[i],
                    layout,
                    self.rotary_emb,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(tcfg.num_hidden_layers)
            ]
        )
        self.norm = _NormWeight(tcfg.hidden_size)
        self._kv_gid = None  # MUSE_SPARK_SWA_KV=1: KV group id per layer (bound by swa_kv.after_bind)

    def finalize(self) -> None:
        self.g_final = gain(self.norm.weight.data.cpu(), 0.0).to(self.norm.weight.device)
        for layer in self.layers:
            layer.finalize()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """x0 = weightless RMS of the embedding rows, fp32 (the only embedding norm)."""
        return rms(self.embed_tokens(ids), self.eps)

    def forward(
        self, ids: torch.Tensor, pos: torch.Tensor, inputs_embeds: torch.Tensor | None = None, decode: bool = False
    ) -> torch.Tensor:
        x = self.embed(ids) if inputs_embeds is None else inputs_embeds.float()
        bound = x.shape[0] >= PREFILL_INFLIGHT_MIN_T
        events = []
        # MUSE_SPARK_SWA_KV=1: each sliding layer reads and writes its own KV group's blocks, through a metadata
        # view with that group's slot mapping and window block list
        fc = views = None
        if swa_kv.active():
            from vllm.forward_context import get_forward_context

            fc = get_forward_context()
            views = swa_kv.layer_views(fc.attn_metadata)
            if views is not None:
                base_md = fc.attn_metadata
                if self._kv_gid is None:
                    self._kv_gid = [int(layer.self_attn.attn._muse_kv_gid) for layer in self.layers]
        group = DECODE_LAYER_GROUP if decode else 1
        try:
            for i, layer in enumerate(self.layers):
                if views is not None:
                    fc.attn_metadata = views[self._kv_gid[i]]
                if group > 1:
                    # forward() skips the runner's per-layer mark_step hook; mark once per group of layers
                    x = layer.forward(x, pos, decode)
                    if (i + 1) % group == 0 or i + 1 == len(self.layers):
                        import habana_frameworks.torch as htorch

                        htorch.core.mark_step()
                else:
                    x = layer(x, pos, decode)  # the runner's per-layer hook launches this layer's graph
                if bound:
                    import habana_frameworks.torch as htorch

                    ev = htorch.hpu.Event()
                    ev.record()
                    events.append(ev)
                    if len(events) > 1:
                        events.pop(0).synchronize()
        finally:
            if views is not None:
                fc.attn_metadata = base_md
        return (rms(x, self.eps) * self.g_final).to(torch.bfloat16)


@register_omni_processor
class MuseSparkHPUForConditionalGeneration(nn.Module, SupportsMultiModal):
    """The text model under the checkpoint's multimodal architecture name. The image and audio towers are never
    built: the server must run with --language-model-only."""

    get_placeholder_str = MuseSpark1xOmniMixin.get_placeholder_str

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        runner_hooks.install()
        if os.environ.get("PT_HPU_SDPA_QKV_SLICE_MODE_FWD"):
            raise RuntimeError(
                "muse_spark_hpu: unset PT_HPU_SDPA_QKV_SLICE_MODE_FWD (its window semantics are unverified)"
            )
        mm = vllm_config.model_config.multimodal_config
        if mm is not None and any((mm.get_limit_per_prompt(m) or 0) > 0 for m in ("image", "video", "audio")):
            raise ValueError("muse_spark_hpu: this port serves text only; add --language-model-only")
        if get_pp_group().world_size > 1:
            raise ValueError("muse_spark_hpu: pipeline parallelism is not supported")
        torch.set_float32_matmul_precision("highest")
        mc = vllm_config.model_config
        self.config = mc.hf_config
        tcfg = getattr(mc.hf_config, "text_config", None) or mc.hf_config
        self.tcfg = tcfg
        tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        split = int(os.environ.get("MUSE_SPARK_MOE_TP_SPLIT", "1"))
        if split < 1 or tp % split:
            raise ValueError(f"MUSE_SPARK_MOE_TP_SPLIT={split} does not divide tp={tp}")
        self.layout = Layout(tp, rank, tp // split)
        self.ckpt = checkpoint_dir(mc.model)
        specs, kinds = specs_for(self.ckpt, self.layout)
        self.model = MuseSparkHPUTextModel(
            tcfg, kinds, self.layout, vllm_config.cache_config, vllm_config.quant_config, max_position=mc.max_model_len
        )
        self.lm_head = ParallelLMHead(tcfg.vocab_size, tcfg.hidden_size, params_dtype=torch.bfloat16, prefix="lm_head")
        self.logits_processor = LogitsProcessor(tcfg.vocab_size, scale=1.0, soft_cap=None)
        self.mult, self.cap = float(tcfg.output_multiplier), float(tcfg.final_logit_softcapping)
        self._check_params(specs)
        n_rope = sum(1 for m in self.modules() if type(m).__name__.endswith("RotaryEmbedding"))
        if n_rope != 1:
            raise RuntimeError(f"muse_spark_hpu: {n_rope} rotary modules, expected 1")
        logger.warning(
            "muse_spark_hpu: model built; %d layers, tp %d, ep %d (MoE split %d), rank %d, expert layers %s, "
            "decode layer group %d",
            tcfg.num_hidden_layers,
            tp,
            self.layout.ep_size,
            split,
            rank,
            {k: kinds.count(k) for k in sorted(set(kinds))},
            DECODE_LAYER_GROUP,
        )

    def _check_params(self, specs) -> None:
        """Every loader parameter exists with its per-rank shape and dtype, and nothing else needs loading."""
        params = dict(self.named_parameters())
        bad = [
            f"{n}: loader {list(s.shape)} {s.dtype}, model "
            + (f"{list(params[n].shape)} {params[n].dtype}" if n in params else "nothing")
            for n, s in specs.items()
            if n not in params or tuple(params[n].shape) != s.shape or params[n].dtype != _TORCH[s.dtype]
        ]
        extra = sorted(set(params) - set(specs))
        if bad or extra:
            raise ValueError(
                "muse_spark_hpu: parameters do not match the loader table:\n  "
                + "\n  ".join(bad[:20] + [f"{n}: not filled by the loader" for n in extra[:20]])
            )

    def embed_input_ids(self, input_ids: torch.Tensor, multimodal_embeddings=None, *, is_multimodal=None):
        """fp32 normed embeddings, the same x0 as the input_ids path (the norm is applied once)."""
        if multimodal_embeddings is not None and len(multimodal_embeddings):
            raise RuntimeError("muse_spark_hpu: multimodal embeddings for a text-only server")
        return self.model.embed(input_ids)

    def embed_multimodal(self, **kwargs):
        return []

    def get_language_model(self) -> nn.Module:
        return self

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
        H = self.tcfg.hidden_size
        lead = tuple(input_ids.shape) if input_ids is not None else tuple(inputs_embeds.shape[:-1])
        ids = input_ids.reshape(-1) if input_ids is not None else None
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(-1, H)
        decode = len(lead) == 2 and lead[1] == 1
        h = self.model(ids, positions.reshape(-1), inputs_embeds, decode)
        return h.reshape(*lead, H)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        z = self.logits_processor(self.lm_head, hidden_states)
        return None if z is None else softcap(z, self.mult, self.cap)

    def load_weights(self, weights) -> set[str]:
        """Ignores vLLM's iterator (it would read every shard on every rank) and reads this rank's slices."""
        params = dict(self.named_parameters())
        rep = load_into(params, self.ckpt, self.layout)
        missing = set(params) - rep["loaded"]
        if missing:
            raise ValueError(f"muse_spark_hpu: parameters not loaded: {sorted(missing)[:10]}")
        import habana_frameworks.torch as htorch

        htorch.core.mark_step()
        self.model.finalize()
        return set(params)
