# SPDX-License-Identifier: Apache-2.0
"""KDA attention layer for GLM-5.3-Flash on vllm-gaudi.

A subclass of vLLM's `KimiGatedDeltaNetAttention`: the parameters (projections, `dt_bias`, `A_log`, the three short
convolutions, `o_norm`, `o_proj`) and the state shape and dtype are inherited, so checkpoint names match. Only
`_forward` is replaced. The upstream one reads a per-layer dict of `GDNAttentionMetadata` and calls Triton kernels;
vllm-gaudi publishes one metadata object with attributes (`is_prompt`, `load_indices_tensor`, `query_start_loc_p`,
`has_initial_states_p`, `padding_mask_flat`), like its own Qwen3.5 GDN layer.

Differences from vllm-gaudi's GDN layer that matter here: KDA has three separate q/k/v short convolutions over one
conv state split three ways, the decay is per channel, and the gate is GLM's bounded sigmoid.
"""

from __future__ import annotations

import os

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn, hpu_causal_conv1d_update

from .kda import KdaConsts, hpu_chunk_kda, hpu_chunk_kda_scan, hpu_kda_gate, hpu_recurrent_kda, make_kda_consts

# Prefill chunk geometry. The layer builds the kernel's constant buffers for exactly this geometry.
_KDA_CHUNK = 64
_KDA_SUB = 16

# GLM5_NEXT_KDA_SCAN=1: prefill carries the state between chunks with the log-depth scan; 0 (default) runs the
# reference loop. Read once per process so every rank takes the same path.
KDA_SCAN = os.environ.get("GLM5_NEXT_KDA_SCAN", "0") == "1"


@torch._dynamo.disable
def _save_ssm_state(core_attn_out, ssm_state, final_state, state_indices):
    """Write the final recurrent states into the cache and pass `core_attn_out` through.

    Same shape as vllm-gaudi's GDN layer: outside dynamo (torch.compile drops an in-place index_copy_ into an aliased
    state), and the caller must use the return value (a call whose result is unused can be eliminated, which also
    broke graph replay with "Neither storage attached to input tensor").
    """
    safe = torch.remainder(state_indices, ssm_state.shape[0]).long()
    ssm_state.index_copy_(0, safe, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
    return core_attn_out


def _a_log_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Shard `A_log` over heads. The checkpoint stores it flat, [num_heads]; vLLM's Kimi layer declares
    [1, 1, local_heads, 1] and narrows axis 2, which fails on a 1-D tensor (only visible at TP > 1)."""
    if loaded_weight.dim() == 1:
        loaded_weight = loaded_weight.view(1, 1, -1, 1)
    shard = param.data.shape[2]
    start = get_tensor_model_parallel_rank() * shard
    return default_weight_loader(param, loaded_weight.narrow(2, start, shard))


class Glm5NextHPUKDAAttention(KimiGatedDeltaNetAttention):
    def __init__(self, *args, gate_lower_bound: float, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the inherited A_log loader through set_weight_attrs, which is where vllm-gaudi wraps loaders with the
        # torch.hpu.synchronize() every HPU weight load needs. It refuses to overwrite, so drop the old one first.
        if hasattr(self.A_log, "weight_loader"):
            del self.A_log.weight_loader
        set_weight_attrs(self.A_log, {"weight_loader": _a_log_weight_loader})

        # Kernel constants as non-persistent module buffers: owned and placed like weights, so they are stable across
        # every captured graph, and absent from the state dict.
        consts = make_kda_consts(_KDA_CHUNK, _KDA_SUB, torch.float32, self.A_log.device)
        self.register_buffer("kda_eye", consts.eye, persistent=False)
        self.register_buffer("kda_keep_strict", consts.keep_strict, persistent=False)
        self.register_buffer("kda_keep_diag", consts.keep_diag, persistent=False)
        if KDA_SCAN:
            self.register_buffer(
                "kda_eye_k", torch.eye(self.head_dim, dtype=torch.float32, device=self.A_log.device), persistent=False
            )

        # This layer implements the state contract without prefix caching: one index for load and store, one final
        # state per sequence. With prefix caching vllm-gaudi splits the load and store indices and stores one state per
        # chunk, so a reused prefix would start from another request's state. Refuse rather than corrupt silently.
        if getattr(getattr(self, "cache_config", None), "enable_prefix_caching", False):
            raise ValueError(
                "GLM-5.3-Flash KDA does not support prefix caching on vllm-gaudi; serve with --no-enable-prefix-caching"
            )

        self.gate_lower_bound = gate_lower_bound
        # Set by the model runner (cache-group-idx patch): which row of the 2-D state index tensor is this layer's.
        self.cache_group_idx = None
        self._cg_int = None

    @property
    def kda_consts(self) -> KdaConsts:
        return KdaConsts(self.kda_eye, self.kda_keep_strict, self.kda_keep_diag)

    def _resolve_state_indices(self, attn_metadata):
        """This layer's row of the per-cache-group state index tensor, as a static slice (resolved to an int once)."""
        idx = attn_metadata.load_indices_tensor
        if idx is not None and idx.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None, "cache_group_idx not set for a 2-D state index"
            if self._cg_int is None:
                self._cg_int = int(cg.item()) if torch.is_tensor(cg) else int(cg)
            idx = idx[self._cg_int]
        return idx

    def _forward(self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out) -> None:
        md = get_forward_context().attn_metadata
        if md is None:
            return  # profile run
        if isinstance(md, dict):
            md = md.get(self.prefix, md)

        for apc in ("mamba_chunks_to_block_mapping", "blocks_caching_range", "seqlens_offsets_for_blocks"):
            if getattr(md, apc, None) is not None:
                raise RuntimeError(
                    f"GLM-5.3-Flash KDA saw `{apc}`: prefix caching is on, which this layer does not support. "
                    "Serve with --no-enable-prefix-caching."
                )

        is_prompt = bool(getattr(md, "is_prompt", False))
        state_indices = self._resolve_state_indices(md)
        query_start_loc = getattr(md, "query_start_loc_p", None)
        if query_start_loc is None:
            query_start_loc = getattr(md, "query_start_loc", None)
        has_initial_state = getattr(md, "has_initial_states_p", None)
        # vllm-gaudi pads a prompt batch to its bucket. Upstream's _forward trims the padding itself; this override
        # must zero it, or the padded tail runs through the recurrence into the state decode starts from.
        padding_mask_flat = getattr(md, "padding_mask_flat", None)

        if state_indices is None and query_start_loc is None:
            return  # warmup / profile metadata carries no state map

        conv_state, ssm_state = self.kv_cache[0], self.kv_cache[1]
        n = q_proj_states.size(0)

        # Mask before the conv as well as after: the padded tail can hold NaN, and the conv writes its state from the
        # tail of its input. nan_to_num because 0 * NaN is NaN.
        if padding_mask_flat is not None:
            m = padding_mask_flat.reshape(-1, 1)
            q_proj_states = torch.nan_to_num(q_proj_states * m.to(q_proj_states.dtype))
            k_proj_states = torch.nan_to_num(k_proj_states * m.to(k_proj_states.dtype))
            v_proj_states = torch.nan_to_num(v_proj_states * m.to(v_proj_states.dtype))

        # The conv state is (kernel - 1, 3 * dim) with q, k, v side by side on the last axis, the layout vllm-gaudi's
        # conv kernels read. The kernels update their state in place, and an in-place write through a strided view of
        # a graph input fails at replay, so they work on contiguous copies that are written back as one tensor.
        dim3 = conv_state.shape[-1] // 3
        cs_q = conv_state[..., 0:dim3].contiguous()
        cs_k = conv_state[..., dim3 : 2 * dim3].contiguous()
        cs_v = conv_state[..., 2 * dim3 :].contiguous()

        def cw(conv):
            return conv.weight.view(conv.weight.size(0), conv.weight.size(2))

        if is_prompt:
            conv_args = dict(
                query_start_loc=query_start_loc,
                cache_indices=state_indices,
                has_initial_state=has_initial_state,
                activation=self.activation,
                block_size_to_align=self.cache_config.mamba_block_size,
                is_prompt=True,
            )
            q, k, v = (
                hpu_causal_conv1d_fn(
                    x=x.transpose(0, 1), weight=cw(conv), bias=conv.bias, conv_states=cs, **conv_args
                ).transpose(0, 1)
                for x, conv, cs in (
                    (q_proj_states, self.q_conv1d, cs_q),
                    (k_proj_states, self.k_conv1d, cs_k),
                    (v_proj_states, self.v_conv1d, cs_v),
                )
            )
        else:
            upd = dict(
                activation=self.activation,
                conv_state_indices=state_indices,
                query_start_loc=query_start_loc,
                validate_data=False,
            )
            q, k, v = (
                hpu_causal_conv1d_update(x=x, conv_state=cs, weight=cw(conv), bias=conv.bias, **upd)
                for x, conv, cs in (
                    (q_proj_states, self.q_conv1d, cs_q),
                    (k_proj_states, self.k_conv1d, cs_k),
                    (v_proj_states, self.v_conv1d, cs_v),
                )
            )

        conv_state.copy_(torch.cat((cs_q, cs_k, cs_v), dim=-1))

        d = self.head_dim
        if padding_mask_flat is not None:
            m_flat = padding_mask_flat.reshape(-1, 1).to(q.dtype)
            q = torch.nan_to_num(q * m_flat)
            k = torch.nan_to_num(k * m_flat)
            v = torch.nan_to_num(v * m_flat)

        q = q.reshape(1, n, -1, d).contiguous()
        k = k.reshape(1, n, -1, d).contiguous()
        v = v.reshape(1, n, -1, d).contiguous()

        g = hpu_kda_gate(g1.reshape(1, n, -1), self.A_log, d, g_bias=self.dt_bias, lower_bound=self.gate_lower_bound)
        if padding_mask_flat is not None:
            # the gate is never 0 on padding (lower_bound * sigmoid(...)), so it must be masked too
            g = torch.nan_to_num(g * padding_mask_flat.reshape(1, -1, 1, 1).to(g.dtype))
            beta = torch.nan_to_num(beta * padding_mask_flat.reshape(1, -1, 1).to(beta.dtype))

        if is_prompt:
            init = None
            n_seqs = 1
            if state_indices is not None:
                init = ssm_state.index_select(0, state_indices.long()).contiguous()
                if has_initial_state is not None:
                    # a multiply, not a boolean index (which lowers to scatter_nd)
                    init = init * has_initial_state.bool().view(-1, 1, 1, 1).to(init.dtype)
                n_seqs = int(state_indices.numel())

            # One recurrence per sequence: fold the flat prompt block into [n_seqs, seq_len, H, D].
            seq_len = n // n_seqs if n_seqs > 0 else n
            if n_seqs > 1 and seq_len * n_seqs == n:
                qkv = [t.reshape(n_seqs, seq_len, -1, d).contiguous() for t in (q, k, v)]
                gg = g.reshape(n_seqs, seq_len, -1, d).contiguous()
                bb = beta.reshape(n_seqs, seq_len, -1).contiguous()
            else:
                qkv, gg, bb = [q, k, v], g, beta

            kw = dict(
                chunk_size=_KDA_CHUNK,
                sub_block=_KDA_SUB,
                initial_state=init,
                output_final_state=True,
                consts=self.kda_consts,
            )
            if KDA_SCAN:
                out, final = hpu_chunk_kda_scan(qkv[0], qkv[1], qkv[2], gg, bb, eye_k=self.kda_eye_k, **kw)
            else:
                out, final = hpu_chunk_kda(qkv[0], qkv[1], qkv[2], gg, bb, **kw)
            out = out.reshape(1, n, -1, d).contiguous()
        else:
            state = ssm_state.index_select(0, state_indices.long()).contiguous() if state_indices is not None else None

            def to_seq(t):
                # [1, n, H, D] -> [n, 1, H, D], contiguous: strided inputs to a captured graph fail at replay
                return t.reshape(-1, *t.shape[2:]).unsqueeze(1).contiguous()

            out, final = hpu_recurrent_kda(
                to_seq(q),
                to_seq(k),
                to_seq(v),
                to_seq(g),
                beta.reshape(-1, beta.shape[-1]).unsqueeze(1).contiguous(),
                initial_state=state,
                output_final_state=True,
            )
            out = out.reshape(1, -1, *out.shape[2:]).contiguous()
        if state_indices is not None:
            out = _save_ssm_state(out, ssm_state, final, state_indices)

        out = torch.nan_to_num(out).to(core_attn_out.dtype)
        if out.shape == core_attn_out.shape:
            core_attn_out.copy_(out)
        else:
            core_attn_out[0, :n] = out[0, :n]
