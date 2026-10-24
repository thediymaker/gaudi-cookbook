# SPDX-License-Identifier: Apache-2.0
"""Short-conv state for Inkling: four depthwise causal convs per layer (kernel 4, residual, no activation)
on the raw K and V streams and on the TP-summed attention and MLP deltas.

Each conv needs the raw inputs of the previous 3 positions. Those are kept as one snapshot row per
attention KV block id: slot[b] holds the inputs after the last position processed in block b. A full block
therefore carries the state at its end, which is what a prefix-cache hit on that block needs, and every
request sharing a cached block reads the same snapshot. A request's partial last block holds its running
state.

The state is not a KV-cache layer. As one (a MambaSpec) the scheduler would give it its own block per 128
tokens from the same pool, which the model never reads: half the pool is wasted and the admission check
leaves any prompt over half the pool waiting forever. Instead the attention page is padded by the conv
bytes and model.install_kv_hooks allocates the rows next to the attention cache.

Plain torch, no vLLM imports (the CPU tests run the real class).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .ops import step_cached

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = 0, 1, 2, 3

# Every conv state of the model in this worker process, in layer order; install_kv_hooks allocates them.
CONV_STATES: list = []


class InklingConvState(nn.Module):
    def __init__(self, vllm_config, kv_dim: int, hidden: int, kernel_size: int, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_dim = kv_dim
        self.hidden = hidden
        self.taps = kernel_size - 1
        self.kv_cache = (torch.tensor([]),)
        self._vllm_config = vllm_config

    def register(self) -> None:
        """Record this state for install_kv_hooks, never in the forward context (that would make it KV)."""
        if self.prefix in self._vllm_config.compilation_config.static_forward_context:
            raise ValueError(f"{self.prefix} must not be a KV-cache layer")
        if any(m.prefix == self.prefix for m in CONV_STATES):
            raise ValueError(f"Duplicate conv state: {self.prefix}")
        CONV_STATES.append(self)

    def get_state_shape(self):
        return (
            (self.taps, self.kv_dim),  # K stream (this rank's KV heads)
            (self.taps, self.kv_dim),  # V stream
            (self.taps, self.hidden),  # attention delta (full width)
            (self.taps, self.hidden),  # MLP delta (full width)
        )

    def get_state_dtype(self):
        return (torch.bfloat16,) * 4

    def state_bytes_per_block(self) -> int:
        return sum(
            math.prod(shape) * torch.tensor([], dtype=dt).element_size()
            for shape, dt in zip(self.get_state_shape(), self.get_state_dtype(), strict=True)
        )

    def _states(self):
        kv = self.kv_cache
        if isinstance(kv, tuple | list) and len(kv) and hasattr(kv[0], "numel") and kv[0].numel():
            return kv
        return None  # profiling run: no rows allocated yet

    @staticmethod
    def _masked_store(state: torch.Tensor, idx: torch.Tensor, new_rows: torch.Tensor) -> None:
        """index_copy_ that leaves rows with idx < 0 (batch padding) unchanged instead of writing pad garbage
        into slot 0."""
        safe = idx.clamp(min=0).to(torch.long)
        valid = (idx >= 0).view(-1, 1, 1)
        cur = state.index_select(0, safe)
        state.index_copy_(0, safe, torch.where(valid, new_rows.to(state.dtype), cur))

    # The runner does not see a mamba-like model, so it fills no state indices; slots are derived from the
    # attention metadata. Decode needs block_list in logical order (VLLM_CONTIGUOUS_PA=false, checked at
    # startup). Rows are num_blocks + 1, so every attention block id is a valid slot.

    def _prefill_slots(self, attn_metadata, bsz: int, seqlen: int):
        """-> (load_slot [B], store_tok [B, K], store_slot [B, K]) or None.

        load_slot: the block holding position ctx - 1 (-1 on a first chunk). store_tok: chunk-local index
        of the last position this chunk processes in each block it touches (block ends, then the final
        token); store_slot: that position's block. Entries past the chunk clamp onto the final token, so
        duplicates write identical rows. Padding rows (qlen == 0) get -1."""
        sm = attn_metadata.slot_mapping
        ctx = getattr(attn_metadata, "context_lens_tensor", None)
        qlen = attn_metadata.seq_lens_tensor
        if sm is None or ctx is None or qlen is None:
            return None
        bs = attn_metadata.block_size
        dev = sm.device
        sm = sm.view(bsz, seqlen).to(torch.long)
        ctx = ctx[:bsz].to(torch.long)
        qlen = qlen[:bsz].to(torch.long)
        neg = torch.full((bsz,), -1, dtype=torch.long, device=dev)

        bl = attn_metadata.block_list
        if bl is not None:
            bl = bl.view(bsz, -1).to(torch.long)
            col = torch.div((ctx - 1).clamp(min=0), bs, rounding_mode="floor").clamp(max=bl.shape[1] - 1)
            prev = bl.gather(1, col.view(bsz, 1)).view(bsz)
            load_slot = torch.where((ctx > 0) & (prev >= 0), prev, neg)
        else:
            load_slot = neg

        k_max = seqlen // bs + 1
        ks = torch.arange(k_max, device=dev).view(1, k_max)
        first = (bs - 1) - torch.remainder(ctx, bs)
        last = (qlen - 1).clamp(min=0).view(bsz, 1)
        tok = torch.cat([torch.minimum(first.view(bsz, 1) + ks * bs, last), last], dim=1)  # [B, K]
        slot = torch.div(sm.gather(1, tok).clamp(min=0), bs, rounding_mode="floor")
        valid = (qlen > 0).view(bsz, 1) & (sm.gather(1, tok) >= 0)
        return load_slot, tok, torch.where(valid, slot, torch.full_like(slot, -1))

    def _decode_slots(self, attn_metadata, bsz: int):
        """-> (load_slot [B], store_slot [B]) or None. The new token lives in the request's last listed block
        (store). The previous position is in the same block unless the new token opens a block
        (block_usage == 1); then it is in the block listed just before (load)."""
        bl = attn_metadata.block_list
        bg = attn_metadata.block_groups
        bu = attn_metadata.block_usage
        if bl is None or bg is None or bu is None:
            return None
        nb = bg.shape[0]
        ar = torch.arange(nb, device=bg.device)
        member = bg.view(1, nb).to(torch.long) == torch.arange(bsz, device=bg.device).view(bsz, 1)
        last_idx = torch.where(member, ar.view(1, nb), torch.full_like(ar, -1).view(1, nb)).amax(dim=1)
        has_any = last_idx >= 0
        last_idx = last_idx.clamp(min=0)
        bl = bl.to(torch.long)
        store = bl[last_idx]
        load_idx = torch.where(bu[last_idx] == 1, (last_idx - 1).clamp(min=0), last_idx)
        load = bl[load_idx]
        neg = torch.full_like(store, -1)
        return torch.where(has_any, load, neg), torch.where(has_any, store, neg)

    def conv(self, stream: int, x: torch.Tensor, weight: torch.Tensor, is_prompt: bool, attn_metadata):
        """Depthwise causal conv1d + residual on one stream. x: prompt [B, L, C] or decode [B, C]; weight
        [C, W]. Without state (the profiling run) the history is zeros, still causal within the chunk."""
        states = self._states()
        taps = self.taps
        if is_prompt:
            bsz, seqlen, ch = x.shape
            hist = x.new_zeros(bsz, taps, ch)
            slots = None
            if states is not None and attn_metadata is not None:
                slots = step_cached(
                    ("pslots", id(attn_metadata.slot_mapping), bsz, seqlen),
                    lambda: self._prefill_slots(attn_metadata, bsz, seqlen),
                )
            if slots is not None:
                load_slot, store_tok, store_slot = slots
                # Continuation chunk or prefix-cache hit: resume from the snapshot of the block holding
                # position ctx - 1; first chunks (load_slot == -1) keep zeros.
                stored = states[stream].index_select(0, load_slot.clamp(min=0))
                hist = torch.where((load_slot >= 0).view(bsz, 1, 1), stored.to(x.dtype), hist)
            xin = torch.cat([hist.to(x.dtype), x], dim=1)  # [B, taps + L, C]
            y = F.conv1d(xin.transpose(1, 2), weight.unsqueeze(1), groups=weight.shape[0]).transpose(1, 2)
            out = x + y.to(x.dtype)
            if slots is not None:
                # The snapshot after chunk token t is the raw inputs of t-2..t = xin rows t+1..t+taps (for
                # t < 2 this slides the loaded history in). Real rows only, never the bucket's pad tail.
                k = store_tok.shape[1]
                rows = (store_tok.view(bsz, k, 1) + 1 + torch.arange(taps, device=x.device).view(1, 1, taps)).view(
                    bsz, k * taps
                )
                snap = xin.gather(1, rows.unsqueeze(-1).expand(-1, -1, ch))
                self._masked_store(states[stream], store_slot.reshape(-1), snap.view(bsz * k, taps, ch))
            return out
        bsz, ch = x.shape
        if states is None or attn_metadata is None:
            return x + x * weight[:, -1].to(x.dtype)
        slots = step_cached(
            ("dslots", id(attn_metadata.block_list), bsz),
            lambda: self._decode_slots(attn_metadata, bsz),
        )
        st = states[stream]
        if slots is None:
            hist = st.new_zeros(bsz, taps, ch)
        else:
            load_slot, store_slot = slots
            hist = st.index_select(0, load_slot.clamp(min=0))
            hist = torch.where((load_slot >= 0).view(bsz, 1, 1), hist, torch.zeros_like(hist))
        xin = torch.cat([hist.to(x.dtype), x.unsqueeze(1)], dim=1)  # [B, W, C]
        out = x + (xin * weight.t().unsqueeze(0).to(x.dtype)).sum(dim=1)
        if slots is not None:
            self._masked_store(st, store_slot, xin[:, -taps:, :])
        return out
