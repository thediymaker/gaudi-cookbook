"""Mistral Large 4 FFNs: the routed MoE on mxk after the virtual-expert split, the shared expert, the layer-0 dense
FFN.

Router (HF MistralLarge4TopkRouter + the PR's config: softmax scoring, no bias, one group, norm_topk_prob, scale 1.0):
  p = softmax(fp32(f) @ fp32(gate)^T) over 128;  E = top4(p);  g = p_E / (sum p_E + 1e-20)
vllm-gaudi's stock MXFP4 MoE method would route with its own softmax/top-k and no renormalisation control (the V4
port lost 42% PPL to stock routing, DSV), so every call site routes here.

Virtual experts: each real expert e (I 6144) is two experts of I 3072, h = 0, 1, with
  w13[v] = [w1 rows 3072h .. +3072 ; w3 the same rows]    packed [6144, 3072] + E8M0 [6144, 192]
  w2[v]  = w2 columns 3072h .. +3072                       packed [6144, 1536] + E8M0 [6144, 96]
and every route (e, g) is two routes (v(e, 0), g), (v(e, 1), g): top-8 of 256. The kernel's per-token fp32 sum
adds the two w2 halves. The loader (loader.py) does the re-slice on the host; no byte is stored twice.
  placement A (bring-up): v = 2e + h; rank r holds real experts 16r .. 16r+15 = virtual 32r .. 32r+31 (e_min 32r)
  placement B:  v = 128h + e; rank r holds half h = r div 4 of experts 32g .. 32g+31, g = r mod 4
                           (e_min 128h + 32g)
  placement F (fitted): B's layout with each layer's experts reordered by a fitted order
                           (ML4_PLACEMENT_FILE, JSON {"layers": {"<layer>": [128 real expert ids]}}): group g holds
                           order[32g .. 32g+31]; v = 128h + pos(e) (pos = the inverse order, one gather per layer).
                           A layer missing from the file keeps B's order. Same math per route as B (each route's two
                           halves still go to ranks g and g + 4); only which experts share a rank changes.

Expert calls by token count T (this rank's virtual experts, bf16 activations, W4A16):
  T <= 16              mxk_w13x (SiLU, unclamped) + mxk_w2x                       (04: 4-6x stock at decode)
  17..ML4_MXK_SPLIT_MAX  the same in 16-token pieces (one more ~16 us call per piece; DSpark verify rows,
                       B x 4 <= 64 at 16 streams, stay on mxk)
  256 <= T <= 4096     v8p: 32 + 32 mxk_deq, mark_step, the stock bf16 fused MoE  (bit-identical to stock)
  otherwise            the stock mxfp4_fused_weights op on the same virtual tensors
A rank holding more local experts than one call takes (TP1 / TP2 test ranks) runs them in groups and adds the
groups' bf16 outputs; an EP8 rank (32 virtual experts) is always one call, bit-identical to 04's harness.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from . import mxk
from .numerics import mxfp4_dequant

N_EXPERTS = 128
TOPK = 4
SPLIT_MAX = int(os.environ.get("ML4_MXK_SPLIT_MAX", "64"))
GROUP_MXK = 128  # kernel limit on local experts per call (TP1 / TP2 test ranks hold 256 / 128)
GROUP_V8P = int(os.environ.get("ML4_V8P_GROUP", "32"))  # bf16 unpack transient: 3.375 GiB per 32 experts
# ML4_V8P_GROUP=16 on an EP8 rank: "v8 unpack in two halves". Each half runs to the
# end (mark_step after its MoE call) before the next half unpacks, so only one half's bf16 experts are live: -1.69 GiB
# at the prefill peak. The halves' bf16 outputs are added (one more bf16 rounding than the single call).
V8P_MIN, V8P_MAX = 256, 4096
# Set by graphs.DecodeGraphs while it captures a prompt chunk (ML4_PREFILL_GRAPHS): the v8p branch flushes the lazy
# graph between its unpack and the fused op, which a captured graph cannot hold, so those calls take the stock
# MXFP4 fused op (the same values eager).
GRAPH_PROMPT = [False]
# Inside a prompt graph, chunks of up to ML4_PREFILL_MXK_MAX rows (default 64: the 64-token bucket) take the mxk decode
# kernels (v9g, 64-row calls) instead of the stock op. Their cost grows with the units of routes (~50 us per unit, a
# unit = up to 2 routes of one local expert): about 0.9 ms per layer at 16 random rows, 2.5 at 64, 5.5 at 128, where the
# stock op is ~2.9 / 6.3 / 5.4 ms; above 64 rows the two cost the same. Control key
# prefill_mxk_max overrides it (read at capture).
PREFILL_MXK_MAX = int(os.environ.get("ML4_PREFILL_MXK_MAX", "64"))
# Controls (tests only): the loader swaps the w1 / w3 halves; stock routing drops the top-k renormalisation
# (must cost >= 5 % PPL).
FAULTS = {
    "swap_w13": os.environ.get("ML4_FAULT_SWAP_W13", "0") == "1",
    "stock_routing": os.environ.get("ML4_FAULT_STOCK_ROUTING", "0") == "1",
}


# ML4_ROUTER_PAD=1 (control key router_pad): a one-row router runs its top-4 on two copies of the row and keeps row 0.
# The HPU topk node for [1, 128] took ~32 us per layer (2.3 ms per 1-stream step) against ~3 us at >= 2 rows.
# Exact: the same fp32 values, the same top-4 (greedy 8/8 identical on the server). On by default.
ROUTER_PAD = os.environ.get("ML4_ROUTER_PAD", "1") == "1"
# ML4_MOE_DECODE_CHAIN=1 (default): decode / verify rows take ML4MoE._forward_decode (same values, kernels start
# sooner; 4 streams 18.10 -> 18.30 tok/s steady, 17.2 -> 17.5 whole). Read at capture time; "0" restores forward().
DECODE_CHAIN = os.environ.get("ML4_MOE_DECODE_CHAIN", "1") == "1"
# ML4_HOT_REPLICAS=N with ML4_HOT_REPLICAS_FILE (placement B only; off by default): each
# layer's N hottest real experts (from recorded routes) get a second copy, both halves, in another group, N / 4 extra
# slots per rank. Rank (h, g) then holds S = 32 + N / 4 slots: its 32 own halves, then its group's copies; virtual id
# of slot j on rank (h, g) = (4h + g) S + j (S = 32 is B's 128h + e). Per step, a replicated expert's routes go to
# the copy whose group has the lower load from the step's non-replicated routes (ties: the owner). Control key
# hot_replicas (int) uses only the first k replicas of each layer's list (0 = B's routing on the same slots); a change
# needs a decode-graph re-capture. File: {"layers": {"<layer>": [[expert, group], ...]}}, hottest first.
HOT_N = int(os.environ.get("ML4_HOT_REPLICAS", "0"))
_HOT: dict | None = None


def hot_replicas(layer: int | None) -> list | None:
    global _HOT
    if HOT_N <= 0 or layer is None:
        return None
    if _HOT is None:
        import json

        _HOT = {int(k): v for k, v in json.load(open(os.environ["ML4_HOT_REPLICAS_FILE"]))["layers"].items()}
    rep = _HOT.get(layer)
    return None if rep is None else [(int(e), int(g)) for e, g in rep[:HOT_N]]


def hot_active() -> int:
    from .control import get as ctl_get

    return int(ctl_get("hot_replicas", HOT_N))


def _prefill_mxk_max() -> int:
    from .control import ctl

    return int(ctl().get("prefill_mxk_max", PREFILL_MXK_MAX))


# ML4_MOE_PAD_SKIP=1 (default): rows of padding (position -1: the bucket's tail of a short prompt, padding requests of
# a decode batch) get a virtual id no rank holds, so the mxk kernels skip their routes. Padding rows are all one
# token and share their experts, so on the ranks that hold those experts they added units of 2 routes. Only the mxk
# paths (<= 64 rows) use it; the stock and v8p ops get the ids unchanged. Real rows are bit-identical (the kernels
# keep each token's own summation order). Control key moe_pad_skip overrides it.
PAD_SKIP = os.environ.get("ML4_MOE_PAD_SKIP", "1") == "1"
PAD_ID = 1 << 24


def pad_skip() -> bool:
    from .control import ctl

    return bool(ctl().get("moe_pad_skip", PAD_SKIP))


def _mask_pad(vid: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """vid [T, 8] int32 with padding rows (valid [T] False) set to PAD_ID."""
    if valid is None:
        return vid
    return torch.where(valid.reshape(-1, 1), vid, torch.full_like(vid, PAD_ID))


def router_pad() -> bool:
    from .control import get as ctl_get

    return bool(ctl_get("router_pad", ROUTER_PAD))


def route(f: torch.Tensor, gate_w: torch.Tensor, topk: int = TOPK) -> tuple[torch.Tensor, torch.Tensor]:
    """f [T, D] -> (weights [T, topk] fp32, ids [T, topk] long)."""
    p = torch.softmax(F.linear(f.float(), gate_w.float()), dim=-1)
    if p.shape[0] == 1 and p.device.type == "hpu" and router_pad():
        w, ids = torch.cat([p, p]).topk(topk, dim=-1)
        w, ids = w[:1], ids[:1]
    else:
        w, ids = p.topk(topk, dim=-1)
    if FAULTS["stock_routing"]:
        return w, ids
    return w / (w.sum(-1, keepdim=True) + 1e-20), ids


def to_virtual(
    ids: torch.Tensor, w: torch.Tensor, placement: str = "A", pos: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real routes [T, k] -> virtual routes [T, 2k] (ids int32, weights repeated). pos [128] (placement F): the
    position of each real expert in the layer's fitted order."""
    if placement == "A":
        vid = torch.stack([2 * ids, 2 * ids + 1], -1)
    else:
        if pos is not None:
            ids = pos[ids]
        vid = torch.stack([ids, N_EXPERTS + ids], -1)
    return vid.flatten(-2).to(torch.int32), torch.stack([w, w], -1).flatten(-2)


_ORDERS: dict | None = None


def placement_order(layer: int | None) -> list[int] | None:
    """The fitted expert order of one layer (placement F), or None (B's order)."""
    global _ORDERS
    if _ORDERS is None:
        import json

        path = os.environ.get("ML4_PLACEMENT_FILE", "")
        _ORDERS = {int(k): v for k, v in json.load(open(path))["layers"].items()} if path else {}
    order = _ORDERS.get(layer) if layer is not None else None
    if order is not None:
        assert sorted(order) == list(range(N_EXPERTS)), f"placement file: layer {layer} is not a permutation"
    return order


def local_experts(
    ep_rank: int, ep_size: int, placement: str = "A", order: list[int] | None = None
) -> tuple[list[tuple[int, int]], int]:
    """([(real expert e, half h) for local virtual slot 0..], e_min) of one EP rank."""
    n_real = N_EXPERTS // ep_size
    if placement == "A":
        e0 = ep_rank * n_real
        return [(e0 + i // 2, i % 2) for i in range(2 * n_real)], 2 * e0
    assert ep_size == 8, "placement B / F is defined for EP8"
    h, g = ep_rank // 4, ep_rank % 4
    order = order or list(range(N_EXPERTS))
    return [(order[32 * g + i], h) for i in range(32)], 128 * h + 32 * g


class GatedFFN(nn.Module):
    """w2(silu(w1 f) * (w3 f)) with bf16 roundings as HF's MLP (silu output, product); w1/w3 merged into one
    column-parallel w13 [2 I_local, D] (rows [w1 local; w3 local]); w2 row-parallel (partial sum)."""

    def __init__(self, prefix: str, dim: int, inter: int, linear):
        super().__init__()
        self.w13 = linear(f"{prefix}.w13", dim, 2 * inter, "column", torch.bfloat16)
        self.w2 = linear(f"{prefix}.w2", inter, dim, "row", torch.bfloat16)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        gu = _lin(self.w13, f)
        g, u = gu.chunk(2, dim=-1)
        return _lin(self.w2, F.silu(g) * u)


class ML4MoE(nn.Module):
    """Routed experts of one EP rank + the TP-split shared expert. Returns this rank's partial sum (the caller
    all-reduces once, with attention's output already added before the FFN)."""

    def __init__(self, cfg, linear, ep_rank: int = 0, ep_size: int = 1, placement: str = "A", layer: int | None = None):
        super().__init__()
        D, I = cfg.hidden_size, cfg.moe_intermediate_size
        self.D, self.IV = D, I // 2
        self.placement = placement
        order = placement_order(layer) if placement == "F" else None
        self.slots, self.e_min = local_experts(ep_rank, ep_size, placement, order)
        self.n_global = 2 * N_EXPERTS
        self._hot = hot_replicas(layer) if placement == "B" and ep_size == 8 else None
        if self._hot:
            X = HOT_N // 4
            h, g = ep_rank // 4, ep_rank % 4
            per_g = [[e for e, gg in self._hot if gg == q] for q in range(4)]
            assert all(len(v) == X for v in per_g), f"hot replicas file: layer {layer} needs {X} copies per group"
            assert all(e // 32 != gg for e, gg in self._hot), f"hot replicas file: layer {layer} copies its own group"
            self.S = 32 + X
            self.slots = self.slots + [(e, h) for e in per_g[g]]
            self.e_min = (4 * h + g) * self.S
            self.n_global = 8 * self.S
            self._hot_slot = {e: 32 + per_g[gg].index(e) for e, gg in self._hot}
            self._hot_k = None
        self._pos_host = None
        self._pos = None
        if order is not None:
            self._pos_host = torch.empty(N_EXPERTS, dtype=torch.long)
            self._pos_host[torch.tensor(order)] = torch.arange(N_EXPERTS)
        n = len(self.slots)
        self.n_local = n
        u8 = dict(dtype=torch.uint8)
        self.gate_w = nn.Parameter(torch.empty(N_EXPERTS, D, dtype=torch.bfloat16), requires_grad=False)
        self.w13 = nn.Parameter(torch.empty(n, 2 * self.IV, D // 2, **u8), requires_grad=False)
        self.w13_scale = nn.Parameter(torch.empty(n, 2 * self.IV, D // 32, **u8), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(n, D, self.IV // 2, **u8), requires_grad=False)
        self.w2_scale = nn.Parameter(torch.empty(n, D, self.IV // 32, **u8), requires_grad=False)
        self.shared = GatedFFN("shared_experts", D, I * getattr(cfg, "n_shared_experts", 1), linear)

    def routed(self, f: torch.Tensor, vid: torch.Tensor, vw: torch.Tensor) -> torch.Tensor:
        """f [T, D] bf16, virtual routes [T, 8] -> this rank's routed sum [T, D] bf16."""
        if f.device.type == "cpu":
            return self.routed_ref(f, vid, vw)
        T = f.shape[0]
        if mxk.ENABLED and T <= SPLIT_MAX:

            def dec(g0, W, e):
                if T <= mxk.MAX_TOKENS or mxk.v9g_for(T):  # v9g: up to 64 tokens in one call
                    return mxk.decode(f, vid, vw, *W, vid.shape[-1], e)
                return torch.cat(
                    [
                        mxk.decode(
                            f[t0 : t0 + mxk.MAX_TOKENS],
                            vid[t0 : t0 + mxk.MAX_TOKENS],
                            vw[t0 : t0 + mxk.MAX_TOKENS],
                            *W,
                            vid.shape[-1],
                            e,
                        )
                        for t0 in range(0, T, mxk.MAX_TOKENS)
                    ]
                )

            return self._grouped(GROUP_MXK, dec)
        if GRAPH_PROMPT[0] and mxk.ENABLED and mxk.v9g_flag() and T <= _prefill_mxk_max():

            def dec64(g0, W, e):
                return torch.cat(
                    [
                        mxk.decode(f[t0 : t0 + 64], vid[t0 : t0 + 64], vw[t0 : t0 + 64], *W, vid.shape[-1], e)
                        for t0 in range(0, T, 64)
                    ]
                )

            return self._grouped(GROUP_MXK, dec64)
        if mxk.ENABLED and V8P_MIN <= T <= V8P_MAX and not GRAPH_PROMPT[0]:
            return self._grouped(
                GROUP_V8P, lambda g0, W, e: mxk.prefill_v8p(f, vid, vw, *W, e, self.n_global), flush_each=True
            )
        return self._grouped(GROUP_MXK, lambda g0, W, e: mxk.stock(f, vid, vw, *W, e, self.n_global))

    def _grouped(self, G: int, call, flush_each: bool = False) -> torch.Tensor:
        """call(g0, (W13, S13, W2, S2) of local experts g0..g0+G-1, their e_min) summed over groups. One group (the
        whole tensors, no views) whenever the rank holds <= G experts: EP8 is always one call."""
        W = (self.w13, self.w13_scale, self.w2, self.w2_scale)
        if self.n_local <= G:
            return call(0, W, self.e_min)
        starts = list(range(0, self.n_local, G))
        if self.n_local - starts[-1] == 1:
            starts.pop()  # no 1-expert group: the fused MoE op fails on one expert (moe25/odd_slots_check.py)
        out = None
        for i, g0 in enumerate(starts):
            g1 = starts[i + 1] if i + 1 < len(starts) else self.n_local
            y = call(g0, tuple(t[g0:g1] for t in W), self.e_min + g0)
            out = y if out is None else out + y
            if flush_each and g1 < self.n_local:
                from .indexer import _mark_step

                _mark_step()  # this group's unpacked experts die before the next group unpacks
        return out

    def routed_ref(self, f: torch.Tensor, vid: torch.Tensor, vw: torch.Tensor) -> torch.Tensor:
        """CPU reference of routed(): fp32 dequant, fp32 GEMV, silu * up rounded to bf16 (the kernels' stored
        intermediate), fp32 down and routing sum, one rounding."""
        y = torch.zeros(f.shape[0], self.D)
        for v in range(self.n_local):
            tok, slot = torch.where(vid == self.e_min + v)
            if tok.numel() == 0:
                continue
            h = f[tok].float() @ mxfp4_dequant(self.w13[v], self.w13_scale[v]).t()
            a = (F.silu(h[:, : self.IV]) * h[:, self.IV :]).to(torch.bfloat16).float()
            y.index_add_(
                0, tok, vw[tok, slot].float().unsqueeze(-1) * (a @ mxfp4_dequant(self.w2[v], self.w2_scale[v]).t())
            )
        return y.to(f.dtype)

    def _mxk_rows(self, T: int) -> bool:
        """routed() takes the mxk decode kernels for T rows."""
        return mxk.ENABLED and (T <= SPLIT_MAX or (GRAPH_PROMPT[0] and mxk.v9g_flag() and T <= _prefill_mxk_max()))

    def forward(self, f: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """f [..., D] (RMSNorm'ed) -> routed + shared partial [..., D]. valid [...] bool (positions >= 0): padding rows
        skip the mxk kernels (ML4_MOE_PAD_SKIP)."""
        shape = f.shape
        f2 = f.reshape(-1, shape[-1])
        T = f2.shape[0]
        if valid is not None and (T == 1 or f2.device.type != "hpu" or not self._mxk_rows(T) or not pad_skip()):
            valid = None
        if (
            DECODE_CHAIN
            and f2.device.type == "hpu"
            and T <= 8
            and self.placement != "A"
            and mxk.ENABLED
            and self.n_local <= GROUP_MXK
            and (T <= mxk.MAX_TOKENS or mxk.v9g_for(T))
        ):
            return self._forward_decode(f2, valid).reshape(shape)
        w, ids = route(f2, self.gate_w)
        if self._pos_host is not None and (self._pos is None or self._pos.device != ids.device):
            self._pos = self._pos_host.to(ids.device)
        vid, vw = to_virtual(ids, w, self.placement, self._pos if self._pos_host is not None else None)
        if self._hot:
            vid = self._hot_vid(ids)  # vw as to_virtual: the same weight on both halves
        y = self.routed(f2, _mask_pad(vid, valid), vw) + self.shared(f2)
        return y.reshape(shape)

    def _hot_tables(self, k: int, device) -> None:
        """Device tables for the first k replicas: owner / copy virtual id (h = 0), the owner-group one-hot of each
        non-replicated expert, the half offset, and each expert's (owner, copy) cell in the 4 x 4 load compare."""
        S = self.S
        e = torch.arange(N_EXPERTS)
        base = (e // 32) * S + e % 32
        g0 = e // 32
        rep_b, g1, isrep = base.clone(), g0.clone(), torch.zeros(N_EXPERTS, dtype=torch.bool)
        for ex, gg in self._hot[:k]:
            rep_b[ex] = gg * S + self._hot_slot[ex]
            g1[ex] = gg
            isrep[ex] = True
        own = torch.nn.functional.one_hot(g0, 4).masked_fill(isrep.unsqueeze(-1), 0)  # [128, 4] non-replicated
        pair = torch.where(isrep, g0 * 4 + g1, 0)  # C[0, 0] is always False: non-replicated keep the owner
        off = torch.tensor([0, 4 * S])
        self._hot_t = tuple(t.to(device=device, dtype=torch.int32) for t in (base, rep_b, own, off)) + (
            pair.to(device),
        )
        self._hot_k = (k, str(device))

    def _hot_vid(self, ids: torch.Tensor) -> torch.Tensor:
        """Real ids [T, 4] -> virtual ids [T, 8] int32 under the replicas (ML4_HOT_REPLICAS)."""
        k = hot_active()
        if self._hot_k != (k, str(ids.device)):
            self._hot_tables(k, ids.device)
        base, rep_b, own, off, pair = self._hot_t
        load = own[ids].sum((0, 1))  # [4] routes of non-replicated experts per group
        c = (load.unsqueeze(1) > load.unsqueeze(0)).flatten()  # c[4 go + gc]: the copy's group is less loaded
        tab = torch.where(c[pair], rep_b, base)  # [128] the step's virtual id (h = 0) per expert
        return (tab[ids].unsqueeze(-1) + off).flatten(-2)  # [T, 8] int32

    def _forward_decode(self, f2: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """forward() for decode / verify rows (ML4_MOE_DECODE_CHAIN): the same ops and values, issued so the kernels
        start sooner: the activation permute and the shared expert before the router, virtual ids as ids + [0, 128]
        (one add instead of a stack). The router chain sat on every rank's critical path, ~78 us per layer before the
        kernels at 4 streams. Greedy 8/8 identical to forward."""
        xp = mxk.permute(f2)
        sh = self.shared(f2)
        from . import r5k

        if (
            not FAULTS["stock_routing"]
            and self._pos_host is None
            and not self._hot
            and f2.device.type == "hpu"
            and r5k.ROUTER
            and r5k.ready()
        ):  # ML4_R5_ROUTER: top-4 .. bf16 weights in one kernel, same bits
            vid, vw = r5k.router(torch.softmax(F.linear(f2.float(), self.gate_w.float()), dim=-1))
            vid = _mask_pad(vid, valid)
            yr = mxk.decode(
                f2, vid, vw, self.w13, self.w13_scale, self.w2, self.w2_scale, vid.shape[-1], self.e_min, xp=xp
            )
            return yr + sh
        w, ids = route(f2, self.gate_w)
        if self._pos_host is not None:
            if self._pos is None or self._pos.device != ids.device:
                self._pos = self._pos_host.to(ids.device)
            ids = self._pos[ids]
        off = getattr(self, "_vid_off", None)
        if off is None or off.device != ids.device:
            off = self._vid_off = torch.tensor([0, N_EXPERTS], dtype=ids.dtype, device=ids.device)
        vid = self._hot_vid(ids) if self._hot else (ids.unsqueeze(-1) + off).flatten(-2).to(torch.int32)  # [T, 8]
        vid = _mask_pad(vid, valid)
        vw = w.unsqueeze(-1).expand(*w.shape, 2).flatten(-2)
        yr = mxk.decode(f2, vid, vw, self.w13, self.w13_scale, self.w2, self.w2_scale, vid.shape[-1], self.e_min, xp=xp)
        return yr + sh


def _lin(mod, x):
    out = mod(x)
    return out[0] if isinstance(out, tuple) else out
