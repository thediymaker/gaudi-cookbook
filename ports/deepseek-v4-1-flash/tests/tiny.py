# SPDX-License-Identifier: Apache-2.0
"""A tiny random-weight DeepSeek-V4.1 stack on the CPU, for the port's tests.

Same topology as the real model (40 layers, the real compress ratios, kv / index / candidate sources, engram on
layers 1 and 14, hc_mult 4, YaRN parameters) at small widths, with a small window, top-k and candidate blocks so
every mechanism engages within a few hundred tokens. Pool is a toy vLLM: shuffled physical blocks (block 0 = the
null block), per-request block tables, and a TaggedCache per (layer, kind).
"""

from __future__ import annotations

import math
import random
import types

import numpy as np
import torch

RATIOS = [0, 0] + [2] * 18 + [1] * 20
VOCAB = 512


def config(**over):
    c = dict(
        num_hidden_layers=40,
        hidden_size=256,
        num_attention_heads=4,
        head_dim=128,
        qk_rope_head_dim=32,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        rms_norm_eps=1e-20,
        sliding_window=16,
        compress_ratios=list(RATIOS),
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=8,
        candidate_block_size=4,
        index_n_heads=4,
        index_head_dim=64,
        index_topk=16,
        compress_rope_theta=160000.0,
        rope_theta=10000.0,
        rope_scaling={"original_max_position_embeddings": 65536, "factor": 16, "beta_fast": 32, "beta_slow": 1},
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=[1, 14],
        engram_head_dim=64,
        engram_n_heads=2,
        engram_max_ngram_size=4,
        engram_vocab_size=1000,
        engram_pad_token_id=2,
        moe_intermediate_size=64,
        n_routed_experts=16,
        num_experts_per_tok=3,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        vocab_size=VOCAB,
    )
    c.update(over)
    return types.SimpleNamespace(**c)


class Lin(torch.nn.Module):
    def __init__(self, i, o, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(o, i, dtype=dtype or torch.bfloat16), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x.to(self.weight.dtype), self.weight)


def build(cfg, seed=0):
    """(stack, head [V, D] fp32, hasher) with random weights."""
    from dsv41_hpu.core import DSV41Engram, DSV41Stack
    from dsv41_hpu.engram import EngramHasher, bucket_primes
    from dsv41_hpu.moe import RefMoE

    def lin(name, i, o, kind, dtype):
        return Lin(i, o, dtype)

    def moe(_):
        return RefMoE(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.n_routed_experts,
            cfg.num_experts_per_tok,
            cfg.routed_scaling_factor,
            cfg.swiglu_limit,
        )

    primes = bucket_primes(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_vocab_size)
    ncols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads

    def eng(lid):
        return DSV41Engram(cfg, lid, primes[cfg.engram_layer_ids.index(lid)], 0, ncols, lin)

    def emb():
        return torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)

    st = DSV41Stack(cfg, lin, moe, emb, 1, eng, max_pos=1024)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in st.named_parameters():
            if p.dtype == torch.uint8:
                if name.endswith("table_scale"):
                    p.copy_(torch.randint(118, 122, p.shape, generator=g, dtype=torch.int32).to(torch.uint8))
                else:  # engram rows: e4m3fn codes without the NaN code 0x7F / 0xFF
                    c = torch.randint(0, 254, p.shape, generator=g, dtype=torch.int32)
                    p.copy_(torch.where(c >= 0x7F, c + 1, c).to(torch.uint8))
            elif "norm" in name or name.endswith(("q_weight", "k_weight")):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("attn_sink"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif "hc_" in name and name.endswith("_fn"):
                p.copy_(0.02 * torch.randn(p.shape, generator=g))
            elif "hc_" in name and name.endswith("_base"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            elif "hc_" in name and name.endswith("_scale"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("gate_bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            elif p.dim() >= 2:
                # large enough that some expert pre-activations pass the SwiGLU limit
                gain = 3.0 if name.endswith((".w1", ".w3")) else 1.0
                if name.endswith("embed.weight"):
                    gain = math.sqrt(p.shape[-1])
                p.copy_(gain * torch.randn(p.shape, generator=g) / math.sqrt(p.shape[-1]))
            else:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    head = 4.0 * torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) / math.sqrt(cfg.hidden_size)
    token_map = [t % 300 for t in range(cfg.vocab_size)]
    hasher = EngramHasher(
        cfg.engram_layer_ids,
        cfg.engram_max_ngram_size,
        cfg.engram_n_heads,
        cfg.engram_vocab_size,
        token_map,
        300,
        cfg.engram_pad_token_id,
    )
    return st.eval(), head, hasher


class Pool:
    """Toy vLLM block pool. use_epoch tags rows with block allocation numbers, as the serving runner does (fresh
    blocks get a new number, prefix-cache hits keep theirs); otherwise rows are tagged with request uids."""

    def __init__(self, cfg, n_blocks, block_size, dtype=torch.bfloat16, seed=0, max_blocks_per_req=None, epoch=True):
        from dsv41_hpu.csa import csa_roles
        from dsv41_hpu.paged import TaggedCache

        self.bs = block_size
        rows = n_blocks * block_size
        self.free = list(range(1, n_blocks))
        random.Random(seed).shuffle(self.free)
        self.tables, self.uids, self.next_uid = {}, {}, 1000
        self.max_blocks = max_blocks_per_req or n_blocks
        self.use_epoch = epoch
        self.epoch = torch.zeros(n_blocks, dtype=torch.int32)
        self.next_epoch = 1
        pend_w = 2 * cfg.head_dim * (4 // torch.tensor([], dtype=dtype).element_size())

        def cache(width, ratio=1, label="cache"):
            data = torch.zeros(rows, width, dtype=dtype)
            return TaggedCache(data, torch.full((rows, 2), -1, dtype=torch.int32), ratio, label)

        self.caches = {}
        for r in csa_roles(cfg):
            c = {"swa": cache(cfg.head_dim, label="swa")}
            if r.is_kv_source:
                c["comp"] = cache(cfg.head_dim, r.ratio, "comp")
                c["idx_k"] = cache(cfg.index_head_dim, r.ratio, "idx_k")
                if r.ratio > 1:
                    c["pend"] = cache(pend_w, label="pend")
            self.caches[r.layer_id] = c

    def admit(self, rid):
        self.tables[rid] = []
        self.uids[rid] = self.next_uid
        self.next_uid += 1

    def release(self, rid):
        self.free.extend(self.tables.pop(rid))

    def ensure(self, rid, upto):
        need = -(-upto // self.bs)
        while len(self.tables[rid]) < need:
            b = self.free.pop()
            self.tables[rid].append(b)
            self.epoch[b] = self.next_epoch  # a fresh allocation
            self.next_epoch += 1

    def share(self, dst, src, n_blocks):
        """dst starts with src's first n_blocks, as a vLLM prefix-cache hit."""
        self.tables[dst] = list(self.tables[src][:n_blocks])

    def ctx(self, rids, positions):
        from dsv41_hpu.paged import BatchCtx

        bt = torch.full((len(rids), self.max_blocks), -1, dtype=torch.long)
        for i, rid in enumerate(rids):
            t = self.tables[rid]
            bt[i, : len(t)] = torch.tensor(t)
        uid = torch.tensor([self.uids[r] for r in rids], dtype=torch.int32)
        ep = torch.where(bt >= 0, self.epoch[bt.clamp(min=0)], torch.full_like(bt, -1)).to(torch.int32)
        return BatchCtx(
            positions=positions,
            block_table=bt,
            req_uid=uid,
            block_size=self.bs,
            block_epoch=ep if self.use_epoch else None,
        )


def engram_rows(hasher, cfg, histories, positions):
    """{layer: [B, S, ncols] long} from host hashing of each request's history."""
    B, S = positions.shape
    out = {lid: torch.zeros(B, S, hasher.n_cols, dtype=torch.long) for lid in cfg.engram_layer_ids}
    for b in range(B):
        pos = positions[b]
        ok = pos >= 0
        if not ok.any():
            continue
        rows = hasher.hash_rows(np.asarray(histories[b]), pos[ok].numpy())  # [n, Lg, ncols]
        for gi, lid in enumerate(cfg.engram_layer_ids):
            out[lid][b, ok] = torch.from_numpy(rows[:, gi])
    return out


class Runner:
    """Drives one stack over one pool: prompt chunks then teacher-forced decode steps, returning logits."""

    def __init__(self, st, head, hasher, cfg, pool, replay_keep=0, compact=True, pad=0):
        from dsv41_hpu import replay

        self.st, self.head, self.hasher, self.cfg, self.pool = st, head, hasher, cfg, pool
        self.keep, self.compact, self.pad = replay_keep, compact, pad
        self.rp = replay.Replay(st) if replay_keep else None

    def forward(self, rid, seq, start, end):
        from dsv41_hpu import replay

        n = end - start
        pad = -n % self.pad if n > 1 and self.pad else 0  # the runner pads prompt rows to a bucket
        pos = torch.cat([torch.arange(start, end), torch.full((pad,), -1)]).view(1, n + pad)
        rows = engram_rows(self.hasher, self.cfg, [seq[:end]], pos)
        ids = torch.tensor(seq[start:end] + [0] * pad).view(1, n + pad)
        self.pool.ensure(rid, end)
        ctx = self.pool.ctx([rid], pos)
        with torch.inference_mode():
            if self.keep and n > 1:
                h = replay.run(self.st, self.rp, ids, ctx, self.pool.caches, rows, self.keep, compact=self.compact)
            else:
                h = self.st(ids, ctx, self.pool.caches, rows)
        return h[0, n - 1 : n].float() @ self.head.t().to(h.device)

    def request(self, rid, seq, chunks, dec, admit=True):
        """Logits at each chunk's last token, then at every decode step of the tokens in dec."""
        if admit:
            self.pool.admit(rid)
        out = [self.forward(rid, seq, a, b) for a, b in chunks]
        full = seq + dec
        for i in range(len(dec)):
            p = len(seq) + i
            out.append(self.forward(rid, full, p, p + 1))
        return torch.cat(out)
