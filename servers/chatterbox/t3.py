# SPDX-License-Identifier: Apache-2.0
"""On-device, batched T3 decoding for Chatterbox (Llama 520M, with CFG) and Chatterbox Turbo (GPT-2, no CFG).

The stock loop samples on the CPU and reads every token back to the host, so each token costs a host round trip
(33 to 45 ms per token on Gaudi 2). Here a decode step is

    embed last token -> transformer layers on a static KV cache -> speech head -> CFG mix ->
    repetition penalty / temperature / min-p / top-k / top-p -> sample -> write the token back

all on the device with no data-dependent shapes, captured as one HPU graph. The host replays the graph and reads
tokens back every `check_every` steps (to stream them and to see end of speech).

The engine has `slots` independent decode slots (one request each, with its own KV rows, position, sampling
parameters and noise) and one step advances the first k slots together (one graph per k in `sizes`), so
concurrent requests share each step instead of taking turns.

The layer math re-implements the forward passes of the transformers Llama and GPT-2 blocks that the chatterbox-tts
package (MIT, https://github.com/resemble-ai/chatterbox) builds T3 from, and the sampling order follows its
inference loops.
"""

from __future__ import annotations

import logging
import math

import torch

try:
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.core as htcore
except ImportError:  # CPU (tests)
    ht = None
    htcore = None

NEG = -1.0e30
log = logging.getLogger("chatterbox")


def mark_step():
    if htcore is not None:
        htcore.mark_step()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def top_k_threshold(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask everything below the k-th largest value. No boolean scatter_ (the lazy bridge syncs the host on it):
    a comparison against a threshold. Returns (masked logits, the sorted top-k values)."""
    vals = torch.topk(logits, k, dim=-1).values  # descending
    return torch.where(logits < vals[:, -1:], NEG, logits), vals


def top_p_threshold(logits: torch.Tensor, sorted_vals: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of highest values whose mass reaches top_p (Hugging Face's TopPLogitsWarper), as a
    threshold: the smallest value whose higher-ranked mass is still below top_p."""
    p = torch.softmax(sorted_vals, dim=-1)
    above = torch.cumsum(p, dim=-1) - p  # mass ranked strictly above each value
    thr = torch.where(above < top_p, sorted_vals, torch.full_like(sorted_vals, -NEG)).min(dim=-1, keepdim=True).values
    return torch.where(logits < thr, NEG, logits)


def exponential_race(probs: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """argmax(p / E) with E ~ Exp(1) picks index i with probability p_i: an exact categorical draw with no
    multinomial. The noise is drawn per request up front (seedable), so the graph replays no RNG."""
    return torch.argmax(probs / e, dim=-1)


class T3Engine:
    """`slots` concurrent T3 decodes over a T3 model already on `device`. The original model uses CFG (2 rows per
    slot: conditioned and unconditioned); Turbo has 1 row per slot."""

    def __init__(
        self,
        t3,
        *,
        slots: int = 1,
        max_new: int = 1000,
        max_prefill: int = 640,
        prefill_bucket: int = 64,
        device: str = "hpu",
        use_graphs: bool = True,
        check_every: int = 8,
        top_k: int = 0,
        top_p: float = 1.0,
        use_min_p: bool = True,
    ):
        self.t3 = t3
        self.hp = t3.hp
        self.gpt = bool(t3.is_gpt)
        self.cfg = not self.gpt
        self.r1 = 2 if self.cfg else 1
        self.S = slots
        self.R = slots * self.r1
        self.device = torch.device(device)
        self.max_new = max_new
        self.max_prefill = max_prefill
        self.bucket = prefill_bucket
        self.check_every = check_every
        self.top_k = top_k
        self.top_p = top_p
        self.use_min_p = use_min_p
        self.use_graphs = use_graphs and ht is not None and self.device.type == "hpu"
        self.sizes = sorted({s for s in (1, 2, 4, 8) if s < slots} | {slots})
        tf = t3.tfmr
        self.tf = tf
        if self.gpt:
            self.layers = list(tf.h)
            self.H = tf.config.n_head
            self.C = tf.config.n_embd
            self.D = self.C // self.H
            self.KVH = self.H
        else:
            self.layers = list(tf.layers)
            self.H = tf.config.num_attention_heads
            self.C = tf.config.hidden_size
            self.D = getattr(tf.config, "head_dim", None) or self.C // self.H
            self.KVH = tf.config.num_key_value_heads
        self.V = t3.speech_head.out_features
        self.Lc = max_prefill + max_new + 8  # cache positions per row
        self.N = max_new + check_every + 1  # per-slot token and noise rows
        dt = t3.speech_head.weight.dtype
        self.dtype = dt
        dev = self.device
        R, S = self.R, self.S
        self.Kc = [torch.zeros(R, self.KVH, self.Lc, self.D, device=dev, dtype=dt) for _ in self.layers]
        self.Vc = [torch.zeros(R, self.KVH, self.Lc, self.D, device=dev, dtype=dt) for _ in self.layers]
        self.ar = torch.arange(self.Lc, device=dev)
        self.row_base = torch.arange(R * self.KVH, device=dev) * self.Lc  # flat cache row offsets
        self.slot_ar = torch.arange(S, device=dev)
        if not self.gpt:  # RoPE tables for every cache position (llama3 scaling is static)
            with torch.inference_mode():
                cos, sin = tf.rotary_emb(self.Kc[0][:1, 0].transpose(0, 1), torch.arange(self.Lc, device=dev)[None])
            self.cos_tab = cos[0].to(dt)  # (Lc, D)
            self.sin_tab = sin[0].to(dt)
        # CPU copies of the small embedding tables: prefill embeddings are assembled on the CPU, so a new text
        # length compiles nothing.
        self.text_emb_cpu = t3.text_emb.weight.detach().float().cpu()
        self.speech_emb_cpu = t3.speech_emb.weight.detach().float().cpu()
        self.text_pos_cpu = t3.text_pos_emb.emb.weight.detach().float().cpu() if t3.text_pos_emb is not None else None
        self.speech_pos_cpu = (
            t3.speech_pos_emb.emb.weight.detach().float().cpu() if t3.speech_pos_emb is not None else None
        )

        # Per-slot device state; the graphs read and write these buffers.
        self.cur_tok = torch.zeros(S, dtype=torch.long, device=dev)  # last sampled token
        self.pos = torch.zeros(S, dtype=torch.long, device=dev)  # cache position of the next input
        self.step = torch.zeros(S, dtype=torch.long, device=dev)  # index of the token being sampled
        self.out_tok = torch.zeros(S, self.N, dtype=torch.long, device=dev)
        self.seen = torch.zeros(S, self.V, dtype=dt, device=dev)
        self.noise = torch.ones(S, self.N, self.V, dtype=torch.float32, device=dev)
        self.temp = torch.ones(S, 1, dtype=dt, device=dev)
        self.cfg_w = torch.zeros(S, 1, dtype=dt, device=dev)
        self.rep = torch.ones(S, 1, dtype=dt, device=dev)
        self.min_p = torch.zeros(S, 1, dtype=dt, device=dev)
        self.graphs: dict[int, object] = {}
        self.prefill_in: dict[int, dict] = {}
        # Host bookkeeping.
        self.busy = [False] * S  # slot holds a request
        self.decoding = [False] * S  # slot still produces tokens
        self.steps = [0] * S  # tokens produced so far (host mirror of `step`)
        self.n_read = [0] * S  # tokens handed out so far
        self.max_new_slot = [max_new] * S

    # ------------------------------------------------------------------ model
    def _layer(self, i, x, cos, sin, mask, widx, prefill: bool, rows: int):
        """One transformer layer on x (rows, T, C); writes K and V at flat cache indices widx."""
        R, T, _ = x.shape
        H, KVH, D = self.H, self.KVH, self.D
        lyr = self.layers[i]
        if self.gpt:
            q, k, v = lyr.attn.c_attn(lyr.ln_1(x)).split(self.C, dim=2)
        else:
            h = lyr.input_layernorm(x)
            a = lyr.self_attn
            q, k, v = a.q_proj(h), a.k_proj(h), a.v_proj(h)
        q = q.view(R, T, H, D).transpose(1, 2)
        k = k.view(R, T, KVH, D).transpose(1, 2)
        v = v.view(R, T, KVH, D).transpose(1, 2)
        if not self.gpt:
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        Kc, Vc = self.Kc[i], self.Vc[i]
        Kc.view(-1, D).index_copy_(0, widx, k.reshape(-1, D))
        Vc.view(-1, D).index_copy_(0, widx, v.reshape(-1, D))
        keys, vals = (k, v) if prefill else (Kc[:rows], Vc[:rows])
        if KVH != H:
            keys = keys.repeat_interleave(H // KVH, dim=1)
            vals = vals.repeat_interleave(H // KVH, dim=1)
        att = torch.matmul(q, keys.transpose(-1, -2)) * (1.0 / math.sqrt(D)) + mask
        att = torch.softmax(att.float(), dim=-1).to(q.dtype)
        o = torch.matmul(att, vals).transpose(1, 2).reshape(R, T, H * D)
        if self.gpt:
            x = x + lyr.attn.c_proj(o)
            return x + lyr.mlp(lyr.ln_2(x))
        x = x + lyr.self_attn.o_proj(o)
        return x + lyr.mlp(lyr.post_attention_layernorm(x))

    def _final(self, x):
        return self.t3.speech_head(self.tf.ln_f(x) if self.gpt else self.tf.norm(x))

    def _sample(self, logits_rows, sidx, j):
        """logits_rows (k*r1, V), slot indices sidx (k,), token indices j (k,) -> tokens (k,)."""
        k = sidx.shape[0]
        if self.cfg:
            lr = logits_rows.view(k, 2, self.V)
            cond, uncond = lr[:, 0], lr[:, 1]
            logits = cond + self.cfg_w.index_select(0, sidx) * (cond - uncond)
        else:
            logits = logits_rows
        logits = logits.float()
        seen = self.seen.index_select(0, sidx).float() > 0
        rep = self.rep.index_select(0, sidx).float()
        temp = self.temp.index_select(0, sidx).float()

        def rep_pen(lg):
            return torch.where(seen, torch.where(lg < 0, lg * rep, lg / rep), lg)

        if self.gpt:  # Turbo's order: temperature, top-k, top-p, repetition penalty
            logits = logits / temp
        else:  # the original's order: repetition penalty, temperature, min-p, top-p
            logits = rep_pen(logits) / temp
            if self.use_min_p:
                probs = torch.softmax(logits, dim=-1)
                pmax = probs.max(dim=-1, keepdim=True).values
                logits = torch.where(probs < self.min_p.index_select(0, sidx).float() * pmax, NEG, logits)
        vals = None
        if self.top_k and self.top_k < self.V:
            logits, vals = top_k_threshold(logits, self.top_k)
        if self.top_p < 1.0:
            if vals is None:
                vals = torch.sort(logits, dim=-1, descending=True).values
            logits = top_p_threshold(logits, vals, self.top_p)
        if self.gpt:
            logits = rep_pen(logits)
        e = self.noise.view(-1, self.V).index_select(0, sidx * self.N + j)  # (k, V) Exp(1)
        return exponential_race(torch.softmax(logits, dim=-1), e)

    def _commit(self, sidx, j, tok):
        """out_tok[s, j] = tok, seen[s, tok] = 1, cur_tok[s] = tok."""
        self.out_tok.view(-1).index_copy_(0, sidx * self.N + j, tok)
        self.seen.view(-1).index_fill_(0, sidx * self.V + tok, 1.0)
        self.cur_tok.index_copy_(0, sidx, tok)

    def _decode_step(self, k):
        """Advance slots [0, k) by one token."""
        r1, rows = self.r1, k * self.r1
        sidx = self.slot_ar[:k]
        tok, pos, j = self.cur_tok[:k], self.pos[:k], self.step[:k]
        e = self.t3.speech_emb(tok)  # (k, C)
        if self.speech_pos_cpu is not None:
            e = e + self.t3.speech_pos_emb.emb(j)
        if self.gpt:
            e = e + self.tf.wpe(pos)
        x = e[:, None, :].expand(k, r1, self.C).reshape(rows, 1, self.C).to(self.dtype)
        posr = pos[:, None].expand(k, r1).reshape(rows)
        mask = torch.where(self.ar.view(1, -1) <= posr.view(-1, 1), 0.0, NEG).to(self.dtype).view(rows, 1, 1, self.Lc)
        if self.gpt:
            cos = sin = None
        else:
            cos = self.cos_tab.index_select(0, posr).view(rows, 1, 1, self.D)
            sin = self.sin_tab.index_select(0, posr).view(rows, 1, 1, self.D)
        widx = self.row_base[: rows * self.KVH] + posr[:, None].expand(rows, self.KVH).reshape(-1)
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, mask, widx, prefill=False, rows=rows)
        new = self._sample(self._final(x)[:, -1, :], sidx, j)
        self._commit(sidx, j, new)
        # Idle slots in [0, k) step too; clamp so they never index out of range.
        self.pos[:k].copy_(torch.clamp(pos + 1, max=self.Lc - 1))
        self.step[:k].copy_(torch.clamp(j + 1, max=self.N - 1))

    def _replay(self, k):
        if not self.use_graphs:
            self._decode_step(k)
            mark_step()
        elif k not in self.graphs:
            log.info("t3: capturing the decode graph for %d slots", k)
            g = ht.hpu.HPUGraph()  # capturing also runs it once: that is this step
            with ht.hpu.stream(ht.hpu.Stream()):
                g.capture_begin()
                self._decode_step(k)
                g.capture_end()
            self.graphs[k] = g
        else:
            self.graphs[k].replay()

    # --------------------------------------------------------------- prefill
    def build_prefill(self, cond_emb: torch.Tensor, text_tokens: list[int], cfg_weight: float) -> torch.Tensor:
        """Prefill embeddings assembled on the CPU: (r1, L, C) float32. cond_emb: (1, Lcond, C) CPU float, from
        T3.prepare_conditioning."""
        te = self.text_emb_cpu[torch.tensor(text_tokens, dtype=torch.long)]  # (Lt, C)
        if self.gpt:
            sp = self.speech_emb_cpu[self.hp.start_speech_token][None]
            return torch.cat([cond_emb[0].float(), te, sp], 0)[None]
        tpos = self.text_pos_cpu[: len(text_tokens)]
        sp = self.speech_emb_cpu[self.hp.start_speech_token][None] + self.speech_pos_cpu[0][None]
        # The stock loop appends the BOS embedding twice (in prepare_input_embeds and again in inference); keep
        # that so outputs match the reference model.
        cond_row = torch.cat([cond_emb[0].float(), te + tpos, sp, sp], 0)
        uncond_row = torch.cat([cond_emb[0].float(), tpos, sp, sp], 0) if cfg_weight > 0 else cond_row
        return torch.stack([cond_row, uncond_row])

    def _prefill_fn(self, Lp):
        b = self.prefill_in[Lp]
        x, last, sidx, widx = b["x"], b["last"], b["slot"], b["widx"]
        if self.gpt:
            x = x + self.tf.wpe(self.ar[:Lp].view(1, Lp))
            cos = sin = None
        else:
            cos = self.cos_tab[:Lp].view(1, 1, Lp, self.D)
            sin = self.sin_tab[:Lp].view(1, 1, Lp, self.D)
        ar = self.ar[:Lp]
        mask = torch.where(ar.view(1, Lp) <= ar.view(Lp, 1), 0.0, NEG).to(self.dtype).view(1, 1, Lp, Lp)
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, mask, widx, prefill=True, rows=self.r1)
        logits = self._final(x.index_select(1, last))[:, -1, :]  # at the last real position
        j = torch.zeros(1, dtype=torch.long, device=self.device)
        self._commit(sidx, j, self._sample(logits, sidx, j))

    @torch.inference_mode()
    def start(
        self,
        slot: int,
        embeds: torch.Tensor,
        *,
        temperature=0.8,
        cfg_weight=0.5,
        repetition_penalty=1.2,
        min_p=0.05,
        seed=None,
        max_new=None,
    ):
        """Begin a request in `slot`: set its parameters and noise, run the prefill (a lazy-mode graph per
        64-position bucket, shared by all slots) and sample the first token. No host sync."""
        assert not self.decoding[slot], f"slot {slot} is busy"
        _, L, C = embeds.shape
        Lp = ((L + self.bucket - 1) // self.bucket) * self.bucket
        if Lp > self.max_prefill:
            raise ValueError(f"prompt is {L} positions; the limit is {self.max_prefill}")
        dev = self.device
        sd = torch.tensor([slot], dtype=torch.long).to(dev)

        def one(v):
            return torch.tensor([[float(v)]], dtype=self.dtype).to(dev)

        self.temp.index_copy_(0, sd, one(temperature))
        self.cfg_w.index_copy_(0, sd, one(cfg_weight))
        self.rep.index_copy_(0, sd, one(repetition_penalty))
        self.min_p.index_copy_(0, sd, one(min_p))
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            noise = (-torch.log(torch.rand(1, self.N, self.V, generator=g).clamp_(min=1e-20))).to(dev)
        else:
            noise = torch.empty(1, self.N, self.V, device=dev).uniform_(1e-20, 1.0).log_().neg_()
        self.noise.index_copy_(0, sd, noise)
        seen = torch.zeros(1, self.V, dtype=self.dtype)
        seen[0, self.hp.start_speech_token] = 1.0
        self.seen.index_copy_(0, sd, seen.to(dev))
        if Lp not in self.prefill_in:
            log.info("t3: first prefill of bucket %d", Lp)
            self.prefill_in[Lp] = {
                "x": torch.zeros(self.r1, Lp, C, device=dev, dtype=self.dtype),
                "last": torch.zeros(1, dtype=torch.long, device=dev),
                "slot": torch.zeros(1, dtype=torch.long, device=dev),
                "widx": torch.zeros(self.r1 * self.KVH * Lp, dtype=torch.long, device=dev),
            }
        b = self.prefill_in[Lp]
        xpad = torch.zeros(self.r1, Lp, C, dtype=torch.float32)
        xpad[:, :L] = embeds
        b["x"].copy_(xpad.to(self.dtype).to(dev))
        b["last"].copy_(torch.tensor([L - 1], dtype=torch.long).to(dev))
        b["slot"].copy_(sd)
        rows = torch.arange(slot * self.r1, (slot + 1) * self.r1)
        widx = (
            (rows[:, None, None] * self.KVH + torch.arange(self.KVH)[None, :, None]) * self.Lc
            + torch.arange(Lp)[None, None, :]
        ).reshape(-1)
        b["widx"].copy_(widx.to(dev))
        self._prefill_fn(Lp)
        mark_step()
        self.pos.index_copy_(0, sd, torch.tensor([L], dtype=torch.long).to(dev))
        self.step.index_copy_(0, sd, torch.ones(1, dtype=torch.long).to(dev))
        mark_step()
        self.busy[slot] = self.decoding[slot] = True
        self.steps[slot] = 1
        self.n_read[slot] = 0
        self.max_new_slot[slot] = min(max_new or self.max_new, self.max_new)

    def release(self, slot: int):
        self.busy[slot] = self.decoding[slot] = False

    @torch.inference_mode()
    def run_block(self) -> dict[int, tuple[list[int], bool]]:
        """Advance every decoding slot by `check_every` tokens (one batched step per token, on the smallest graph
        that covers them), then read the new tokens back with one sync. Returns {slot: (new tokens without end
        of speech, finished)}."""
        live = [s for s in range(self.S) if self.decoding[s]]
        if not live:
            return {}
        k = next(s for s in self.sizes if s >= max(live) + 1)
        n = self.check_every
        for _ in range(n):
            self._replay(k)
        toks = self.out_tok.cpu()  # the whole buffer: a per-k slice would be a new lazy graph
        eos = self.hp.stop_speech_token
        out = {}
        for s in live:
            self.steps[s] = min(self.steps[s] + n, self.N - 1)
            lim = min(self.steps[s], self.max_new_slot[s])
            new = toks[s, self.n_read[s] : lim].tolist()
            fin = lim >= self.max_new_slot[s]
            if eos in new:
                new = new[: new.index(eos)]
                fin = True
            self.n_read[s] += len(new)
            if fin:
                self.decoding[s] = False
            out[s] = (new, fin)
        return out
