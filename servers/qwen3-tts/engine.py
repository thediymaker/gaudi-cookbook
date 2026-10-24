# SPDX-License-Identifier: Apache-2.0
"""Fast, streaming generation for Qwen3-TTS-12Hz (CustomVoice) on one Gaudi card.

The stock path calls transformers' generate() for the talker and, inside every talker step, a second generate()
for the 15-codebook code predictor. In lazy mode that is a growing dynamic cache (new graph shapes every step),
dozens of host syncs per frame and generate()'s own per-step machinery. This engine drives the same weights from
the `qwen_tts` package with its own loop:

* Talker KV cache: static [slots * kv_heads * cap, head_dim] per layer, written with index_copy_ at each slot's
  position. Attention reads a fixed-length bucket of it, so a graph's shape depends only on (slot bucket, kv bucket).
* One frame step = code predictor (15 unrolled sub-steps with a local KV) + talker step + sampling on the device
  (repetition penalty, suppression, min-new-tokens, temperature, top-k by Gumbel-max) + state update, captured as
  ONE HPU graph per (slot bucket, kv bucket). A frame costs one graph replay and no host sync.
* The host syncs once every `sync_every` frames to read the new codes, find end of speech and feed the decoder.
* The 12 Hz codec decoder is causal (causal convolutions, causal sliding-window attention), so audio is decoded in
  chunks with left context, right-padded to a fixed window; the padding cannot reach earlier samples.
* Several requests share the frame step (continuous batching over slots).

The prompt layout and sampling follow `Qwen3TTSForConditionalGeneration.generate` in qwen-tts 0.1.1 (Apache-2.0,
https://github.com/QwenLM/Qwen3-TTS) with non_streaming_mode=True, which is what generate_custom_voice uses.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.core as htcore
except ImportError:  # CPU (tests)
    ht = None
    htcore = None

log = logging.getLogger("qwen3-tts")

NEG = -1.0e30
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920  # 24 kHz / 12.5 Hz


def mark_step() -> None:
    if htcore is not None:
        htcore.mark_step()


def hpu_sync() -> None:
    mark_step()
    if hasattr(torch, "hpu") and hasattr(torch.hpu, "synchronize"):
        torch.hpu.synchronize()


def _rot(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def bucket(n: int, buckets) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {buckets[-1]}")


def gumbel_topk(logits: torch.Tensor, top_k: int, temperature: float, u: torch.Tensor) -> torch.Tensor:
    """An exact categorical draw from softmax(top-k(logits / temperature)), with no multinomial and no host sync.

    argmax(log p + g) with g = -log(-log(u)) Gumbel noise picks index i with probability p_i. `u` is uniform noise
    [B, top_k] drawn in the graph. (torch.multinomial on the HPU samples a visibly distorted distribution.)
    """
    vals, idx = torch.topk(logits / temperature, top_k, dim=-1)
    g = -torch.log(-torch.log(u.clamp(1e-10, 1.0 - 1e-7)))
    return idx.gather(1, torch.argmax(vals + g, dim=-1, keepdim=True)).squeeze(1)


class Captured:
    """A function captured as one HPU graph (capture also runs it once), or run eagerly without graphs."""

    def __init__(self, fn, use_graph: bool):
        self.fn = fn
        self.graph = None
        if use_graph:
            mark_step()
            self.graph = ht.hpu.HPUGraph()
            with ht.hpu.stream(ht.hpu.Stream()):
                self.graph.capture_begin()
                self.out = fn()
                self.graph.capture_end()
            hpu_sync()
        else:
            self.out = fn()
            mark_step()

    def __call__(self):
        if self.graph is None:
            self.out = self.fn()
            mark_step()
        else:
            mark_step()  # flush host-side lazy ops queued before this replay
            self.graph.replay()
        return self.out


@dataclass
class Job:
    """One text segment being generated in a slot."""

    req: Request
    tid: list
    cid: list
    cmask: list
    max_frames: int
    slot: int = -1
    prompt_len: int = 0
    frames: list = field(default_factory=list)  # np.int64[16] each
    decoded: int = 0  # frames already turned into audio
    chunks_sent: int = 0
    done: bool = False


class Request:
    """A whole speech request: one or more segments, audio out through `push` (float32 chunks, then None)."""

    def __init__(self, segments: list[dict], pause_s: list[float], push=None):
        self.segments = segments  # each: tid, cid, cmask, max_frames
        self.pause_s = pause_s
        self.out: queue.Queue = queue.Queue()
        self.push = push or self.out.put
        self.cancelled = False
        self.error: str | None = None
        self.frames = 0
        self.seg_index = 0
        self.hit_cap = False

    def cancel(self) -> None:
        self.cancelled = True


class FastQwen3TTS:
    def __init__(
        self,
        tts,
        max_slots: int = 16,
        slot_buckets=(4, 16),  # a frame step costs about 10 ms at 1, 4 or 16 slots, so few buckets
        cap: int = 4096,
        kv_buckets=(512, 1024, 2048, 4096),
        prefill_buckets=(64, 128, 192, 256, 384, 512, 768, 1024),
        sync_every: int = 4,
        # Left context for chunked decoding. Against a whole-clip decode, 25 frames (upstream's chunked_decode)
        # gave clicks at chunk edges (20 to 22 dB SNR), 72 gave 37 dB, 100 gives 42 dB (bf16 floor about 43.5).
        dec_left: int = 100,
        dec_chunks=(4, 8, 16, 24, 48),  # first audio after 4 frames (0.32 s), then growing chunks
        dec_windows=(16, 32, 64, 128, 160),
        use_graphs: bool = True,
        step_guard=contextlib.nullcontext,  # wraps each round of device work (the server's watchdog)
    ):
        self.tts = tts
        self.m = tts.model
        talker = self.m.talker
        self.talker = talker
        cfg = self.m.config
        tc = cfg.talker_config
        cpc = tc.code_predictor_config
        self.cfg, self.tc = cfg, tc
        p = next(talker.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self.use_graphs = use_graphs and ht is not None and self.dev.type == "hpu"
        self.step_guard = step_guard

        self.H = tc.hidden_size
        self.NH = tc.num_attention_heads
        self.NKV = tc.num_key_value_heads
        self.HD = getattr(tc, "head_dim", self.H // self.NH)
        self.V = tc.vocab_size
        self.G = tc.num_code_groups
        self.NHc = cpc.num_attention_heads
        self.NKVc = cpc.num_key_value_heads
        self.HDc = getattr(cpc, "head_dim", cpc.hidden_size // self.NHc)
        self.EOS = tc.codec_eos_token_id

        self.max_slots = max_slots
        self.slot_buckets = tuple(b for b in slot_buckets if b < max_slots) + (max_slots,)
        self.cap = cap
        self.kv_buckets = tuple(b for b in kv_buckets if b <= cap)
        self.prefill_buckets = tuple(b for b in prefill_buckets if b < cap)
        self.sync_every = sync_every
        self.dec_left = dec_left
        self.dec_chunks = dec_chunks
        self.dec_windows = dec_windows

        gen = dict(self.tts.generate_defaults or {})
        self.temperature = float(gen.get("temperature", 0.9))
        self.top_k = int(gen.get("top_k", 50))
        self.rep_penalty = float(gen.get("repetition_penalty", 1.05))
        self.sub_temperature = float(gen.get("subtalker_temperature", 0.9))
        self.sub_top_k = int(gen.get("subtalker_top_k", 50))
        self.noise_k = max(self.top_k, self.sub_top_k)
        if float(gen.get("top_p", 1.0)) < 1.0 or float(gen.get("subtalker_top_p", 1.0)) < 1.0:
            log.warning("generate_config asks for top_p < 1; this engine samples top-k only")

        self._prepare_weights()
        self._alloc_state()
        self.graphs_frame: dict = {}
        self.graphs_prefill: dict = {}
        self.prefill_parts: dict = {}
        self.graphs_decode: dict = {}

        self.lock = threading.Condition()
        self.pending: list[Job] = []
        self.slots: list[Job | None] = [None] * max_slots
        self.stop = False
        self.on_fatal = None  # called with the exception if the engine loop dies
        self.stats = {"frames": 0, "prefills": 0, "decodes": 0}

    # ------------------------------------------------------------------ setup
    def _prepare_weights(self) -> None:
        t = self.talker
        dev = self.dev
        self.t_layers = list(t.model.layers)
        self.t_norm = t.model.norm
        self.codec_head = t.codec_head
        self.codec_emb = t.model.codec_embedding
        self.text_emb = t.model.text_embedding
        self.text_proj = t.text_projection
        cp = t.code_predictor
        self.cp_layers = list(cp.model.layers)
        self.cp_norm = cp.model.norm
        self.cp_proj = cp.small_to_mtp_projection
        self.cp_emb = list(cp.model.codec_embedding)
        self.cp_heads = list(cp.lm_head)

        with torch.no_grad():
            # RoPE tables for every position, computed like the stock rotary modules: fp32 angle, cos/sin, then the
            # model dtype.
            def rope(rot, n):
                inv = rot.inv_freq.float().to(dev)
                sc = float(getattr(rot, "attention_scaling", 1.0))
                f = torch.arange(n, device=dev, dtype=torch.float32)[:, None] * inv[None, :]
                emb = torch.cat((f, f), dim=-1)
                return (emb.cos() * sc).to(self.dtype), (emb.sin() * sc).to(self.dtype)

            self.t_cos, self.t_sin = rope(t.model.rotary_emb, self.cap)
            self.c_cos, self.c_sin = rope(cp.model.rotary_emb, self.G + 1)

            ids = torch.tensor([[self.cfg.tts_pad_token_id]], device=dev)
            self.tts_pad_embed = self.text_proj(self.text_emb(ids))[0, 0].clone()  # [H]
            # The last 1024 codec ids are control tokens the talker may not emit, except end of speech.
            sup = torch.zeros(self.V, device=dev, dtype=torch.float32)
            sup[self.V - 1024 :] = NEG
            sup[self.EOS] = 0.0
            self.suppress_bias = sup
            eos_only = torch.zeros(self.V, device=dev, dtype=torch.float32)
            eos_only[self.EOS] = NEG
            self.eos_bias = eos_only  # min_new_tokens = 2
            self.vocab_iota = torch.arange(self.V, device=dev)
            self.slot_iota = torch.arange(self.max_slots, device=dev)
            self.kv_iota = torch.arange(self.NKV, device=dev)
            self.cap_iota = torch.arange(self.cap, device=dev)
        mark_step()

    def _alloc_state(self) -> None:
        S, dev = self.max_slots, self.dev
        rows = S * self.NKV * self.cap
        self.kcache = [torch.zeros(rows, self.HD, device=dev, dtype=self.dtype) for _ in self.t_layers]
        self.vcache = [torch.zeros(rows, self.HD, device=dev, dtype=self.dtype) for _ in self.t_layers]
        self.s_code0 = torch.zeros(S, device=dev, dtype=torch.long)  # first codebook of the next frame
        self.s_hid = torch.zeros(S, self.H, device=dev, dtype=self.dtype)  # talker hidden of the last step
        self.s_pos = torch.zeros(S, device=dev, dtype=torch.long)
        self.s_step = torch.zeros(S, device=dev, dtype=torch.long)
        self.s_fidx = torch.zeros(S, device=dev, dtype=torch.long)
        self.s_active = torch.zeros(S, device=dev, dtype=torch.long)
        self.s_presence = torch.zeros(S, self.V, device=dev, dtype=torch.float32)
        self.frames = torch.zeros(S * self.cap, self.G, device=dev, dtype=torch.long)
        mark_step()

    # ------------------------------------------------------------- sampling
    def _sample_talker(self, logits, presence, step, u):
        """logits [B,V] fp32 -> token [B], in Hugging Face's order: repetition penalty, suppression,
        min_new_tokens, temperature, top-k, then the draw."""
        pen = torch.where(logits < 0, logits * self.rep_penalty, logits / self.rep_penalty)
        logits = torch.where(presence > 0, pen, logits)
        logits = logits + self.suppress_bias
        logits = logits + self.eos_bias * (step < 2).to(torch.float32).unsqueeze(1)
        return gumbel_topk(logits, self.top_k, self.temperature, u)

    # ------------------------------------------------------------ talker ops
    def _talker_layers_decode(self, x, Sb, L, pos):
        """x [Sb,1,H]; one decode step at per-slot positions `pos` [Sb]."""
        NH, NKV, HD = self.NH, self.NKV, self.HD
        cos = self.t_cos.index_select(0, pos).unsqueeze(1)  # [Sb,1,HD]
        sin = self.t_sin.index_select(0, pos).unsqueeze(1)
        widx = ((self.slot_iota[:Sb, None] * NKV + self.kv_iota[None, :]) * self.cap + pos[:, None]).reshape(-1)
        mask = torch.where(self.cap_iota[None, :L] <= pos[:, None], 0.0, NEG).to(self.dtype).view(Sb, 1, 1, L)
        scale = HD**-0.5
        for li, layer in enumerate(self.t_layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(Sb, NH, HD))
            k = a.k_norm(a.k_proj(h).view(Sb, NKV, HD))
            v = a.v_proj(h).view(Sb, NKV, HD)
            q = q * cos + _rot(q) * sin
            k = k * cos + _rot(k) * sin
            kc, vc = self.kcache[li], self.vcache[li]
            kc.index_copy_(0, widx, k.reshape(Sb * NKV, HD))
            vc.index_copy_(0, widx, v.reshape(Sb * NKV, HD))
            K = kc.view(self.max_slots, NKV, self.cap, HD)[:Sb, :, :L]
            Vv = vc.view(self.max_slots, NKV, self.cap, HD)[:Sb, :, :L]
            qg = q.view(Sb, NKV, NH // NKV, HD)
            att = torch.matmul(qg, K.transpose(-1, -2)) * scale + mask
            att = torch.softmax(att.float(), dim=-1).to(self.dtype)
            x = x + a.o_proj(torch.matmul(att, Vv).reshape(Sb, 1, NH * HD))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.t_norm(x)

    def _talker_layers_prefill(self, x, P, widx):
        """x [1,P,H] for one slot; writes positions 0..P-1 of that slot's cache."""
        NH, NKV, HD = self.NH, self.NKV, self.HD
        cos = self.t_cos[:P].unsqueeze(0).unsqueeze(1)  # [1,1,P,HD]
        sin = self.t_sin[:P].unsqueeze(0).unsqueeze(1)
        causal = torch.where(self.cap_iota[None, :P] <= self.cap_iota[:P, None], 0.0, NEG).to(self.dtype)
        scale = HD**-0.5
        for li, layer in enumerate(self.t_layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(1, P, NH, HD)).transpose(1, 2)  # [1,NH,P,HD]
            k = a.k_norm(a.k_proj(h).view(1, P, NKV, HD)).transpose(1, 2)  # [1,NKV,P,HD]
            v = a.v_proj(h).view(1, P, NKV, HD).transpose(1, 2)
            q = q * cos + _rot(q) * sin
            k = k * cos + _rot(k) * sin
            self.kcache[li].index_copy_(0, widx, k[0].reshape(NKV * P, HD))
            self.vcache[li].index_copy_(0, widx, v[0].reshape(NKV * P, HD))
            qg = q.view(1, NKV, NH // NKV, P, HD)
            att = torch.matmul(qg, k.unsqueeze(2).transpose(-1, -2)) * scale + causal
            att = torch.softmax(att.float(), dim=-1).to(self.dtype)
            o = torch.matmul(att, v.unsqueeze(2))  # [1,NKV,g,P,HD]
            x = x + a.o_proj(o.reshape(1, NH, P, HD).transpose(1, 2).reshape(1, P, NH * HD))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.t_norm(x)

    # ------------------------------------------------------ code predictor
    def _code_predictor(self, hid, code0, u):
        """hid [B,H] (talker hidden of the previous step), code0 [B] -> codes [B,15], unrolled with a local KV."""
        B = hid.shape[0]
        NH, NKV, HD = self.NHc, self.NKVc, self.HDc
        scale = HD**-0.5
        x = self.cp_proj(torch.stack([hid, self.codec_emb(code0)], dim=1))  # [B,2,Hc]
        ks = [None] * len(self.cp_layers)
        vs = [None] * len(self.cp_layers)
        causal2 = torch.tensor([[0.0, NEG], [0.0, 0.0]], device=self.dev).to(self.dtype)
        out = []
        for j in range(self.G - 1):
            if j == 0:
                T, p0 = 2, 0
            else:
                x = self.cp_proj(self.cp_emb[j - 1](out[-1])).unsqueeze(1)  # [B,1,Hc]
                T, p0 = 1, j + 1
            cos = self.c_cos[p0 : p0 + T].view(1, 1, T, HD)
            sin = self.c_sin[p0 : p0 + T].view(1, 1, T, HD)
            for li, layer in enumerate(self.cp_layers):
                a = layer.self_attn
                h = layer.input_layernorm(x)
                q = a.q_norm(a.q_proj(h).view(B, T, NH, HD)).transpose(1, 2)
                k = a.k_norm(a.k_proj(h).view(B, T, NKV, HD)).transpose(1, 2)
                v = a.v_proj(h).view(B, T, NKV, HD).transpose(1, 2)
                q = q * cos + _rot(q) * sin
                k = k * cos + _rot(k) * sin
                ks[li] = k if ks[li] is None else torch.cat([ks[li], k], dim=2)
                vs[li] = v if vs[li] is None else torch.cat([vs[li], v], dim=2)
                if T > 1:
                    qg = q.reshape(B, NKV, NH // NKV, T, HD)
                    att = torch.matmul(qg, ks[li].unsqueeze(2).transpose(-1, -2)) * scale + causal2
                    att = torch.softmax(att.float(), dim=-1).to(self.dtype)
                    o = torch.matmul(att, vs[li].unsqueeze(2)).reshape(B, NH, T, HD).transpose(1, 2)
                    o = o.reshape(B, T, NH * HD)
                else:
                    qg = q.reshape(B, NKV, NH // NKV, HD)
                    att = torch.softmax((torch.matmul(qg, ks[li].transpose(-1, -2)) * scale).float(), dim=-1)
                    o = torch.matmul(att.to(self.dtype), vs[li]).reshape(B, 1, NH * HD)
                x = x + a.o_proj(o)
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            logits = self.cp_heads[j](self.cp_norm(x)[:, -1]).float()
            out.append(gumbel_topk(logits, self.sub_top_k, self.sub_temperature, u[:, j]))
        return torch.stack(out, dim=1)  # [B,15]

    # ------------------------------------------------------------ frame step
    def _frame_fn(self, Sb: int, L: int):
        def fn():
            code0, hid, pos = self.s_code0[:Sb], self.s_hid[:Sb], self.s_pos[:Sb]
            step, fidx, act, pres = self.s_step[:Sb], self.s_fidx[:Sb], self.s_active[:Sb], self.s_presence[:Sb]
            u = torch.rand(Sb, self.G, self.noise_k, device=self.dev, dtype=torch.float32)
            rest = self._code_predictor(hid, code0, u[:, 1:, : self.sub_top_k])  # [Sb,15]
            codes = torch.cat([code0.unsqueeze(1), rest], dim=1)  # [Sb,16]
            embs = [self.codec_emb(code0)] + [self.cp_emb[i](rest[:, i]) for i in range(self.G - 1)]
            e = torch.stack(embs, dim=1).sum(1) + self.tts_pad_embed  # [Sb,H]
            hidden = self._talker_layers_decode(e.unsqueeze(1), Sb, L, pos)  # [Sb,1,H]
            logits = self.codec_head(hidden[:, 0]).float()
            new0 = self._sample_talker(logits, pres, step, u[:, 0, : self.top_k])
            self.frames.index_copy_(0, self.slot_iota[:Sb] * self.cap + fidx, codes)
            onehot = (self.vocab_iota[None, :] == new0[:, None]).to(torch.float32) * act[:, None].to(torch.float32)
            pres.copy_(torch.maximum(pres, onehot))
            code0.copy_(torch.where(act > 0, new0, code0))
            hid.copy_(hidden[:, 0])
            pos.add_(act)
            step.add_(act)
            fidx.add_(act)

        return fn

    def frame_graph(self, Sb: int, L: int):
        """(graph, fresh). Capture runs one real frame step, so a fresh graph has advanced every active slot once."""
        g = self.graphs_frame.get((Sb, L))
        if g is not None:
            return g, False
        t = time.time()
        g = self.graphs_frame[(Sb, L)] = Captured(self._frame_fn(Sb, L), self.use_graphs)
        log.info("frame graph slots=%d kv=%d captured in %.1fs", Sb, L, time.time() - t)
        return g, True

    # --------------------------------------------------------------- prefill
    def _prefill_parts(self, P: int):
        parts = self.prefill_parts.get(P)
        if parts is not None:
            return parts
        d = self.dev
        buf = {
            "tid": torch.zeros(1, P, device=d, dtype=torch.long),
            "cid": torch.zeros(1, P, device=d, dtype=torch.long),
            "cm": torch.zeros(1, P, 1, device=d, dtype=self.dtype),
            "widx": torch.zeros(self.NKV * P, device=d, dtype=torch.long),
            "last": torch.zeros(1, device=d, dtype=torch.long),
            "slot": torch.zeros(1, device=d, dtype=torch.long),
            "n": torch.zeros(1, device=d, dtype=torch.long),
        }

        def fn():
            x = self.text_proj(self.text_emb(buf["tid"])) + self.codec_emb(buf["cid"]) * buf["cm"]
            hidden = self._talker_layers_prefill(x, P, buf["widx"])  # [1,P,H]
            last = hidden[0].index_select(0, buf["last"])  # [1,H]
            logits = self.codec_head(last).float()
            zeros = torch.zeros(1, self.V, device=d, dtype=torch.float32)
            u = torch.rand(1, self.top_k, device=d, dtype=torch.float32)
            tok = self._sample_talker(logits, zeros, torch.zeros(1, device=d, dtype=torch.long), u)
            slot = buf["slot"]
            self.s_code0.index_copy_(0, slot, tok)
            self.s_hid.index_copy_(0, slot, last)
            self.s_pos.index_copy_(0, slot, buf["n"])
            self.s_step.index_copy_(0, slot, torch.ones_like(slot))
            self.s_fidx.index_copy_(0, slot, torch.zeros_like(slot))
            self.s_active.index_copy_(0, slot, torch.ones_like(slot))
            self.s_presence.index_copy_(0, slot, (self.vocab_iota[None, :] == tok[:, None]).to(torch.float32))

        parts = self.prefill_parts[P] = (buf, fn)
        return parts

    def run_prefill(self, P: int) -> None:
        """Replay (or capture, which also runs it) the prefill graph of bucket P on whatever its buffers hold."""
        g = self.graphs_prefill.get(P)
        if g is not None:
            g()
            return
        t = time.time()
        mark_step()
        self.graphs_prefill[P] = Captured(self._prefill_parts(P)[1], self.use_graphs)
        log.info("prefill graph len=%d captured in %.1fs", P, time.time() - t)

    def _fill_prefill(self, P: int, slot: int, tid: list, cid: list, cm: list) -> int:
        n = len(tid)
        pad = P - n
        buf, _ = self._prefill_parts(P)
        buf["tid"].copy_(torch.tensor([tid + [self.cfg.tts_pad_token_id] * pad], dtype=torch.long))
        buf["cid"].copy_(torch.tensor([cid + [0] * pad], dtype=torch.long))
        buf["cm"].copy_(torch.tensor(cm + [0] * pad, dtype=torch.float32).view(1, P, 1).to(self.dtype))
        w = ((slot * self.NKV + np.arange(self.NKV)[:, None]) * self.cap + np.arange(P)[None, :]).reshape(-1)
        buf["widx"].copy_(torch.from_numpy(w.astype(np.int64)))
        buf["last"].copy_(torch.tensor([n - 1], dtype=torch.long))
        buf["slot"].copy_(torch.tensor([slot], dtype=torch.long))
        buf["n"].copy_(torch.tensor([n], dtype=torch.long))
        return n

    # --------------------------------------------------------------- decoder
    def decode_graph(self, W: int):
        g = self.graphs_decode.get(W)
        if g is None:
            t = time.time()
            dec = self.m.speech_tokenizer.model.decoder
            buf = torch.zeros(1, self.G, W, device=self.dev, dtype=torch.long)
            mark_step()
            g = self.graphs_decode[W] = (buf, Captured(lambda: dec(buf)[0, 0], self.use_graphs))  # [W*1920]
            log.info("decoder graph window=%d captured in %.1fs", W, time.time() - t)
        return g

    def decode_window(self, codes: np.ndarray, start: int, end: int) -> np.ndarray:
        """Audio for frames [start, end) of `codes` [F,16], decoded with up to dec_left frames of left context."""
        ctx = min(self.dec_left, start)
        n = end - start + ctx
        W = bucket(n, self.dec_windows)
        win = np.zeros((self.G, W), dtype=np.int64)
        win[:, :n] = codes[start - ctx : end].T
        buf, g = self.decode_graph(W)
        buf.copy_(torch.from_numpy(win).unsqueeze(0))
        wav = g()
        # Whole static buffer to the CPU, sliced there: a device-side slice of varying size is a new lazy graph.
        a = wav.float().cpu().numpy()[ctx * SAMPLES_PER_FRAME : n * SAMPLES_PER_FRAME]
        self.stats["decodes"] += 1
        return a

    def max_decode_frames(self) -> int:
        return self.dec_windows[-1] - self.dec_left

    # -------------------------------------------------------------- prompts
    def build_segment(self, text: str, speaker: str, language: str, instruct: str | None) -> dict:
        """The talker prompt for one text segment, as generate_custom_voice builds it: text ids, codec ids and the
        mask saying which positions add a codec embedding."""
        cfg, tc = self.cfg, self.tc
        proc = self.tts.processor

        def ids_of(s):
            ids = proc(text=s, return_tensors="pt")["input_ids"]
            return ids[0].tolist() if ids.dim() == 2 else ids.tolist()

        ids = ids_of(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
        ins = ids_of(f"<|im_start|>user\n{instruct}<|im_end|>\n") if instruct else []
        spk = speaker.lower()
        lang = (language or "auto").lower()
        language_id = None if lang == "auto" else tc.codec_language_id[lang]
        if lang in ("chinese", "auto") and tc.spk_is_dialect.get(spk, False) not in (False, None):
            language_id = tc.codec_language_id[tc.spk_is_dialect[spk]]
        if language_id is None:
            pre = [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]
        else:
            pre = [tc.codec_think_id, tc.codec_think_bos_id, language_id, tc.codec_think_eos_id]
        pre = pre + [tc.spk_id[spk], tc.codec_pad_id]
        tid, cid, cm = [], [], []
        for t in ins + ids[:3]:  # instructions and the role header: text only
            tid.append(t)
            cid.append(0)
            cm.append(0)
        for i, c in enumerate(pre):  # codec control prefix over pad text, the last over tts_bos
            tid.append(cfg.tts_pad_token_id if i < len(pre) - 1 else cfg.tts_bos_token_id)
            cid.append(c)
            cm.append(1)
        for t in ids[3:-5] + [cfg.tts_eos_token_id]:  # the text itself over codec pad
            tid.append(t)
            cid.append(tc.codec_pad_id)
            cm.append(1)
        tid.append(cfg.tts_pad_token_id)
        cid.append(tc.codec_bos_id)
        cm.append(1)
        return {"tid": tid, "cid": cid, "cmask": cm, "text_tokens": len(ids) - 8}

    # ---------------------------------------------------------------- warmup
    def warmup(self) -> None:
        """Capture every graph a request can reach, so no request compiles."""
        t0 = time.time()
        with torch.no_grad():
            for P in self.prefill_buckets:
                pad = [self.cfg.tts_pad_token_id] * 8
                self._fill_prefill(P, 0, pad, [0] * 8, [0] * 8)
                self.run_prefill(P)
            for Sb in self.slot_buckets:
                for L in self.kv_buckets:
                    self.frame_graph(Sb, L)
            for W in self.dec_windows:
                self.decode_graph(W)
            for t in (self.s_code0, self.s_pos, self.s_step, self.s_fidx, self.s_active, self.s_presence):
                t.zero_()
            hpu_sync()
        log.info(
            "warmup: %d prefill, %d frame, %d decoder graphs in %.1fs",
            len(self.graphs_prefill),
            len(self.graphs_frame),
            len(self.graphs_decode),
            time.time() - t0,
        )

    # ------------------------------------------------------------ scheduling
    def submit(self, req: Request) -> None:
        with self.lock:
            req.seg_index = 0
            self.pending.append(self._job_for(req, 0))
            self.lock.notify()

    def _job_for(self, req: Request, i: int) -> Job:
        s = req.segments[i]
        return Job(req=req, tid=s["tid"], cid=s["cid"], cmask=s["cmask"], max_frames=s["max_frames"])

    def start(self) -> None:
        threading.Thread(target=self._run, name="tts-engine", daemon=True).start()

    def _admit(self, job: Job, slot: int) -> None:
        P = bucket(len(job.tid), self.prefill_buckets)
        job.prompt_len = self._fill_prefill(P, slot, job.tid, job.cid, job.cmask)
        self.run_prefill(P)
        job.slot = slot
        # The cache must hold the prompt, every frame and one sync round of overrun.
        job.max_frames = max(2, min(job.max_frames, self.cap - job.prompt_len - self.sync_every - 1))
        self.slots[slot] = job
        self.stats["prefills"] += 1

    def _release(self, slot: int) -> None:
        self.slots[slot] = None
        idx = torch.tensor([slot], dtype=torch.long).to(self.dev)
        z = torch.zeros(1, dtype=torch.long).to(self.dev)
        self.s_active.index_copy_(0, idx, z)
        self.s_pos.index_copy_(0, idx, z)

    def _run(self) -> None:
        try:
            with torch.no_grad():
                self._loop()
        except Exception as exc:  # the device is gone or the loop is broken: fail everything
            log.exception("engine loop died")
            if self.on_fatal:
                self.on_fatal(exc)
            with self.lock:
                jobs = [j for j in self.slots if j] + self.pending
                self.pending.clear()
            for j in jobs:
                j.req.error = f"{type(exc).__name__}: {exc}"
                j.req.push(None)

    def _loop(self) -> None:
        K = self.sync_every
        while not self.stop:
            with self.lock:
                while not self.pending and not any(self.slots) and not self.stop:
                    self.lock.wait()
                new = []
                for i in range(self.max_slots):
                    if self.slots[i] is None and self.pending:
                        new.append((self.pending.pop(0), i))
                        self.slots[i] = new[-1][0]  # reserve
            with self.step_guard():
                self._round(new, K)

    def _round(self, new, K: int) -> None:
        for job, i in new:
            if job.req.cancelled:
                self.slots[i] = None
                job.req.push(None)
                continue
            if len(job.tid) > self.prefill_buckets[-1]:  # the server checks; never kill the loop over it
                self.slots[i] = None
                job.req.error = f"prompt of {len(job.tid)} tokens exceeds {self.prefill_buckets[-1]}"
                job.req.push(None)
                continue
            self._admit(job, i)
        active = [i for i, j in enumerate(self.slots) if j is not None]
        if not active:
            return
        Sb = bucket(max(active) + 1, self.slot_buckets)
        max_pos = max(self.slots[i].prompt_len + len(self.slots[i].frames) for i in active)
        g, fresh = self.frame_graph(Sb, bucket(max_pos + K + 1, self.kv_buckets))
        for _ in range(K - 1 if fresh else K):
            g()
        # One sync: read the K new frames of every active slot.
        rows = [i * self.cap + len(self.slots[i].frames) + k for i in active for k in range(K)]
        idx = torch.tensor(rows, dtype=torch.long).to(self.dev)
        got = self.frames.index_select(0, idx).cpu().numpy().reshape(len(active), K, self.G)
        for n, i in enumerate(active):
            self._take_frames(i, got[n])
        self.stats["frames"] += K * len(active)
        self._emit_audio()

    def _take_frames(self, slot: int, new: np.ndarray) -> None:
        job = self.slots[slot]
        for fr in new:  # frames after end of speech (seen up to K-1 steps late) are dropped
            if job.done:
                break
            if fr[0] == self.EOS:
                job.done = True
                break
            job.frames.append(fr.copy())
            if len(job.frames) >= job.max_frames:
                job.done = True
                job.req.hit_cap = True
        if job.req.cancelled:
            job.done = True

    def _emit_audio(self) -> None:
        for i, job in enumerate(self.slots):
            if job is None:
                continue
            req = job.req
            codes = None
            while not req.cancelled:
                avail = len(job.frames) - job.decoded
                want = self.dec_chunks[min(job.chunks_sent, len(self.dec_chunks) - 1)]
                if avail <= 0 or (avail < want and not job.done):
                    break
                take = min(avail, want if not job.done else self.max_decode_frames())
                if codes is None or len(codes) < job.decoded + take:
                    codes = np.stack(job.frames)
                req.push(self.decode_window(codes, job.decoded, job.decoded + take))
                job.decoded += take
                job.chunks_sent += 1
            if job.done and (job.decoded >= len(job.frames) or req.cancelled):
                req.frames += len(job.frames)
                self._release(i)
                nxt = req.seg_index + 1
                if not req.cancelled and nxt < len(req.segments):
                    req.seg_index = nxt
                    pause = req.pause_s[nxt - 1] if nxt - 1 < len(req.pause_s) else 0.0
                    if pause > 0:
                        req.push(np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32))
                    with self.lock:
                        self.pending.insert(0, self._job_for(req, nxt))  # next segment goes first
                else:
                    req.push(None)

    # ---------------------------------------------------------- convenience
    def synthesize(self, segments: list[dict], pause_s: list[float] | None = None):
        """Blocking generator of float32 audio chunks (startup smoke test, scripts)."""
        req = Request(segments, pause_s or [])
        self.submit(req)
        while (item := req.out.get()) is not None:
            yield item
        if req.error:
            raise RuntimeError(req.error)
