# SPDX-License-Identifier: Apache-2.0
"""Fast, streaming Chatterbox and Chatterbox Turbo on one Gaudi card.

    model = FastChatterbox("turbo")  # or "original"
    for pcm in model.stream("Hello there."):
        ...  # float32 24 kHz chunks

T3 (the speech-token LM), the S3Gen flow and the HiFT vocoder run on the HPU (t3.py, s3gen.py); the text tokenizer,
the reference-voice encoders and the watermark run on the CPU. Device work is not thread-safe: one thread drives a
FastChatterbox (server.py's DeviceWorker). The checkpoints load through the chatterbox-tts package.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import s3gen
import t3

log = logging.getLogger("chatterbox")

SAMPLE_RATE = 24000
SPEECH_VOCAB = 6561  # valid S3 speech tokens are < 6561
S3GEN_SIL = 4299  # the S3Gen silence token (Turbo pads the end with 3 of them)
LOOKAHEAD = 3  # the flow encoder looks 3 tokens ahead
REPOS = {"turbo": "ResembleAI/chatterbox-turbo", "original": "ResembleAI/chatterbox"}


@dataclass
class Voice:
    name: str
    cond_emb: torch.Tensor  # (1, Lcond, C) CPU float32: the T3 conditioning prefix
    ref_tokens: torch.Tensor  # (P,) CPU long: S3Gen prompt tokens
    ref_mel: torch.Tensor  # (80, 2P) CPU float32: S3Gen prompt mel
    spks: torch.Tensor  # (1, 80) device: projected speaker x-vector


class StreamWatermark:
    """Resemble's Perth watermark, applied per vocoder window (the window has left context, and its right edge is
    held back and crossfaded, which also hides the watermark's own edge effects)."""

    def __init__(self, wm):
        self.wm = wm

    def __call__(self, y: np.ndarray) -> np.ndarray:
        out = np.asarray(self.wm.apply_watermark(y, sample_rate=SAMPLE_RATE), dtype=np.float32)
        return out[: len(y)] if len(out) >= len(y) else np.concatenate([out, y[len(out) :]])


class FastChatterbox:
    def __init__(
        self,
        variant: str = "turbo",
        *,
        device: str = "hpu",
        use_graphs: bool = True,
        slots: int = 1,
        check_every: int = 8,
        flow_ctx: int = 25,
        first_chunk: int = 12,
        max_chunk: int = 100,
        watermark: bool = True,
    ):
        """`slots` requests decode together; the flow chunk schedule starts at `first_chunk` tokens (25 per
        second), doubles, and stops growing at `max_chunk`; each chunk sees `flow_ctx` tokens of spoken context."""
        from huggingface_hub import hf_hub_download, snapshot_download

        self.variant = variant
        self.turbo = variant == "turbo"
        self.device = device
        self.check_every = check_every
        self.flow_ctx = flow_ctx
        self.first_chunk = first_chunk
        self.max_chunk = max_chunk
        s3gen.install_sync_free_masks()
        t0 = time.time()
        if self.turbo:
            from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm

            m = ChatterboxTurboTTS.from_local(Path(snapshot_download(REPOS[variant])), "cpu")
        else:
            from chatterbox.tts import ChatterboxTTS, punc_norm

            m = ChatterboxTTS.from_local(Path(hf_hub_download(REPOS[variant], "conds.pt")).parent, "cpu")
        m.device = "cpu"  # the stock model object stays on the CPU, only to enroll reference voices
        self.punc_norm = punc_norm
        self.m = m
        self.tokenizer = m.tokenizer
        self.watermark = StreamWatermark(m.watermarker) if watermark else None
        self.t3 = m.t3.to(device).eval()
        s3 = m.s3gen
        s3.flow.to(device)  # the flow runs on the HPU; the S3 tokenizer and CAMPPlus stay on the CPU
        s3.mel2wav.to("cpu").float()
        self.trim_fade = s3.trim_fade.detach().cpu().float()
        self.vocoder = s3gen.HiftVocoder(s3.mel2wav, device=device, use_graphs=use_graphs)
        flow_batches = tuple(sorted({x for x in (1, 2, 4, 8) if x < slots} | {slots}))
        if self.turbo:  # Turbo: GPT-2 T3 sampled with top-k 1000 and top-p 0.95; the distilled decoder at 2 steps
            self.t3e = t3.T3Engine(
                self.t3,
                slots=slots,
                max_prefill=640,
                device=device,
                use_graphs=use_graphs,
                check_every=check_every,
                top_k=1000,
                top_p=0.95,
                use_min_p=False,
            )
            self.flow = s3gen.FlowRunner(
                s3, device=device, meanflow=True, n_timesteps=2, use_graphs=use_graphs, batch_sizes=flow_batches
            )
        else:  # the original: Llama T3 with CFG and min-p; 10 flow steps with CFG
            self.t3e = t3.T3Engine(
                self.t3, slots=slots, max_prefill=448, device=device, use_graphs=use_graphs, check_every=check_every
            )
            self.flow = s3gen.FlowRunner(
                s3, device=device, meanflow=False, n_timesteps=10, use_graphs=use_graphs, batch_sizes=flow_batches
            )
        self.voices: dict[str, Voice] = {"default": self._voice_from_conds("default", m.conds)}
        log.info("loaded %s in %.1fs", variant, time.time() - t0)

    def free_slot(self) -> int | None:
        """The lowest idle T3 slot (low slots keep the batched step small)."""
        return next((i for i, busy in enumerate(self.t3e.busy) if not busy), None)

    # ------------------------------------------------------------------ voices
    @torch.inference_mode()
    def _voice_from_conds(self, name, conds) -> Voice:
        t3c = conds.t3.to(device=self.device)
        t3c.cond_prompt_speech_emb = None
        cond_emb = self.t3.prepare_conditioning(t3c).float().cpu()
        gen = conds.gen
        ref_tokens = torch.as_tensor(gen["prompt_token"]).view(-1).long().cpu()
        ref_mel = torch.as_tensor(gen["prompt_feat"]).float().cpu()[0].transpose(0, 1).contiguous()
        assert ref_mel.shape[1] == 2 * ref_tokens.numel(), (ref_mel.shape, ref_tokens.shape)
        spks = self.flow.set_speaker(torch.as_tensor(gen["embedding"]).float())
        return Voice(name, cond_emb, ref_tokens, ref_mel, spks)

    @torch.inference_mode()
    def enroll(self, name: str, wav_path: str) -> Voice:
        """A voice from a reference clip (5 to 10 s of clean speech; Turbo needs more than 5 s). The stock
        prepare_conditionals runs the voice encoder, the S3 tokenizer and CAMPPlus, all on the CPU here."""
        if self.turbo:  # Turbo's loudness normalization hands float64 audio to the float32 encoders
            m, orig = self.m, type(self.m).norm_loudness
            m.norm_loudness = lambda wav, sr, target_lufs=-27: np.asarray(orig(m, wav, sr, target_lufs), np.float32)
        self.m.prepare_conditionals(wav_path, exaggeration=0.5)
        self.voices[name] = self._voice_from_conds(name, self.m.conds)
        return self.voices[name]

    # ------------------------------------------------------------------- text
    def text_ids(self, text: str) -> list[int]:
        text = self.punc_norm(text)
        if self.turbo:
            return self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        hp = self.t3.hp
        return [hp.start_text_token] + self.tokenizer.text_to_tokens(text)[0].tolist() + [hp.stop_text_token]

    def prefill_len(self, voice: Voice, n_text_ids: int) -> int:
        return voice.cond_emb.shape[1] + n_text_ids + (1 if self.turbo else 2)

    def _cond_for(self, voice: Voice, exaggeration: float) -> torch.Tensor:
        ce = voice.cond_emb
        if not self.turbo and self.t3.cond_enc.emotion_adv_fc is not None:  # the emotion input is the last row
            ce = ce.clone()
            ce[0, -1] = exaggeration * self.t3.cond_enc.emotion_adv_fc.weight.detach().float().cpu()[:, 0]
        return ce

    # ----------------------------------------------------------------- stream
    def utterance(self, text: str, voice: Voice | None = None, **kw) -> Utterance:
        return Utterance(self, text, voice or self.voices["default"], **kw)

    def stream(self, text: str, voice: Voice | None = None, *, slot: int = 0, **kw):
        """Single-request driver: float32 audio for one text chunk as it is made, using T3 slot `slot` alone."""
        u = self.utterance(text, voice, **kw)
        eng = self.t3e
        eng.start(slot, u.embeds, **u.t3_kw)
        try:
            while not u.finished:
                if eng.decoding[slot]:
                    u.feed(*eng.run_block().get(slot, ([], True)))
                elif not u.t3_finished:
                    u.feed([], True)
                while u.ready():
                    req = u.plan()
                    pcm = u.finish(self.flow.run_many([req])[0] if req is not None else None)
                    if pcm.size:
                        yield pcm
        finally:
            eng.release(slot)

    # ----------------------------------------------------------------- warmup
    @torch.inference_mode()
    def warmup(self):
        """Compile every graph a request can reach, so no request pays a compile: the batched decode graph for each
        slot count, every T3 prefill bucket, every flow bucket for each voice's prompt length and batch size, and
        every vocoder bucket."""
        voices = list(self.voices.values())
        eng = self.t3e
        t0 = time.time()
        for _ in self.stream("Warm up the graphs before serving traffic.", voices[0]):
            pass

        def block(k, embeds):  # occupy slots [0, k) so run_block uses the size-k graph
            for s in range(k):
                eng.start(s, embeds(), max_new=eng.check_every)
            eng.run_block()
            for s in range(k):
                eng.release(s)

        for k in eng.sizes:
            block(k, lambda: eng.build_prefill(voices[0].cond_emb, [1, 2, 3], 0.5))
        t1 = time.time()
        # Each prefill bucket followed by a decode block of each batch size: the graph compiler picks the KV cache's
        # memory layout per producing graph, so a decode graph first replayed after a new prefill bucket would
        # recompile once.
        lo = min(self.prefill_len(v, 3) for v in voices)
        for k in eng.sizes:
            for Lp in range(eng.bucket, eng.max_prefill + 1, eng.bucket):
                if Lp >= lo:
                    block(k, lambda Lp=Lp: torch.randn(eng.r1, Lp, eng.C) * 0.02)
        t2 = time.time()
        C, mc = self.flow_ctx, self.max_chunk
        for P in sorted({int(v.ref_tokens.numel()) for v in voices}):
            mel = torch.zeros(80, 2 * P) - 5
            hi = P + C + mc + LOOKAHEAD + self.check_every + 3
            for TTp in range(self.flow.bucket_of(P + 1), self.flow.bucket_of(hi) + 1, self.flow.bucket):
                for nb in self.flow.batch_sizes:
                    req = dict(
                        tokens=torch.randint(0, SPEECH_VOCAB, (TTp,)),
                        n_prompt=P,
                        prompt_mel=mel,
                        spks=voices[0].spks,
                        n_keep=1,
                    )
                    self.flow.run_many([req] * nb)
        t3_ = time.time()
        hp = torch.zeros(1, self.vocoder.H, 1)
        top = 2 * (mc + LOOKAHEAD + self.check_every + 3) + 16 + 8 + 4  # largest window: chunk mel + ctx + hold
        for Fb in range(self.vocoder.bucket, self.vocoder.bucket_of(top) + 1, self.vocoder.bucket):
            self.vocoder.vocode(torch.zeros(1, 80, Fb) - 5, 0, None, Fb * s3gen.HOP - 1000, hp)
        log.info(
            "%s warm-up: decode %.1fs, prefill buckets %.1fs, flow buckets %.1fs, vocoder %.1fs",
            self.variant,
            t1 - t0,
            t2 - t1,
            t3_ - t2,
            time.time() - t3_,
        )


class Utterance:
    """One text chunk: T3 tokens in (the caller drives the T3 slot and feeds them), streamed audio out. `ready()`
    says a flow chunk can be made; `plan()` returns its flow request (to batch with other requests' chunks in
    FlowRunner.run_many) and `finish(mel)` vocodes it and returns the audio that is now final."""

    def __init__(
        self,
        m: FastChatterbox,
        text: str,
        voice: Voice,
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        seed: int | None = None,
    ):
        self.m = m
        self.voice = voice
        ids = m.text_ids(text)
        self.n_input = len(ids)
        self.embeds = m.t3e.build_prefill(
            m._cond_for(voice, exaggeration), ids, cfg_weight=0.0 if m.turbo else cfg_weight
        )
        self.t3_kw = dict(
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            seed=seed,
        )
        self.gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        self.hift = s3gen.HiftStreamer(m.vocoder, m.vocoder.H, m.trim_fade, generator=self.gen, post=m.watermark)
        self.toks: list[int] = []
        self.done = 0  # generated tokens whose mel is committed
        self.mel_hist = torch.zeros(80, 0)  # committed mel of the last flow_ctx tokens
        self.want = m.first_chunk
        self.t3_finished = False
        self.finished = False

    def feed(self, toks: list[int], finished: bool):
        self.toks += [x for x in toks if x < SPEECH_VOCAB]
        self.t3_finished = self.t3_finished or finished

    def ready(self) -> bool:
        return not self.finished and (len(self.toks) - self.done >= self.want + LOOKAHEAD or self.t3_finished)

    def plan(self) -> dict | None:
        """The next chunk's flow request, or None when only the vocoder's held-back tail is left."""
        m, v, C = self.m, self.voice, self.m.flow_ctx
        toks, done = self.toks, self.done
        if len(toks) - done >= self.want + LOOKAHEAD:
            n_new, final = self.want, False
        else:
            n_new, final = len(toks) - done, True
            if n_new == 0 and not (m.turbo and done > 0):
                self._pending = (0, True, False)
                return None
        ctx = toks[max(0, done - C) : done]
        new = toks[done : done + n_new]
        la = [] if final else toks[done + n_new : done + n_new + LOOKAHEAD]
        if final and m.turbo:
            new = new + [S3GEN_SIL] * 3  # Turbo's own end padding (kept as audio)
        self._pending = (len(new), final, True)
        prompt_tok = torch.cat([v.ref_tokens, torch.tensor(ctx, dtype=torch.long)])
        prompt_mel = (
            torch.cat([v.ref_mel, self.mel_hist[:, self.mel_hist.shape[1] - 2 * len(ctx) :]], 1) if ctx else v.ref_mel
        )
        return dict(
            tokens=torch.cat([prompt_tok, torch.tensor(new + la, dtype=torch.long)]),
            n_prompt=prompt_tok.numel(),
            prompt_mel=prompt_mel,
            spks=v.spks,
            n_keep=len(new),
            generator=self.gen,
        )

    def finish(self, mel: torch.Tensor | None) -> np.ndarray:
        """Take the planned chunk's mel (None for a vocoder-only flush); return the audio that is now final."""
        n_new, final, has_flow = self._pending
        if not has_flow:
            self.finished = True
            return self.hift.push(None, final=True) if self.done > 0 else np.zeros(0, np.float32)
        if self.m.flow_ctx:
            self.mel_hist = torch.cat([self.mel_hist, mel], 1)[:, -2 * self.m.flow_ctx :]
        self.done = min(self.done + n_new, len(self.toks))
        if final:
            self.finished = True
        else:
            self.want = min(self.m.max_chunk, self.want * 2)
        return self.hift.push(mel, final=final)
