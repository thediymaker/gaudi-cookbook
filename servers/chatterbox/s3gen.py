# SPDX-License-Identifier: Apache-2.0
"""Chunked S3Gen for streaming: speech tokens -> mel (flow) and mel -> 24 kHz audio (HiFT), both on the HPU.

Flow. Each chunk runs the flow on

    [reference prompt tokens | context tokens | new tokens | lookahead tokens | pad]

with the reference mel AND the mel already generated for the context tokens as the conditioning prefix, so the new
mel continues the audio already sent instead of starting a fresh utterance. Only the new tokens' mel is kept; the
lookahead tokens (the encoder looks 3 tokens ahead) are generated again with full context in the next chunk. Inputs
are padded to `bucket`-token sizes and every shape-dependent tensor (masks, conditioning, noise) is built on the
CPU, so each bucket is one static graph. The encoder and ONE ODE step are captured per (bucket, batch size) and the
step graph is replayed n times: all 10 steps as one graph compiled for more than 15 minutes.

HiFT. Each push vocodes the new mel plus `ctx` frames of left context and holds back the last `hold` frames (they
lack right context) plus a crossfade window, which the next push vocodes again and crossfades. HiFT's receptive
field is roughly 15 mel frames each way: with 16 frames of left context and 8 held back, audio near the seams
matches whole-clip vocoding as closely as the rest does (about 40 dB SNR at seams, 37 dB elsewhere; with 4 held
back the seams measured 23 dB). The NSF sine source keeps its phase across pushes (re-anchored at the crossfade
start, with per-request harmonic phases), so the two renderings of the overlap agree.

The vocoder re-implements HiFTGenerator.decode and its sine source from the chatterbox-tts package (MIT,
https://github.com/resemble-ai/chatterbox), which adapts CosyVoice's HiFT generator (Apache-2.0,
https://github.com/FunAudioLLM/CosyVoice), with the STFT and inverse STFT as convolutions.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.core as htcore
except ImportError:  # CPU (tests)
    ht = None
    htcore = None

log = logging.getLogger("chatterbox")
HOP = 480  # samples per mel frame at 24 kHz


def _mark_step(device):
    if htcore is not None and device.type == "hpu":
        htcore.mark_step()


def _capture_or_replay(graphs: dict, key, fn, use_graphs: bool, device):
    """Run fn() through the HPU graph for `key`, capturing it on first use (capture also runs it once)."""
    if not use_graphs:
        fn()
        _mark_step(device)
        return
    g = graphs.get(key)
    if g is None:
        log.info("capturing graph %s", key)
        g = graphs[key] = ht.hpu.HPUGraph()
        with ht.hpu.stream(ht.hpu.Stream()):
            g.capture_begin()
            fn()
            g.capture_end()
    else:
        g.replay()


def _sync_free_chunk_mask(
    xs,
    masks,
    use_dynamic_chunk,
    use_dynamic_left_chunk,
    decoding_chunk_size,
    static_chunk_size,
    num_decoding_left_chunks,
    enable_full_context=True,
):
    """Inference-only add_optional_chunk_mask. S3Gen always uses full context (static_chunk_size 0, no dynamic
    chunks), where the stock function returns the padding mask after a sanity check that reads a tensor back to
    the host, about 14 times per estimator call. The static-chunk branch is kept, minus the check."""
    if static_chunk_size > 0 and not use_dynamic_chunk:
        pos = torch.arange(xs.size(1), device=xs.device)
        block = (torch.div(pos, static_chunk_size, rounding_mode="trunc") + 1) * static_chunk_size
        return masks & (pos.unsqueeze(0) < block.unsqueeze(1)).unsqueeze(0)
    return masks


def install_sync_free_masks():
    import chatterbox.models.s3gen.decoder as dec
    import chatterbox.models.s3gen.transformer.upsample_encoder as enc

    dec.add_optional_chunk_mask = _sync_free_chunk_mask
    enc.add_optional_chunk_mask = _sync_free_chunk_mask


class FlowRunner:
    """Speech tokens -> mel for chunks of one or more requests: one encoder graph and one ODE-step graph per
    (token bucket, batch size)."""

    def __init__(
        self,
        s3gen,
        *,
        device="hpu",
        meanflow=False,
        n_timesteps=10,
        bucket=25,
        use_graphs=True,
        cfg_rate=0.7,
        batch_sizes=(1, 2, 4),
    ):
        self.flow = s3gen.flow
        self.est = s3gen.flow.decoder.estimator
        self.device = torch.device(device)
        self.meanflow = meanflow  # Turbo's distilled decoder: no CFG, takes the step end `r` as an input
        self.n = n_timesteps
        self.bucket = bucket
        self.cfg_rate = cfg_rate
        self.batch_sizes = tuple(sorted(batch_sizes))
        self.dtype = self.est.dtype
        self.use_graphs = use_graphs and ht is not None and self.device.type == "hpu"
        self.bufs = {}
        self.graphs = {}
        # The relative-position encoders cache their tables as plain attributes and move them to the input's device
        # on first use; do it now so no graph capture records the copy.
        for mod in self.flow.encoder.modules():
            if getattr(mod, "pe", None) is not None and torch.is_tensor(mod.pe):
                mod.pe = mod.pe.to(self.device, self.dtype)

    @torch.inference_mode()
    def set_speaker(self, embedding):
        """Speaker x-vector (1, 192) -> projected conditioning (1, 80) on the device."""
        e = F.normalize(torch.atleast_2d(embedding).to(self.device, torch.float32), dim=1)
        return self.flow.spk_embed_affine_layer(e).to(self.dtype)

    def _buffers(self, TTp, B):
        b = self.bufs.get((TTp, B))
        if b is None:
            d, dt = self.device, self.dtype
            T = 2 * TTp  # two mel frames per token
            rows = B if self.meanflow else 2 * B  # CFG: [conditioned rows | unconditioned rows]
            ts = torch.linspace(0, 1, self.n + 1, dtype=torch.float64)
            if not self.meanflow:
                ts = 1 - torch.cos(ts * 0.5 * math.pi)  # the stock cosine schedule
            b = self.bufs[(TTp, B)] = {
                "tok": torch.zeros(B, TTp, dtype=torch.long, device=d),
                "len": torch.ones(B, dtype=torch.long, device=d),
                "cond": torch.zeros(rows, 80, T, dtype=dt, device=d),  # unconditioned rows stay 0
                "spks": torch.zeros(rows, 80, dtype=dt, device=d),
                "mu": torch.zeros(rows, 80, T, dtype=dt, device=d),
                "mask": torch.zeros(rows, 1, T, dtype=dt, device=d),
                "x": torch.zeros(B, 80, T, dtype=dt, device=d),
                "i": torch.zeros(1, dtype=torch.long, device=d),
                "ts": ts.to(dt).to(d),
            }
        return b

    def _encode_fn(self, b):
        """Tokens -> mu (encoder output) and the mel mask, written into b."""
        tok, tok_len = b["tok"], b["len"]
        B, TTp = tok.shape
        d = self.device
        tmask = (torch.arange(TTp, device=d).view(1, TTp) < tok_len.view(B, 1)).to(self.dtype)
        h, _ = self.flow.encoder(self.flow.input_embedding(tok) * tmask.unsqueeze(-1), tok_len)
        b["mu"][:B].copy_(self.flow.encoder_proj(h).transpose(1, 2))  # (B, 80, 2TTp)
        T = 2 * TTp
        mask = (torch.arange(T, device=d).view(1, 1, T) < (2 * tok_len).view(B, 1, 1)).to(self.dtype)
        b["mask"].copy_(mask if self.meanflow else torch.cat([mask, mask], 0))
        b["i"].zero_()

    def _step_fn(self, b):
        """One Euler step of the flow ODE on b["x"] (with CFG for the original model)."""
        i = b["i"]
        t = b["ts"].index_select(0, i)
        r = b["ts"].index_select(0, i + 1)
        x = b["x"]
        B = x.shape[0]
        if self.meanflow:
            dxdt = self.est(x, mask=b["mask"], mu=b["mu"], t=t.expand(B), spks=b["spks"], cond=b["cond"], r=r.expand(B))
        else:
            d2 = self.est(
                torch.cat([x, x], 0),
                mask=b["mask"],
                mu=b["mu"],
                t=t.expand(2 * B),
                spks=b["spks"],
                cond=b["cond"],
                r=None,
            )
            dxdt = (1.0 + self.cfg_rate) * d2[:B] - self.cfg_rate * d2[B:]
        b["x"].copy_(x + (r - t) * dxdt)
        b["i"].add_(1)

    def bucket_of(self, n_tokens):
        return ((n_tokens + self.bucket - 1) // self.bucket) * self.bucket

    @torch.inference_mode()
    def run_many(self, reqs: list[dict]) -> list[torch.Tensor]:
        """Chunks from several requests in one batched pass, padded to the largest bucket and to a batch size in
        `batch_sizes`. Each request: tokens (1-D long CPU, prompt + generated), n_prompt, prompt_mel (80, 2*n_prompt)
        CPU, spks (1, 80) device, n_keep, generator. Returns each request's mel (80, 2*n_keep)."""
        n = len(reqs)
        B = next(s for s in self.batch_sizes if s >= n)
        TTp = max(self.bucket_of(int(r["tokens"].numel())) for r in reqs)
        b = self._buffers(TTp, B)
        tok = torch.zeros(B, TTp, dtype=torch.long)
        lens = torch.ones(B, dtype=torch.long)
        cond = torch.zeros(B, 80, 2 * TTp, dtype=torch.float32)
        z = torch.zeros(B, 80, 2 * TTp, dtype=torch.float32)
        for i, r in enumerate(reqs):
            TT = int(r["tokens"].numel())
            tok[i, :TT] = r["tokens"]
            lens[i] = TT
            cond[i, :, : 2 * r["n_prompt"]] = r["prompt_mel"]
            z[i] = torch.randn(80, 2 * TTp, generator=r.get("generator"))
        b["tok"].copy_(tok.to(self.device))
        b["len"].copy_(lens.to(self.device))
        b["cond"][:B].copy_(cond.to(self.dtype).to(self.device))
        b["x"].copy_(z.to(self.dtype).to(self.device))
        b["spks"][:B].copy_(torch.cat([r["spks"] for r in reqs] + [reqs[0]["spks"]] * (B - n), 0))
        _mark_step(self.device)  # the input copies as their own small graph, the same every time
        _capture_or_replay(self.graphs, ("enc", TTp, B), lambda: self._encode_fn(b), self.use_graphs, self.device)
        for _ in range(self.n):
            _capture_or_replay(self.graphs, ("step", TTp, B), lambda: self._step_fn(b), self.use_graphs, self.device)
        # Read the whole static buffer and slice on the CPU: a device-side slice of varying size would be a new
        # lazy-mode graph (and a compile) per size.
        x = b["x"].cpu().float()
        return [x[i, :, 2 * r["n_prompt"] : 2 * (r["n_prompt"] + r["n_keep"])] for i, r in enumerate(reqs)]


class ConvSTFT:
    """torch.stft / torch.istft for HiFT's tiny transform (n_fft 16, hop 4, periodic Hann) as fixed convolutions.
    They need complex dtypes the HPU lacks; with n_fft 16 the STFT is a strided conv1d with windowed DFT kernels and
    the inverse a conv_transpose1d with the windowed inverse kernels, divided by the summed squared window."""

    def __init__(self, window: torch.Tensor, n_fft: int, hop: int, device):
        self.n_fft, self.hop = n_fft, hop
        w = window.detach().cpu().double()
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        ang = 2 * math.pi * k[:, None] * n[None, :] / n_fft  # (bins, n_fft)
        c = torch.full((n_fft // 2 + 1, 1), 2.0, dtype=torch.float64)
        c[0] = c[-1] = 1.0  # DC and Nyquist appear once in a one-sided spectrum
        fwd = torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w], 0)
        inv = torch.cat([c * torch.cos(ang) * w, -c * torch.sin(ang) * w], 0) / n_fft
        self.k_fwd = fwd.float().unsqueeze(1).to(device)  # (2*bins, 1, n_fft): real rows, then imaginary
        self.k_inv = inv.float().unsqueeze(1).to(device)
        self.w2 = (w**2).float().view(1, 1, n_fft).to(device)

    def stft(self, x):
        """(1, 1, N) -> (1, 2*bins, 1 + N/hop), like torch.stft(center=True, pad_mode='reflect') as real, imag."""
        x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        return F.conv1d(x, self.k_fwd, stride=self.hop)

    def istft(self, spec):
        """(1, 2*bins, T) -> (1, hop*(T-1)), like torch.istft(center=True)."""
        T = spec.shape[-1]
        y = F.conv_transpose1d(spec, self.k_inv, stride=self.hop)
        env = F.conv_transpose1d(torch.ones(1, 1, T, device=spec.device), self.w2, stride=self.hop)
        y = y / torch.clamp(env, min=1e-11)
        p = self.n_fft // 2
        return y[:, 0, p : y.shape[-1] - p]


def _remove_weight_norms(module):
    from torch.nn.utils import parametrize

    for m in [m for m in module.modules() if parametrize.is_parametrized(m)]:
        for name in list(m.parametrizations.keys()):
            parametrize.remove_parametrizations(m, name, leave_parametrized=True)


class HiftVocoder:
    """The HiFT vocoder on the device: a weight-norm-folded copy of S3Gen's HiFT, the STFTs as convolutions
    (ConvSTFT), the NSF sine source built in fp32 with its phase re-anchored at every push. Windows are padded with
    silence to `bucket`-frame sizes; one HPU graph per bucket."""

    def __init__(self, hift_cpu, *, device="hpu", bucket=16, use_graphs=True, pad_value=-11.5129):
        from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
        from chatterbox.models.s3gen.hifigan import HiFTGenerator

        self.device = torch.device(device)
        # A fresh copy (deepcopy of weight-norm-parametrized modules shares state with the original), with the same
        # configuration as S3Token2Wav's.
        h = HiFTGenerator(
            sampling_rate=hift_cpu.sampling_rate,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=ConvRNNF0Predictor(),
        )
        h.load_state_dict(hift_cpu.state_dict())
        h = h.float()
        _remove_weight_norms(h)
        self.h = h.to(self.device).eval()
        self.bucket = bucket
        self.pad_value = pad_value  # log-mel of silence
        self.use_graphs = use_graphs and ht is not None and self.device.type == "hpu"
        self.stft = ConvSTFT(h.stft_window, h.istft_params["n_fft"], h.istft_params["hop_len"], self.device)
        sg = h.m_source.l_sin_gen
        self.H = sg.harmonic_num + 1
        self.mult = torch.arange(1, self.H + 1, dtype=torch.float32, device=self.device).view(1, self.H, 1)
        self.sr = float(sg.sampling_rate)
        self.sine_amp = float(sg.sine_amp)
        self.noise_std = float(sg.noise_std)
        self.vth = float(sg.voiced_threshold)
        self.bufs = {}
        self.graphs = {}

    def bucket_of(self, frames):
        return ((frames + self.bucket - 1) // self.bucket) * self.bucket

    def _decode(self, x, s):
        """HiFTGenerator.decode with the convolution STFTs."""
        h = self.h
        s_stft = self.stft.stft(s)
        x = h.conv_pre(x)
        for i in range(h.num_upsamples):
            x = h.ups[i](F.leaky_relu(x, h.lrelu_slope))
            if i == h.num_upsamples - 1:
                x = h.reflection_pad(x)
            x = x + h.source_resblocks[i](h.source_downs[i](s_stft))
            xs = None
            for j in range(h.num_kernels):
                r = h.resblocks[i * h.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / h.num_kernels
        x = h.conv_post(F.leaky_relu(x))
        nb = self.stft.n_fft // 2 + 1
        mag = torch.clamp(torch.exp(x[:, :nb, :]), max=1e2)
        ph = torch.sin(x[:, nb:, :])
        y = self.stft.istft(torch.cat([mag * torch.cos(ph), mag * torch.sin(ph)], 1))
        return torch.clamp(y, -h.audio_limit, h.audio_limit)

    def _fn(self, b):
        h = self.h
        mel = b["mel"]
        f0s = torch.repeat_interleave(h.f0_predictor(mel), HOP, dim=-1).unsqueeze(1)  # (1, 1, Ns)
        cum = torch.cumsum(f0s * self.mult / self.sr, dim=-1)  # (1, H, Ns) phase in cycles
        cum = cum + b["use"] * (b["aval"] - cum.index_select(2, b["aidx"]))  # continue the previous push's phase
        b["nval"].copy_(torch.remainder(cum.index_select(2, b["nidx"]), 1.0))  # anchor for the next push
        sine = self.sine_amp * torch.sin(2 * math.pi * torch.remainder(cum, 1.0) + b["hp"])
        uv = (f0s > self.vth).float()
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        sine = sine * uv + noise_amp * torch.randn_like(sine)
        src = h.m_source.l_tanh(h.m_source.l_linear(sine.transpose(1, 2))).transpose(1, 2)
        b["y"].copy_(self._decode(mel, src))

    def _buffers(self, Fb):
        b = self.bufs.get(Fb)
        if b is None:
            d = self.device
            b = self.bufs[Fb] = {
                "mel": torch.zeros(1, 80, Fb, device=d),
                "aidx": torch.zeros(1, dtype=torch.long, device=d),
                "aval": torch.zeros(1, self.H, 1, device=d),
                "use": torch.zeros(1, 1, 1, device=d),
                "nidx": torch.zeros(1, dtype=torch.long, device=d),
                "hp": torch.zeros(1, self.H, 1, device=d),
                "nval": torch.zeros(1, self.H, 1, device=d),
                "y": torch.zeros(1, Fb * HOP, device=d),
            }
        return b

    @torch.inference_mode()
    def vocode(self, mel_w, a, anchor, anchor_at, harm_phase):
        """mel_w (1, 80, Fw) CPU starting at absolute frame a. `anchor` (absolute sample, phase) continues the sine
        source; `anchor_at` asks for the phase at that sample for the next push. Returns (audio, new anchor)."""
        Fw = mel_w.shape[-1]
        Fb = self.bucket_of(Fw)
        b = self._buffers(Fb)
        mel = torch.full((1, 80, Fb), self.pad_value)
        mel[..., :Fw] = mel_w
        b["mel"].copy_(mel.to(self.device))
        off = a * HOP
        use = anchor is not None and 0 <= anchor[0] - off < Fw * HOP
        ni = (anchor_at - off) if anchor_at is not None else 0
        b["aidx"].copy_(torch.tensor([anchor[0] - off if use else 0]).to(self.device))
        b["aval"].copy_((anchor[1].float() if use else torch.zeros(1, self.H, 1)).to(self.device))
        b["use"].copy_(torch.full((1, 1, 1), 1.0 if use else 0.0).to(self.device))
        b["nidx"].copy_(torch.tensor([min(max(ni, 0), Fw * HOP - 1)]).to(self.device))
        b["hp"].copy_(harm_phase.float().to(self.device))
        _mark_step(self.device)
        _capture_or_replay(self.graphs, Fb, lambda: self._fn(b), self.use_graphs, self.device)
        y = b["y"].cpu()[0, : Fw * HOP].numpy()  # whole static buffer to the CPU, sliced there
        return y, ((anchor_at, b["nval"].cpu()) if anchor_at is not None else None)


class HiftStreamer:
    """Streaming mel -> 24 kHz audio for one utterance: left context, held-back frames and a crossfade."""

    def __init__(
        self,
        vocoder,
        harmonics: int,
        trim_fade: torch.Tensor | None,
        *,
        ctx=16,
        hold=8,
        xfade=960,
        generator: torch.Generator | None = None,
        post=None,
    ):
        self.vocoder = vocoder  # anything with HiftVocoder.vocode's signature
        self.post = post  # optional per-window audio transform (the watermark)
        self.ctx, self.hold, self.W = ctx, hold, xfade
        self.trim_fade = trim_fade.float().cpu().numpy() if trim_fade is not None else None
        self.mel = torch.zeros(80, 0)
        self.mel_off = 0  # absolute frame index of self.mel[:, 0]
        self.sent = 0  # samples emitted
        self.tail = None  # pending overlap (W samples starting at self.sent)
        self.anchor = None  # (absolute sample, phase (1, H, 1))
        ph = (torch.rand(1, harmonics, 1, generator=generator, dtype=torch.float64) * 2 - 1) * math.pi
        ph[:, 0] = 0
        self.harm_phase = ph
        w = np.hanning(2 * self.W + 2)[1:-1].astype(np.float32)
        self.fade_in, self.fade_out = w[: self.W], w[self.W :]

    def push(self, mel_new: torch.Tensor | None, final: bool = False) -> np.ndarray:
        """Add committed mel frames (80, n); return the audio that is now final."""
        if mel_new is not None and mel_new.shape[-1]:
            self.mel = torch.cat([self.mel, mel_new.float()], dim=1)
        f_end = self.mel_off + self.mel.shape[1]
        W = self.W
        cur = self.sent + (W if self.tail is not None else 0)
        X = HOP * (f_end - self.hold) - W  # where the next pending overlap starts
        if not final and cur >= X:
            return np.zeros(0, dtype=np.float32)  # wait for more mel
        a = max(self.mel_off, self.sent // HOP - self.ctx)
        y, new_anchor = self.vocoder.vocode(
            self.mel[:, a - self.mel_off :][None], a, self.anchor, None if final else X, self.harm_phase
        )
        if self.post is not None:
            y = self.post(y)
        off = a * HOP
        if a == 0 and self.trim_fade is not None:  # the stock fade-in over the first samples
            n = min(len(self.trim_fade), len(y))
            y[:n] *= self.trim_fade[:n]
        parts = []
        if self.tail is not None:
            parts.append(self.tail * self.fade_out + y[self.sent - off : self.sent - off + W] * self.fade_in)
        if final:
            parts.append(y[cur - off :])
            self.sent = off + len(y)
            self.tail = None
        else:
            parts.append(y[cur - off : X - off])
            self.tail = y[X - off : X - off + W].copy()
            self.sent = X
            self.anchor = new_anchor
        keep_from = max(self.mel_off, self.sent // HOP - self.ctx - 1)  # keep only what the next window can reach
        if keep_from > self.mel_off:
            self.mel = self.mel[:, keep_from - self.mel_off :]
            self.mel_off = keep_from
        return (np.concatenate(parts) if parts else np.zeros(0)).astype(np.float32, copy=False)
