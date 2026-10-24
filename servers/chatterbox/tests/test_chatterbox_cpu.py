# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Chatterbox pieces that need no checkpoint and no device: `python3 -m pytest servers/chatterbox`.

* the convolution STFT and inverse STFT against torch.stft / torch.istft (HiFT's n_fft 16, hop 4, periodic Hann);
* the threshold form of top-p against the sort-and-scatter definition, and the exponential race as a draw;
* the streaming vocoder bookkeeping (context, held-back frames, crossfade) with a stand-in vocoder: every sample is
  emitted exactly once and matches a whole-clip rendering, whatever the chunk sizes.
"""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from server_modules import load  # noqa: E402

s3gen, t3 = load(__file__, "s3gen", "t3")


def test_conv_stft_matches_torch():
    window = torch.hann_window(16, periodic=True)
    c = s3gen.ConvSTFT(window, 16, 4, "cpu")
    x = torch.randn(1, 1, 4 * 300, dtype=torch.float32)
    ref = torch.stft(x[0, 0], 16, 4, window=window, center=True, pad_mode="reflect", return_complex=True)
    got = c.stft(x)
    assert torch.allclose(got[0, :9], ref.real, atol=1e-5) and torch.allclose(got[0, 9:], ref.imag, atol=1e-5)
    back = c.istft(torch.cat([ref.real, ref.imag], 0)[None])[0]
    want = torch.istft(ref, 16, 4, window=window, center=True)
    assert back.shape == want.shape and torch.allclose(back, want, atol=1e-5)


def test_top_p_threshold_matches_the_definition():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(6, 50, generator=g) * 3
    for top_p in (0.5, 0.9, 0.95):
        vals = torch.sort(logits, dim=-1, descending=True).values
        got = t3.top_p_threshold(logits, vals, top_p) > t3.NEG / 2
        # Reference (Hugging Face TopPLogitsWarper): drop the tail whose cumulative mass, ascending, is <= 1 - p.
        s, idx = torch.sort(logits, dim=-1, descending=False)
        drop_sorted = torch.cumsum(torch.softmax(s, -1), -1) <= (1 - top_p)
        drop_sorted[..., -1] = False
        drop = drop_sorted.scatter(1, idx, drop_sorted)
        assert torch.equal(got, ~drop), top_p


def test_top_k_threshold():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    masked, vals = t3.top_k_threshold(logits, 2)
    assert (masked > t3.NEG / 2).tolist() == [[False, True, False, True, False]] and vals.tolist() == [[5.0, 4.0]]


def test_exponential_race_is_an_exact_draw():
    g = torch.Generator().manual_seed(2)
    p = torch.softmax(torch.randn(10, generator=g), -1)
    n = 200_000
    e = -torch.log(torch.rand(n, 10, generator=g).clamp_min(1e-20))
    got = torch.bincount(t3.exponential_race(p.expand(n, -1), e), minlength=10).float() / n
    assert 0.5 * (got - p).abs().sum() < 0.01


class LinearVocoder:
    """Stands in for HiftVocoder: frame f of the clip renders as a ramp unique to f, independent of the window, so
    any correct streaming reproduces the whole-clip rendering."""

    H = 9

    def vocode(self, mel_w, a, anchor, anchor_at, harm_phase):
        frames = mel_w[0, 0].numpy()  # channel 0 carries the absolute frame index
        y = np.concatenate([f + np.linspace(0, 0.5, s3gen.HOP, endpoint=False) for f in frames]).astype(np.float32)
        return y, ((anchor_at, torch.zeros(1, self.H, 1)) if anchor_at is not None else None)


@pytest.mark.parametrize("chunks", [[200], [24, 48, 96, 32], [1] * 60 + [40], [8, 8, 8, 8, 8]])
def test_streamer_emits_every_sample_once(chunks):
    voc = LinearVocoder()
    st = s3gen.HiftStreamer(voc, voc.H, None)
    total = sum(chunks)
    mel = torch.zeros(80, total)
    mel[0] = torch.arange(total, dtype=torch.float32)
    out, at = [], 0
    for i, n in enumerate(chunks):
        out.append(st.push(mel[:, at : at + n], final=i == len(chunks) - 1))
        at += n
    y = np.concatenate(out)
    want, _ = voc.vocode(mel[None], 0, None, None, None)
    assert len(y) == total * s3gen.HOP
    assert np.max(np.abs(y - want)) < 2e-3 * (total + 1)  # crossfade weights sum to 1 within about 0.1%


def test_sync_free_mask_full_context_is_the_padding_mask():
    xs = torch.zeros(2, 7, 4)
    masks = torch.ones(2, 1, 7, dtype=torch.bool)
    assert s3gen._sync_free_chunk_mask(xs, masks, False, False, 0, 0, -1) is masks
    chunked = s3gen._sync_free_chunk_mask(xs, masks, False, False, 0, 3, -1)
    assert chunked.shape == (2, 7, 7) and bool(chunked[0, 0, 2]) and not bool(chunked[0, 0, 3])
