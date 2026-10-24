# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Orpheus stream helpers (no model, no device): `python3 -m pytest servers/orpheus/tests`."""

from server_modules import load

(ss,) = load(__file__, "speech_stream")
CODEBOOK_SIZE, SAMPLES_PER_FRAME = ss.CODEBOOK_SIZE, ss.SAMPLES_PER_FRAME
CodeParser, FrameDecoder, prompt_ids, snac_layers = ss.CodeParser, ss.FrameDecoder, ss.prompt_ids, ss.snac_layers


def token(slot: int, code: int) -> str:
    return f"<custom_token_{10 + slot * CODEBOOK_SIZE + code}>"


def frame_text(codes: list[int]) -> str:
    return " ".join(token(i, c) for i, c in enumerate(codes))


def test_whole_frames():
    p = CodeParser()
    frames = p.feed(frame_text([1, 2, 3, 4, 5, 6, 7]) + frame_text([8, 9, 10, 11, 12, 13, 14]))
    assert frames == [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]
    assert p.tokens == 14


def test_token_split_across_deltas():
    p = CodeParser()
    text = frame_text([4095, 0, 1, 2, 3, 4, 5])
    out = []
    for i in range(0, len(text), 5):
        out += p.feed(text[i : i + 5])
    assert out == [[4095, 0, 1, 2, 3, 4, 5]]


def test_non_audio_tokens_skipped_without_advancing():
    # <custom_token_2> maps to a negative code for slot 3, so it is skipped and slot 3 waits for the next token.
    text = (
        "<custom_token_1>" + frame_text([1, 2, 3]) + " <custom_token_2> " + " ".join(token(i, 9) for i in range(3, 7))
    )
    assert CodeParser().feed(text) == [[1, 2, 3, 9, 9, 9, 9]]


def test_plain_text_ignored():
    assert CodeParser().feed("hello < world") == []


class Decoder:
    """Each frame "decodes" to SAMPLES_PER_FRAME copies of its first code, so the samples show which frames went out."""

    def __init__(self):
        self.calls = []
        self.dec = FrameDecoder(self.decode)

    def decode(self, frames):
        self.calls.append(len(frames))
        return [f[0] for f in frames for _ in range(SAMPLES_PER_FRAME)]

    def run(self, n):
        chunks = [self.dec.push([i, 0, 0, 0, 0, 0, 0]) for i in range(n)] + [self.dec.flush()]
        out = [s for c in chunks if c for s in c]
        return [out[i] for i in range(0, len(out), SAMPLES_PER_FRAME)]


def test_every_frame_once_in_order():
    for n in (1, 2, 3, 4, 5, 9, 30):
        assert Decoder().run(n) == list(range(n)), n


def test_first_audio_after_four_frames():
    d = Decoder().dec
    for i in range(3):  # two frames go out once two more have arrived as right context
        assert d.push([i] * 7) is None
    assert len(d.push([3] * 7)) == 2 * SAMPLES_PER_FRAME


def test_sends_grow_and_decode_with_context():
    d = Decoder()
    sizes = [len(c) // SAMPLES_PER_FRAME for c in (d.dec.push([i] * 7) for i in range(40)) if c]
    assert sizes[:5] == [2, 2, 4, 4, 8]
    assert all(n <= s + 4 for n, s in zip(d.calls, sizes, strict=False)), (d.calls, sizes)  # at most 2 each side


def test_empty_flush():
    assert Decoder().dec.flush() is None


def test_snac_layers():
    c0, c1, c2 = snac_layers([[0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15, 16]])
    assert (c0, c1, c2) == ([0, 10], [1, 4, 11, 14], [2, 3, 5, 6, 12, 13, 15, 16])


def test_prompt_ids():
    assert prompt_ids([128000, 5, 6]) == [128259, 128000, 5, 6, 128009, 128260, 128261, 128257]
