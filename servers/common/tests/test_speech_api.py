# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the shared speech surface: `python3 -m pytest servers/common`. Skips what needs fastapi or
prometheus-client when they are not installed."""

import asyncio
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("fastapi")
prom = pytest.importorskip("prometheus_client")

import speech_api as sa  # noqa: E402


# ------------------------------------------------------------------ split_text
def test_short_text_one_chunk():
    assert sa.split_text("Hello there. How are you?", 100) == [("Hello there. How are you?", True)]


def test_sentences_packed_to_limit():
    text = "One two three. Four five six. Seven eight nine."
    assert sa.split_text(text, 30) == [("One two three. Four five six.", False), ("Seven eight nine.", True)]


def test_paragraphs_marked():
    assert sa.split_text("First para.\n\nSecond para.", 100) == [("First para.", True), ("Second para.", True)]


def test_long_sentence_breaks_at_clauses_then_words():
    sentence = "alpha beta gamma, delta epsilon zeta, " + " ".join(["word"] * 40) + "."
    chunks = sa.split_text(sentence, 40)
    assert all(len(c) <= 40 for c, _ in chunks)
    assert " ".join(c for c, _ in chunks).split() == sentence.split()


def test_whitespace_collapsed_and_nothing_lost():
    text = "  A   line\nwrapped here.  Next!\n\n\n Last one?  "
    assert " ".join(c for c, _ in sa.split_text(text, 1000)) == "A line wrapped here. Next! Last one?"


def test_tags_and_quotes_survive():
    text = 'She said "wait." <laugh> Then left.'
    assert " ".join(c for c, _ in sa.split_text(text, 20)).split() == text.split()


def test_cjk_sentences_and_hard_cut():
    text = "今天天气很好。我们去公园散步吧！" + "长" * 50
    chunks = [c for c, _ in sa.split_text(text, 10)]
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text
    assert chunks[:2] == ["今天天气很好。", "我们去公园散步吧！"]


def test_empty():
    assert sa.split_text("   \n\n  ", 100) == []


# --------------------------------------------------------------- formats
def test_wav_header():
    h = sa.wav_header(24000, 100)
    assert len(h) == 44 and h[:4] == b"RIFF" and h[8:16] == b"WAVEfmt "
    assert int.from_bytes(h[4:8], "little") == 136
    assert int.from_bytes(h[24:28], "little") == 24000
    assert sa.wav_header(24000)[40:44] == b"\xff\xff\xff\xff"


def test_to_pcm16_clips():
    pcm = np.frombuffer(sa.to_pcm16(np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32)), dtype="<i2")
    assert pcm.tolist() == [0, 16383, 32767, -32767]


@pytest.mark.parametrize(
    ("fmt", "stream", "stream_format", "mode"),
    [
        ("pcm", None, None, "stream"),
        ("wav", None, None, "whole"),
        ("wav", None, "audio", "stream"),
        ("pcm", True, None, "sse"),
        ("wav", None, "sse", "sse"),
        ("mp3", None, None, "whole"),
        ("aac", False, None, "whole"),
    ],
)
def test_output_mode(fmt, stream, stream_format, mode):
    assert sa.output_mode(fmt, stream, stream_format) == mode


@pytest.mark.parametrize(
    ("fmt", "stream", "stream_format"), [("ogg", None, None), ("mp3", True, None), ("pcm", None, "chunked")]
)
def test_output_mode_refuses(fmt, stream, stream_format):
    with pytest.raises(sa.SpeechError):
        sa.output_mode(fmt, stream, stream_format)


def test_encode_wav_and_flac():
    x = (np.sin(np.arange(24000) / 10) * 0.3).astype(np.float32)
    pcm = sa.to_pcm16(x)
    wav = sa.encode(pcm, "wav", 24000)
    assert wav[:4] == b"RIFF" and wav[44:] == pcm
    sf = pytest.importorskip("soundfile")
    import io

    y, sr = sf.read(io.BytesIO(sa.encode(pcm, "flac", 24000)), dtype="int16")
    assert sr == 24000 and y.tobytes() == pcm


# ------------------------------------------------------------- speech_response
def metrics():
    return sa.SpeechMetrics(prom.CollectorRegistry())


class Producer:
    """An async iterator of PCM chunks that records whether it was closed (how servers free a slot)."""

    def __init__(self, chunks, fail_at=None):
        self.chunks, self.fail_at, self.closed = chunks, fail_at, False

    async def gen(self):
        try:
            for i, c in enumerate(self.chunks):
                if i == self.fail_at:
                    raise RuntimeError("device broke")
                yield c
        finally:
            self.closed = True


def respond(p, mode, fmt, usage=None, **kw):
    return sa.speech_response(
        p.gen(), mode=mode, fmt=fmt, sample_rate=24000, headers={}, usage=usage or {}, metrics=metrics(), **kw
    )


async def drain(resp):
    return b"".join([c if isinstance(c, bytes) else c.encode() async for c in resp.body_iterator])


def test_whole_wav():
    async def run():
        p = Producer([b"\x01\x00" * 10, b"\x02\x00" * 10])
        resp = await respond(p, "whole", "wav")
        return p, resp

    p, resp = asyncio.run(run())
    assert resp.status_code == 200 and resp.body[:4] == b"RIFF" and len(resp.body) == 44 + 40 and p.closed


def test_stream_wav_has_open_header():
    async def run():
        resp = await respond(Producer([b"\x01\x00" * 4, b"\x02\x00" * 4]), "stream", "wav")
        return await drain(resp)

    body = asyncio.run(run())
    assert body[40:44] == b"\xff\xff\xff\xff" and len(body) == 44 + 16


def test_sse_events_and_usage():
    async def run():
        resp = await respond(Producer([b"\x01\x00" * 4]), "sse", "pcm", usage={"input_tokens": 3, "output_tokens": 5})
        return await drain(resp)

    events = [json.loads(line[6:]) for line in asyncio.run(run()).decode().split("\n\n") if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["speech.audio.delta", "speech.audio.done"]
    assert events[1]["usage"] == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}


def test_failure_before_audio_is_a_real_status():
    seen = []

    async def run():
        return await respond(
            Producer([b"x"], fail_at=0), "stream", "pcm", error_status=lambda exc: seen.append(exc) or 503
        )

    resp = asyncio.run(run())
    assert resp.status_code == 503 and seen and "device broke" in json.loads(resp.body)["error"]["message"]


def test_failure_mid_stream_reported_in_band():
    async def run():
        resp = await respond(Producer([b"\x01\x00", b"\x02\x00"], fail_at=1), "sse", "pcm")
        return await drain(resp)

    events = [json.loads(x[6:]) for x in asyncio.run(run()).decode().split("\n\n") if x.startswith("data: ")]
    assert events[-1]["type"] == "speech.audio.error"


def test_hang_up_closes_the_producer():
    async def run():
        p = Producer([b"\x01\x00"] * 10)
        resp = await respond(p, "stream", "pcm")
        it = resp.body_iterator
        await it.__anext__()
        await it.aclose()  # what Starlette does when the client disconnects
        return p

    assert asyncio.run(run()).closed
