# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the EmbeddingGemma 2 server's model-free parts: `python3 -m pytest servers/embeddinggemma-2`."""

import base64

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from server_modules import load  # noqa: E402

engine, media = load(__file__, "engine", "media")


def stock_unfold(hidden, past, future, chunk, context):
    """Gemma4AudioAttention._extract_block_context as transformers 5.20 writes it."""
    hidden = torch.nn.functional.pad(hidden, (0, 0, 0, 0, past, future + chunk - 1))
    return torch.movedim(hidden.unfold(1, context, chunk), -1, 2).contiguous()


@pytest.mark.parametrize("length", [1, 11, 12, 13, 100, 257])
def test_block_context_matches_unfold(length):
    # The checkpoint's audio config: chunk 12, 13 frames of left context, none to the right.
    chunk, past, future = 12, 12, 0
    context = chunk + past + future
    x = torch.randn(2, length, 8, 4)
    assert torch.equal(
        engine.block_context(x, past, future, chunk, context), stock_unfold(x, past, future, chunk, context)
    )


def test_bucket():
    assert engine.bucket(65, engine.LEN_BUCKETS) == 128
    with pytest.raises(ValueError):
        engine.bucket(8193, engine.LEN_BUCKETS)


def test_load_bytes_forms():
    data = b"\x89PNG fake"
    b64 = base64.b64encode(data).decode()
    assert media.load_bytes(b64) == data
    assert media.load_bytes("data:image/png;base64," + b64) == data
    for bad in ("not base64!!", "data:text/plain,hello", ""):
        with pytest.raises(media.MediaError):
            media.load_bytes(bad)


@pytest.mark.parametrize("host", ["127.0.0.1", "169.254.169.254", "10.1.2.3", "192.168.0.1", "::1"])
def test_private_hosts_refused(host):
    with pytest.raises(media.MediaError):
        media._check_host(host)


def test_decode_audio_resamples_to_16k_mono():
    pytest.importorskip("av")
    sr = 44100
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    stereo = np.stack([x, x], 1)
    pcm = (stereo * 32767).astype("<i2").tobytes()
    head = b"RIFF" + (36 + len(pcm)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    head += (1).to_bytes(2, "little") + (2).to_bytes(2, "little") + sr.to_bytes(4, "little")
    head += (4 * sr).to_bytes(4, "little") + (4).to_bytes(2, "little") + (16).to_bytes(2, "little")
    wave = media.decode_audio(head + b"data" + len(pcm).to_bytes(4, "little") + pcm, 10)
    assert wave.dtype == np.float32 and abs(len(wave) - 16000) < 400
    with pytest.raises(media.MediaError):
        media.decode_audio(head + b"data" + len(pcm).to_bytes(4, "little") + pcm, 0.5)


def test_parse_items():
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    (server,) = load(__file__, "server")

    req = server.EmbeddingRequest
    assert server.parse_items(req(input="hi")) == [("text", "hi")]
    assert server.parse_items(req(input=[1, 2, 3])) == [("ids", [1, 2, 3])]
    items = server.parse_items(req(input=["a", [{"type": "image_url", "image_url": {"url": "u"}}, {"text": "t"}]]))
    assert items == [("text", "a"), ("parts", [("image", "u"), ("text", "t")])]
    msgs = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "QQ=="}}, "x"]}]
    assert server.parse_items(req(messages=msgs)) == [("parts", [("audio", "QQ=="), ("text", "x")])]
    for bad in (req(), req(input=[]), req(input="x", messages=msgs), req(input=[[{"type": "pdf"}]])):
        with pytest.raises(server.BadRequest):
            server.parse_items(bad)
