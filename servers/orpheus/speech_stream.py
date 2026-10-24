# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for streaming Orpheus speech: no torch, no numpy, no I/O.

Kept apart from server.py so the logic that decides what the listener hears (which tokens are audio, which samples
go out when) can be unit-tested without a model. The prompt layout and the code-to-layer mapping follow the
reference decoder in https://github.com/canopyai/Orpheus-TTS (Apache-2.0).
"""

import re

# Orpheus' control tokens are Llama 3 added tokens: <custom_token_N> has id 128256 + N.
START_OF_HUMAN = 128259  # <custom_token_3>
END_OF_TEXT = 128009  # <|eot_id|>
END_OF_HUMAN = 128260  # <custom_token_4>
START_OF_AI = 128261  # <custom_token_5>
START_OF_SPEECH = 128257  # <custom_token_1>
END_OF_SPEECH = 128258  # <custom_token_2>

# Audio codes start at <custom_token_10>. Each SNAC frame is 7 codes, one per
# slot, and slot i's codes are offset by i * 4096 so the model can tell them apart.
AUDIO_TOKEN_OFFSET = 10
CODEBOOK_SIZE = 4096
CODES_PER_FRAME = 7
# One SNAC 24 kHz frame decodes to 2048 samples (~85 ms).
SAMPLES_PER_FRAME = 2048

_TOKEN = re.compile(r"<custom_token_(\d+)>")


def prompt_ids(text_ids: list[int]) -> list[int]:
    """Wrap the tokenized "voice: text" (with its BOS) in Orpheus' speaker turn."""
    return [START_OF_HUMAN, *text_ids, END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]


class CodeParser:
    """Turns streamed completion text into SNAC frames.

    vLLM streams the generated tokens back as text, so audio codes arrive as
    "<custom_token_N>" strings, possibly split across deltas. Tokens that are
    not audio codes (a stray control token) are skipped without advancing the
    slot, which is how the reference decoder keeps its place.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._slot = 0
        self._frame: list[int] = []
        self.tokens = 0

    def feed(self, text: str) -> list[list[int]]:
        data = self._pending + text
        frames: list[list[int]] = []
        end = 0
        for match in _TOKEN.finditer(data):
            end = match.end()
            self.tokens += 1
            code = int(match.group(1)) - AUDIO_TOKEN_OFFSET - self._slot * CODEBOOK_SIZE
            if not 0 <= code < CODEBOOK_SIZE:
                continue
            self._frame.append(code)
            self._slot += 1
            if self._slot == CODES_PER_FRAME:
                frames.append(self._frame)
                self._frame, self._slot = [], 0
        # Keep an unfinished "<custom_tok" for the next delta.
        rest = data[end:]
        cut = rest.rfind("<")
        self._pending = rest[cut:] if cut >= 0 and ">" not in rest[cut:] else ""
        return frames


def snac_layers(frames: list[list[int]]) -> tuple[list[int], list[int], list[int]]:
    """Split frames into SNAC's three code layers (12, 23 and 47 Hz)."""
    c0 = [f[0] for f in frames]
    c1 = [x for f in frames for x in (f[1], f[4])]
    c2 = [x for f in frames for x in (f[2], f[3], f[5], f[6])]
    return c0, c1, c2


class FrameDecoder:
    """Emits each frame's audio once, decoded with neighbours for context.

    SNAC decodes better with context on both sides, so each decode covers the
    frames being sent plus up to `left` frames before and `right` after, and
    only the middle goes out. Decoding several frames at once keeps the CPU
    cost per frame low; the first sends are small so the listener hears
    something quickly, and later ones grow once there is audio in hand.
    `flush` sends whatever is left, so every frame is heard exactly once.

    `decode(frames)` returns `len(frames) * SAMPLES_PER_FRAME` samples in any
    sliceable sequence.
    """

    def __init__(self, decode, left: int = 2, right: int = 2, strides: tuple[int, ...] = (2, 2, 4, 4, 8)) -> None:
        self._decode = decode
        self._left, self._right, self._strides = left, right, strides
        self._frames: list[list[int]] = []
        self._sent = 0
        self._sends = 0

    def _stride(self) -> int:
        return self._strides[min(self._sends, len(self._strides) - 1)]

    def _emit(self, end: int, upto: int):
        start = max(0, self._sent - self._left)
        audio = self._decode(self._frames[start:end])
        f = SAMPLES_PER_FRAME
        out = audio[(self._sent - start) * f : (upto - start) * f]
        self._sent = upto
        self._sends += 1
        return out

    def push(self, frame: list[int]):
        self._frames.append(frame)
        upto = self._sent + self._stride()
        end = upto + self._right
        if len(self._frames) < end:
            return None
        return self._emit(end, upto)

    def flush(self):
        n = len(self._frames)
        if self._sent >= n:
            return None
        return self._emit(n, n)

    @property
    def frames(self) -> int:
        return len(self._frames)
