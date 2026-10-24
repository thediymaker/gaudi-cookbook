# SPDX-License-Identifier: Apache-2.0
"""response_format / structured_outputs on Muse: start the grammar at the final answer channel.

The problem (vLLM v0.30): the engine's StructuredOutputManager holds a request's grammar back until its
reasoning parser says reasoning has ended (`is_reasoning_end` / `is_reasoning_end_streaming` on the request's
token ids), then constrains every later token. The vendored MuseReasoningParser answers True only once a TOOL
channel opens: a `to=user` answer is content it surfaces itself, and the serving-side phase machine (reasoning
parser -> tool parser) depends on exactly that. So for a plain answer the grammar never engaged, and
json_schema / json_object / regex / choice requests came back as free text (even on /v1/completions).

The fix, engine side only: the manager gets its own subclass of the Muse parser
(MuseStructuredOutputReasoner) whose `is_reasoning_end` is True when the current assistant segment is the
final answer channel, judged on token ids:

  ... <|start|> "assistant" " to" "=user" <|message|>      -> ended: the grammar applies from the next token
  ... <|start|> "assistant" " to" "=self" <|message|> ...  -> reasoning: not ended
  ... <|start|> "assistant" " to=<tool>" / "=commentary"   -> not ended (tool calls and notes stay free)
  no <|start|> at all (a raw /v1/completions prompt)        -> ended: no channel framing, constrain from token 0

The segment is the ids after the LAST <|start|> token, so earlier turns in the prompt never decide it, and only
the header (before the first <|message|> of the segment) is decoded, so `to=user` inside reasoning text cannot
trigger it. The model may reason first (any effort) or open the answer channel at once; both reach the same
header. The API server keeps the stock MuseReasoningParser (reasoning/content split, tool hand-off unchanged).
Forced tool_choice keeps its own grammar (tool_parser.py, `_grammar_from_parser`: reasoning_ended=True up front,
so the manager never asks the reasoner). With response_format and tools in one request, a tool call is not
constrained; a final answer is.

Second part, the xgrammar backend (vLLM v0.30 XgrammarBackend). Measured once the grammar engaged:
  * <|eot|> inside a JSON string -> "grammar rejected tokens [200008]", the request is terminated (HTTP 500).
    vLLM builds the xgrammar TokenizerInfo with stop_token_ids = [tokenizer.eos_token_id] = [200001] only, so the
    bitmask treats <|eot|> (200008, Muse's end of turn, in generation_config eos_token_id) as plain text "<|eot|>",
    which a JSON string may contain; the matcher's override_stop_tokens does count it as a stop token, and xgrammar
    accepts a stop token only on a complete grammar. Fix: the tokenizer info gets every generation-config eos id.
  * The other framing tokens (<|eom|>, <|start|>, <|message|>, ...) were allowed inside strings as literal text
    too; the content parser would then cut the answer at them. Fix: JSON / regex / grammar / choice requests are
    compiled against a second tokenizer info in which every special token has an empty string (xgrammar never
    matches an empty token as text); stop tokens keep working by id. Structural tags (forced tool_choice,
    tool_parser.py) keep the first info: their `token` elements need the special tokens matchable.
  * A JSON schema whose type the model does not want (a string where it would write a number) could make it pad
    with whitespace until max_tokens at temperature 0 (any_whitespace). Muse compiles JSON with
    any_whitespace=False (single spaces after ':' and ',').
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

logger = logging.getLogger("vllm.muse_spark_hpu")

MUSE_MARKERS = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>")
_ANSWER_HEADER_RE = re.compile(r"^\s*assistant\s*to=user\s*$")
_installed = False


def _special_id(tokenizer, tok: str) -> int | None:
    try:
        i = tokenizer.convert_tokens_to_ids(tok)
    except Exception:  # noqa: BLE001
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    return None if i is None or (unk is not None and i == unk) else int(i)


def answer_started(ids: Sequence[int], tokenizer, start_id: int, message_id: int) -> bool:
    """True if the final answer channel is open: the ids after the last <|start|> are
    'assistant to=user' + <|message|> (+ anything), or there is no <|start|> at all."""
    n = len(ids)
    s = n - 1
    while s >= 0 and ids[s] != start_id:
        s -= 1
    if s < 0:
        return True
    m = s + 1
    while m < n and ids[m] != message_id:
        m += 1
    if m >= n:
        return False  # header not complete yet
    header = tokenizer.decode(list(ids[s + 1 : m]))
    return _ANSWER_HEADER_RE.match(header) is not None


def make_reasoner_class(base):
    class MuseStructuredOutputReasoner(base):
        """Engine-side reasoner for the structured-output gate only (see module docstring)."""

        def __init__(self, tokenizer, *args, **kwargs):
            super().__init__(tokenizer, *args, **kwargs)
            self._so_tok = tokenizer
            self._so_start = _special_id(tokenizer, "<|start|>")
            self._so_message = _special_id(tokenizer, "<|message|>")

        def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
            if self._so_start is None or self._so_message is None:
                return super().is_reasoning_end(input_ids)
            return answer_started(input_ids, self._so_tok, self._so_start, self._so_message)

        def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
            return self.is_reasoning_end(input_ids)

    MuseStructuredOutputReasoner.__name__ = MuseStructuredOutputReasoner.__qualname__ = "MuseStructuredOutputReasoner"
    return MuseStructuredOutputReasoner


def is_muse_tokenizer(tokenizer) -> bool:
    return all(_special_id(tokenizer, m) is not None for m in MUSE_MARKERS)


def muse_stop_ids(tokenizer) -> list[int]:
    """eos of the tokenizer plus every generation-config eos id, and <|eot|> (Muse's end of turn) in any case."""
    ids = {int(tokenizer.eos_token_id)} if getattr(tokenizer, "eos_token_id", None) is not None else set()
    gen = getattr(tokenizer, "name_or_path", None)
    try:
        from transformers import GenerationConfig

        eos = GenerationConfig.from_pretrained(gen).eos_token_id if gen else None
        ids.update([eos] if isinstance(eos, int) else list(eos or []))
    except Exception:  # noqa: BLE001
        pass
    eot = _special_id(tokenizer, "<|eot|>")
    if eot is not None:
        ids.add(eot)
    return sorted(int(i) for i in ids)


def special_ids(tokenizer) -> set[int]:
    out = {i for i in (_special_id(tokenizer, m) for m in MUSE_MARKERS) if i is not None}
    dec = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(dec, dict):
        out.update(int(i) for i, t in dec.items() if getattr(t, "special", False))
    out.update(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
    return out


def tokenizer_infos(xgr, tokenizer, vocab_size: int):
    """(info for structural tags, info for JSON / regex / grammar): both stop on every Muse eos id; the second has
    every special token blanked, so no grammar can spell one out as text."""
    stops = muse_stop_ids(tokenizer)
    base = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size, stop_token_ids=stops)
    sp = special_ids(tokenizer)
    dv = list(base.decoded_vocab)
    text = xgr.TokenizerInfo(
        [b"" if i in sp else dv[i] for i in range(len(dv))],
        xgr.VocabType.RAW,
        vocab_size=vocab_size,
        stop_token_ids=stops,
    )
    return base, text, stops, len(sp)


def _install_xgrammar_fix() -> None:
    try:
        import copy
        import vllm.envs as envs
        from vllm.v1.structured_output import backend_xgrammar as BX
        from vllm.v1.structured_output.backend_types import StructuredOutputOptions as O
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_spark_hpu: xgrammar backend fix not installed (%s)", e)
        return
    Backend = BX.XgrammarBackend
    orig_post, orig_compile = Backend.__post_init__, Backend.compile_grammar

    def __post_init__(self):
        orig_post(self)
        if not is_muse_tokenizer(self.tokenizer):
            return
        xgr = BX.xgr
        base, text, stops, nsp = tokenizer_infos(xgr, self.tokenizer, self.vocab_size)
        kw = dict(max_threads=8, cache_enabled=True, cache_limit_bytes=envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024)
        self.compiler = xgr.GrammarCompiler(base, **kw)
        self._muse_text_compiler = xgr.GrammarCompiler(text, **kw)
        self.disable_any_whitespace = True
        logger.warning(
            "muse_spark_hpu: xgrammar backend: stop tokens %s, %d special tokens never matched as text in "
            "JSON/regex/grammar requests, compact JSON",
            stops,
            nsp,
        )

    def compile_grammar(self, request_type, grammar_spec, stop_token_ids=None):
        text = getattr(self, "_muse_text_compiler", None)
        if text is None or request_type == O.STRUCTURAL_TAG:
            return orig_compile(self, request_type, grammar_spec, stop_token_ids)
        view = copy.copy(self)  # thread-safe: compiles run in a pool; only the view's compiler differs
        view.compiler = text
        return orig_compile(view, request_type, grammar_spec, stop_token_ids)

    Backend.__post_init__ = __post_init__
    Backend.compile_grammar = compile_grammar


def install() -> None:
    """Give vLLM's StructuredOutputManager the answer-channel gate when it is set up with the Muse parser, and the
    xgrammar backend the Muse stop / special-token handling."""
    global _installed
    if _installed:
        return
    _install_xgrammar_fix()
    try:
        from vllm.v1.structured_output import StructuredOutputManager
    except Exception as e:  # noqa: BLE001
        logger.warning("muse_spark_hpu: structured-output gate not installed (%s)", e)
        return
    _installed = True
    orig_init = StructuredOutputManager.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        cls = getattr(self, "reasoner_cls", None)
        if cls is not None and cls.__name__ == "MuseReasoningParser":
            self.reasoner_cls = make_reasoner_class(cls)
            logger.warning("muse_spark_hpu: structured outputs start at the final answer channel")

    StructuredOutputManager.__init__ = __init__
