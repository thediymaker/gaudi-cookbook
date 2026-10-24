#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gemma4-bos-tokenizer: build a Gemma 4 instruct tokenizer that adds <bos> to raw prompts.

This is a startup step, not a source patch. Gemma 4 instruct checkpoints ship a tokenizer.json
whose post-processor is "$A" (no <bos>) and no add_bos_token, because the chat template writes a
literal <bos> itself. Raw prompts sent to /v1/completions therefore reach the model without the
<bos> every Gemma sequence starts with, and the output degenerates ("LLLL...", or the prompt
repeated). Chat requests are unaffected.

The step writes OUT_DIR: every tokenizer, config and template file of the model's snapshot as a
symlink, except tokenizer.json (post-processor "<bos> $A" / "<bos> $A <bos>:1 $B:1", as in the
base checkpoint) and tokenizer_config.json ("add_bos_token": true). Serve with
`--tokenizer OUT_DIR`; weights, config and the processor still load from `--model`.

Before exiting, the result is checked with the installed transformers: a raw prompt gains exactly
one leading <bos>, add_special_tokens=False is unchanged, the vocabulary and chat template are
identical, and a rendered chat prompt tokenizes to the same ids as with the stock tokenizer. Any
failure exits 2 and leaves no usable directory, so the server cannot start with a half-built
tokenizer. OUT_DIR is rebuilt from scratch on every run.

Usage: apply.py MODEL_ID_OR_DIR OUT_DIR
Works offline when the snapshot is in the Hugging Face cache; otherwise only the small tokenizer
and config files are downloaded.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys

FIX_ID = "gemma4-bos-tokenizer"
MARKER = f"[gaudi-fix:{FIX_ID}]"
KIND = "startup"

SMALL = ["*.json", "*.jinja", "*.txt", "*.model", "tokenizer*"]
REWRITTEN = {"tokenizer.json", "tokenizer_config.json"}


class Failure(Exception):
    """The tokenizer could not be built or did not pass the checks."""


def snapshot(model: str) -> pathlib.Path:
    p = pathlib.Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    try:
        return pathlib.Path(snapshot_download(model, allow_patterns=SMALL, local_files_only=True))
    except Exception as e:  # not cached yet
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise Failure(f"{model} is not in the Hugging Face cache and HF_HUB_OFFLINE=1 ({e})") from e
        return pathlib.Path(snapshot_download(model, allow_patterns=SMALL))


def bos_of(cfg: dict, tok: dict) -> tuple[str, int]:
    b = cfg.get("bos_token")
    b = b.get("content") if isinstance(b, dict) else b
    if not b:
        raise Failure("tokenizer_config.json has no bos_token")
    ids = [t["id"] for t in tok.get("added_tokens", []) if t.get("content") == b]
    if not ids:
        v = tok.get("model", {}).get("vocab", {})
        ids = [v[b]] if isinstance(v, dict) and b in v else []
    if len(ids) != 1:
        raise Failure(f"bos token {b!r} resolves to {ids}")
    return b, ids[0]


def adds_bos(pp: dict | None, bos: str) -> bool:
    single = (pp or {}).get("single") or []
    return bool(single) and single[0].get("SpecialToken", {}).get("id") == bos


def bos_post_processor(bos: str, bos_id: int) -> dict:
    return {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": bos, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [
            {"SpecialToken": {"id": bos, "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": bos, "type_id": 1}},
            {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": {bos: {"id": bos, "ids": [bos_id], "tokens": [bos]}},
    }


def build(src: pathlib.Path, out: pathlib.Path) -> tuple[str, int]:
    """Write OUT_DIR from the snapshot. Returns (bos, bos_id)."""
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    tok = json.loads((src / "tokenizer.json").read_text())
    bos, bos_id = bos_of(cfg, tok)
    pp = tok.get("post_processor")

    if out.exists() or out.is_symlink():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.name in REWRITTEN or f.name.endswith(".safetensors") or f.name.endswith(".safetensors.index.json"):
            continue
        (out / f.name).symlink_to(f.resolve())

    if adds_bos(pp, bos):
        print(f"{MARKER} {src} already adds {bos}; using its tokenizer unchanged", flush=True)
        for name in REWRITTEN:
            (out / name).symlink_to((src / name).resolve())
        return bos, bos_id
    if (
        not isinstance(pp, dict)
        or pp.get("type") != "TemplateProcessing"
        or pp.get("single") != [{"Sequence": {"id": "A", "type_id": 0}}]
    ):
        raise Failure(f"unexpected post_processor {json.dumps(pp)[:300]}")
    tok["post_processor"] = bos_post_processor(bos, bos_id)
    cfg["add_bos_token"] = True
    (out / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    return bos, bos_id


def verify(src: pathlib.Path, out: pathlib.Path, bos: str, bos_id: int) -> str:
    from transformers import AutoTokenizer

    stock = AutoTokenizer.from_pretrained(str(src))
    fixed = AutoTokenizer.from_pretrained(str(out))
    text = "def fibonacci(n):\n    "
    plain = stock(text, add_special_tokens=False)["input_ids"]
    got = fixed(text)["input_ids"]
    if got != [bos_id] + plain:
        raise Failure(f"raw prompt -> {got[:6]}..., expected [{bos_id}] + {plain[:5]}...")
    if fixed(text, add_special_tokens=False)["input_ids"] != plain:
        raise Failure("add_special_tokens=False output changed")
    if len(fixed) != len(stock) or fixed.get_vocab() != stock.get_vocab():
        raise Failure("vocabulary differs from the snapshot's")
    if fixed.chat_template != stock.chat_template:
        raise Failure("chat template differs from the snapshot's")
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}]
    rendered = fixed.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if not rendered.startswith(bos) or rendered.count(bos) != 1:
        raise Failure(f"chat template renders {rendered.count(bos)} {bos} (expected one, first)")
    chat_fixed = fixed(rendered, add_special_tokens=False)["input_ids"]
    chat_stock = stock(rendered, add_special_tokens=False)["input_ids"]
    if chat_fixed != chat_stock or chat_fixed.count(bos_id) != 1:
        raise Failure("chat prompt ids differ from the stock tokenizer's")
    return f"raw prompts get one {bos} (id {bos_id}); chat ids unchanged; vocab {len(fixed)}"


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0].startswith("-"):
        print(f"{MARKER} usage: apply.py MODEL_ID_OR_DIR OUT_DIR", file=sys.stderr)
        return 2
    out = pathlib.Path(argv[1])
    try:
        src = snapshot(argv[0])
        bos, bos_id = build(src, out)
        summary = verify(src, out, bos, bos_id)
    except Exception as e:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        print(f"{MARKER} failed: {e}", file=sys.stderr, flush=True)
        return 2
    print(f"{MARKER} {out}: {summary}; source {src}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
