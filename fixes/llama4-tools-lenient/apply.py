#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""llama4-tools-lenient: check the Llama 4 tool-parser plugin and print the serve flags.

This fix is a vLLM tool-parser plugin plus a chat template, not a source patch, so nothing is
written. The script compiles the plugin, checks that the template exists, and checks that the
installed vLLM (or the copy under --root) still has the parser API the plugin builds on:
ToolParserManager.register_module, Llama4PythonicToolParser and the protocol classes it imports.
On success it prints the flags to pass to `vllm serve`.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

FIX_ID = "llama4-tools-lenient"
MARKER = f"[gaudi-fix:{FIX_ID}]"
KIND = "plugin"

HERE = Path(__file__).resolve().parent
PLUGIN = HERE / "llama4_tools_lenient.py"
TEMPLATE = HERE / "llama4-tools.jinja"
PARSER_NAME = "llama4_lenient"

# (path inside the vllm package, text that must be present)
ANCHORS = (
    ("tool_parsers/__init__.py", "    ToolParserManager,\n"),
    ("tool_parsers/abstract_tool_parser.py", "    def register_module(\n"),
    ("tool_parsers/abstract_tool_parser.py", "        self.prev_tool_call_arr: list[dict] = []\n"),
    ("tool_parsers/abstract_tool_parser.py", "        self.streamed_args_for_tool: list[str] = []\n"),
    ("tool_parsers/llama4_pythonic_tool_parser.py", "class Llama4PythonicToolParser(ToolParser):\n"),
)
# The plugin imports these from the first module that exists.
PROTOCOL_MODULES = (
    "entrypoints/generate/base/protocol.py",
    "entrypoints/openai/engine/protocol.py",
    "entrypoints/openai/protocol.py",
)
PROTOCOL_CLASSES = (
    "DeltaFunctionCall",
    "DeltaMessage",
    "DeltaToolCall",
    "ExtractedToolCallInformation",
    "FunctionCall",
    "ToolCall",
)


def vllm_dir(root: str | None) -> Path | None:
    if root:
        p = Path(root) / "vllm"
        return p if p.is_dir() else None
    spec = importlib.util.find_spec("vllm")
    locs = list(spec.submodule_search_locations or []) if spec else []
    return Path(locs[0]) if locs else None


def problems(root: str | None) -> list[str]:
    out = []
    try:
        compile(PLUGIN.read_text(), str(PLUGIN), "exec", dont_inherit=True)
    except (OSError, SyntaxError) as e:
        out.append(f"plugin does not compile: {e}")
    if not TEMPLATE.is_file():
        out.append(f"chat template missing: {TEMPLATE}")
    pkg = vllm_dir(root)
    if pkg is None:
        return out + ["vllm package not found"]
    for rel, needle in ANCHORS:
        p = pkg / rel
        if not p.is_file() or needle not in p.read_text():
            out.append(f"vllm/{rel} lacks {needle.strip()!r}")
    proto = next((pkg / m for m in PROTOCOL_MODULES if (pkg / m).is_file()), None)
    if proto is None:
        out.append("no tool-call protocol module found")
    else:
        text = proto.read_text()
        missing = [c for c in PROTOCOL_CLASSES if f"\nclass {c}(" not in text]
        if missing:
            out.append(f"{proto.name} lacks {', '.join(missing)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="same as a plain run: this fix never writes")
    ap.add_argument("--root", help="directory holding vllm/ (default: the installed package)")
    args = ap.parse_args()
    found = problems(args.root)
    if found:
        for msg in found:
            print(f"{MARKER} {msg}", file=sys.stderr)
        print(f"{MARKER} source drifted; the plugin may not load", file=sys.stderr)
        return 2
    print(f"{MARKER} plugin ok; serve with:")
    print(f"  --enable-auto-tool-choice --tool-parser-plugin {PLUGIN}")
    print(f"  --tool-call-parser {PARSER_NAME} --chat-template {TEMPLATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
