#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write docs/llms.txt, the index agents read, from the mkdocs nav, the recipe YAML files and the fix pages.

Run from the repository root after adding or changing a recipe, fix or page:

    python3 scripts/gen_llms_txt.py           # rewrite docs/llms.txt
    python3 scripts/gen_llms_txt.py --check   # exit 1 if docs/llms.txt is out of date (CI)
"""

import glob
import re
import sys
from pathlib import Path

R = "https://raw.githubusercontent.com/thediymaker/gaudi-cookbook/main/"


def paras(p):
    txt = re.sub(r"^---\n.*?\n---\n", "", Path(p).read_text(), flags=re.S)
    out = []
    for b in txt.split("\n\n"):
        b = b.strip()
        if not b or b.startswith(("#", "|", "```", "!!!", "<", "Source:", "-", "1.")):
            continue
        out.append(" ".join(b.split()))
    return out


def clean(s):
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s).replace("**", "")


def sentence(s):
    s = clean(s)
    m = re.match(r"(.+?\.)(\s|$)", s)
    return m.group(1) if m else s


def title(p):
    for line in Path(p).read_text().splitlines():
        if line.startswith("# "):
            return line[2:].strip()


def affects(p):
    m = re.search(r"^\| Affects \| (.+?) \|$", Path(p).read_text(), re.M)
    return clean(m.group(1)) if m else ""


nav = re.findall(
    r"^[ \t]+-[ \t]+(?:[^:\n]+:[ \t]+)?((?:recipes|fixes|kernels|ports|results|kubernetes)/[\w./-]+\.md)$",
    Path("mkdocs.yml").read_text(),
    re.M,
)
rdesc = {}
for y in sorted(glob.glob("recipes/**/recipe.yaml", recursive=True)):
    s = Path(y).read_text()
    page = re.search(r"Page: docs/recipes/([\w.-]+\.md)", s).group(1)
    m = re.search(r'^  description: "(.+)"$', s, re.M)
    if m:
        d = m.group(1)
    else:  # a folded block: description: >- and indented lines
        m = re.search(r"^  description: >-\n((?:    .*\n)+)", s, re.M)
        d = " ".join(line.strip() for line in m.group(1).splitlines())
    rdesc.setdefault(page, []).append((y, d))


def section(prefix):
    return [p for p in nav if p.startswith(prefix) and not p.endswith("/index.md")]


L = [
    "# Gaudi Cookbook",
    "",
    "> Tested recipes, runtime fixes, kernels and fleet practices for serving open LLMs with vLLM on Intel Gaudi 2"
    " through the vllm-gaudi plugin. Gaudi software 1.24.1, vllm-gaudi v0.26.0 and releases/v0.30.0, lazy mode"
    " unless a recipe sets compile mode.",
    "",
    "Agents: read AGENTS.md first. It maps tasks to files and gives the rules: check the image before patching, keep"
    " verified and untested apart, and ask before touching shared hardware. The links below point at raw Markdown"
    " and YAML.",
    "",
    "## Start here",
    "",
    f"- [AGENTS.md]({R}AGENTS.md): where to look for each task, how to turn a recipe into a command, and the rules.",
    f"- [Getting started]({R}docs/getting-started.md): software baseline, image build steps, container flags.",
    f"- [Recipe format]({R}recipes/README.md): the recipe.yaml schema and its gaudi block.",
    f"- [Fixes contract]({R}fixes/README.md): how the fix pack applies, profiles, exit codes, adding a fix.",
    f"- [Port contract]({R}ports/README.md): how a model port is laid out and built from a stock image.",
    "",
    "## Recipes",
    "",
    f"- [Recipe index]({R}docs/recipes/index.md): every model with cards, precision, image, features and status.",
]
for p in section("recipes/"):
    pg = p.split("/")[1]
    entries = rdesc[pg]
    desc = entries[0][1] if len(entries) == 1 else sentence(paras("docs/" + p)[0])
    L.append(f"- [{title('docs/' + p)}]({R}docs/{p}): {desc}")
    for y, _ in entries:
        L.append(f"  - [{y.split('/', 1)[1].rsplit('/', 1)[0]} recipe.yaml]({R}{y})")
L += [
    "",
    "## Fixes",
    "",
    f"- [Fix index]({R}docs/fixes/index.md): status table with affected setups, releases, profiles and upstream links.",
]
for p in section("fixes/"):
    i = p.split("/")[1][:-3]
    L.append(f"- [{i}]({R}fixes/{i}/README.md): {sentence(paras('docs/' + p)[0])} Affects: {affects('docs/' + p)}.")
SECTIONS = (("Kernels", "kernels/"), ("Ports", "ports/"), ("Results", "results/"), ("Kubernetes", "kubernetes/"))
for name, prefix in SECTIONS:
    L += ["", f"## {name}", ""]
    for p in [p for p in nav if p.startswith(prefix)]:
        L.append(f"- [{title('docs/' + p)}]({R}docs/{p}): {sentence(paras('docs/' + p)[0])}")
L += [
    "",
    "## Optional",
    "",
    f"- [Kubernetes manifests and tools]({R}kubernetes/README.md): what each manifest and tool does;"
    " placeholders to replace.",
    f"- [MXFP4 kernel sources and build]({R}kernels/mxfp4-moe/README.md): layout, toolchain, build, enable, verify.",
    f"- [Servers]({R}servers/README.md): models that do not run on vLLM (text-to-speech, EmbeddingGemma 2).",
    f"- [Contributing]({R}CONTRIBUTING.md): ground rules for changes.",
    "",
]
text = "\n".join(L)
if "--check" in sys.argv[1:]:
    current = Path("docs/llms.txt").read_text() if glob.glob("docs/llms.txt") else ""
    if current != text:
        sys.exit("docs/llms.txt is out of date: run python3 scripts/gen_llms_txt.py")
else:
    Path("docs/llms.txt").write_text(text)
