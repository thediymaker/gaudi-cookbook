#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the recipe catalog: a static site in the style of recipes.vllm.ai, generated from recipes/*/recipe.yaml.

The catalog is the front page of the GitHub Pages site; the MkDocs documentation is published under docs/ next to
it. Every launch command on the catalog is generated here from the recipe YAML and the fix pack, for every
combination of the recipe's feature toggles, and --check compares the full command with the Launch block of the
recipe's docs page, so the two cannot drift apart.

    python3 catalog/build.py --out _site              # build (fetches provider avatars from Hugging Face)
    python3 catalog/build.py --out _site --no-fetch   # build without network access (letter tiles instead)
    python3 catalog/build.py --check                  # compare generated commands with the docs pages only

Needs PyYAML, Markdown and pymdown-extensions (all in requirements-docs.txt).
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import itertools
import json
import posixpath
import re
import shlex
import shutil
import sys
import textwrap
import urllib.request
from pathlib import Path

import markdown
import yaml
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/thediymaker/gaudi-cookbook"
SITE_TITLE = "Gaudi Recipes"
DOCS_DIR = "docs/"  # where the MkDocs site is published, relative to the catalog root
CONTAINER_COOKBOOK = "/cookbook"
IMAGE_COOKBOOK = "/opt/gaudi-cookbook"
INVERT_IN_DARK = {"openai"}  # monochrome black marks
TASK_LABELS = {"text": "Text", "multimodal": "Multimodal", "embedding": "Embedding", "omni": "Omni"}
FEATURE_LABELS = {"tool_calling": "Tool calling", "reasoning": "Reasoning", "spec_decoding": "Spec decoding"}
RELEASES = ("v0.26.0", "v0.30.0")

# The MXFP4 MoE decode kernel, featured on the home page and on the recipes it speeds up. Every number and its
# "measured as" label comes from docs/kernels/mxfp4-moe.md; keep them in step.
KERNEL = {
    "name": "MXFP4 MoE decode kernel",
    "headline": "Up to 2.5\u00d7 faster MXFP4 MoE decode on Gaudi 2",
    "lead": "Custom TPC kernels run the routed experts straight from the packed MXFP4 weights, with no bf16 unpack "
    "round trip. On one 8-card node they took GLM-5.3 from 23.3 to 59.1 tok/s per stream and Kimi-K2.7 from 17.9 to "
    "44.5, with lower numerical error than the stock op, and also serve DeepSeek-V4.1-Flash, GLM-5.2, Muse Spark Open "
    "and MiMo-V2.6-Pro. Time to first token is unchanged, except on GLM-5.x, where a prefill unpack kernel cuts it "
    "by 10% to 19%.",
    "stats": [
        ("2.5\u00d7", "GLM-5.3 single stream: 23.3 \u2192 59.1 tok/s on one 8-card node"),
        ("2.5\u00d7", "Kimi-K2.7 single stream at 16K context: 17.9 \u2192 44.5 tok/s, same pod"),
        ("2.4\u00d7", "Muse Spark Open, 4 streams aggregate: 43.3 \u2192 102.8 tok/s, same pod"),
        ("2\u20134\u00d7", "lower error than the stock op against an fp32 reference"),
    ],
    # the recipe the spotlight's second button opens
    "featured": ("zai-org/GLM-5.3-BF16", "GLM-5.3 recipe"),
    # (model, setup, stock, with the kernel, gain, how it was measured, where it lives: recipe:<id>, port:<id>
    # (links the port's recipe once one exists, else its write-up) or doc:<path>)
    "models": [
        (
            "GLM-5.3 (MXFP4)",
            "8 cards, TP 8 + EP 8, 1 stream",
            "23.3 tok/s",
            "59.1",
            "2.5\u00d7",
            "3 same-node A/Bs chained",
            "recipe:zai-org/GLM-5.3-BF16",
        ),
        (
            "Kimi-K2.7-Code (MXFP4)",
            "8 cards, TP 8 + EP 8, 16K, 1 stream",
            "17.9 tok/s",
            "44.5",
            "2.5\u00d7",
            "same-pod A/B, published recipe",
            "recipe:amd/Kimi-K2.7-Code-MXFP4",
        ),
        (
            "Muse Spark Open",
            "8 cards, 1 stream",
            "22.7 tok/s",
            "40.8",
            "1.8\u00d7",
            "same-pod A/B, published recipe",
            "recipe:amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4",
        ),
        (
            "Muse Spark Open",
            "8 cards, 4 streams, aggregate",
            "43.3 tok/s",
            "102.8",
            "2.4\u00d7",
            "same-pod A/B, published recipe",
            "recipe:amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4",
        ),
        (
            "DeepSeek-V4.1-Flash (MXFP4)",
            "8 cards, TP 8, 1 stream",
            "20.7 tok/s",
            "37.6",
            "1.8\u00d7",
            "same-pod A/B, published recipe",
            "recipe:deepseek-ai/DeepSeek-V4.1-Flash-1017",
        ),
        (
            "GLM-5.2 (MXFP4)",
            "8 cards, 1 stream",
            "23.1 tok/s",
            "47.8",
            "2.1\u00d7",
            "stock pod vs kernel pods",
            "recipe:amd/GLM-5.2-MXFP4",
        ),
        (
            "MiMo-V2.6-Pro (MXFP4)",
            "8 cards, 1 stream",
            "18.1 tok/s",
            "39.7",
            "2.2\u00d7",
            "same node: stock pod, then kernel pod",
            "doc:kernels/mxfp4-moe/#beyond-gpt-oss",
        ),
        (
            "openai/gpt-oss-120b",
            "2 cards, TP 2 + EP, 4 in flight",
            "16.6 tok/s per stream",
            "52.4",
            "3.2\u00d7",
            "same-node A/B",
            "recipe:openai/gpt-oss-120b",
        ),
    ],
    "models_note": "GLM-5.x, DeepSeek-V4.1-Flash, Kimi-K2.7 and Muse Spark Open run the kernel through their ports, "
    "each with a recipe here; MiMo-V2.6-Pro's integration is not published. GLM-5.3 reaches 69.6 tok/s single stream "
    "with decode-step and expert-placement work on top of the kernel.",
    "proof": "Every model above passes its recipe's gates with the kernel on: tool calls, long-context needles, and "
    "GSM8K or perplexity against a reference.",
    "doc": "kernels/mxfp4-moe/",
    "source": "kernels/mxfp4-moe",
    "recipes": {
        "openai/gpt-oss-120b": {
            "headline": "Up to 3× faster decode with the MXFP4 MoE kernel",
            "stats": [
                ("2.96×", "per stream at 4 in flight: 59.6 vs 20.1 tok/s"),
                ("2.25×", "per stream at 16 in flight: 22.4 vs 10.0 tok/s"),
                (
                    "2 cards",
                    "TP 2 with the kernel: 52 tok/s per stream at 4 in flight, vs 20 for this stock recipe on 4",
                ),
            ],
            "how": "Build it in your image, then add <code>HPU_MXFP4_NATIVE=1 MXFP4_TPC_MOE=1</code>. Off by default; "
            "any mismatch keeps the stock op.",
        }
    },
}

# Ports of models that released images cannot serve, and fleet studies, shown on the home page. Each stat and its
# "measured as" label comes from the linked page under docs/; keep them in step.
PORTS = [
    (
        "Muse Spark Open",
        "816B MoE, MXFP4, 8 cards",
        "Needles pass at 256K, 512K and 1M tokens",
        "perplexity within 0.4% of an fp32 reference",
        "ports/muse-spark-open/",
    ),
    (
        "GLM-5.3-Flash",
        "KDA + DSA + mHC MoE, 8 cards",
        "22 to 24 tok/s from 2K to 517K context",
        "decode flat with context, same image and code",
        "ports/glm-5-3-flash/",
    ),
    (
        "DeepSeek-V4.1-Flash",
        "New attention and experts, 8 cards",
        "23.6 → 35.0 tok/s decode",
        "same-node A/B; prefill 1.35 to 1.55x",
        "ports/deepseek-v4-1-flash/",
    ),
    (
        "GLM-5.2 and GLM-5.3 MXFP4",
        "744B MoE, sparse DSA, 8 cards",
        "23.25 → 45.6 tok/s, first kernel",
        "same-pod A/B; needles fixed to 128K",
        "ports/glm-5-mxfp4/",
    ),
    (
        "Beam-1.0",
        "501B MoE, 256K, 8 cards",
        "Weight load 2177 → 356 s",
        "same-node A/B, bit-identical state",
        "ports/beam-1-0/",
    ),
    (
        "MiMo-V2.6-Flash",
        "Sliding window with sinks, 512K",
        "5.33x the KV token capacity",
        "identical outputs; needles 3/3 at 500K",
        "ports/mimo-v2-6-flash/",
    ),
    (
        "Kimi-K2.7-Code",
        "1T MoE, TP 16 over two nodes",
        "FP8 experts about 3x MXFP4 at 8 streams",
        "different conditions, named on the page",
        "ports/kimi-k2-7/",
    ),
]
RESULTS = [
    (
        "Platform findings",
        "NIC stalls, FusedSDPA, fp32 trig, sampling",
        "7.56 → 15.45 tok/s from one NIC fix",
        "same-node A/B",
        "results/platform-findings/",
    ),
    (
        "Prefix caching on GDN hybrids",
        "Qwen3.5 / 3.6 / 3.8 style models",
        "Later-turn TTFT 19.8 → 2.6 s",
        "same-pod A/B, agent replay",
        "results/gdn-prefix-caching/",
    ),
    (
        "Anatomy of a MoE decode step",
        "8-rank critical path, EP layouts",
        "51.6 → 57.8 tok/s, bit-identical",
        "same-node A/B",
        "results/moe-decode-step/",
    ),
    (
        "Two-node pipeline parallel",
        "PP 2 x TP 8 over the host NIC",
        "709,888-token KV pool",
        "needles pass at 637K tokens",
        "results/two-node-pipeline/",
    ),
    (
        "Gemma-4-E2B fast prefill",
        "KV-shared layers skip prefill work",
        "72.8k → 169.7k tok/s prefill",
        "same card, token-exact",
        "results/gemma-4-e2b-prefill/",
    ),
    (
        "Fast text-to-speech",
        "Qwen3-TTS, Chatterbox, Orpheus",
        "Chatterbox Turbo 10 to 13x real time",
        "per-call, one card",
        "results/tts-on-gaudi/",
    ),
    (
        "EmbeddingGemma 2",
        "Bidirectional multimodal embedder",
        "95k to 235k text tok/s per card",
        "per-call, one card; cosine ≥ 0.9995",
        "results/embeddinggemma-2/",
    ),
]

# ---------------------------------------------------------------------------------------------------------------------
# Inputs


def load_fix_pack():
    spec = importlib.util.spec_from_file_location("apply_all", ROOT / "fixes" / "apply_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ORDER, mod.STANDALONE, mod.PROFILES


def load_fix_status():
    """Affects / upstream / status per fix, from the status table in docs/fixes/index.md."""
    out = {}
    row = re.compile(r"^\| \[([\w-]+)\]\([\w-]+\.md\) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \|$")
    for line in (ROOT / "docs" / "fixes" / "index.md").read_text().splitlines():
        m = row.match(line)
        if m:
            fid, affects, releases, profiles, upstream, status = m.groups()
            out[fid] = {"affects": strip_md(affects), "status": strip_md(status), "profiles": profiles}
    return out


def strip_md(s):
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s).replace("`", "")


def load_recipes():
    recipes = []
    for path in sorted((ROOT / "recipes").glob("*/*/recipe.yaml")):
        text = path.read_text()
        data = yaml.safe_load(text)
        org, repo = path.parent.parent.name, path.parent.name
        page = re.search(r"Page: docs/recipes/([\w.-]+\.md)", text)
        if data["model"]["model_id"] != f"{org}/{repo}":
            sys.exit(f"{path}: model_id {data['model']['model_id']} does not match the directory")
        recipes.append(
            {
                "id": f"{org}/{repo}",
                "org": org,
                "repo": repo,
                "yaml": path.relative_to(ROOT).as_posix(),
                "page": page.group(1) if page else None,
                **data,
            }
        )
    return recipes


# ---------------------------------------------------------------------------------------------------------------------
# Commands


def release_of(r):
    return r["gaudi"]["image"]["vllm_gaudi"].split("/")[-1]


def shell_value(v):
    v = str(v)
    if v.startswith("fixes/"):
        v = "$COOKBOOK/" + v
    if "$" in v:
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return shlex.quote(v)


def arg_pairs(args):
    pairs, i = [], 0
    while i < len(args):
        a = str(args[i])
        if a.startswith("--") and i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
            pairs.append((a, str(args[i + 1])))
            i += 2
        else:
            pairs.append((a, None))
            i += 1
    return pairs


def serve_pairs(r, enabled):
    pairs = arg_pairs(r["gaudi"]["serve_args"])
    for key, feat in (r.get("features") or {}).items():
        if key not in enabled:
            drop = set(arg_pairs(feat.get("args") or []))
            pairs = [p for p in pairs if p not in drop]
    return pairs


def fix_block(r, order, standalone, profiles):
    req = r["gaudi"].get("required_fixes") or []
    if not req:
        return []
    prof = release_of(r)
    optin = [f for f in order if f in req and f not in profiles[prof]]
    head = (
        f"# Fixes: the {prof} profile"
        + (" plus this recipe's opt-in fixes" if optin else "")
        + ", in the pack's order."
    )
    cmd = f'python3 "$COOKBOOK/fixes/apply_all.py" --profile {prof}'
    if optin and len(cmd) + len(" ".join(optin)) + 12 > 100:
        cmd += " \\\n    " + " ".join(optin)
    elif optin:
        cmd += " " + " ".join(optin)
    lines = [head, "# Stops with a non-zero exit code if the image does not match.", cmd + " || exit 1"]
    for fid in standalone:
        if fid in req and fid == "gemma4-bos-tokenizer":
            tok = dict(p for p in arg_pairs(r["gaudi"]["serve_args"]) if p[1] is not None).get("--tokenizer")
            if not tok:
                sys.exit(f"{r['yaml']}: gemma4-bos-tokenizer needs a --tokenizer OUT_DIR in serve_args")
            lines += [
                "# Separate step, not run by apply_all.py: a tokenizer copy that adds <bos>",
                f'python3 "$COOKBOOK/fixes/gemma4-bos-tokenizer/apply.py" {r["id"]} {tok} || exit 1',
            ]
        elif fid in req:
            sys.exit(f"{r['yaml']}: no launch step known for standalone fix {fid}")
    return lines


def serve_block(model, pairs, indent="", exec_=False, extra=()):
    items = list(extra) + [f"{a} {shell_value(v)}" if v is not None else a for a, v in pairs]
    lines = [f"{indent}{'exec ' if exec_ else ''}vllm serve {model}" + (" \\" if items else "")]
    for i, item in enumerate(items):
        lines.append(f"{indent}  {item}" + (" \\" if i < len(items) - 1 else ""))
    return lines


def serve_target(r):
    """The model argument of `vllm serve`, and the extra args it implies.

    A recipe that serves a converted local checkpoint sets gaudi.serve_model (for example "$CHECKPOINT_DIR"); the
    served name then stays the public model id.
    """
    local = r["gaudi"].get("serve_model")
    if not local:
        return r["id"], ()
    return (f'"{local}"' if local.startswith("$") else local), (f"--served-model-name {r['id']}",)


def port_of(r):
    return r["gaudi"].get("port")


def image_of(r):
    """The image the launch runs in: the stock image, or the port image built from it."""
    port = port_of(r)
    return f"<your-{port}-image>" if port else r["gaudi"]["image"]["placeholder"]


def uses_cookbook(r):
    g = r["gaudi"]
    return bool(port_of(r) or g.get("required_fixes")) or any(str(a).startswith("fixes/") for a in g["serve_args"])


def build_commands(r, enabled, fix_pack):
    g = r["gaudi"]
    pairs = serve_pairs(r, enabled)
    env = g.get("env") or {}
    setup = g.get("setup") or []
    fixes = fix_block(r, *fix_pack)

    shell = [f"export {k}={shell_value(v)}" for k, v in env.items()]
    for block in (setup, fixes):
        if block:
            shell += [""] + block
    model, named = serve_target(r)
    shell += [""] + serve_block(model, pairs, extra=named)
    shell_text = "\n".join(shell).strip("\n")

    image = image_of(r)
    cb = uses_cookbook(r)
    port = port_of(r)
    docker = []
    if port:
        docker += [
            f"# 0. Build the port image from a stock image (ports/{port}/README.md).",
            f"docker build -f gaudi-cookbook/ports/{port}/Dockerfile \\",
            f"  --build-arg BASE={g['image']['placeholder']} -t {image} gaudi-cookbook",
            "",
        ]
    docker += [
        "# 1. Write the launch script (the Shell tab)" + (" next to a checkout of this repository." if cb else "."),
        "cat > launch.sh <<'EOF'",
        shell_text,
        "EOF",
        "",
        "# 2. Run it in a vllm-gaudi container.",
        "docker run -it --rm --runtime=habana \\",
        "  -e HABANA_VISIBLE_DEVICES=all \\",
        "  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \\",
        "  -e HF_TOKEN \\",
        "  --cap-add=sys_nice --ipc=host --net=host \\",
        "  -v ~/.cache/huggingface:/root/.cache/huggingface \\",
    ]
    if cb:
        docker.append(f'  -v "$PWD/gaudi-cookbook:{CONTAINER_COOKBOOK}:ro" -e COOKBOOK={CONTAINER_COOKBOOK} \\')
    docker += ['  -v "$PWD/launch.sh:/launch.sh:ro" \\', f"  {image} \\", "  bash /launch.sh"]

    script = ["set -euo pipefail"]
    for block in (setup, fixes):
        if block:
            script += block
    slug = r["meta"].get("slug") or r["repo"].lower()
    model, named = serve_target(r)
    extra = (*(named or (f"--served-model-name {slug}",)), "--host 0.0.0.0", "--port 8000", "--shutdown-timeout 120")
    script += serve_block(model, pairs, exec_=True, extra=extra)
    cards = str(g["cards"]["total"])
    k8s = [
        "# Container for kubernetes/manifests/vllm-deployment.yaml: keep the template's probes, volumes,",
        "# hugepages and drain settings, and size CPU and memory per card (docs/kubernetes/pod-spec.md).",
    ]
    if port:
        k8s.append(f"# The port image (ports/{port}/Dockerfile) carries this repository at {IMAGE_COOKBOOK}.")
    elif cb:
        k8s.append(f"# Assumes this repository is copied into the image at {IMAGE_COOKBOOK}.")
    k8s += [
        "containers:",
        "  - name: vllm",
        f"    image: {image}",
        '    command: ["/bin/bash", "-c"]',
        "    args:",
        "      - |",
        *[f"        {line}" if line else "" for line in "\n".join(script).split("\n")],
        "    env:",
    ]
    if cb:
        k8s += ["      - name: COOKBOOK", f"        value: {IMAGE_COOKBOOK}"]
    for k, v in env.items():
        k8s += [f"      - name: {k}", f"        value: {json.dumps(str(v))}"]
    k8s += [
        "    resources:",
        "      requests:",
        f'        habana.ai/gaudi: "{cards}"',
        "      limits:",
        f'        habana.ai/gaudi: "{cards}"',
        "    ports:",
        "      - name: http",
        "        containerPort: 8000",
    ]
    return {"shell": shell_text, "docker": "\n".join(docker), "kubernetes": "\n".join(k8s)}


def is_pooling(r):
    return ("--runner", "pooling") in arg_pairs(r["gaudi"]["serve_args"])


def client_commands(r):
    model = r["id"]
    pairs = dict(p for p in arg_pairs(r["gaudi"]["serve_args"]) if p[1] is not None)
    if is_pooling(r):
        body = json.dumps({"model": model, "input": ["What is a Gaudi accelerator?"]})
        return {
            "curl": f"curl -s http://localhost:8000/v1/embeddings \\\n  -H 'Content-Type: application/json' \\\n  -d '{body}'"
        }
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 512}
    )
    curl = (
        f"curl -s http://localhost:8000/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{body}'"
    )
    bench = (
        f"vllm bench serve --model {model} \\\n  --dataset-name random --random-input-len 1024 --random-output-len 256 \\\n"
        f"  --num-prompts 64 --max-concurrency {pairs.get('--max-num-seqs', '8')}"
    )
    return {"curl": curl, "bench": bench}


def variants(r, fix_pack):
    keys = list((r.get("features") or {}).keys())
    out = {}
    for n in range(len(keys) + 1):
        for combo in itertools.combinations(keys, n):
            out[",".join(sorted(combo))] = build_commands(r, set(combo), fix_pack)
    return keys, out


# ---------------------------------------------------------------------------------------------------------------------
# --check: generated command vs the docs Launch block


def shell_tokens(script):
    script = script.replace("\\\n", " ")
    toks = []
    for line in script.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks.append(shlex.split(line, comments=True))
    return toks


def docs_launch_blocks(page):
    text = (ROOT / "docs" / "recipes" / page).read_text()
    m = re.search(r"^## Launch\n(.*?)(?=^## )", text, re.M | re.S)
    if not m:
        return []
    blocks = re.findall(r"^( *)```bash\n(.*?)^\1```", m.group(1), re.M | re.S)
    return [textwrap.dedent(body) for _, body in blocks]


def check(recipes, fix_pack):
    bad = 0
    for r in recipes:
        if not r["page"]:
            print(f"{r['yaml']}: no 'Page: docs/recipes/...' header comment")
            bad += 1
            continue
        mine = shell_tokens(build_commands(r, set(r.get("features") or {}), fix_pack)["shell"])
        # Compare as shell tokens, so `vllm serve "$X"` and `vllm serve $X` both match.
        target = shlex.split(serve_target(r)[0])[0]
        serve = f"vllm serve {target}"
        blocks = [
            b
            for b in docs_launch_blocks(r["page"])
            if any(t[t[:1] == ["exec"] :][:3] == ["vllm", "serve", target] for t in shell_tokens(b))
        ]
        if len(blocks) != 1:
            print(f"{r['id']}: expected one Launch block with '{serve}' in {r['page']}, found {len(blocks)}")
            bad += 1
            continue
        theirs = shell_tokens(blocks[0])
        if mine != theirs:
            bad += 1
            print(f"{r['id']}: generated command differs from docs/recipes/{r['page']}")
            for a, b in itertools.zip_longest(mine, theirs):
                if a != b:
                    print(
                        f"    catalog: {' '.join(a) if a else '(nothing)'}\n    docs:    {' '.join(b) if b else '(nothing)'}"
                    )
    print(f"catalog check: {len(recipes) - bad}/{len(recipes)} recipes match their docs Launch block")
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# HTML helpers

E = html.escape

ICONS = {
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
    '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "text": '<path d="M4 7V4h16v3M9 20h6M12 4v16"/>',
    "multimodal": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "embedding": '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
    "chevron": '<path d="m6 9 6 6 6-6"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "terminal": '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "pencil": '<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>',
    "bug": '<rect x="8" y="6" width="8" height="14" rx="4"/><path d="M12 20v-9M2 13h6M16 13h6M4 7l4 2M20 7l-4 2M4 19l4-2M20 19l-4-2"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9a2.1 2.1 0 0 1-3-3'
    'l6.9-6.9a6 6 0 0 1 7.9-7.9z"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.7 9a1 1 0 0 1-.7 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.2-2.7a1.2 1.2 0 0 1 '
    '1.5 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "hexagon": '<path d="M12 2 21 7v10l-9 5-9-5V7z"/>',
    "bot": '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2M20 14h2M15 13v2M9 13v2"/>',
    "box": '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5M12 13v8"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
}


def spotlight(prefix, recipes):
    stats = "".join(
        f'<div class="stat"><div class="stat-num">{E(n)}</div><div class="stat-label">{E(t)}</div></div>'
        for n, t in KERNEL["stats"]
    )
    featured, featured_label = KERNEL["featured"]

    def tag(where):
        kind, _, target = where.partition(":")
        if kind == "port":
            r = next((r for r in recipes if port_of(r) == target), None)
            if r is None:
                return f'<a class="tag" href="{prefix}{DOCS_DIR}ports/{target}/">write-up</a>'
            target = r["id"]
        elif kind == "doc":
            return f'<a class="tag" href="{prefix}{DOCS_DIR}{target}">write-up</a>'
        return f'<a class="tag ok" href="{prefix}{target}/">recipe</a>'

    rows = "".join(
        f'<tr><td class="mono">{E(m)}</td><td>{E(setup)}</td><td class="num">{E(a)}</td><td class="num"><b>{E(b)}</b></td>'
        f'<td class="num gain">{E(g)}</td><td class="muted">{E(how)}</td><td>{tag(where)}</td></tr>'
        for m, setup, a, b, g, how, where in KERNEL["models"]
    )
    return f"""<section class="spotlight">
  <div class="spot-text">
    <div class="spot-eyebrow">{icon("zap", 12)} New kernel</div>
    <h2>{E(KERNEL["headline"])}</h2>
    <p>{E(KERNEL["lead"])}</p>
    <p class="spot-proof">{icon("shield", 12)} {E(KERNEL["proof"])}</p>
    <div class="spot-links">
      <a class="spot-btn primary" href="{prefix}{DOCS_DIR}{KERNEL["doc"]}">Read the write-up</a>
      <a class="spot-btn" href="{prefix}{featured}/">{E(featured_label)}</a>
      <a class="spot-btn" href="{REPO_URL}/tree/main/{KERNEL["source"]}">Source {icon("external", 11)}</a>
    </div>
  </div>
  <div class="spot-stats">{stats}</div>
  <div class="spot-table">
    <table>
      <thead><tr><th>Model</th><th>Setup</th><th>Stock op</th><th>With the kernel</th><th>Gain</th><th>Measured as</th><th></th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p class="spot-note">{E(KERNEL["models_note"])}</p>
  </div>
</section>"""


def kernel_callout(r, prefix):
    k = KERNEL["recipes"].get(r["id"])
    if not k:
        return ""
    stats = "".join(
        f'<div class="stat"><div class="stat-num">{E(n)}</div><div class="stat-label">{E(t)}</div></div>'
        for n, t in k["stats"]
    )
    return f"""<aside class="kernel-callout">
  <div class="callout-head">{icon("zap", 14)}<b>{E(k["headline"])}</b>
    <a href="{prefix}{DOCS_DIR}{KERNEL["doc"]}">How it works &rarr;</a></div>
  <div class="callout-stats">{stats}</div>
  <div class="callout-how">{k["how"]} Measured A/B on the same node and image: TP 4 + EP unless noted, about 3.2K-token prompts.</div>
</aside>"""


def icon(name, size=14, cls=""):
    return (
        f'<svg class="icon {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


def ctx_label(n):
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{round(n / 1_000_000)}M"
    if n >= 1000:
        return f"{round(n / 1024) if n % 1024 == 0 else round(n / 1000)}K"
    return str(n)


def params_label(r):
    m = r["model"]
    p, a = m.get("parameter_count", ""), m.get("active_parameters")
    return f"{p}/{a}" if m.get("architecture") == "moe" and a and a != p else p


def status_of(r):
    return (r["meta"].get("hardware") or {}).get("gaudi2", "supported")


def precision_of(r):
    return ((r.get("variants") or {}).get("default") or {}).get("precision", "")


def provider_logo(org, prefix, logos, size=28, cls="logo"):
    inv = " invert-dark" if org.lower() in INVERT_IN_DARK else ""
    if org in logos:
        return f'<img src="{prefix}{logos[org]}" alt="" width="{size}" height="{size}" class="{cls}{inv}">'
    return f'<span class="{cls} monogram" style="width:{size}px;height:{size}px">{E(org[:1].upper())}</span>'


# ---------------------------------------------------------------------------------------------------------------------
# Page frame


def page(title, description, prefix, body, search_index, canonical=""):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)}</title>
<meta name="description" content="{E(description)}">
<meta property="og:title" content="{E(title)}">
<meta property="og:description" content="{E(description)}">
<link rel="icon" href="{prefix}assets/logo.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{prefix}assets/catalog.css">
<link rel="stylesheet" href="{prefix}assets/pygments.css">
<script>
(function () {{
  var t = null;
  try {{ t = localStorage.getItem("gc-theme"); }} catch (e) {{}}
  if (t === "dark" || (!t && window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches)) {{
    document.documentElement.classList.add("dark");
  }}
}})();
</script>
</head>
<body>
<header class="site-header">
  <div class="wrap header-row">
    <a class="brand" href="{prefix}index.html">
      <img src="{prefix}assets/logo.svg" alt="" width="30" height="30">
      <span class="brand-name">Gaudi Cookbook</span>
      <span class="brand-slash">/</span>
      <span class="brand-sub">Recipes</span>
    </a>
    <div class="search" data-prefix="{prefix}">
      {icon("search", 14, "search-icon")}
      <input type="search" placeholder="Search models" aria-label="Search recipes" autocomplete="off">
      <kbd>/</kbd>
      <div class="search-results" role="listbox" hidden></div>
    </div>
    <nav class="header-nav">
      <a href="{prefix}index.html#all">Recipes</a>
      <a href="{prefix}{DOCS_DIR}">Docs</a>
      <a href="{prefix}{DOCS_DIR}fixes/">Fixes</a>
      <a href="{REPO_URL}">GitHub</a>
      <button class="theme-toggle" type="button" aria-label="Toggle dark mode">{icon("moon", 16, "i-moon")}{icon("sun", 16, "i-sun")}</button>
    </nav>
  </div>
</header>
<div class="site-body">
{body}
</div>
<footer class="site-footer">
  <div class="wrap footer-row">
    <a href="{REPO_URL}">GitHub</a>
    <a href="{REPO_URL}/issues">Request a recipe</a>
    <a href="{prefix}{DOCS_DIR}">Documentation</a>
    <a href="{prefix}{DOCS_DIR}getting-started/">Build an image</a>
    <a href="{prefix}{DOCS_DIR}agents/">Use with an agent</a>
    <a href="{prefix}recipes.json">JSON API</a>
    <a href="{prefix}{DOCS_DIR}llms.txt">llms.txt</a>
    <span class="footer-note">Independent project, not affiliated with Intel or the vLLM project. Apache-2.0.</span>
  </div>
</footer>
<script>window.GC_INDEX = {json.dumps(search_index, separators=(",", ":"))};</script>
<script src="{prefix}assets/catalog.js"></script>
</body>
</html>
"""


# ---------------------------------------------------------------------------------------------------------------------
# Cards and tiles


def recipe_card(r, prefix, logos):
    tasks = r["meta"].get("tasks") or []
    moe = r["model"].get("architecture") == "moe"
    cards = r["gaudi"]["cards"]["total"]
    status = status_of(r)
    badges = [
        f'<span class="badge">{icon("layers" if moe else "cpu", 10)}<span class="mono">{E(params_label(r))}</span></span>',
        f'<span class="badge mono upper">{E(precision_of(r))}</span>',
        f'<span class="badge mono">{ctx_label(r["gaudi"]["max_model_len"])} ctx</span>',
        f'<span class="badge mono">{cards}&times; Gaudi 2</span>',
    ] + [
        f'<span class="badge subtle">{icon(t if t in ICONS else "text", 10)}{E(TASK_LABELS.get(t, t))}</span>'
        for t in tasks
    ]
    if port_of(r):
        badges.append(f'<span class="badge port">{icon("box", 10)}Port</span>')
    if r["id"] in KERNEL["recipes"]:
        badges.append(
            f'<span class="badge boost">{icon("zap", 10)}{E(KERNEL["recipes"][r["id"]]["stats"][0][0])} with kernel</span>'
        )
    data = {
        "tasks": " ".join(tasks),
        "cards": str(cards),
        "precision": precision_of(r).lower(),
        "image": release_of(r),
        "text": " ".join(
            [r["id"], r["meta"].get("title", ""), r["meta"].get("provider", ""), r["meta"].get("description", "")]
        ).lower(),
    }
    attrs = " ".join(f'data-{k}="{E(v)}"' for k, v in data.items())
    return f"""<a class="recipe-card" href="{prefix}{r["org"]}/{r["repo"]}/" {attrs}>
  <div class="card-head">
    {provider_logo(r["org"], prefix, logos)}
    <div class="card-title-wrap">
      <div class="card-title">{E(r["repo"])}</div>
      <div class="card-org mono">{E(r["org"])}</div>
    </div>
    <span class="status-dot {status}" title="{"Verified on Gaudi 2" if status == "verified" else "Supported: not re-run as written"}"></span>
  </div>
  <div class="badges">{"".join(badges)}</div>
  <div class="card-desc">{E(r["meta"].get("description", ""))}</div>
</a>"""


def provider_tile(org, rs, prefix, logos):
    name = rs[0]["meta"].get("provider") or org
    n = len(rs)
    return f"""<a class="provider-tile" href="{prefix}{org}/">
  <div class="card-head">
    {provider_logo(org, prefix, logos)}
    <div class="card-title-wrap">
      <div class="card-title">{E(name)}</div>
      <div class="card-org mono">{E(org)}</div>
    </div>
  </div>
  <div class="tile-foot"><span><b>{n}</b> {"recipe" if n == 1 else "recipes"}</span><span class="arrow">&rarr;</span></div>
</a>"""


def by_org(recipes):
    groups = {}
    for r in recipes:
        groups.setdefault(r["org"], []).append(r)
    return sorted(groups.items(), key=lambda kv: kv[0].lower())


def section_head(label, extra=""):
    return f'<div class="section-head"><span class="eyebrow">{label}</span>{extra}</div>'


# ---------------------------------------------------------------------------------------------------------------------
# Pages


def home_page(recipes, logos):
    p = ""
    chips = []
    groups = [
        (
            "tasks",
            "Task",
            [(t, TASK_LABELS.get(t, t)) for t in sorted({t for r in recipes for t in r["meta"].get("tasks") or []})],
        ),
        (
            "cards",
            "Cards",
            [
                (str(c), f"{c} card" + ("s" if c > 1 else ""))
                for c in sorted({r["gaudi"]["cards"]["total"] for r in recipes})
            ],
        ),
        ("precision", "Precision", [(x.lower(), x.upper()) for x in sorted({precision_of(r) for r in recipes})]),
        ("image", "Image", [(x, x) for x in RELEASES if any(release_of(r) == x for r in recipes)]),
    ]
    for key, label, opts in groups:
        pills = "".join(
            f'<button type="button" class="pill" data-filter="{key}" data-value="{E(v)}">{E(t)}</button>'
            for v, t in opts
        )
        chips.append(
            f'<div class="filter-row"><span class="filter-label">{label}</span><div class="pill-group">{pills}</div></div>'
        )
    cards = "".join(
        recipe_card(r, p, logos) for r in sorted(recipes, key=lambda r: (r["org"].lower(), r["repo"].lower()))
    )
    tiles = "".join(provider_tile(org, rs, p, logos) for org, rs in by_org(recipes))
    verified = sum(status_of(r) == "verified" for r in recipes)
    more = [
        (
            "wrench",
            "Fixes",
            "Fail-closed runtime patches for bugs in released images, each tied to an upstream pull request.",
            f"{DOCS_DIR}fixes/",
        ),
        (
            "zap",
            "Kernels",
            "MXFP4 MoE decode TPC kernels: up to 2.5x faster decode on GLM-5.x, Kimi-K2.7, Muse Spark Open and DeepSeek-V4.1.",
            f"{DOCS_DIR}kernels/",
        ),
        (
            "hexagon",
            "Kubernetes",
            "Pod spec, bin-pack scheduling, warm-up, health, rollouts, autoscaling and routing.",
            f"{DOCS_DIR}kubernetes/",
        ),
        (
            "bot",
            "Use with an agent",
            "Point your coding agent at the repository; AGENTS.md tells it where everything is.",
            f"{DOCS_DIR}agents/",
        ),
    ]
    showcase = "".join(
        f'<a class="more-card port-card" href="{DOCS_DIR}{doc}"><div class="card-title">{E(t)}</div>'
        f'<div class="card-org">{E(sub)}</div><div class="port-stat">{E(stat)}</div><div class="card-desc">{E(how)}</div></a>'
        for t, sub, stat, how, doc in PORTS + RESULTS
    )
    more_html = "".join(
        f'<a class="more-card" href="{href}">{icon(ic, 18)}<div><div class="card-title">{E(t)}</div><div class="card-desc">{E(d)}</div></div></a>'
        for ic, t, d, href in more
    )
    body = f"""<main class="wrap home">
  <header class="hero">
    <h1>{SITE_TITLE}</h1>
    <p>Pick a model, copy the <code>vllm serve</code> line that runs on <b>Intel Gaudi 2</b>. Every recipe uses a released
    vllm-gaudi image and a public checkpoint, lists the fixes it needs, and shows the checks that passed.
    <a href="{DOCS_DIR}getting-started/">Build an image &rarr;</a></p>
    <div class="hero-stats">
      <span><b>{len(recipes)}</b> recipes</span><span><b>{verified}</b> verified</span>
      <span><b>{len(by_org(recipes))}</b> providers</span><span>Gaudi SW <b>1.24.1</b></span>
      <span>vllm-gaudi <b>v0.26.0</b> &middot; <b>v0.30.0</b></span>
    </div>
  </header>
  {spotlight(p, recipes)}
  <section id="all">
    {section_head("All recipes", f'<span class="eyebrow-sub" data-count>{len(recipes)} shown</span><button type="button" class="link-btn" data-clear hidden>Clear filters</button>')}
    <div class="filters">
      <div class="filter-search">{icon("search", 14)}<input type="search" data-filter-text placeholder="Filter by model, provider or description" aria-label="Filter recipes"></div>
      {"".join(chips)}
    </div>
    <div class="grid cards-grid" data-grid>{cards}</div>
    <p class="empty" data-empty hidden>No recipe matches these filters.</p>
  </section>
  <section id="ports">
    {section_head("Ports and results", f'<span class="eyebrow-sub"><a href="{DOCS_DIR}ports/">All ports &rarr;</a> &middot; <a href="{DOCS_DIR}results/">All results &rarr;</a></span>')}
    <div class="grid more-grid">{showcase}</div>
  </section>
  <section>
    {section_head("Browse by provider")}
    <div class="grid tiles-grid">{tiles}</div>
  </section>
  <section>
    {section_head("More in the cookbook")}
    <div class="grid more-grid">{more_html}</div>
  </section>
</main>"""
    return body


def org_page(org, rs, logos):
    p = "../"
    name = rs[0]["meta"].get("provider") or org
    cards = "".join(recipe_card(r, p, logos) for r in rs)
    return f"""<main class="wrap">
  <nav class="crumbs"><a href="{p}index.html">Recipes</a>{icon("chevron-right", 12)}<span>{E(name)}</span></nav>
  <header class="org-head">
    {provider_logo(org, p, logos, 44, "logo logo-lg")}
    <div><h1>{E(name)}</h1><a class="mono muted" href="https://huggingface.co/{E(org)}">huggingface.co/{E(org)} {icon("external", 11)}</a></div>
  </header>
  {section_head(f"{len(rs)} {'recipe' if len(rs) == 1 else 'recipes'}")}
  <div class="grid cards-grid">{cards}</div>
</main>"""


def sidebar(recipes, current, prefix, logos):
    out = ['<nav class="sidebar"><div class="sidebar-inner"><div class="eyebrow pad">Providers</div>']
    for org, rs in by_org(recipes):
        open_ = org == current["org"]
        items = "".join(
            f'<a class="side-model{" active" if r["id"] == current["id"] else ""}" href="{prefix}{r["org"]}/{r["repo"]}/">'
            f"{icon((r['meta'].get('tasks') or ['text'])[0] if (r['meta'].get('tasks') or ['text'])[0] in ICONS else 'text', 10)}"
            f"<span>{E(r['repo'])}</span></a>"
            for r in rs
        )
        out.append(
            f'<details class="side-org"{" open" if open_ else ""}><summary>{provider_logo(org, prefix, logos, 18, "logo logo-sm")}'
            f"<span>{E(rs[0]['meta'].get('provider') or org)}</span>{icon('chevron-right', 12, 'side-chev')}</summary>"
            f'<div class="side-models">{items}</div></details>'
        )
    out.append("</div></nav>")
    return "".join(out)


def config_row(label, content, hint=""):
    h = f' title="{E(hint)}"' if hint else ""
    return f'<div class="config-row"><div class="config-label"{h}>{label}</div><div class="config-body">{content}</div></div>'


def builder(r, fix_pack, fix_status, prefix):
    g = r["gaudi"]
    keys, combos = variants(r, fix_pack)
    clients = client_commands(r)
    cards = g["cards"]
    rel = release_of(r)
    status = status_of(r)
    layout = (
        f"{cards['total']} card"
        + ("s" if cards["total"] > 1 else "")
        + f" &middot; TP {cards['tp']}"
        + (" &middot; EP" if cards.get("ep") else "")
    )
    hw = (
        f'<div class="pill-group"><button type="button" class="pill active" title="Intel Gaudi 2 (HL-225), 96 GB HBM per card">'
        f'<span class="status-dot {status}"></span><b>Gaudi 2</b><span class="mono muted">&nbsp;96G/card</span></button>'
        f'<button type="button" class="pill" disabled title="Not tested on Gaudi 3"><b>Gaudi 3</b><span class="muted">&nbsp;not tested</span></button></div>'
        f'<div class="config-sub mono">{layout}</div>'
    )
    img = g["image"]
    image_pills = "".join(
        f'<button type="button" class="pill{" active" if x == rel else ""}"{"" if x == rel else " disabled"} '
        f'title="{E(("vllm-gaudi " + img["vllm_gaudi"] + " at " + img["vllm_gaudi_commit"] + ", vLLM " + img["vllm_commit"]) if x == rel else "Not verified with this recipe")}">'
        f"<b>{x}</b></button>"
        for x in RELEASES
    )
    image = (
        f'<div class="pill-group">{image_pills}</div>'
        f'<div class="config-sub mono">vllm-gaudi {E(img["vllm_gaudi"])} @ {E(img["vllm_gaudi_commit"])} &middot; vLLM @ {E(img["vllm_commit"])} &middot; Gaudi SW {E(img["gaudi_sw"])}</div>'
    )
    v = (r.get("variants") or {}).get("default") or {}
    variant = (
        f'<div class="pill-group"><button type="button" class="pill active"><b class="upper">{E(v.get("precision", ""))}</b>'
        f'<span class="mono muted">&nbsp;&ge;{E(str(v.get("vram_minimum_gb", "")))} GB</span></button></div>'
        f'<div class="config-sub">{E(v.get("description", ""))}</div>'
    )
    rows = [config_row("Hardware", hw), config_row("Image", image), config_row("Variant", variant)]
    if keys:
        feats = "".join(
            f'<button type="button" class="pill toggle active" data-feature="{k}" aria-pressed="true" title="{E(r["features"][k].get("description", ""))}">'
            f'<span class="tick">{icon("check", 11)}</span>{E(FEATURE_LABELS.get(k, k))}</button>'
            for k in keys
        )
        rows.append(
            config_row(
                "Features",
                f'<div class="pill-group">{feats}</div>',
                "Turn off a feature to drop its flags from the command",
            )
        )
    req = g.get("required_fixes") or []
    order, standalone, profiles = fix_pack
    if req:
        optin = [f for f in order if f in req and f not in profiles[rel]] + [f for f in standalone if f in req]
        chips = [
            f'<a class="chip" href="{prefix}{DOCS_DIR}fixes/" title="The recommended generic fixes for {rel}">profile {rel}</a>'
        ]
        for f in optin:
            st = fix_status.get(f, {})
            chips.append(
                f'<a class="chip optin" href="{prefix}{DOCS_DIR}fixes/{f}/" title="{E(st.get("affects", ""))} ({E(st.get("status", ""))})">{E(f)}</a>'
            )
        fixes = f'<div class="chip-row">{"".join(chips)}</div><div class="config-sub">{len(req)} fixes this configuration depends on; opt-in fixes are named.</div>'
    else:
        fixes = '<div class="config-sub">None: this recipe runs on the stock image.</div>'
    rows.append(config_row("Fixes", fixes))
    tabs = "".join(
        f'<button type="button" class="pill{" active" if t == "shell" else ""}" data-target="{t}">{label}</button>'
        for t, label in (("shell", "Shell"), ("docker", "Docker"), ("kubernetes", "Kubernetes"))
    )
    rows.append(config_row("Launch", f'<div class="pill-group">{tabs}</div>'))
    pop = "".join(
        f'<div class="popover-wrap"><button type="button" class="cmd-btn" data-popover="{k}">{icon(ic, 12)}{label}</button>'
        f'<div class="popover" data-popover-body="{k}" hidden><div class="popover-head"><span>{label}</span>'
        f'<button type="button" class="cmd-btn" data-copy-text="{E(clients[k])}">{icon("copy", 12)}Copy</button></div>'
        f'<pre class="code-block"><code>{E(clients[k])}</code></pre></div></div>'
        for k, ic, label in (("curl", "terminal", "cURL"), ("bench", "gauge", "Bench"))
        if k in clients
    )
    notes = {
        "shell": "Run inside a vllm-gaudi container. <code>$COOKBOOK</code> is a checkout of this repository there."
        if uses_cookbook(r)
        else "Run inside a vllm-gaudi container.",
        "docker": "Writes the Shell tab to <code>launch.sh</code> and runs it in a container with all cards visible.",
        "kubernetes": f'A container to merge into <a href="{REPO_URL}/blob/main/kubernetes/manifests/vllm-deployment.yaml">vllm-deployment.yaml</a>. '
        f'See the <a href="{prefix}{DOCS_DIR}kubernetes/pod-spec/">pod spec</a> page.',
    }
    payload = {"features": keys, "commands": combos, "notes": notes}
    first = combos[",".join(sorted(keys))]["shell"]
    return f"""<div class="builder" data-builder>
  <div class="config">{"".join(rows)}</div>
  <div class="command">
    <div class="command-head">
      <span class="eyebrow" data-target-label>Shell</span>
      <div class="command-actions">{pop}<button type="button" class="cmd-btn primary" data-copy-command>{icon("copy", 12)}<span>Copy</span></button></div>
    </div>
    <pre class="command-body"><code data-command>{E(first)}</code></pre>
    <div class="command-note" data-note>{notes["shell"]}</div>
  </div>
  <script type="application/json" data-builder-payload>{json.dumps(payload).replace("</", "<\\/")}</script>
</div>"""


MD_EXTENSIONS = [
    "admonition",
    "attr_list",
    "def_list",
    "md_in_html",
    "tables",
    "toc",
    "pymdownx.details",
    "pymdownx.superfences",
    "pymdownx.tabbed",
    "pymdownx.highlight",
    "pymdownx.inlinehilite",
]
MD_CONFIG = {"pymdownx.tabbed": {"alternate_style": True}, "pymdownx.highlight": {"use_pygments": True}}


def guide_html(r, recipes, prefix):
    text = (ROOT / "docs" / "recipes" / r["page"]).read_text()
    text = re.sub(r"\A# .*\n", "", text)
    text = re.sub(r"^## Launch\n.*?(?=^## )", "", text, flags=re.M | re.S)
    out = markdown.markdown(text, extensions=MD_EXTENSIONS, extension_configs=MD_CONFIG)
    page_to_recipe = {}
    for x in recipes:
        page_to_recipe.setdefault(x["page"], x)

    def link(m):
        href = m.group(1)
        if re.match(r"^[a-z]+:|^#", href):
            return m.group(0)
        path, _, frag = href.partition("#")
        target = posixpath.normpath(posixpath.join("docs/recipes", path))
        frag = "#" + frag if frag else ""
        if target == "docs/recipes/index.md":
            return f'href="{prefix}index.html{frag or "#all"}"'
        name = target.rsplit("/", 1)[-1]
        if target.startswith("docs/recipes/") and name in page_to_recipe:
            x = page_to_recipe[name]
            return f'href="{prefix}{x["org"]}/{x["repo"]}/{frag}"'
        if target.startswith("docs/") and target.endswith(".md"):
            rel = target[len("docs/") : -len(".md")]
            rel = rel[: -len("index")] if rel == "index" or rel.endswith("/index") else rel + "/"
            return f'href="{prefix}{DOCS_DIR}{rel}{frag}"'
        return f'href="{REPO_URL}/blob/main/{target}{frag}"'

    return re.sub(r'href="([^"]+)"', link, out)


def port_links(r, prefix):
    port = port_of(r)
    if not port:
        return ""
    return (
        f'<a href="{prefix}{DOCS_DIR}ports/{port}/">How it was ported {icon("book", 11)}</a>'
        f'<a href="{REPO_URL}/tree/main/ports/{port}">Port code {icon("external", 10)}</a>'
    )


def recipe_page(r, recipes, fix_pack, fix_status, logos):
    p = "../../"
    m, g, meta = r["model"], r["gaudi"], r["meta"]
    status = status_of(r)
    moe = m.get("architecture") == "moe"
    tags = (
        [
            f'<span class="badge lg">{icon("layers" if moe else "cpu", 11)}{E(m.get("architecture", ""))}</span>',
            f'<span class="badge lg mono">{E(m.get("parameter_count", ""))}'
            + (f" / {E(m['active_parameters'])} active" if moe and m.get("active_parameters") else "")
            + "</span>",
            f'<span class="badge lg mono">{int(g["max_model_len"]):,} ctx served</span>',
            f'<span class="badge lg mono upper">{E(precision_of(r))}</span>',
            f'<span class="badge lg mono">vllm-gaudi {E(release_of(r))}</span>',
        ]
        + ([f'<span class="badge lg port">{icon("box", 11)}Port</span>'] if port_of(r) else [])
        + [f'<span class="badge lg secondary">{E(TASK_LABELS.get(t, t))}</span>' for t in meta.get("tasks") or []]
    )
    status_html = (
        f'<span class="status-pill verified">{icon("shield", 12)}Verified on Gaudi 2</span>'
        if status == "verified"
        else f'<span class="status-pill supported">{icon("shield", 12)}Supported: checks not re-run as written</span>'
    )
    checks = "".join(f"<li>{E(c)}</li>" for c in g.get("verified") or [])
    notes = "".join(f"<li>{E(n)}</li>" for n in g.get("notes") or [])
    glance = ""
    if checks:
        glance += f'<h3>{icon("check", 14)} Checks that passed</h3><ul class="checks">{checks}</ul>'
    if notes:
        glance += f"<h3>Notes</h3><ul>{notes}</ul>"
    related_ids = [x for x in recipes if x["page"] == r["page"] and x["id"] != r["id"]]
    related_ids += [x for x in recipes if x["id"] in (meta.get("related_recipes") or []) and x not in related_ids]
    related = " &middot; ".join(f'<a href="{p}{x["org"]}/{x["repo"]}/">{E(x["id"])}</a>' for x in related_ids)
    doc_url = f"{p}{DOCS_DIR}recipes/{r['page'][:-3]}/"
    body = f"""<div class="wrap recipe-layout">
{sidebar(recipes, r, p, logos)}
<main class="recipe-main">
  <nav class="crumbs"><a href="{p}index.html">Recipes</a>{icon("chevron-right", 12)}<a href="{p}{r["org"]}/">{E(meta.get("provider") or r["org"])}</a>{icon("chevron-right", 12)}<span>{E(r["repo"])}</span></nav>
  <header class="recipe-head">
    <div class="recipe-title-row">
      {provider_logo(r["org"], p, logos, 44, "logo logo-lg")}
      <div class="recipe-title-wrap">
        <h1 class="mono"><span class="muted">{E(r["org"])}/</span>{E(r["repo"])}</h1>
        <p class="recipe-desc">{E(meta.get("description", ""))}</p>
        <div class="recipe-links">
          {status_html}
          <a href="https://huggingface.co/{E(r["id"])}">Hugging Face {icon("external", 10)}</a>
          <a href="{doc_url}">Full guide {icon("book", 11)}</a>
          {port_links(r, p)}
        </div>
      </div>
    </div>
    <div class="badges">{"".join(tags)}</div>
  </header>
  {kernel_callout(r, p)}
  <section class="builder-section">
    {builder(r, fix_pack, fix_status, p)}
  </section>
  <section class="accordions">
    {f'<details class="accordion" open><summary>At a glance{icon("chevron", 16, "acc-chev")}</summary><div class="accordion-body prose">{glance}</div></details>' if glance else ""}
    <details class="accordion" open><summary>Guide{icon("chevron", 16, "acc-chev")}</summary><div class="accordion-body prose">{guide_html(r, recipes, p)}</div></details>
  </section>
  <footer class="recipe-foot">
    <span>Updated {E(str(meta.get("date_updated", "")))}</span>
    <a href="{REPO_URL}/edit/main/{r["yaml"]}">{icon("pencil", 12)} Edit recipe</a>
    <a href="{REPO_URL}/blob/main/{r["yaml"]}">{icon("box", 12)} recipe.yaml</a>
    <a href="{REPO_URL}/issues">{icon("bug", 12)} Report issue</a>
    {f'<span class="related">Related: {related}</span>' if related else ""}
  </footer>
</main>
</div>"""
    return body


# ---------------------------------------------------------------------------------------------------------------------
# Assets


def fetch_logos(orgs, out):
    logos = {}
    dest = out / "assets" / "providers"
    dest.mkdir(parents=True, exist_ok=True)
    for org in orgs:
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/{org}", headers={"User-Agent": "gaudi-cookbook-catalog"}
            )
            page_html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
            m = re.search(r"https://cdn-avatars\.huggingface\.co/[^\"&\s)]+", page_html)
            if not m:
                continue
            data = urllib.request.urlopen(m.group(0), timeout=20).read()
            ext = Path(m.group(0).split("?")[0]).suffix.lower() or ".png"
            if ext not in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
                ext = ".png"
            (dest / f"{org}{ext}").write_bytes(data)
            logos[org] = f"assets/providers/{org}{ext}"
        except Exception as exc:  # noqa: BLE001  a missing avatar falls back to a letter tile
            print(f"logo {org}: {exc}", file=sys.stderr)
    return logos


def pygments_css():
    light = HtmlFormatter(style="default").get_style_defs(".highlight")
    dark = HtmlFormatter(style="github-dark").get_style_defs("html.dark .highlight")
    strip = re.compile(r"^.*\{ background: [^}]*\}\n?", re.M)  # code backgrounds come from catalog.css
    return strip.sub("", light) + "\n" + strip.sub("", dark)


def search_index(recipes):
    return [
        {
            "id": r["id"],
            "org": r["org"],
            "repo": r["repo"],
            "provider": r["meta"].get("provider", ""),
            "desc": r["meta"].get("description", ""),
            "tasks": r["meta"].get("tasks") or [],
            "precision": precision_of(r),
            "cards": r["gaudi"]["cards"]["total"],
        }
        for r in recipes
    ]


def api_json(recipes, fix_pack):
    out = []
    for r in recipes:
        cmds = build_commands(r, set(r.get("features") or {}), fix_pack)
        out.append(
            {
                "id": r["id"],
                "title": r["meta"].get("title"),
                "provider": r["meta"].get("provider"),
                "description": r["meta"].get("description"),
                "status": status_of(r),
                "tasks": r["meta"].get("tasks") or [],
                "architecture": r["model"].get("architecture"),
                "parameters": r["model"].get("parameter_count"),
                "active_parameters": r["model"].get("active_parameters"),
                "precision": precision_of(r),
                "cards": r["gaudi"]["cards"],
                "image": r["gaudi"]["image"],
                "max_model_len": r["gaudi"]["max_model_len"],
                "features": list((r.get("features") or {}).keys()),
                "required_fixes": r["gaudi"].get("required_fixes") or [],
                "verified": r["gaudi"].get("verified") or [],
                "notes": r["gaudi"].get("notes") or [],
                "recipe_yaml": f"{REPO_URL}/blob/main/{r['yaml']}",
                "raw_yaml": f"https://raw.githubusercontent.com/thediymaker/gaudi-cookbook/main/{r['yaml']}",
                "page": f"{r['org']}/{r['repo']}/",
                "guide": f"{DOCS_DIR}recipes/{r['page'][:-3]}/",
                "commands": cmds,
                "clients": client_commands(r),
            }
        )
    return {"generator": "gaudi-cookbook catalog", "repository": REPO_URL, "recipes": out}


# ---------------------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="output directory (the Pages root; the MkDocs site goes in its docs/ folder)")
    ap.add_argument("--no-fetch", action="store_true", help="do not fetch provider avatars from Hugging Face")
    ap.add_argument("--check", action="store_true", help="compare generated commands with the docs Launch blocks")
    a = ap.parse_args()
    fix_pack = load_fix_pack()
    recipes = load_recipes()
    if a.check:
        bad = check(recipes, fix_pack)
        if not a.out:
            sys.exit(1 if bad else 0)
        if bad:
            sys.exit(1)
    if not a.out:
        ap.error("--out or --check is required")
    out = Path(a.out)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    for name in ("catalog.css", "catalog.js"):
        shutil.copyfile(HERE / "assets" / name, out / "assets" / name)
    shutil.copyfile(ROOT / "docs" / "assets" / "logo.svg", out / "assets" / "logo.svg")
    (out / "assets" / "pygments.css").write_text(pygments_css())
    orgs = sorted({r["org"] for r in recipes})
    logos = {} if a.no_fetch else fetch_logos(orgs, out)
    fix_status = load_fix_status()
    index = search_index(recipes)

    desc = (
        "Pick a model, copy the vllm serve line that runs on Intel Gaudi 2. Tested recipes with their fixes and checks."
    )
    (out / "index.html").write_text(
        page(f"{SITE_TITLE}: vLLM on Intel Gaudi", desc, "", home_page(recipes, logos), index)
    )
    for org, rs in by_org(recipes):
        d = out / org
        d.mkdir(parents=True, exist_ok=True)
        name = rs[0]["meta"].get("provider") or org
        (d / "index.html").write_text(
            page(
                f"{name} | {SITE_TITLE}",
                f"{name} models on Intel Gaudi 2 with vLLM.",
                "../",
                org_page(org, rs, logos),
                index,
            )
        )
    for r in recipes:
        d = out / r["org"] / r["repo"]
        d.mkdir(parents=True, exist_ok=True)
        title = f"{r['id']} on Gaudi 2 | {SITE_TITLE}"
        html_ = page(
            title,
            r["meta"].get("description", ""),
            "../../",
            recipe_page(r, recipes, fix_pack, fix_status, logos),
            index,
        )
        (d / "index.html").write_text(html_)
    (out / "recipes.json").write_text(json.dumps(api_json(recipes, fix_pack), indent=1))
    (out / ".nojekyll").write_text("")
    print(f"catalog: {len(recipes)} recipes, {len(orgs)} providers, {len(logos)} logos -> {out}")


if __name__ == "__main__":
    main()
