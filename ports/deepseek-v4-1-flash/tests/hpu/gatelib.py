# SPDX-License-Identifier: Apache-2.0
"""Shared client helpers for the device gates (standard library only). Every gate talks to a running server.

Env: DSV41_GATE_URL (default http://localhost:8000). The served model name is read from /v1/models.
"""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request

URL = os.environ.get("DSV41_GATE_URL", "http://localhost:8000").rstrip("/")
NO_THINK = {"chat_template_kwargs": {"thinking": False}}
_model: list[str] = []


def model() -> str:
    if not _model:
        with urllib.request.urlopen(URL + "/v1/models", timeout=60) as r:
            _model.append(json.load(r)["data"][0]["id"])
    return _model[0]


def post(path: str, body: dict, timeout: float = 3600) -> dict:
    body = {"model": model(), **body}
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post_status(path: str, body: dict, timeout: float = 1800) -> tuple[int, dict | str]:
    """(HTTP status, JSON body) or (status, error text); -1 when the connection fails (a dead engine)."""
    try:
        return 200, post(path, body, timeout)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]
    except Exception as e:
        return -1, repr(e)[:300]


def chat(text: str, max_tokens: int, **extra) -> str:
    r = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            **NO_THINK,
            **extra,
        },
    )
    return r["choices"][0]["message"].get("content") or ""


def metric(name: str) -> float:
    with urllib.request.urlopen(URL + "/metrics", timeout=600) as r:
        txt = r.read().decode()
    vals = [float(ln.split()[-1]) for ln in txt.splitlines() if ln.startswith(name) and not ln.startswith("#")]
    return sum(vals)


_SUBJ = [
    "the archive",
    "a courier",
    "the old bridge",
    "our committee",
    "the river pilot",
    "a small museum",
    "the harbor office",
    "the night train",
    "a weather station",
    "the orchard crew",
    "the town library",
    "a ferry captain",
    "the survey team",
    "the clock tower",
    "a print shop",
    "the mountain road",
]
_VERB = [
    "recorded",
    "repaired",
    "described",
    "measured",
    "delivered",
    "inspected",
    "catalogued",
    "painted",
    "postponed",
    "announced",
    "translated",
    "planted",
    "photographed",
    "rebuilt",
    "mapped",
    "audited",
]
_OBJ = [
    "the spring schedule",
    "a set of brass gauges",
    "the eastern wall",
    "forty crates of paper",
    "the lighthouse lamp",
    "a list of tide tables",
    "the winter accounts",
    "three wooden benches",
    "the northern field",
    "a box of old letters",
    "the drainage canal",
    "the market stalls",
]
_TAIL = [
    "before the rain arrived",
    "after a long delay",
    "with unusual care",
    "for the third time this year",
    "while the square was empty",
    "as the committee had asked",
    "without any fuss",
    "late in the evening",
]


def filler(n_words: int, seed: int) -> list[str]:
    """Deterministic, plain English filler of about n_words words that carries no facts a question could ask for."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n_words:
        s = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} {rng.choice(_TAIL)}."
        out.extend((s[0].upper() + s[1:]).split())
    return out[:n_words]


def verdict(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{name} {'PASS' if ok else 'FAIL'}{(' ' + detail) if detail else ''}", flush=True)
    return ok
