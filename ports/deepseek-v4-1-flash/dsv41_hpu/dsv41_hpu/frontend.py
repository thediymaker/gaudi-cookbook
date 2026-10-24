# SPDX-License-Identifier: Apache-2.0
"""Serving-frontend corrections for DeepSeek-V4.1 in vLLM v0.30 (API server process).

Reasoning-effort aliases: vLLM's copy of the encoder maps low/high/xhigh/max to
25/50/75/100, while the reference encoder shipped with the checkpoint
(encoding/encoding.py, and the model card) maps low/high/max to 50/75/100 with
"high" the default. Unpatched, every thinking request without an explicit effort
renders "Reasoning Effort: 50" instead of 75. The alias table is updated in place
(deepseek_v41.py imported the same dict object); "xhigh", which the reference does
not define, stays accepted at vLLM's 75 so such requests do not start failing.

OpenAI clients also send request-level reasoning_effort "medium" (OpenAI's default)
and "minimal", which the V4.1 encoder rejects with a 400. They are accepted as
aliases of the reference default ("high", 75) and of "low" (50).
"""

REFERENCE_EFFORT = {"low": 50, "high": 75, "max": 100}
OPENAI_COMPAT = {"medium": 75, "minimal": 50}


def install() -> None:
    try:
        from vllm.tokenizers import deepseek_v41_encoding as enc
    except ImportError:
        return
    enc.REASONING_EFFORT_MAPPINGS.update(REFERENCE_EFFORT)
    enc.REASONING_EFFORT_MAPPINGS.update(OPENAI_COMPAT)
