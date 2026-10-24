# SPDX-License-Identifier: Apache-2.0
"""Kimi-K2.7-Code on Intel Gaudi 2.

vLLM serves the model with its own KimiK25 / DeepSeek-V3 code; this package only holds the MXFP4 MoE decode
kernel hook (mxk.py, switch KIMI_MXK). patches/mxk_hooks.py calls it from vllm_gaudi, so it is not a vLLM plugin.
"""
