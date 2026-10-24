# SPDX-License-Identifier: Apache-2.0
"""Muse Spark numerics shared by the model and the CPU tests (pure torch, no vLLM import).

Every norm computes in fp32 and the caller rounds once. A zero-centred gain is built as
w.float() + 1, which is exact in fp32 for a bf16 weight. The transformers reference rounds 1 + w
in bf16 first (relative error up to 2**-9, large for post_expert_norm, whose weights reach about
+12), so this differs from it on purpose and matches the fp32 oracle the gates compare against.
"""

from __future__ import annotations

import torch


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) over the last dim, fp32 out."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def gain(w: torch.Tensor, c: float) -> torch.Tensor:
    """The effective norm gain c + w, as w.float() + c: c = 1 for the centred norms, 0 for the final norm."""
    return w.float() + c


def gate_coeffs(g: torch.Tensor, temperature: float, eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Norm-preserving residual gate coefficients (alpha, beta), fp32 per channel.

    beta = sigmoid(g / T); alpha = sqrt(max(eps, sigmoid(-g / T) * (1 + beta))) = sqrt(1 - beta^2).
    """
    w = g.float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=eps))
    return alpha, beta


def softcap(z: torch.Tensor, mult: float, cap: float) -> torch.Tensor:
    """Final logit softcap: cap * tanh(z * mult / cap), fp32."""
    return cap * torch.tanh(z.float() * mult / cap)
