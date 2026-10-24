# SPDX-License-Identifier: Apache-2.0
"""Engram conditional memory for DeepSeek-V4.1 (layers 1 and 14), HPU version.

Hashing runs on the HOST in int64 numpy from each request's full token history
(the runner has it), so there is no device int64 arithmetic (HPU narrows int64
silently in places) and no n-gram state carried across chunks: a chunk boundary
is just an index into the history. The device only gathers table rows.

Hash ids follow the reference NgramHashState (inference/engram.py in the checkpoint): token ids
go through the compressed-vocab map; position p hashes the n-grams ending at p
(2..4 tokens), looking back up to 3 tokens and padding with the pad token's
compressed id before the sequence start; products with per-layer odd multipliers
are XORed one look-back at a time; each (n-gram size, head) column takes the
result modulo its own prime. The converted checkpoint stores one U8 bucket tensor
per column, so the modulo IS the row inside that column's bucket.

Table rows are FP8 e4m3fn bytes with one E8M0 scale per 32; they are decoded by a
256-entry LUT (Gaudi2 would read ~9% of these codes as NaN through a float8 cast).
"""

from __future__ import annotations

import numpy as np
import torch


# ------------------------------------------------------------------- layout
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def bucket_primes(layer_ids, max_ngram_size, n_heads, vocab_size) -> list[list[int]]:
    """Per engram layer, its (max_ngram_size-1)*n_heads prime bucket sizes in hash-column
    order (n-gram size major, head minor); primes are drawn in order and never reused."""
    seen, out = set(), []
    for _ in layer_ids:
        cols = []
        for _ in range(max_ngram_size - 1):
            cur = vocab_size - 1
            for _ in range(n_heads):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                cols.append(cur)
        out.append(cols)
    return out


def hash_multipliers(layer_ids, max_ngram_size, compressed_vocab_size) -> np.ndarray:
    """[n_layers, max_ngram_size] int64, odd, bounded so id*multiplier fits int64."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = [
        np.random.default_rng(10007 * lid).integers(0, bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1
        for lid in layer_ids
    ]
    return np.stack(rows)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Same normalization as upstream/reference build_compressed_token_map."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tokenizer)
    for tid in range(len(tokenizer)):
        text = backend.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(tid)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text
        lookup[tid] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


class EngramHasher:
    """Host-side hash rows. hash_rows(history, positions) -> [n_tok, n_layers, n_cols]
    int64 rows inside each column's bucket."""

    def __init__(self, layer_ids, max_ngram_size, n_heads, vocab_size, token_map, compressed_vocab_size, pad_token_id):
        self.layer_ids = tuple(layer_ids)
        self.n = max_ngram_size
        self.token_map = np.asarray(token_map, dtype=np.int64)
        assert int(self.token_map.max()) + 1 == compressed_vocab_size, "compressed vocab size mismatch"
        self.pad = int(self.token_map[pad_token_id])
        self.mult = hash_multipliers(self.layer_ids, max_ngram_size, compressed_vocab_size)  # [Lg, n]
        primes = bucket_primes(self.layer_ids, max_ngram_size, n_heads, vocab_size)
        self.primes = np.asarray(primes, dtype=np.int64).reshape(len(self.layer_ids), max_ngram_size - 1, n_heads)
        self.n_cols = (max_ngram_size - 1) * n_heads

    def hash_rows(self, history: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """history: this request's token ids (at least up to max(positions));
        positions: absolute positions to hash."""
        comp = self.token_map[np.asarray(history, dtype=np.int64)]
        pos = np.asarray(positions, dtype=np.int64)
        toks = []
        for shift in range(self.n):
            src = pos - shift
            toks.append(np.where(src >= 0, comp[np.clip(src, 0, None)], self.pad))
        toks = np.stack(toks, -1)  # [n_tok, n]
        prod = toks[:, None, :] * self.mult[None]  # [n_tok, Lg, n]
        rolling, out = prod[..., 0], []
        for i in range(1, self.n):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])  # [n_tok, Lg, heads]
        return np.concatenate(out, axis=-1)  # [n_tok, Lg, n_cols]


# ----------------------------------------------------------------- device side
_E4M3FN_LUT: dict = {}


def e4m3fn_lut(device) -> torch.Tensor:
    key = str(device)
    if key not in _E4M3FN_LUT:
        lut = torch.arange(256, dtype=torch.uint8, device="cpu").view(torch.float8_e4m3fn).float()
        _E4M3FN_LUT[key] = lut.nan_to_num(0.0).to(device)
    return _E4M3FN_LUT[key]


def lookup_rows(
    weight_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    rows: torch.Tensor,
    block: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather table rows [..] from U8 storage and decode: e4m3fn(byte) * 2^(scale-127),
    rounded to bf16 once (exact: the product has <= 4 significant bits)."""
    flat = rows.reshape(-1)
    w = weight_u8.index_select(0, flat)
    s = scale_u8.index_select(0, flat)
    vals = e4m3fn_lut(w.device)[w.long()]
    scale = torch.pow(2.0, s.float() - 127.0)
    vals = (vals.unflatten(-1, (-1, block)) * scale.unsqueeze(-1)).flatten(-2)
    return vals.to(out_dtype).reshape(*rows.shape, weight_u8.shape[-1])


def engram_gate(
    h: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    hc: int,
    eps: float,
    clamp_value: float = 1e-6,
) -> torch.Tensor:
    """Reference Engram.forward after wkv. h [..., hc, D]; kv [..., (hc+1)*D] bf16."""
    D = h.shape[-1]
    key, value = kv.split([hc * D, D], dim=-1)
    key = key.float().unflatten(-1, (hc, D))
    weight = q_weight.float() * k_weight.float()
    hf = h.float()
    from .numerics import mean_sq

    rstd = torch.rsqrt(mean_sq(hf, False) + eps) * torch.rsqrt(mean_sq(key, False) + eps)
    dot = (hf * weight * key).sum(-1) * rstd * D**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (hf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)
