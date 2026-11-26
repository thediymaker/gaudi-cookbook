# `mistral_large4_hpu.ref`: pure-torch Mistral Large 4 reference

CPU reference for the Gaudi 2 port (the perplexity bars in docs/ports/mistral-large-4.md come from it). Pure torch:
no vllm import, so it runs from `PYTHONPATH` alone. It follows the
vLLM PR (#64534) where the PR and HF differ (FP8 indexer fake-quant, bf16 SSMax rounding) and matches
HF `model.float()` to 1e-5 in fp32 when the FP8 indexer is off.

Names below are stable; the HPU ops and the DSpark drafter are tested against them.

## Run the stack

```python
import sys

sys.path.insert(0, "<cookbook>/ports/mistral-large-4/ml4_hpu")
import torch
from mistral_large4_hpu.ref import RefConfig, RefModel, RefOptions, TensorSource

ckpt = "<checkpoint dir>"  # a small synthetic checkpoint, or the AMD snapshot dir
src = TensorSource(ckpt)  # pread + page-cache drop, Mistral-native names
cfg = RefConfig.load(ckpt)  # config.json (HF text_config) or params.json
m = RefModel(src, cfg, RefOptions(dtype="bf16"))  # "fp32" for exact math; PR defaults otherwise
out = m.forward(ids)  # ids: LongTensor [T], positions 0..T-1
out.logits  # [T, 204800] fp32, soft-capped (30*tanh(z/30))
```

- `forward(ids, chunks=[n0, n1, ...])` runs a chunked prefill with per-layer caches (same result as one pass).
- `forward(ids, logits=False)` skips the head; `out.hidden` is the final residual stream (pre final norm).
- `RefOptions` switches: `dtype`, `fp8_indexer`, `attn_form` ("unabsorbed" | "absorbed"), `act_mxfp4`
  (W4A4), `rope_table_dtype`, `topk_ties`, plus the negative controls (`CONTROLS` maps the CLI names).
- Real checkpoint: `RefModel(..., keep_weights=False)` loads one layer at a time (a large CPU host is still
  needed for the full 72 layers).

## Hidden-state taps (DSpark, layer parity)

```python
out = m.forward(ids, taps=(1, 18, 36, 54, 71), logits=False)
out.taps[18]  # [T, 6144]: residual stream AFTER layer 18 (= entering layer 19), pre final norm
```

DSpark's target taps are the streams after layers 1, 18, 36, 54, 71 (per the upstream pull request), concatenated in that
order. For per-layer inputs, outputs and internals use a hook:

```python
def hook(l, x_in, x_out, rec):  # rec: sel [T, S] mask, index_scores, router_ids, router_w, boundary_ties
    ...


m.forward(ids, hook=hook)
```

## Per-op functions (signatures are the contract)

| module | function | what |
|---|---|---|
| `rope` | `yarn_inv_freq(dim, base, factor, orig_max, beta_fast, beta_slow, alpha=None)` | [32] fp32 YaRN x exp(alpha) (PR form) |
| `rope` | `inv_freq_from_cfg(cfg, alpha)`, `cos_sin(pos, inv_freq, dtype=fp32)` | tables, [T, 32] |
| `rope` | `rope_interleaved(x, cos, sin)` / `rope_neox` (control) | GPT-J pairs, output stays interleaved |
| `numerics` | `rmsnorm(x, w, eps, dt)`, `layernorm(x, w, b, eps, dt)`, `linear(x, w, dt)` | fp32 compute, one `dt` rounding |
| `numerics` | `log_positions(pos)`, `ssmax_scale(logp, s, b, dt)`, `apply_ssmax(q, lam, dt)` | lambda = s log(pos+1) + b (bf16-rounded in bf16) |
| `numerics` | `ue8m0_scale(amax)`, `fp8_fakequant_ue8m0(x, group)` | PR indexer FP8 e4m3, power-of-two scale |
| `numerics` | `fp8_block_dequant(w, scale)`, `mxfp4_dequant(u8, e8m0)`, `act_mxfp4(x)`, `softcap(z, cap)` | weights, W4A4, head |
| `indexer` | `indexer_q(a, w, lam_idx, cos, sin, use_rope, rope_dim, dt, fp8)` | [T, 32, 128], rope on dims [0:64] |
| `indexer` | `indexer_k(a, w, cos, sin, use_rope, rope_dim, dt, fp8)` | [T, 128], the index-K cache row |
| `indexer` | `indexer_weights(a, w, dt)` | [T, 32] with 128^-0.5 * 32^-0.5 folded |
| `indexer` | `indexer_scores(q, k, wts, q_pos, k_pos)`, `select_topk(scores, q_pos, k_pos, topk, skip_short, ties)` | [T, S] scores; selection mask |
| `indexer` | `mask_to_indices(sel, topk)` | PR form: [T, topk] rows, -1 padded |
| `attention` | `mla_q(a, w, use_rope, H, rope_dim, lam, cos, sin, dt)` | [T, 64, 128], rope on the last 64 |
| `attention` | `mla_kv(a, w, use_rope, 512, 64, cos, sin, dt, eps)` | latent row (c [T, 512], k_pe [T, 64]; zeros on NoPE) |
| `attention` | `attend_unabsorbed(q, c, kpe, wkv_b, sel, 128, dt, use_rope)` / `attend_absorbed(...)` | [T, 64*128], both forms |
| `moe` | `route(f, gate_w, 4)`, `swiglu_ffn(x, w1, w3, w2, dt)`, `moe_forward(f, gate_w, experts, shared, 4, dt)` | softmax top-4, renormalised |
| `stack` | `RefModel.layer_forward(l, x, pos, state, sels)` | one chunk through one layer |

Weight dicts use Mistral-native tensor names (see `RefModel.load_layer`). KV-shared layers read
`state.c[spec.kv_source]` (layer 35 in the real model); shared-indexer layers reuse
`sels[spec.index_source]`. `get_layer_specs(cfg)` gives the per-layer table.

Top-k ties break toward the earlier key (`topk_ties="stable"`). Exact ties occur at score 0 (a key
whose q.k is negative on every indexer head); the HPU top-k must use the same rule to be
bit-comparable at the boundary.
