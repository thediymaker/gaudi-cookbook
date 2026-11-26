"""DSpark draft oracle for Mistral Large 4 (`mtp_dspark.*`), pure torch, no vllm import.

Ground truth: vLLM PR #64534 (bd4fa2633):
  D   = vllm/models/mistral_large4/nvidia/dspark.py         (drafter: block embed, context K/V, layers, RNN step)
  SPK = vllm/v1/worker/gpu/spec_decode/dspark/speculator.py  (anchor-as-first block, sequential Markov sampling)
  DFL = vllm/v1/worker/gpu/spec_decode/dflash/speculator.py  (_prepare_dflash_inputs_kernel: positions, slots)
  Q3D = vllm/model_executor/models/qwen3_dspark.py           (DSparkMarkovHead: w1 embedding, w2 vocab bias)
  M   = vllm/models/mistral_large4/nvidia/model.py           (target taps: aux_hidden_state_layers)

What one draft computes, for a request whose target has processed positions 0 .. m-1 and whose last sampled token
x_m (the anchor) sits at position m:

  taps_t   = residual stream AFTER target layers 1, 18, 36, 54, 71 (pre final norm) at position t, concatenated
             [5 * 6144] in that order (M:243-246, 269-270: aux layers = ids + 1, the stream entering them)
  ctx_t    = target_norm(target_projection(taps_t))                       one GEMM over 30720, bf16 out (D:189-192)
  ring_j,t = [kv_a_norm_j(c) | rope(k_pe)] of wkv_a_j(ctx_t)              no input norm on ctx (D:194-227)
  block    = [E[x_m], mask_embedding[0], mask_embedding[1]] at positions m, m+1, m+2   (position_specific, D:170-182)
  layer j  = pre-norm dense MLA block, 64 heads x [64 nope | 64 rope], scale 128^-0.5, NO SSMax, plain YaRN (no
             LeRoPE alpha), NON-causal: every block query sees the ring rows of positions m-2048 .. m-1 and all 3
             block rows (window 2048 + 3 counted back from the block's end: D:64-79 draft_attention_windows), then
             a dense SiLU MLP 24576; residual adds as vLLM's fused add + RMSNorm (fp32 sum feeds the norm, the
             residual is stored rounded)
  hd       = final_norm(x + residual)                                      [3, 6144]
  base_i   = 30 tanh(lm_head(hd_i) / 30)                                   the target's lm_head, soft-capped
  RNN head (head_type "rnn", D:229-244, SPK:179-225): state = 0, prev = x_m; for i in 0..2:
             e = W1[prev]; gate, cand, out = split3(joint_proj([state, e, hd_i]) + b)
             state = sig(gate) state + (1 - sig(gate)) tanh(cand);  bias = W2 tanh(out)   (bias NOT soft-capped)
             d_i = argmax(base_i + bias);  prev = d_i
  drafts   = d_0, d_1, d_2 for positions m+1, m+2, m+3 (sample_pos = query_pos + 1)

Greedy only (argmax; the PR samples drafts with Gumbel noise at temperature > 0, SPK:124-162: not modelled here,
the HPU path drafts greedily and the verify sampler keeps the target's distribution either way).

Numerics profiles (`Numerics`): the PR on CUDA rounds activations to bf16 at every linear / norm / residual output
and computes the lm_head, the Markov bias and the RNN cell in bf16. Its RoPE table is fp32 on CUDA with flashinfer
(RotaryEmbeddingBase keeps the fp32 cache when use_flashinfer; the rotation is fp32 with one rounding, as the
plugin's rope.py) and bf16 everywhere else (the cache cast to the model dtype, each op rounded: forward_native).
Profile "pr" is the CUDA deployment (fp32 table), "pr_native" the bf16 table; "hpu" is what
mistral_large4_hpu/dspark.py computes (fp32 table, fp32 head / Markov bias / RNN as the target's fp32 lm_head,
the port's choice); "fp32" runs everything in fp32. Attention itself (scores,
softmax, P.V) is fp32 in every profile, output rounded to the activation dtype (the CUDA MLA kernels accumulate in
fp32; their internal bf16 roundings of the absorbed query are not reproduced).

Controls (tests only): `faults` = {"no_carry"} resets the RNN state at every position; {"shared_mask"} feeds
mask row 0 to both mask slots; {"lerope"} applies a LeRoPE alpha (passed in) to the draft RoPE; {"causal"} masks
the block causally; {"window+1"} lets the queries see one more context position. Each must change the drafts.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os

import torch
import torch.nn.functional as F

F32 = torch.float32
BF16 = torch.bfloat16
PREFIX = "mtp_dspark."


# --------------------------------------------------------------------------------------------- config
@dataclasses.dataclass
class DSparkSpec:
    dim: int = 6144
    n_layers: int = 3
    depth: int = 3  # block size = num_speculative_tokens (position-specific masks need it)
    window: int = 2048
    target_layers: tuple = (1, 18, 36, 54, 71)
    markov_rank: int = 512
    head_type: str = "rnn"
    draft_input: str = "position_specific"
    n_heads: int = 64
    kv_rank: int = 512
    qk_nope: int = 64
    qk_rope: int = 64
    v_dim: int = 128
    ffn: int = 24576
    vocab: int = 204800
    eps: float = 1e-5
    softcap: float = 30.0
    rope_theta: float = 10000.0
    yarn_orig: int = 512
    yarn_factor: float = 2048.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    @property
    def qk_dim(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def aux_layers(self) -> tuple:
        """Target layer indices whose INPUT stream is a tap (E3U:118-121: ids + 1; 72 = the final pre-norm one)."""
        return tuple(t + 1 for t in self.target_layers)

    @classmethod
    def from_params(cls, path: str) -> "DSparkSpec":
        """Mistral-native params.json (file or directory) -> spec (P:86-98 mtp block; CFG:477-545)."""
        if os.path.isdir(path):
            path = os.path.join(path, "params.json")
        p = json.load(open(path))
        mtp = p["mtp"]
        ds = mtp["dspark"]
        windows = {int(w) for w in str(mtp.get("ragged_attention") or "").split(",") if w.strip()}
        assert len(windows) <= 1, windows
        y = p.get("yarn") or {}
        return cls(
            dim=p["dim"],
            n_layers=int(mtp["n_layers"]),
            depth=int(mtp["depth"]),
            window=windows.pop() if windows else 0,
            target_layers=tuple(int(t) for t in str(mtp["target_activation_layers"]).split(",")),
            markov_rank=int(ds["markov_rank"]),
            head_type=ds.get("head_type", "vanilla"),
            draft_input=ds.get("draft_input", "shared_mask"),
            n_heads=p["n_heads"],
            kv_rank=p.get("kv_lora_rank", 512),
            qk_nope=p.get("qk_nope_head_dim", 64),
            qk_rope=p.get("qk_rope_head_dim", 64),
            v_dim=p.get("v_head_dim", 128),
            ffn=p.get("hidden_dim", 24576),
            vocab=p["vocab_size"],
            eps=p.get("norm_eps", 1e-5),
            softcap=float(p.get("softmax_tanh_gating") or 0.0),
            rope_theta=p.get("rope_theta", 10000.0),
            yarn_orig=int(y.get("original_max_position_embeddings", 512)),
            yarn_factor=float(y.get("factor", 2048)),
            beta_fast=float(y.get("beta", 32)),
            beta_slow=float(y.get("alpha", 1)),
        )


@dataclasses.dataclass(frozen=True)
class Numerics:
    act: torch.dtype = BF16  # activation dtype between ops
    rope_bf16: bool = True  # RoPE cos/sin table stored in bf16 (the PR) or fp32 (the plugin)
    head32: bool = False  # base logits, Markov bias and RNN cell in fp32

    @classmethod
    def profile(cls, name: str) -> "Numerics":
        return {
            "pr": cls(BF16, False, False),
            "pr_native": cls(BF16, True, False),
            "hpu": cls(BF16, False, True),
            "fp32": cls(F32, False, True),
        }[name]


# --------------------------------------------------------------------------------------------- weights
def checkpoint_names(spec: DSparkSpec) -> dict:
    """Every `mtp_dspark.*` tensor with its expected shape (the header has exactly these 38)."""
    D, L, R, V = spec.dim, spec.n_layers, spec.markov_rank, spec.vocab
    H, P, Rd, Vd, C = spec.n_heads, spec.qk_nope, spec.qk_rope, spec.v_dim, spec.kv_rank
    out = {
        "backbone.target_projection.weight": (D, D * len(spec.target_layers)),
        "backbone.target_norm.weight": (D,),
        "backbone.final_norm.weight": (D,),
        "backbone.mask_embedding": (spec.depth - 1, D) if spec.draft_input == "position_specific" else (D,),
        "autoregressive_head.embedding.weight": (V, R),
        "autoregressive_head.output.weight": (V, R),
    }
    if spec.head_type == "rnn":
        out["autoregressive_head.joint_proj.weight"] = (3 * R, D + 2 * R)
        out["autoregressive_head.joint_proj.bias"] = (3 * R,)
    for j in range(L):
        p = f"backbone.layers.{j}."
        out.update(
            {
                p + "attention.wq.weight": (H * (P + Rd), D),
                p + "attention.wkv_a_with_mqa.weight": (C + Rd, D),
                p + "attention.kv_a_norm.weight": (C,),
                p + "attention.wkv_b.weight": (H * (P + Vd), C),
                p + "attention.wo.weight": (D, H * Vd),
                p + "attention_norm.weight": (D,),
                p + "ffn_norm.weight": (D,),
                p + "feed_forward.w1.weight": (spec.ffn, D),
                p + "feed_forward.w3.weight": (spec.ffn, D),
                p + "feed_forward.w2.weight": (D, spec.ffn),
            }
        )
    return {PREFIX + k: v for k, v in out.items()}


def load_weights(ckpt_dir: str, spec: DSparkSpec, extra: tuple = ()) -> dict:
    """{name without the mtp_dspark. prefix: bf16 tensor} from a Mistral-native checkpoint dir (the index tells the
    shards), plus `extra` target tensors by their own names (e.g. tok_embeddings.weight, output.weight)."""
    from safetensors import safe_open

    idx = json.load(open(os.path.join(ckpt_dir, "consolidated.safetensors.index.json")))["weight_map"]
    want = list(checkpoint_names(spec)) + list(extra)
    by_file: dict = {}
    for n in want:
        by_file.setdefault(idx[n], []).append(n)
    W = {}
    for f, names in by_file.items():
        with safe_open(os.path.join(ckpt_dir, f), framework="pt") as fh:
            for n in names:
                W[n.removeprefix(PREFIX)] = fh.get_tensor(n)
    for n, shape in checkpoint_names(spec).items():
        t = W[n.removeprefix(PREFIX)]
        assert tuple(t.shape) == shape and t.dtype == BF16, (n, tuple(t.shape), shape, t.dtype)
    return W


def synthetic_weights(spec: DSparkSpec, seed: int = 0, scale: float = 1.0) -> dict:
    """Random bf16 weights at the spec's shapes (tests without a checkpoint): linears ~ N(0, 1/fan_in), norms ~ 1,
    tables ~ N(0, 1)."""
    g = torch.Generator().manual_seed(seed)
    W = {}
    for n, shape in checkpoint_names(spec).items():
        k = n.removeprefix(PREFIX)
        if k.endswith("norm.weight"):
            t = 1.0 + 0.1 * torch.randn(shape, generator=g)
        elif k.endswith(".bias") or k == "backbone.mask_embedding" or k.endswith("embedding.weight"):
            t = torch.randn(shape, generator=g) * (0.02 if k.endswith(".bias") else 1.0)
        else:
            t = torch.randn(shape, generator=g) * (scale / math.sqrt(shape[-1]))
        W[k] = t.to(BF16)
    return W


# --------------------------------------------------------------------------------------------- numerics
def _rnd(x: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    return x if x.dtype == dt else x.to(dt)


def linear(x: torch.Tensor, w: torch.Tensor, dt: torch.dtype, b: torch.Tensor | None = None) -> torch.Tensor:
    """x @ w.T (+ b): fp32 accumulation, one rounding to dt (a bf16 GEMM; with bias, cuBLAS's epilogue add). Run as
    one 2-D GEMM over the flattened rows, as vLLM's [tokens, features] layout does."""
    y = (x.reshape(-1, x.shape[-1]).to(F32) @ w.to(F32).t()).view(*x.shape[:-1], w.shape[0])
    if b is not None:
        y = y + b.to(F32)
    return _rnd(y, dt)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, dt: torch.dtype) -> torch.Tensor:
    """vllm ir.ops.rms_norm: fp32 statistics, cast to the weight dtype, times the weight, cast to the input dtype."""
    xf = x.to(F32)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return _rnd(_rnd(y, dt) * w.to(dt), dt)


def add_rms_norm(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor, eps: float, dt: torch.dtype):
    """vllm ir.ops.fused_add_rms_norm: s = x + res in fp32; the residual is s rounded, the norm reads s itself."""
    s = x.to(F32) + res.to(F32)
    y = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return _rnd(_rnd(y, dt) * w.to(dt), dt), _rnd(s, dt)


def softcap(z: torch.Tensor, cap: float) -> torch.Tensor:
    """LogitsProcessor soft cap: z / cap, tanh, * cap, each in z's dtype."""
    if not cap:
        return z
    return torch.tanh(z / cap) * cap


def yarn_inv_freq(spec: DSparkSpec, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """MistralRotaryEmbedding._compute_inv_freq without LeRoPE (the draft never calls set_lerope_alpha, D:119-120)."""
    dim, base = spec.qk_rope, spec.rope_theta
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=F32) / dim)
    ext, inter = 1.0 / pos_freqs, 1.0 / (spec.yarn_factor * pos_freqs)
    if alpha is not None:
        s = torch.exp(alpha.to(F32))
        ext, inter = ext * s, inter * s

    def corr(rot):
        return (dim * math.log(spec.yarn_orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = max(math.floor(corr(spec.beta_fast)), 0), min(math.ceil(corr(spec.beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=F32) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return inter * (1 - mask) + ext * mask


def rope_cos_sin(spec: DSparkSpec, pos: torch.Tensor, num: Numerics, alpha=None):
    """cos/sin [..., 32] at positions pos: the fp32 table (t * inv as an einsum, no mscale), stored bf16 (PR) or not."""
    inv = yarn_inv_freq(spec, alpha)
    ang = pos.to(F32).unsqueeze(-1) * inv
    cos, sin = ang.cos(), ang.sin()
    if num.rope_bf16:
        cos, sin = cos.to(BF16), sin.to(BF16)
    return cos, sin


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """Interleaved (GPT-J, is_neox_style=False) rotation of the last dim: pairs (2i, 2i+1). With a bf16 table (the
    PR) the ops run in bf16, each rounded (DeepseekScalingRotaryEmbedding.forward_static and the CUDA kernel's
    scalar_t arithmetic: x*cos, y*sin, then their sum); with an fp32 table, fp32 math and one rounding (rope.py)."""
    if cos.dtype == BF16 and dt == BF16:
        x0, x1 = x[..., 0::2].to(BF16), x[..., 1::2].to(BF16)
        return torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).flatten(-2)
    xf = x.to(F32)
    c, s = cos.to(F32), sin.to(F32)
    x0, x1 = xf[..., 0::2], xf[..., 1::2]
    return _rnd(torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], -1).flatten(-2), dt)


# --------------------------------------------------------------------------------------------- the drafter
class DSparkRef:
    """W: weights by checkpoint name (no prefix), bf16. embed [V, D] and lm_head [V, D]: the target's tables."""

    def __init__(
        self,
        spec: DSparkSpec,
        W: dict,
        embed: torch.Tensor,
        lm_head: torch.Tensor,
        num: Numerics | str = "pr",
        faults: frozenset = frozenset(),
        lerope_alpha=None,
    ):
        self.spec, self.W = spec, W
        self.embed, self.lm_head = embed, lm_head
        self.num = Numerics.profile(num) if isinstance(num, str) else num
        self.faults = frozenset(faults)
        self.alpha = lerope_alpha if "lerope" in self.faults else None
        self.trace: dict = {}

    # ---- context rows (the target side of a draft: every processed target token)
    def project_taps(self, taps: torch.Tensor) -> torch.Tensor:
        """taps [N, 5 * D] (bf16 residual streams, tap order) -> ctx [N, D] = target_norm(target_projection(taps))."""
        dt = self.num.act
        x = linear(taps, self.W["backbone.target_projection.weight"], dt)
        return rms_norm(x, self.W["backbone.target_norm.weight"], self.spec.eps, dt)

    def latent_rows(self, j: int, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Layer j's cache row [N, 576] = [kv_a_norm(c) | rope(k_pe)] of its input x [N, D] at positions pos [N]."""
        s, dt = self.spec, self.num.act
        p = f"backbone.layers.{j}.attention."
        kv = linear(x, self.W[p + "wkv_a_with_mqa.weight"], dt)
        c = rms_norm(kv[..., : s.kv_rank], self.W[p + "kv_a_norm.weight"], s.eps, dt)
        cos, sin = rope_cos_sin(s, pos, self.num, self.alpha)
        return torch.cat([c, rotate(kv[..., s.kv_rank :], cos, sin, dt)], -1)

    def context_rows(self, taps: torch.Tensor, pos: torch.Tensor) -> list:
        """Every layer's ring rows [N, 576] for target tokens with taps [N, 5D] at positions pos [N]."""
        ctx = self.project_taps(taps)
        return [self.latent_rows(j, ctx, pos) for j in range(self.spec.n_layers)]

    # ---- one draft
    def block_embeds(self, anchor: torch.Tensor) -> torch.Tensor:
        """anchor [B] -> [B, depth, D]: [E[anchor], mask rows] (position_specific: slot k reads row k-1)."""
        s = self.spec
        a = self.embed[anchor].to(self.num.act)
        m = self.W["backbone.mask_embedding"].to(self.num.act)
        if s.draft_input == "position_specific":
            rows = m if "shared_mask" not in self.faults else m[:1].expand(s.depth - 1, -1)
        else:
            rows = m.view(1, -1).expand(s.depth - 1, -1)
        return torch.cat([a.unsqueeze(1), rows.unsqueeze(0).expand(a.shape[0], -1, -1)], 1)

    def attention(self, j: int, a: torch.Tensor, qpos: torch.Tensor, keys: torch.Tensor, kvalid: torch.Tensor):
        """a [B, S, D] (normed), qpos [B, S], keys [B, K, 576] context rows (+ the block's own rows appended here),
        kvalid [B, K] -> attention output [B, S, D] (after wo)."""
        s, dt = self.spec, self.num.act
        p = f"backbone.layers.{j}.attention."
        B, S, _ = a.shape
        H, P, Rd, C = s.n_heads, s.qk_nope, s.qk_rope, s.kv_rank
        q = linear(a, self.W[p + "wq.weight"], dt).view(B, S, H, P + Rd)
        cos, sin = rope_cos_sin(s, qpos, self.num, self.alpha)
        q = torch.cat([q[..., :P], rotate(q[..., P:], cos.unsqueeze(-2), sin.unsqueeze(-2), dt)], -1)
        own = self.latent_rows(j, a, qpos)  # [B, S, 576]
        allk = torch.cat([keys, own], 1)  # [B, K + S, 576]
        allv = torch.cat([kvalid, torch.ones(B, S, dtype=torch.bool)], 1)
        wkv_b = self.W[p + "wkv_b.weight"].to(F32).view(H, P + s.v_dim, C)
        W_UK, W_UV = wkv_b[:, :P], wkv_b[:, P:]  # [H, P, C], [H, V, C]
        c, kpe = allk[..., :C].to(F32), allk[..., C:].to(F32)
        k_nope = torch.einsum("bkc,hpc->bkhp", c, W_UK)  # unabsorbed, fp32
        v = torch.einsum("bkc,hvc->bkhv", c, W_UV)
        sc = (
            torch.einsum("bshp,bkhp->bhsk", q[..., :P].to(F32), k_nope)
            + torch.einsum("bshr,bkr->bhsk", q[..., P:].to(F32), kpe)
        ) * (s.qk_dim**-0.5)
        mask = allv.view(B, 1, 1, -1).expand(B, 1, S, allk.shape[1])
        if "causal" in self.faults:
            K = keys.shape[1]
            tri = torch.ones(S, S, dtype=torch.bool).tril()
            mask = mask.clone()
            mask[..., K:] &= tri.view(1, 1, S, S)
        sc = sc.masked_fill(~mask, float("-inf"))
        pr = torch.softmax(sc, -1)
        o = _rnd(torch.einsum("bhsk,bkhv->bshv", pr, v), dt).reshape(B, S, H * s.v_dim)
        return linear(o, self.W[p + "wo.weight"], dt)

    def mlp(self, j: int, f: torch.Tensor) -> torch.Tensor:
        dt = self.num.act
        p = f"backbone.layers.{j}.feed_forward."
        g = linear(f, self.W[p + "w1.weight"], dt)
        u = linear(f, self.W[p + "w3.weight"], dt)
        h = _rnd(_rnd(F.silu(g.to(F32)), dt).to(F32) * u.to(F32), dt)  # SiluAndMul: silu rounded, then
        #                                                                             the product (CUDA kernel order)
        return linear(h, self.W[p + "w2.weight"], dt)

    def window_mask(self, m: torch.Tensor, ring_pos: torch.Tensor, ring_valid: torch.Tensor) -> torch.Tensor:
        """Context rows a draft at m attends: positions m - window .. m - 1 (the trainer mask: source - t < window,
        t <= source = m - 1; test_sliding_window_matches_trainer_mask)."""
        lo = self.spec.window + (1 if "window+1" in self.faults else 0)
        B = m.shape[0]
        return (ring_pos >= (m.view(B, 1) - lo)) & (ring_pos <= m.view(B, 1) - 1) & ring_valid

    def backbone(
        self, anchor: torch.Tensor, m: torch.Tensor, rings: list, ring_pos: torch.Tensor, ring_valid: torch.Tensor
    ) -> torch.Tensor:
        """anchor [B] at positions m [B]; rings[j] [B, K, 576] context rows at ring_pos [B, K] (valid [B, K]) ->
        hd [B, depth, D] (final-normed block hidden states). Only ring rows with m - window <= pos <= m - 1 count."""
        s, dt = self.spec, self.num.act
        B = anchor.shape[0]
        qpos = m.view(B, 1) + torch.arange(s.depth).view(1, -1)
        inwin = self.window_mask(m, ring_pos, ring_valid)
        x = self.block_embeds(anchor)
        res = None
        for j in range(s.n_layers):
            p = f"backbone.layers.{j}."
            if res is None:
                a, res = rms_norm(x, self.W[p + "attention_norm.weight"], s.eps, dt), x
            else:
                a, res = add_rms_norm(x, res, self.W[p + "attention_norm.weight"], s.eps, dt)
            o = self.attention(j, a, qpos, rings[j], inwin)
            f, res = add_rms_norm(o, res, self.W[p + "ffn_norm.weight"], s.eps, dt)
            x = self.mlp(j, f)
            self.trace[f"layer{j}"] = x.to(F32) + res.to(F32)  # the stream after layer j
        hd, _ = add_rms_norm(x, res, self.W["backbone.final_norm.weight"], s.eps, dt)
        self.trace["hd"] = hd
        return hd

    def base_logits(self, hd: torch.Tensor) -> torch.Tensor:
        """Soft-capped lm_head logits [B, depth, V] (fp32 head in the port; bf16 in the PR profile)."""
        hdt = F32 if self.num.head32 else BF16
        z = linear(hd, self.lm_head, hdt)
        return softcap(z, self.spec.softcap)

    def rnn_step(self, prev: torch.Tensor, hd_i: torch.Tensor, state):
        """D:229-244 markov_step: (tanh(out) [B, R], next state [B, R])."""
        hdt = F32 if self.num.head32 else BF16
        e = self.W["autoregressive_head.embedding.weight"][prev].to(hdt)
        if state is None or "no_carry" in self.faults:
            state = torch.zeros_like(e)
        joint = linear(
            torch.cat([state, e, hd_i.to(hdt)], -1),
            self.W["autoregressive_head.joint_proj.weight"],
            hdt,
            self.W["autoregressive_head.joint_proj.bias"],
        )
        R = self.spec.markov_rank
        g_raw, c_raw, o_raw = joint[..., :R], joint[..., R : 2 * R], joint[..., 2 * R :]
        gate = torch.sigmoid(g_raw)
        nxt = gate * state + (1.0 - gate) * torch.tanh(c_raw)
        return torch.tanh(o_raw), nxt

    def draft_from_hd(self, anchor: torch.Tensor, hd: torch.Tensor) -> torch.Tensor:
        """SPK:179-225 _sample_sequential, greedy: [B] anchors + hd [B, depth, D] -> drafts [B, depth]."""
        hdt = F32 if self.num.head32 else BF16
        base = self.base_logits(hd)
        W2 = self.W["autoregressive_head.output.weight"]
        prev, state, out, logits = anchor, None, [], []
        for i in range(self.spec.depth):
            if self.spec.head_type == "rnn":
                emb, state = self.rnn_step(prev, hd[:, i], state)
            else:
                emb = self.W["autoregressive_head.embedding.weight"][prev].to(hdt)
            lg = base[:, i] + linear(emb, W2, hdt)  # bias after the cap
            logits.append(lg)
            prev = lg.argmax(-1)
            out.append(prev)
        self.trace["logits"] = torch.stack(logits, 1)
        return torch.stack(out, 1)

    def draft(
        self,
        anchor: torch.Tensor,
        m: torch.Tensor,
        rings: list,
        ring_pos: torch.Tensor,
        ring_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ring_valid is None:
            ring_valid = ring_pos >= 0
        hd = self.backbone(anchor, m, rings, ring_pos, ring_valid)
        return self.draft_from_hd(anchor, hd)

    # ---- convenience: a whole sequence's context, one request
    def draft_at(self, taps: torch.Tensor, tokens: torch.Tensor, m: int) -> torch.Tensor:
        """One request: taps [N, 5D] for positions 0 .. N-1 (N >= m), tokens [>= m+1]; the draft at decode point m
        (anchor tokens[m], context positions max(0, m - window) .. m-1). -> drafts [depth]."""
        lo = max(0, m - self.spec.window)
        pos = torch.arange(lo, m)
        rows = self.context_rows(taps[lo:m], pos)
        return self.draft(tokens[m : m + 1], torch.tensor([m]), [r.unsqueeze(0) for r in rows], pos.unsqueeze(0))[0]
