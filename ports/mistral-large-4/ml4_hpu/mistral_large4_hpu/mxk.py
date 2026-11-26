"""The MXFP4 MoE decode kernels for Mistral Large 4's routed experts (kernels/mxfp4-moe, build variant ml4).

The ml4 build holds GLM's v8 kernels (mxk_w13x, mxk_w2x, mxk_deq) plus the grouped-route decode kernels (mxk_w13g,
mxk_w2g, mxk_w2c). Mistral's experts (I 6144) fit their limits after the load-time virtual split (moe.py, loader.py):
256 virtual experts of I 3072, top-8, 32 per EP8 rank, K 6144, T <= 16 tokens per v8 decode call (8T <= 128 routes).

Libraries (ML4_MXK_LIB, default /opt/gaudi-cookbook-ml4/mxk, where the port's install.sh builds them):
  libmxk_tpc.so      one kernel library: the variant's GUIDs + forwarding of every other GUID to the stock
                     libtpc_kernels.so, so GC_KERNEL_PATH stays a single file (the recipe cache can stay on);
  libmxk_op_lazy.so  torch.ops.custom_op.mxk_* (lazy-mode legacy custom ops).

setup_env() must run before the first HPU op of every process that opens a device (the graph compiler reads
GC_KERNEL_PATH when the device opens): the port's patches/kernel_env_hook.py puts a call at the top of
vllm_gaudi/__init__.py. It sets
  GC_KERNEL_PATH=<libmxk_tpc.so>, MXK_TPC_FWD_LIB=<the previous single GC_KERNEL_PATH>;
  PT_HPU_RECIPE_CACHE_CONFIG -> a sibling dir <dir>_mxk-<version>/ (cache keys do not cover the kernel library);
  PT_HPU_LAZY_ACC_PAR_MODE=0 (with the lazy accumulation thread on, a custom op fed by a view of a tensor made in
  the same lazy graph fails intermittently).
The MoE has no stock fallback by default (ML4_MXK=0 is the stock op on the same virtual tensors, for A/B): if the
libraries are missing with ML4_MXK=1, setup_env says so and the model refuses to build its MoE.

Lazy legacy custom ops cache by input signature: two DIFFERENT ops with identical input signatures must not share a
lazy graph. mxk_w13x (11 inputs), mxk_w2x (7), the grouped ops (2-D ids) and mxk_deq on w13 ([6144, 3072]) / w2
([6144, 1536]) never collide; kernels/mxfp4-moe/tests/run_parity.sh --variant ml4 checks it (SIG_DEVICE PASS).
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("ML4_MXK", "1") == "1"
LIB_DIR = os.environ.get("ML4_MXK_LIB", "/opt/gaudi-cookbook-ml4/mxk")
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
MAX_TOKENS = 16  # decode calls: 8 routes per token, kernel limit 128 routes
MAX_ROUTES = 128
MAX_K = 6144
MAX_I = 3072
INF = 3.0e38  # swiglu limit: off (Mistral's SiLU is unclamped)
_ops_loaded = False
# The v9g library (a superset of the v8p one:
# the same v8 ELFs plus mxk_w13g / mxk_w2g / mxk_w2c) groups a call's routes by local expert and unpacks each expert
# once per unit of up to 2 routes, bit-identical to v8 per route and per token, T <= 64 per call. ML4_MXK_V9G=1 with
# ML4_MXK_LIB pointing at a v9g library: decode calls of 2..64 tokens (multi-stream decode, DSpark verify rows) take
# it in one call instead of v8's 16-token pieces. The control key mxk_v9g overrides the env (bring-up A/B; the decode
# graphs are re-captured on a switch, vllm_model).
V9G = os.environ.get("ML4_MXK_V9G", "0") == "1"
V9G_MAX_TOKENS = 64
V9G_GMAX = int(os.environ.get("ML4_MXK_V9G_GMAX", "2"))  # the library's units hold at most 2 routes


def tpc_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_tpc.so")


def op_lib() -> str:
    return os.path.join(LIB_DIR, "libmxk_op_lazy.so")


def lib_version(path: str | None = None) -> str | None:
    """'mxk_tpc <version>' string of a single-file kernel library, None if absent."""
    try:
        data = open(path or tpc_lib(), "rb").read()
    except OSError:
        return None
    i = data.find(b"mxk_tpc ")
    if i < 0:
        return None
    return data[i : data.index(b"\0", i)].decode(errors="replace")


def setup_env() -> None:
    """Before the first HPU op: register the kernel library with the graph compiler. Idempotent. Also chains the
    ML4_INDEX_KERNEL=1 decode-indexer library (k1ix.setup_env) behind ours, or alone when the MoE library is off."""
    global ENABLED
    from . import k1ix

    if not ENABLED:
        _k1_cache_tag(k1ix.setup_env(None))
        return
    tpc = tpc_lib()
    ver = lib_version(tpc)
    if not (ver and os.path.isfile(op_lib())):
        ENABLED = False
        print(f"[ml4-mxk] ML4_MXK=1 but no single-file libmxk_tpc.so + libmxk_op_lazy.so in {LIB_DIR}", flush=True)
        _k1_cache_tag(k1ix.setup_env(None))
        return
    cur = os.environ.get("GC_KERNEL_PATH", STOCK)
    rest = [p for p in cur.split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"[ml4-mxk] GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only", flush=True)
        os.environ["MXK_TPC_FWD_LIB"] = STOCK
    os.environ["GC_KERNEL_PATH"] = tpc
    k1tag = k1ix.setup_env(tpc)
    rc = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG")
    if rc:
        path, _, rest_cfg = rc.partition(",")
        tag = "mxk-" + ver.split()[-1] + ("_" + k1tag if k1tag else "")
        if path and tag not in path:
            path = path.rstrip("/") + "_" + tag + "/"
            try:
                os.makedirs(path, exist_ok=True)
                os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = path + ("," + rest_cfg if rest_cfg else "")
            except OSError as e:
                print(f"[ml4-mxk] cannot create recipe cache dir {path}: {e}; keeping {rc}", flush=True)
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def _k1_cache_tag(k1tag: str | None) -> None:
    """MoE library off, decode-indexer library on: its own recipe-cache dir (cache keys do not cover kernel libs)."""
    rc = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG")
    if not (k1tag and rc):
        return
    path, _, rest_cfg = rc.partition(",")
    if path and k1tag not in path:
        path = path.rstrip("/") + "_" + k1tag + "/"
        try:
            os.makedirs(path, exist_ok=True)
            os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = path + ("," + rest_cfg if rest_cfg else "")
        except OSError as e:
            print(f"[ml4-k1] cannot create recipe cache dir {path}: {e}; keeping {rc}", flush=True)


def env_ok() -> bool:
    """True if this process's graph compiler sees our kernel library (setup_env ran before the device opened)."""
    return os.environ.get("GC_KERNEL_PATH") == tpc_lib() and os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE") == "0"


def load_ops() -> None:
    global _ops_loaded
    if not _ops_loaded:
        import torch

        torch.ops.load_library(op_lib())
        _ops_loaded = True


def has_v9g() -> bool:
    import torch

    return _ops_loaded and hasattr(torch.ops.custom_op, "mxk_w13g")


def v9g_flag() -> bool:
    """v9g requested (env, or the control file's mxk_v9g) and present in the loaded library."""
    from .control import ctl

    v = ctl().get("mxk_v9g")
    return bool(V9G if v is None else v) and has_v9g()


def v9g_for(T: int) -> bool:
    return 2 <= T <= V9G_MAX_TOKENS and v9g_flag()


def permute(x):
    """x [R, K] bf16 -> the kernels' per-512-k-chunk layout: xp[kc + 128 p + i] = x[kc + 4 i + p]. K % 512 == 0."""
    R, K = x.shape
    return x.reshape(R, K // 512, 128, 4).transpose(-1, -2).reshape(R, K)


def decode(x, vid, vw, W13, S13, W2, S2, topk: int, e_min: int, xp=None):
    """Routed experts of <= MAX_TOKENS tokens (v8), or of 2..V9G_MAX_TOKENS with v9g on: x [T, K] bf16, vid [T, topk]
    int32 virtual ids, vw [T, topk] routing weights -> [T, K] bf16 (this rank's local routes only; others give 0).
    xp: permute(x) when the caller issued it already."""
    import torch

    co = torch.ops.custom_op
    if xp is None:
        xp = permute(x)
    if v9g_for(x.shape[0]):
        ids = vid.to(torch.int32).reshape(x.shape[0], topk)
        ap = co.mxk_w13g(xp, ids, W13, S13, e_min, V9G_GMAX, 1.0, INF, 0.0, 1.0)
        part = co.mxk_w2g(ap, ids, W2, S2, e_min, V9G_GMAX)
        return co.mxk_w2c(part, ids, vw.reshape(x.shape[0], topk).to(torch.bfloat16), e_min, W2.shape[0])
    fi = vid.reshape(-1).to(torch.int32)
    ap = co.mxk_w13x(xp, fi, W13, S13, topk, e_min, 1.0, INF, 0.0, 1.0, False)  # silu(g) * u
    return co.mxk_w2x(ap, fi, vw.reshape(-1).to(torch.bfloat16), W2, S2, topk, e_min)


def prefill_v8p(x, vid, vw, W13, S13, W2, S2, e_min: int, n_global: int, chunk: int = 256):
    """The GLM v8p prefill branch: unpack this rank's virtual experts to bf16 (mxk_deq, exact), mark_step, then the
    stock bf16 fused MoE on them, mark_step. Transient: one layer's local experts in bf16 (3.375 GiB at 32 x I 3072)."""
    import habana_frameworks.torch as ht
    import torch

    co = torch.ops.custom_op
    n = W13.shape[0]
    q13 = [co.mxk_deq(W13[v], S13[v], 0) for v in range(n)]
    q2 = [co.mxk_deq(W2[v], S2[v], 0) for v in range(n)]
    ht.core.mark_step()
    y = torch.ops.hpu.mixture_of_experts(
        hidden_states=x,
        expert_routing_table=vid.to(torch.int32),
        router_weights=vw.to(torch.bfloat16),
        w12=q13,
        w3=q2,
        permuted_weights=True,
        activation="silu",
        experts_min=e_min,
        experts_max=e_min + n - 1,
        total_experts=n_global,
        chunk_size=chunk,
    )
    # The fused MoE runs in a lazy graph of its own. Left to share the next graph with the shared expert and the next
    # layer's attention, it corrupted other tensors of that graph at some prompt shapes: the shared expert's output
    # on one rank went to NaN (every prompt of 641-768 tokens) or changed run to run (1023-1025, 1600 tokens), so
    # those prompts were not repeatable and moving other flushes moved PPL (keeping q13 / q2
    # referenced longer did not help, this flush does).
    ht.core.mark_step()
    return y


def stock(x, vid, vw, W13, S13, W2, S2, e_min: int, n_global: int, chunk: int = 256):
    """The stock MXFP4 fused MoE op on the same virtual tensors (every other token count, and the ML4_MXK=0 A/B)."""
    import torch

    n = W13.shape[0]
    return torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights(
        x,
        vid.to(torch.int32),
        vw.to(torch.bfloat16),
        [W13[e] for e in range(n)],
        [W2[e] for e in range(n)],
        [S13[e] for e in range(n)],
        [S2[e] for e in range(n)],
        block_size=32,
        permuted_weights=True,
        activation="silu",
        experts_min=e_min,
        experts_max=e_min + n - 1,
        is_fp4=True,
        total_experts=n_global,
        chunk_size=chunk,
    )
