"""Mistral Large 4 as a vLLM model on Gaudi (vllm-gaudi v0.30, lazy mode, TP/EP = TP), DSV41's vllm_model shape.

Wraps the vLLM-independent stack (stack.ML4Stack: attention.py, indexer.py, moe.py over paged.py's caches) with:
  * ML4KVCache   one paged cache per (layer, kind) of paged.cache_layout, declared as an MLAAttentionSpec of its
                 own width (576 latent, 128 index-K, bf16). vLLM groups them all into ONE UniformTypeKVCacheSpecs
                 group (63 + 44 for the real model; the 9 KV-shared layers read layer 35's by id), so every cache
                 shares the request's single block table; the HPU runner allocates each at its own width.
  * the TP group  the stack's two all-reduces per layer and the vocab-parallel embedding / fp32 lm_head go through
                 vLLM's tensor-parallel group (tensor_model_parallel_all_reduce / _all_gather).
  * runner hook   runner_patch: block tables, request uids, block epochs, request slots, the host max position.
  * load_weights  ignores vLLM's iterator (it would read the whole checkpoint on every rank) and loads with
                 loader.load_stack: ranged per-rank reads, the virtual expert re-slice, FP8 block -> BF16 on the host.
                 mtp_dspark.* and vision tensors are never read (the DSpark drafter loads its own).

Checkpoints: the AMD Quark MXFP4 snapshot, or syn8l (the engine test). The
stack reads params.json (Mistral-native, as the loader) from the model directory; config adapter gives
vLLM the same model through --config-format mistral.
Serving: --config-format mistral --load-format mistral --tokenizer-mode mistral, --enforce-eager, prefix
caching optional (prefix.py: hits share every cache kind of a block and keep its tags), no async scheduling;
the mxk kernel env (mxk.setup_env) must run before the device opens (patches/kernel_env_hook.py, or the caller
before importing vllm).
"""

from __future__ import annotations

import json
import logging
import os

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from . import graphs, mxk
from . import runner_patch, spec
from .paged import DCP, TAG_MODE, TaggedCache, cache_layout, dcp_pad, index_cache, latent_cache, tag_violations
from .stack import ML4Stack

logger = logging.getLogger("vllm.mistral_large4_hpu")
TAG_LOG_EVERY = int(os.environ.get("ML4_TAG_LOG_EVERY", "512"))  # count mode: forwards between tag-check lines
# ML4_MEM_LOG=1: every rank logs the allocator's live peak (MaxInUse), in use and reserved whenever the peak has grown
# by >= ML4_MEM_LOG_STEP_GIB since its last line (checked at the start of each forward: lazy mode allocates when a
# step executes, so a line reports the steps before it). hl-smi shows the pool, not live tensors.
MEM_LOG = os.environ.get("ML4_MEM_LOG", "0") == "1"
MEM_LOG_STEP = float(os.environ.get("ML4_MEM_LOG_STEP_GIB", "0.25")) * 2**30
# ML4_CTL_EXEC=1 (test servers only, never when serving): the control file key "exec" names a Python file that
# every rank runs once per change (path, mtime) at the start of its next forward, with `model` (this module) in scope:
# diagnostics and same-server A/B patches without a reload (one load is 15 to 110 min from network storage). Change it only while
# the server is idle, as every control key.
CTL_EXEC = os.environ.get("ML4_CTL_EXEC", "0") == "1"
# ML4_PROMPT_SYNC=1 (default): a device synchronize after every forward that carries a prompt chunk (S > 8 rows).
# Prompt chunks return without a host sync (no token is sampled until the last chunk), so the host builds and queues
# a long prompt's chunks far ahead of the device, and every queued lazy launch holds host memory: one 128K prompt took
# the 8 workers +163 GiB of host RAM (replicated int8) and a 256K prompt took the container past its 440 GiB
# limit (OOM-killed). With the sync: +3.3 GiB, TTFT 394.6 -> 398.7 s. "0": the previous
# behavior. Control key prompt_sync.
PROMPT_SYNC = os.environ.get("ML4_PROMPT_SYNC", "1") == "1"


# ------------------------------------------------------------------- KV caches
class ML4KVCache(nn.Module, AttentionLayerBase):
    """One paged cache view. The HPU runner hands bind_kv_cache a (k, v, k_scales, v_scales) tuple; v is None for
    MLA-type specs. Tags are ours, same row count. dcp (n, r): the 1/n layout (paged.py, ML4_DCP): a block holds
    block_size / n rows of this rank's tokens, declared to vLLM as head_size width / n (width padded to a multiple
    of n), so the block count and pages stay vLLM's and the same bytes hold n times the tokens."""

    def __init__(
        self,
        prefix: str,
        layer_id: int,
        width: int,
        dtype: torch.dtype,
        block_size: int,
        vllm_config: VllmConfig,
        dcp=None,
    ):
        super().__init__()
        self.dcp = dcp
        n = dcp[0] if dcp else 1
        assert block_size % n == 0, (block_size, n)
        width = dcp_pad(width, n)
        self.prefix, self.layer_id, self.width, self.cache_dtype, self.block_size = (
            prefix,
            layer_id,
            width,
            dtype,
            block_size,
        )
        ctx = vllm_config.compilation_config.static_forward_context
        if prefix in ctx:
            raise ValueError(f"duplicate layer name {prefix}")
        ctx[prefix] = self
        self.kv_cache = None
        self.tags = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        n = self.dcp[0] if self.dcp else 1
        return MLAAttentionSpec(
            block_size=self.block_size, num_kv_heads=1, head_size=self.width // n, dtype=self.cache_dtype
        )

    def get_attn_backend(self):
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAAttentionBackend

        return HPUMLAAttentionBackend

    def bind_kv_cache(self, kv_cache) -> None:
        t = kv_cache[0] if isinstance(kv_cache, tuple | list) else kv_cache
        if t.dim() != 2:
            t = t.reshape(-1, self.width)
        assert t.shape[1] == self.width, (self.prefix, tuple(t.shape), self.width)
        self.kv_cache = t
        self.tags = torch.full((t.shape[0], 2), -1, dtype=torch.int32, device=t.device)

    def view(self, device) -> TaggedCache:
        if self.kv_cache is None:  # profiling run before allocation: one dump block
            self.kv_cache = torch.zeros(self.block_size, self.width, dtype=self.cache_dtype, device=device)
            self.tags = torch.full((self.block_size, 2), -1, dtype=torch.int32, device=device)
        label = self.prefix.rsplit("cache_", 1)[-1]
        if label == "latent":
            return latent_cache(self.kv_cache, self.tags, self.layer_id, self.dcp)  # ML4_KV_FP8 (paged.py)
        if label == "index_k":
            return index_cache(self.kv_cache, self.tags, self.layer_id, self.dcp)  # ML4_INDEX_U8 (paged.py)
        return TaggedCache(self.kv_cache, self.tags, label=label, layer=self.layer_id, dcp=self.dcp)


# ---------------------------------------------------------------------- model
class ML4ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        runner_patch.install()
        mc = vllm_config.model_config
        self.vllm_config = vllm_config
        self.model_dir = mc.model
        with open(os.path.join(self.model_dir, "params.json")) as f:
            params = json.load(f)
        self.tp_rank, self.tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        self.block_size = vllm_config.cache_config.block_size
        max_len = mc.max_model_len
        if mxk.ENABLED and mxk.env_ok():
            mxk.load_ops()
        elif mxk.ENABLED:
            raise RuntimeError(
                "ml4: mxk kernel env not set before the device opened (mxk.setup_env / "
                "patches/kernel_env_hook.py; ML4_MXK_LIB=%s)" % mxk.LIB_DIR
            )
        from . import k1ix

        if (k1ix.ENABLED or k1ix.ATTN) and k1ix.load_ops():  # ML4_INDEX_KERNEL / ML4_ATTN_KERNEL (default off)
            print(
                f"[ml4-k1] kernels on: indexer {int(k1ix.ENABLED)} attention {int(k1ix.ATTN)} "
                f"({k1ix.lib_version()}, {k1ix.LIB_DIR})",
                flush=True,
            )
        self.stack = ML4Stack(
            params, self.tp_rank, self.tp_size, placement=os.environ.get("ML4_PLACEMENT", "A"), max_pos=max_len
        )
        self.stack.allreduce = tensor_model_parallel_all_reduce  # vLLM's TP group (instance attribute)
        spec.build(self)  # ML4_SPEC=dspark: the DSpark drafter (T5, spec.py; model.dspark None else)
        self.caches = nn.ModuleDict()
        dcp = (self.tp_size, self.tp_rank) if DCP and self.tp_size > 1 else None
        for name, layer, kind, width in cache_layout(self.stack.roles):
            self.caches[f"{layer}_{kind}"] = ML4KVCache(
                name, layer, width, torch.bfloat16, self.block_size, vllm_config, dcp
            )
        runner_patch.configure(max_len, self.block_size, vllm_config.scheduler_config.max_num_seqs)
        logger.warning(
            "ml4: model built; TP %d, %d layers, %d paged caches, max_model_len %d, mxk %s, DCP %s",
            self.tp_size,
            len(self.stack.layers),
            len(self.caches),
            max_len,
            "ON" if mxk.ENABLED else "OFF",
            f"1/{dcp[0]} layout" if dcp else "off",
        )

    # --------------------------------------------------------------- forward
    def _cache_views(self, device):
        """Views over the bound caches, rebuilt only when a cache tensor changes (the profiling run uses 1-block
        dummies; KV binding replaces them once)."""
        key = tuple(id(c.kv_cache) for c in self.caches.values())
        if getattr(self, "_views_key", None) != key or any(c.kv_cache is None for c in self.caches.values()):
            views: dict = {}
            for k, c in self.caches.items():
                layer, kind = k.split("_", 1)
                views.setdefault(int(layer), {})[kind] = c.view(device)
            self._views = views
            self._views_key = tuple(id(c.kv_cache) for c in self.caches.values())
        return self._views

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.stack.embed_tokens(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.stack.embed_tokens(input_ids)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ) -> torch.Tensor:
        md = get_forward_context().attn_metadata
        if TAG_MODE == "count":
            self._forwards = getattr(self, "_forwards", 0) + 1
            if self._forwards % TAG_LOG_EVERY == 0:  # syncs; the isolation gate greps these lines
                v = tag_violations(by_label=True)
                logger.warning(
                    "ml4: tag check after %d forwards: %d violating rows of %d reads%s",
                    self._forwards,
                    v["rows"],
                    v["reads"],
                    f" {v['by_label']}" if v["by_label"] else "",
                )
        if input_ids.dim() == 1:  # a runner path that flattened anyway: one request row
            input_ids, positions = input_ids.view(1, -1), positions.view(1, -1)
        if MEM_LOG:
            self._mem_log(input_ids.shape)
        if CTL_EXEC:
            self._ctl_exec()
        hold = float(graphs.ctl().get("mem_hold_gib", 0) or 0)  # headroom test: keep this much allocated
        if hold != getattr(self, "_mem_hold_gib", 0.0):
            self._mem_hold = None
            if hold > 0:
                self._mem_hold = torch.empty(int(hold * 2**30), dtype=torch.uint8, device=input_ids.device).fill_(0)
            self._mem_hold_gib = hold
            logger.warning("ml4: rank %d holds %.2f GiB (control file mem_hold_gib)", self.tp_rank, hold)
        out = self._forward(input_ids, positions, md)
        if input_ids.shape[1] > 8 and input_ids.device.type == "hpu" and graphs.ctl().get("prompt_sync", PROMPT_SYNC):
            torch.hpu.synchronize()  # bound the host's queue of prompt-chunk launches (ML4_PROMPT_SYNC)
            if graphs.ctl().get("mem_each"):  # per-chunk device peak (control key mem_each): log, then reset
                st, g = torch.hpu.memory_stats(), 2**30
                logger.warning(
                    "ml4: mem rank %d chunk %s width %d: peak %.2f GiB, in use %.2f",
                    self.tp_rank,
                    tuple(input_ids.shape),
                    md.ml4_block_table.shape[-1] * self.block_size,
                    st.get("MaxInUse", 0) / g,
                    st.get("InUse", 0) / g,
                )
                torch.hpu.reset_peak_memory_stats()
        return out

    def _forward(self, input_ids, positions, md):
        if graphs.BOUNDARY_TRIM and self._trim_ok(input_ids, md):
            return self._forward_trim(input_ids, positions, md)
        ctx = runner_patch.batch_ctx_from_metadata(md, positions, self.block_size)
        T = getattr(md, "ml4_verify_rows", None) or 1
        if (
            T > 1
            and input_ids.shape[1] == 1
            and input_ids.shape[0] % T == 0
            and T <= graphs.DECODE_ROWS_MAX
            and spec.verify_rows_on()
        ):
            # a DSpark verify step in the rows layout (spec.as_rows): [B*T, 1] -> [B, T], one request per row,
            # and the hidden states back in the runner's flat order
            R = input_ids.shape[0]
            ids, rctx = spec.as_rows(input_ids, ctx, T)
            if ids.shape[1] == T:
                return self._forward_ctx(ids, rctx, rows=True).reshape(R, 1, -1)
        return self._forward_ctx(input_ids, ctx)

    def _forward_ctx(self, input_ids, ctx, rows: bool = False):
        spec.before_forward(self, ctx)  # DSpark: ring slots for the tap hooks, spec on/off (control file)
        views = self._cache_views(input_ids.device)
        if (
            graphs.LAYER_GRAPHS
            and self.stack.tap is None
            and (
                input_ids.shape[1] == 1 or rows or graphs.prefill_eligible(input_ids, ctx.block_table, self.block_size)
            )
            and not os.environ.get("ML4_ATTN_FORM")
            and not graphs.ctl().get("decode_eager")
        ):
            self._ensure_graphs(views)
            return self._decode_graphs(input_ids, ctx, self.tp_rank == 0)
        return self.stack(input_ids, ctx, views)

    def _ensure_graphs(self, views) -> None:
        from . import lean
        from . import attention as _attn, indexer as _ix, moe as _moe

        # a lean / v9g / decode-mm / router-pad / pad-skip / dry-run / decode-select switch recaptures the graphs
        gkey = (
            self._views_key,
            lean.enabled(),
            mxk.v9g_flag(),
            _attn.decode_mm(),
            _moe.router_pad(),
            _moe.pad_skip(),
            graphs.ctl().get("decode_graph_dry_run"),
            _ix.decode_select(),
        )
        if getattr(self, "_graphs_key", None) != gkey:
            self._decode_graphs = None
            ds = getattr(self, "dspark", None)
            self._decode_graphs = graphs.DecodeGraphs(
                self.stack, views, self.block_size, slot_buf=ds.slot_buf if ds is not None else None
            )
            self._graphs_key = gkey

    # ML4_BOUNDARY_TRIM=1 (r5): a plain decode step (S == 1, not a verify step) runs batch_ctx_from_metadata's eager
    # ops, spec.before_forward's slot copy and stack.embed_tokens (with its vocab-parallel all-reduce) inside the first
    # decode group graph (graphs.FrontStep): the same ops in the same order, so no eager lazy recipe and no separate
    # embed all-reduce before graph 0 (arm d). Every other forward takes the path above unchanged.
    def _trim_ok(self, input_ids, md) -> bool:
        if not (
            graphs.LAYER_GRAPHS
            and input_ids.shape[1] == 1
            and (getattr(md, "ml4_verify_rows", None) or 1) == 1
            and self.stack.tap is None
            and not os.environ.get("ML4_ATTN_FORM")
            and not graphs.ctl().get("decode_eager")
        ):
            return False
        ds = getattr(self, "dspark", None)
        if ds is None:
            return True
        return getattr(md, "ml4_req_slot", None) is not None and input_ids.shape[0] <= ds.slot_buf.shape[0]

    def _forward_trim(self, input_ids, positions, md):
        views = self._cache_views(input_ids.device)
        self._ensure_graphs(views)
        return self._decode_graphs.run_trim(
            input_ids,
            positions,
            md.ml4_block_table,
            md.ml4_req_uid,
            getattr(md, "ml4_block_epoch", None),
            getattr(md, "ml4_req_slot", None),
            self.tp_rank == 0,
        )

    def _ctl_exec(self) -> None:
        path = graphs.ctl().get("exec")
        if not path:
            return
        try:
            key = (path, os.stat(path).st_mtime)
        except OSError:
            return
        if getattr(self, "_exec_key", None) == key:
            return
        self._exec_key = key
        try:
            with open(path) as f:
                code = compile(f.read(), path, "exec")
            exec(code, {"model": self, "torch": torch, "logger": logger, "__name__": "ml4_ctl_exec"})
            logger.warning("ml4: rank %d ran control hook %s", self.tp_rank, path)
        except Exception:  # noqa: BLE001
            logger.exception("ml4: rank %d control hook %s failed", self.tp_rank, path)

    def _mem_log(self, shape) -> None:
        epoch = graphs.ctl().get("mem_epoch")  # a new value: restart the peak (same-server memory A/B)
        if epoch != getattr(self, "_mem_epoch", None):
            self._mem_epoch = epoch
            torch.hpu.reset_peak_memory_stats()
            self._mem_peak_logged = 0
            logger.warning("ml4: mem rank %d: peak reset (epoch %s)", self.tp_rank, epoch)
        st = torch.hpu.memory_stats()
        peak = st.get("MaxInUse", 0)
        if peak - getattr(self, "_mem_peak_logged", 0) < MEM_LOG_STEP:
            return
        self._mem_peak_logged = peak
        g = 2**30
        logger.warning(
            "ml4: mem rank %d: live peak %.2f GiB, in use %.2f, reserved %.2f, limit %.2f (next batch %s)",
            self.tp_rank,
            peak / g,
            st.get("InUse", 0) / g,
            torch.hpu.memory_reserved() / g,
            st.get("Limit", 0) / g,
            tuple(shape),
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm, fp32 lm_head, 30 tanh(z / 30), vocab shards all-gathered: [N, V] fp32."""
        z = self.stack.logits(hidden_states, gather=False)
        return tensor_model_parallel_all_gather(z.contiguous(), dim=-1) if self.tp_size > 1 else z

    # ----------------------------------------------------------------- load
    def load_weights(self, weights) -> set[str]:
        """Ignores the iterator (never consumed, so nothing is read through it) and reads the checkpoint with ranged
        per-rank reads instead."""
        from .loader import load_stack

        # vLLM runs its workers with one torch thread; the host FP8 -> BF16 dequant and the expert re-slice then pin
        # one core per rank (first start: 100 % CPU per worker, ~45 MB/s per rank). Threads only for the load.
        nt = torch.get_num_threads()
        torch.set_num_threads(int(os.environ.get("ML4_LOAD_THREADS", "10")))
        try:
            st = load_stack(self.stack, self.model_dir)
        finally:
            torch.set_num_threads(nt)
        self.stack.rope()  # tables from the loaded lerope.alpha, on the stack's device
        from . import numerics, r5k

        dev = self.stack.norm.device  # the device the forward's tensors carry (the caches key on it)
        if numerics.NUMERICS_RW:  # ML4_NUMERICS_RW: its device constants, before any graph capture
            numerics.rw_init(dev)
        if r5k.ready():  # ML4_R5_* (r5): the kernels' persistent dummy inputs, before any capture
            r5k.init(dev)
            from . import r6k

            if r6k.ready():  # ML4_R6_*: persistent dummy inputs on the same device, before any capture
                r6k.init(dev)
                if self.tp_rank == 0:
                    cap = getattr(r6k, "DUAL_ROPE_ROWS", "0")  # shown only when set (default line unchanged)
                    cap = f" rope_rows {cap}" if cap not in ("", "0") else ""
                    print(
                        f"[ml4-r6] ffk {r6k.FFK} (mode {r6k.FFK_MODE}) dual {r6k.DUAL}{cap} ropefma {int(r6k.ROPEFMA)} dev {dev}",
                        flush=True,
                    )
            if self.tp_rank == 0:
                print(
                    f"[ml4-r5] rms {int(r5k.RMS)} rope {int(r5k.ROPE)} router {int(r5k.ROUTER)} (mode {r5k.ROUTER_MODE}) "
                    f"ff8 {r5k.FF8} ssl {r5k.SSL} lat {int(r5k.LAT)} k2fold {int(r5k.K2FOLD)} dev {dev}",
                    flush=True,
                )
        spec.load(self)  # DSpark: draft weights (mtp_dspark.*), rings, tap hooks
        logger.warning(
            "ml4: rank %d loaded %.1f GB (fp8 scales not pow2: %s)",
            self.tp_rank,
            st["bytes"] / 1e9,
            st.get("fp8_scales_not_pow2"),
        )
        return {n for n, _ in self.named_parameters()}


__all__ = ["ML4ForCausalLM", "ML4KVCache"]
