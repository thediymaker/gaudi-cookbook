# SPDX-License-Identifier: Apache-2.0
"""Load format `beam` (opt-in with --load-format beam): the public checkpoint, loaded about 6x faster.

The stock loader (load format auto) is bound by host CPU and dispatch, not by storage. Per TP rank it
reads the whole 503 GB checkpoint, runs vllm-gaudi's Gaudi 2 FP8 rescale on every full tensor on the
CPU before the rank's slice is taken, and copies about 76,000 tensors one by one with a device
synchronize after each. This loader reads the same safetensors files in place (no converted copy, no
extra disk) and changes no numerics:

  * ranged, parallel reads: only the expert rows the rank keeps, and under EP only its own experts
    (load_plan.plan_jobs), with BEAM_LOAD_THREADS reader threads (default 16) and at most 4 GiB in
    flight;
  * the identical rescale expression on the kept region only, checked at start against vllm-gaudi's
    wrapper on all 256 FP8 bit patterns (the load refuses to start if they differ);
  * each MoE layer's expert stacks assembled on the host and copied to the device once per parameter
    (load_plan.ExpertStage): 196 synced copies per rank instead of 75,264.

Every other tensor (attention, dense MLP, norms, router, embeddings, scales of non-expert layers) goes
through the model's own load_weights with its full checkpoint shape and the stock weight loaders. While
the model consumes a tensor whose rescale was already applied, vllm-gaudi's scale_adjustment flag (read
by the wrapper on every call) is off for that tensor only.
"""

from __future__ import annotations

import logging
import os
import time

import torch
from torch import nn
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .load_plan import ExpertStage, Reader, gaudi2_rescale, plan_jobs

logger = logging.getLogger("vllm.beam_hpu.loader")

LOAD_FORMAT = "beam"
THREADS = int(os.environ.get("BEAM_LOAD_THREADS", "16"))
WINDOW_BYTES = 4 << 30
FILE_GROUP = 8  # files read interleaved, so cold reads span several files
TORCH_THREADS = 1  # torch intra-op threads while loading; the reader threads give the parallelism


def _moe_plan(model: nn.Module) -> dict:
    """layer id -> (MoE TP size, MoE TP rank, global -> local expert id map) from the built model."""
    out = {}
    for i, layer in enumerate(model.model.layers):
        if not getattr(layer, "is_moe", False):
            continue
        routed = layer.mlp.experts.routed_experts
        mc = routed.moe_config
        if not mc.is_act_and_mul:
            raise NotImplementedError("beam_hpu loader: non-gated experts")
        out[i] = (mc.moe_parallel_config.tp_size, mc.tp_rank, routed._map_global_expert_id_to_local_expert_id)
    return out


def _self_check_rescale() -> None:
    """The pre-applied rescale must equal vllm-gaudi's wrapper bit for bit (catches a changed wrapper)."""
    from vllm_gaudi.extension import ops as hpu_ops

    g = torch.Generator().manual_seed(0)
    w = (torch.randn(64, 257, generator=g) * 100).to(torch.float8_e4m3fn)
    w.view(torch.uint8)[0, :256] = torch.arange(256, dtype=torch.uint8)  # every fp8 bit pattern
    ref = hpu_ops.gaudi_weight_wrapper(lambda p, lw: lw)(None, w)
    if not torch.equal(ref.view(torch.uint8), gaudi2_rescale(w).view(torch.uint8)):
        raise RuntimeError("beam_hpu loader: gaudi_weight_wrapper no longer matches the pre-applied rescale")
    s = torch.rand(8, 1, generator=g).to(torch.bfloat16)
    if not torch.equal(hpu_ops.gaudi_weight_wrapper(lambda p, lw: lw)(None, s), s * 2.0):
        raise RuntimeError("beam_hpu loader: gaudi_weight_wrapper scale path changed")


@register_model_loader(LOAD_FORMAT)
class BeamModelLoader(DefaultModelLoader):
    """DefaultModelLoader with Beam's ranged, parallel weight iterator."""

    def _st_files(self, model_name_or_path: str, revision) -> list[str]:
        # DefaultModelLoader picks files by load format; this one reads plain safetensors
        fmt = self.load_config.load_format
        self.load_config.load_format = "safetensors"
        try:
            _, files, use_st = self._prepare_weights(model_name_or_path, None, revision, False, None)
        finally:
            self.load_config.load_format = fmt
        if not use_st:
            raise RuntimeError("beam_hpu loader: safetensors checkpoint required")
        return files

    def download_model(self, model_config) -> None:
        self._st_files(model_config.model, model_config.revision)

    def load_weights(self, model: nn.Module, model_config) -> None:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise RuntimeError("beam_hpu loader: load format 'beam' is for BeamForCausalLM only")
        from vllm_gaudi.extension import ops as hpu_ops
        from vllm_gaudi.extension.runtime import get_config

        cfg = get_config()
        orig_adjust = cfg.scale_adjustment
        rescale = bool(hpu_ops.is_hpu_gaudi2 and orig_adjust)
        if rescale:
            _self_check_rescale()
        files = self._st_files(model_config.model, model_config.revision)
        jobs, skipped = plan_jobs(files, _moe_plan(model), rescale, FILE_GROUP)
        total = sum(j.read_bytes for j in jobs)
        ckpt = sum(os.path.getsize(f) for f in files)
        logger.warning(
            "beam_hpu loader: %d tensors, reading %.2f GiB of %.2f GiB (%d expert tensors of other EP ranks "
            "skipped), %d threads; Gaudi 2 rescale pre-applied: %s",
            len(jobs),
            total / 2**30,
            ckpt / 2**30,
            skipped,
            THREADS,
            rescale,
        )
        stages = {}
        for i, layer in enumerate(model.model.layers):
            if getattr(layer, "is_moe", False):
                routed = layer.mlp.experts.routed_experts
                why = ExpertStage.check(routed)
                if why is None:
                    stages[i] = ExpertStage(routed, scale_x2=rescale)
                else:
                    logger.warning("beam_hpu loader: layer %d experts take the stock path (%s)", i, why)
        reader = Reader(jobs, THREADS, WINDOW_BYTES, stages)
        t_wait = t_dev = 0.0
        dev_bytes = 0
        old_threads = torch.get_num_threads()
        torch.set_num_threads(TORCH_THREADS)
        t0 = time.perf_counter()

        def stream():
            nonlocal t_wait, t_dev, dev_bytes
            it = iter(reader)
            try:
                while True:
                    tw = time.perf_counter()
                    try:
                        j, t = next(it)
                    except StopIteration:
                        return
                    t_wait += time.perf_counter() - tw
                    if t is None:  # staged expert tensor
                        td = time.perf_counter()
                        dev_bytes += stages[j.exp[0]].done(j)
                        t_dev += time.perf_counter() - td
                        continue
                    # off only while the model's loader consumes a tensor whose rescale is already applied
                    cfg.scale_adjustment = orig_adjust and j.rescale is None
                    yield j.name, t
                    cfg.scale_adjustment = orig_adjust
            finally:
                cfg.scale_adjustment = orig_adjust
                it.close()

        try:
            model.load_weights(stream())
        finally:
            cfg.scale_adjustment = orig_adjust
            torch.set_num_threads(old_threads)
        bad = [i for i, st in stages.items() if not st.complete()]
        if bad:
            raise RuntimeError(f"beam_hpu loader: expert tensors missing for layers {bad[:10]}")
        dt = time.perf_counter() - t0
        logger.warning(
            "beam_hpu loader: experts of %d layers batched (%.2f GiB in %d device copies, %.1f s); loaded %.2f GiB "
            "in %.1f s; reader threads read %.1f s and rescaled %.1f s (summed); waited %.1f s for reads",
            len(stages),
            dev_bytes / 2**30,
            4 * len(stages),
            t_dev,
            reader.bytes / 2**30,
            dt,
            reader.t_read,
            reader.t_rescale,
            t_wait,
        )


__all__ = ["BeamModelLoader", "LOAD_FORMAT"]
