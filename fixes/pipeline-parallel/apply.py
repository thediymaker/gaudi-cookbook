#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""pipeline-parallel: make --pipeline-parallel-size > 1 work on the HPU runner, stages linked over the host network.

On both releases the model classes split their layers by pipeline rank and return IntermediateTensors on a
non-last stage, and _update_states already takes the scheduler's token ids there, but the HPU runner never moves
the hidden states between stages: _execute_model_generic calls the model without intermediate_tensors (a
non-first stage stops at `assert intermediate_tensors is not None`), and a non-last stage would try to compute
logits from IntermediateTensors. This fix adds the missing pieces. With --pipeline-parallel-size 1 every hook
calls the original method, so it changes nothing for single-stage serving.

Runner (vllm_gaudi/v1/worker/hpu_model_runner.py, code appended; existing methods are not edited):
  * A non-last stage's sample_tokens prepares the step exactly as the full one does (the same helpers, in the
    same order, so fixes on those helpers apply to every stage), runs the prefill forwards and then the decode
    forward with the same padded shapes as the last stage, and sends each forward's IntermediateTensors to the
    next stage. No logits and no sampling: the scheduler sends the sampled tokens back, which needs
    --no-async-scheduling. The prompt-order fix's sort runs here too when that fix is applied.
  * A non-first stage receives the previous stage's tensors before each forward and passes them as
    intermediate_tensors.
  * Transfer over gloo on the pipeline group's CPU group (the host NIC; set GLOO_SOCKET_IFNAME): per forward a
    fixed 64 x int64 header (magic, sequence number, dtypes, shapes) and one payload. Payloads of 1 MiB or more
    (prefill chunks) are split: each tensor-parallel rank sends 1/tp over its own connection and the receiving
    stage all-gathers the pieces over its local HCCL group. Sends are asynchronous, receives block, and the
    receiver waits for the received tensors on the device before the forward is queued. A wrong sequence number
    or a token count that is not the receiver's own padded batch stops the engine instead of decoding another
    batch's hidden states.
  * IntermediateTensors gets __iter__ (over its tensors): Habana's module post-forward hook iterates non-tensor
    outputs, and with only __getitem__ that loop never ends.
  * A non-last stage's IntermediateTensors are cloned inside the captured graph: with tensor parallelism they are
    results of in-place collectives, which an HPU graph with the tensor cache disabled frees after replay.
Worker (vllm_gaudi/v1/worker/hpu_worker.py): the start-up world check (an all-reduce over the default HCCL group,
  which spans the nodes) runs over gloo, and HCCL is checked on the node-local tensor-parallel group, so the nodes
  need no RoCE scale-out links. VLLM_HPU_PP_HCCL_WORLD_CHECK=1 keeps the HCCL world check.
Scheduler (vllm/v1/core/sched/scheduler.py): with pipeline parallelism and no async scheduling a step schedules at
  most ceil(running / pp) running requests, so the running requests form pp microbatches that follow each other
  through the stages. Without it one step takes every request, the next finds them all in flight, and the stages
  take turns. --max-num-seqs stays the total over all microbatches.
Loader (vllm/model_executor/model_loader/weight_utils.py): each stage skips the checkpoint tensors of decoder
  layers it does not own (`model.layers.N.*` outside its range; layers >= num_hidden_layers, the MTP layers, except
  on the last stage with speculative decoding) before reading them. Stock vLLM reads every tensor and drops the
  other stage's layers afterwards. Text-only configurations only (a vision tower also has `layers.N`).

VLLM_HPU_PP_STATS_EVERY=N (default 0, off): tensor-parallel rank 0 of each stage logs transfer counters every N
messages.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "pipeline-parallel"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# ---------------------------------------------------------------------------------------------------- runner
RUNNER_ANCHORS = (
    "\nclass HPUModelRunner(",
    "    def sample_tokens(self, grammar_output",
    "    def _ensure_decodes_first(self, scheduler_output",
    "\ndef ensure_multi_token_decodes_last(b",
    "    def _get_prompts_and_decodes(",
    "    def _prepare_inputs(",
    "    def _get_model_mm_inputs(",
    "    def _configure_lora(self, input, requests, req_ids, is_prompt):",
    "\ndef trim_attn_metadata(",
    "\nclass HpuModelAdapter(torch.nn.Module",
    "\ndef shallow_tuple(",
    "\nfrom vllm.distributed.parallel_state import get_pp_group, get_dp_group\n",
    "\nfrom vllm_gaudi.extension.ops import LoraMask as LoraMask\n",
    "        prefill_data, \\\n            dummy_prefill_input_data_batches_across_dp = prefill_input_data\n",
)

# The last stage runs a copy of this method with the received tensors added, so it must be exactly this text.
EXECUTE_MODEL_GENERIC = """    def _execute_model_generic(self,
                               token_ids,
                               position_ids,
                               attn_metadata,
                               logits_indices,
                               kv_caches,
                               lora_logits_mask,
                               lora_mask,
                               warmup_mode=False,
                               inputs_embeds=None,
                               model_mm_kwargs=None):
        # FORWARD.
        batch_size = token_ids.size(0)
        seq_len = self._seq_len(attn_metadata)
        num_blocks = self._num_blocks(attn_metadata)
        self._check_config(batch_size, seq_len, num_blocks, attn_metadata, warmup_mode)
        additional_kwargs = {}
        if htorch.utils.internal.is_lazy():
            use_graphs = self._use_graphs(attn_metadata, batch_size)
            additional_kwargs.update({"bypass_hpu_graphs": not use_graphs})
        else:
            # no hpu graphs for t.compile?
            use_graphs = False
        if self.model_has_chunked_attention:
            additional_kwargs.update({"model_has_chunked_attention": True})
        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)
"""

RUNNER_CODE = r'''

# --- [gaudi-fix:pipeline-parallel] -------------------------------------------------------------------------------
# Pipeline parallelism for the HPU runner: stage-to-stage tensors over gloo on the pipeline group's CPU group.
import time as _gfpp_time  # noqa: E402

from vllm.distributed.parallel_state import get_tp_group as _gfpp_get_tp_group  # noqa: E402
from vllm.sequence import IntermediateTensors as _GfppIntermediateTensors  # noqa: E402

_GFPP_HDR_LEN = 64
_GFPP_MAGIC = 0x50503253
_GFPP_DTYPES = (torch.bfloat16, torch.float32, torch.float16, torch.int32, torch.int64, torch.uint8, torch.int8)
_GFPP_SPLIT_MIN = 1 << 20  # payloads this large are split across the tensor-parallel ranks
_GFPP_ALIGN = 16
_GFPP_STATS_EVERY = int(os.environ.get("VLLM_HPU_PP_STATS_EVERY", "0"))

# Habana's module post-forward hook iterates every non-tensor output (`for o in output`). IntermediateTensors has
# __getitem__ but no __iter__, so Python calls __getitem__(0), (1), ... which returns None forever.
if "__iter__" not in _GfppIntermediateTensors.__dict__:
    _GfppIntermediateTensors.__iter__ = lambda self: iter(self.tensors.values())


class _GfppLink:
    """IntermediateTensors between neighbour stages. One link per worker; the pipeline group pairs this rank with
    the same tensor-parallel rank of the neighbour stages, so the ranks of a stage use independent connections."""

    def __init__(self, runner):
        self.pp = get_pp_group()
        self.tp = _gfpp_get_tp_group()
        self.runner = runner
        self.send_seq = 0
        self.recv_seq = 0
        self.pending = []
        self.bufs = None
        self.keys = None
        self.log = self.tp.rank_in_group == 0
        self.st = dict.fromkeys(("send_n", "send_bytes", "recv_n", "recv_bytes", "split_n"), 0)
        self.st.update(dict.fromkeys(("send_ms", "recv_wait_ms", "recv_ms"), 0.0))

    def _bounds(self, n, split):
        """The piece of a flat n-element tensor this rank sends: all of it, or 1/tp when split."""
        if not split:
            return 0, n, n
        c = -(-n // self.tp.world_size)
        lo = min(self.tp.rank_in_group * c, n)
        return lo, min(lo + c, n), c

    @staticmethod
    def _padded(c, es):
        return -(-(c * es) // _GFPP_ALIGN) * _GFPP_ALIGN

    def _stats(self):
        st = self.st
        if not self.log or _GFPP_STATS_EVERY <= 0 or st["send_n"] + st["recv_n"] < _GFPP_STATS_EVERY:
            return
        sn, rn = max(st["send_n"], 1), max(st["recv_n"], 1)
        logger.info(
            "[pipeline-parallel] stage %d/%d: sent %d (%.3f ms, %.1f KiB/rank), received %d (%.3f ms waiting for "
            "the previous stage, %.3f ms payload, %.1f KiB/rank), %d split",
            self.pp.rank_in_group, self.pp.world_size, st["send_n"], st["send_ms"] / sn, st["send_bytes"] / sn / 1024,
            st["recv_n"], st["recv_wait_ms"] / rn, st["recv_ms"] / rn, st["recv_bytes"] / rn / 1024, st["split_n"])
        for k in st:
            st[k] = 0 if isinstance(st[k], int) else 0.0

    def send(self, it):
        if not isinstance(it, _GfppIntermediateTensors):
            raise RuntimeError(f"[pipeline-parallel] a non-last stage returned {type(it).__name__}, expected "
                               "IntermediateTensors (does this model support pipeline parallelism?)")
        t0 = _gfpp_time.perf_counter()
        for w in self.pending:
            w.wait()
        items = list(it.tensors.items())
        split = self.tp.world_size > 1 and sum(t.numel() * t.element_size() for _, t in items) >= _GFPP_SPLIT_MIN
        hdr = torch.zeros(_GFPP_HDR_LEN, dtype=torch.int64)
        hdr[0], hdr[1], hdr[2], hdr[3] = _GFPP_MAGIC, self.send_seq, len(items), int(split)
        pos, pieces, sizes = 4, [], []
        for _, t in items:
            if t.dtype not in _GFPP_DTYPES:
                raise RuntimeError(f"[pipeline-parallel] unsupported dtype {t.dtype}")
            if pos + 2 + t.dim() > _GFPP_HDR_LEN:
                raise RuntimeError("[pipeline-parallel] header overflow")
            hdr[pos], hdr[pos + 1] = _GFPP_DTYPES.index(t.dtype), t.dim()
            for i, d in enumerate(t.shape):
                hdr[pos + 2 + i] = d
            pos += 2 + t.dim()
            lo, hi, c = self._bounds(t.numel(), split)
            pieces.append(t.reshape(-1)[lo:hi].to("cpu"))
            sizes.append(self._padded(c, t.element_size()))
        payload = torch.zeros(sum(sizes), dtype=torch.uint8)
        off = 0
        for p, sz in zip(pieces, sizes):
            b = p.contiguous().view(torch.uint8)
            payload[off:off + b.numel()].copy_(b)
            off += sz
        dst = self.pp.ranks[(self.pp.rank_in_group + 1) % self.pp.world_size]
        g = self.pp.cpu_group
        self.pending = [torch.distributed.isend(hdr, dst=dst, group=g),
                        torch.distributed.isend(payload, dst=dst, group=g)]
        self.bufs = (hdr, payload)  # alive until the sends complete
        self.send_seq += 1
        st = self.st
        st["send_n"] += 1
        st["send_ms"] += (_gfpp_time.perf_counter() - t0) * 1e3
        st["send_bytes"] += payload.numel()
        st["split_n"] += int(split)
        self._stats()

    def recv(self, token_ids, warmup_mode):
        n_tok = token_ids.numel()
        model = self.runner.get_model()
        if warmup_mode:
            # warmup and profiling forwards run on every stage independently: zeros of the right shape
            it = model.make_empty_intermediate_tensors(batch_size=n_tok, dtype=self.runner.dtype,
                                                       device=self.runner.device)
            shape = (n_tok, ) if get_config().flatten_input else tuple(token_ids.shape)
            return _GfppIntermediateTensors({k: torch.zeros(shape + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
                                             for k, v in it.tensors.items()})
        if self.keys is None:
            empty = model.make_empty_intermediate_tensors(batch_size=1, dtype=self.runner.dtype, device="cpu")
            self.keys = list(empty.tensors.keys())
        src = self.pp.ranks[(self.pp.rank_in_group - 1) % self.pp.world_size]
        g = self.pp.cpu_group
        t0 = _gfpp_time.perf_counter()
        hdr = torch.empty(_GFPP_HDR_LEN, dtype=torch.int64)
        torch.distributed.recv(hdr, src=src, group=g)
        t1 = _gfpp_time.perf_counter()
        h = hdr.tolist()
        if h[0] != _GFPP_MAGIC or h[1] != self.recv_seq or h[2] != len(self.keys):
            raise RuntimeError(f"[pipeline-parallel] stage {self.pp.rank_in_group}: bad header magic={h[0]:#x} "
                               f"seq={h[1]} (expected {self.recv_seq}) keys={h[2]} (expected {len(self.keys)})")
        split = bool(h[3])
        pos, specs = 4, []
        for k in self.keys:
            dt, nd = _GFPP_DTYPES[h[pos]], h[pos + 1]
            shape = tuple(h[pos + 2:pos + 2 + nd])
            pos += 2 + nd
            lead = 1
            for d in shape[:-1]:
                lead *= d
            if lead != n_tok:
                raise RuntimeError(f"[pipeline-parallel] stage {self.pp.rank_in_group} seq {self.recv_seq}: received "
                                   f"{k} {shape} for a batch of {n_tok} tokens; the stages are out of step")
            numel = lead * shape[-1]
            es = torch.empty((), dtype=dt).element_size()
            c = self._bounds(numel, split)[2]
            specs.append((k, dt, shape, numel, c, es, self._padded(c, es)))
        payload = torch.empty(sum(s[-1] for s in specs), dtype=torch.uint8)
        torch.distributed.recv(payload, src=src, group=g)
        out, off = {}, 0
        for k, dt, shape, numel, c, es, sz in specs:
            dev = payload[off:off + c * es].view(dt).to(self.runner.device)
            off += sz
            if split:
                dev = self.tp.all_gather(dev, dim=0)
            out[k] = dev[:numel].view(shape)
        # Wait for the copies (and the all-gather) on the device before the first layer's graph is queued:
        # without it the forward could read the hidden states before they had fully landed.
        htorch.core.mark_step()
        torch.hpu.synchronize()
        self.recv_seq += 1
        st = self.st
        st["recv_n"] += 1
        st["recv_wait_ms"] += (t1 - t0) * 1e3
        st["recv_ms"] += (_gfpp_time.perf_counter() - t1) * 1e3
        st["recv_bytes"] += payload.numel()
        st["split_n"] += int(split)
        self._stats()
        return _GfppIntermediateTensors(out)


def _gfpp_link(self):
    link = getattr(self, "_gfpp_link_obj", None)
    if link is None:
        link = self._gfpp_link_obj = _GfppLink(self)
    return link


def _gfpp_stage_forward(self, token_ids, position_ids, attn_metadata, lora_mask, warmup_mode, inputs_embeds=None,
                        model_mm_kwargs=None):
    """Forward of a non-last stage: receive (unless first), run this stage's layers, send."""
    batch_size = token_ids.size(0)
    seq_len = self._seq_len(attn_metadata)
    num_blocks = self._num_blocks(attn_metadata)
    self._check_config(batch_size, seq_len, num_blocks, attn_metadata, warmup_mode)
    additional_kwargs = {}
    if htorch.utils.internal.is_lazy():
        additional_kwargs["bypass_hpu_graphs"] = not self._use_graphs(attn_metadata, batch_size)
    if self.model_has_chunked_attention:
        additional_kwargs["model_has_chunked_attention"] = True
    link = self._gfpp_link()
    if not get_pp_group().is_first_rank:
        additional_kwargs["intermediate_tensors"] = link.recv(token_ids, warmup_mode)
    out = self.model.forward(input_ids=token_ids, positions=position_ids,
                             attn_metadata=trim_attn_metadata(attn_metadata), kv_caches=self.kv_caches,
                             inputs_embeds=inputs_embeds, model_mm_kwargs=model_mm_kwargs, lora_mask=lora_mask,
                             **additional_kwargs)
    if not warmup_mode:
        link.send(out)


@torch.inference_mode()
def _gfpp_forward_only_sample_tokens(self, grammar_output):
    """sample_tokens of a non-last stage: the full method's step preparation and forwards, in the same order and
    shapes, without logits or sampling."""
    if self.scheduler_output is None:
        return None
    if self.use_async_scheduling:
        raise RuntimeError("[pipeline-parallel] pipeline parallelism on HPU needs --no-async-scheduling")
    scheduler_output = self.scheduler_output
    warmup_mode = self.warmup_mode
    self.scheduler_output = None
    self.warmup_mode = False
    self._ensure_decodes_first(scheduler_output)
    if not self.vllm_config.speculative_config:
        ensure_multi_token_decodes_last(self.input_batch, scheduler_output.num_scheduled_tokens)
    if self.use_prefix_caching and "sort_prompts_by_context" in globals():
        # the prompt-order fix reorders prompts in the full sample_tokens; every stage must see the same order
        sort_prompts_by_context(self.input_batch, self._get_num_decodes(scheduler_output))  # noqa: F821
    pd_info = self._get_prompts_and_decodes(scheduler_output)
    num_decodes = len(pd_info.decode_req_ids)
    num_prefills = len(pd_info.prompt_req_ids)
    prefill_input_data, decode_input_data = self._prepare_inputs(scheduler_output, num_prefills, num_decodes,
                                                                 warmup_mode)
    prefill_data, dummy_prefill = prefill_input_data
    decode_data, dummy_decode = decode_input_data
    # the full sample_tokens sets this here, and the next step's _prepare_inputs reads it
    self.use_structured_output = grammar_output is not None
    if num_prefills > 0:
        htorch.core.mark_step()
        for req_id, _prompt_len, token_ids, position_ids, attn_metadata, _li, _lr in zip(*shallow_tuple(prefill_data)):
            inputs_embeds, model_mm_kwargs = self._get_model_mm_inputs(token_ids, token_ids.shape[-1],
                                                                       scheduler_output, req_id)
            lora_mask, _ = self._configure_lora(token_ids, self.requests, req_id, True)
            htorch.core.mark_step()
            self._gfpp_stage_forward(token_ids, position_ids, attn_metadata, lora_mask, warmup_mode,
                                     inputs_embeds=inputs_embeds, model_mm_kwargs=model_mm_kwargs)
            htorch.core.mark_step()
    if dummy_prefill is not None and len(dummy_prefill.request_ids) > 0:
        for _rid, _pl, token_ids, position_ids, attn_metadata, _li, _lr in zip(*shallow_tuple(dummy_prefill)):
            htorch.core.mark_step()
            self._gfpp_stage_forward(token_ids, position_ids, attn_metadata, None, warmup_mode)
            htorch.core.mark_step()
    if num_decodes > 0:
        lora_mask, _ = self._configure_lora(decode_data.token_ids, self.requests, pd_info.decode_req_ids, False)
        htorch.core.mark_step()
        self._gfpp_stage_forward(decode_data.token_ids, decode_data.position_ids, decode_data.attn_metadata,
                                 lora_mask, warmup_mode)
        htorch.core.mark_step()
    elif dummy_decode is not None:
        htorch.core.mark_step()
        self._gfpp_stage_forward(dummy_decode.token_ids, dummy_decode.position_ids, dummy_decode.attn_metadata, None,
                                 warmup_mode)
        htorch.core.mark_step()
    return None


# A non-last stage's outputs leave the HPU graph as fresh tensors. With tensor parallelism they are results of
# in-place collectives, which an HPU graph with the tensor cache disabled frees after replay ("no storage tensor
# attached to lazy tensor" on the copy to host). The clone is inside the captured graph.
_gfpp_orig_adapter_forward = HpuModelAdapter.forward


def _gfpp_adapter_forward(self, *args, **kwargs):
    out = _gfpp_orig_adapter_forward(self, *args, **kwargs)
    if isinstance(out, _GfppIntermediateTensors):
        out = _GfppIntermediateTensors({k: v.clone() for k, v in out.tensors.items()})
    return out


HpuModelAdapter.forward = _gfpp_adapter_forward

_gfpp_orig_sample_tokens = HPUModelRunner.sample_tokens
_gfpp_orig_execute_model_generic = HPUModelRunner._execute_model_generic


def _gfpp_sample_tokens(self, grammar_output):
    pp = get_pp_group()
    if pp.world_size > 1 and not pp.is_last_rank:
        return self._gfpp_forward_only_sample_tokens(grammar_output)
    return _gfpp_orig_sample_tokens(self, grammar_output)


def _gfpp_execute_model_generic(self, token_ids, position_ids, attn_metadata, logits_indices, kv_caches,
                                lora_logits_mask, lora_mask, warmup_mode=False, inputs_embeds=None,
                                model_mm_kwargs=None):
    pp = get_pp_group()
    if pp.world_size == 1:
        return _gfpp_orig_execute_model_generic(self, token_ids, position_ids, attn_metadata, logits_indices,
                                                kv_caches, lora_logits_mask, lora_mask, warmup_mode=warmup_mode,
                                                inputs_embeds=inputs_embeds, model_mm_kwargs=model_mm_kwargs)
    if not pp.is_last_rank:
        raise RuntimeError("[pipeline-parallel] _execute_model_generic called on a non-last stage")
    # the stock _execute_model_generic (checked when the fix was applied), plus the previous stage's tensors
    batch_size = token_ids.size(0)
    seq_len = self._seq_len(attn_metadata)
    num_blocks = self._num_blocks(attn_metadata)
    self._check_config(batch_size, seq_len, num_blocks, attn_metadata, warmup_mode)
    additional_kwargs = {}
    if htorch.utils.internal.is_lazy():
        use_graphs = self._use_graphs(attn_metadata, batch_size)
        additional_kwargs["bypass_hpu_graphs"] = not use_graphs
    else:
        use_graphs = False
    if self.model_has_chunked_attention:
        additional_kwargs["model_has_chunked_attention"] = True
    additional_kwargs["intermediate_tensors"] = self._gfpp_link().recv(token_ids, warmup_mode)
    trimmed_attn_metadata = trim_attn_metadata(attn_metadata)
    if self.is_driver_worker:
        event = f"model_forward_bs{batch_size}_seq{seq_len}_ctx{num_blocks}_graphs{'T' if use_graphs else 'F'}"
    else:
        event = "model_executable"
    with self.profiler.record_event("internal", event):
        hidden_states = self.model.forward(input_ids=token_ids, positions=position_ids,
                                           attn_metadata=trimmed_attn_metadata, kv_caches=kv_caches,
                                           inputs_embeds=inputs_embeds, model_mm_kwargs=model_mm_kwargs,
                                           lora_mask=lora_mask, **additional_kwargs)
    if self.use_aux_hidden_state_outputs:
        non_flattened_hidden_states, aux_hidden_states = hidden_states
        hidden_states = non_flattened_hidden_states
    else:
        non_flattened_hidden_states = hidden_states
        aux_hidden_states = None
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    hidden_states = hidden_states[logits_indices]
    LoraMask.setLoraMask(lora_logits_mask)
    with self.profiler.record_event("internal", f"compute_logits{batch_size}_seq{seq_len}_ctx{num_blocks}"):
        logits = self.model.compute_logits(hidden_states)
    return non_flattened_hidden_states, aux_hidden_states, hidden_states, logits


HPUModelRunner._gfpp_link = _gfpp_link
HPUModelRunner._gfpp_stage_forward = _gfpp_stage_forward
HPUModelRunner._gfpp_forward_only_sample_tokens = _gfpp_forward_only_sample_tokens
HPUModelRunner.sample_tokens = _gfpp_sample_tokens
HPUModelRunner._execute_model_generic = _gfpp_execute_model_generic
'''

# ---------------------------------------------------------------------------------------------------- worker
WORKER_ANCHORS = (
    "\ndef init_worker_distributed_environment(",
    "torch.distributed.all_reduce(dummy_tensor_hpu)",
    "\nfrom vllm.distributed import (ensure_model_parallel_initialized, init_distributed_environment)\n",
    "\nfrom vllm.distributed.parallel_state import get_pp_group, get_tp_group\n",
    "init_worker_distributed_environment(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank)",
)

WORKER_CODE = r"""

# --- [gaudi-fix:pipeline-parallel] -------------------------------------------------------------------------------
# With pipeline parallelism the default (world) HCCL group spans the nodes, and cross-node HCCL needs RoCE scale-out
# links. Check the world over gloo and HCCL on the node-local tensor-parallel group instead.
_gfpp_orig_init_worker_distributed_environment = init_worker_distributed_environment


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size <= 1 or os.environ.get("VLLM_HPU_PP_HCCL_WORLD_CHECK", "0") == "1":
        return _gfpp_orig_init_worker_distributed_environment(vllm_config, rank, distributed_init_method, local_rank)
    from vllm.distributed.parallel_state import get_world_group
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank, backend="hccl")
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)
    ones = torch.ones(1)
    torch.distributed.all_reduce(ones, group=get_world_group().cpu_group)
    assert ones.item() == parallel_config.world_size * parallel_config.data_parallel_size, ones.item()
    tp = get_tp_group()
    dev = tp.all_reduce(torch.ones(1).to("hpu"))
    assert dev.item() == tp.world_size, dev.item()
    logger.info("[pipeline-parallel] rank %d: PP %d x TP %d; world check over gloo, HCCL on the TP group", rank,
                parallel_config.pipeline_parallel_size, parallel_config.tensor_parallel_size)
"""

# ---------------------------------------------------------------------------------------------------- scheduler
SCHED_OLD = """        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
"""
SCHED_NEW = f"""        # First, schedule the RUNNING requests.
        req_index = 0
        # {MARKER} pipeline microbatches: at most ceil(running / pp) running requests per step, so the next
        # step finds the rest not in flight and the stages overlap.
        _gfpp_cap = None
        if self.use_pp and not self.scheduler_config.async_scheduling:
            _gfpp_cap = max(1, -(-len(self.running) // self.parallel_config.pipeline_parallel_size))
        while req_index < len(self.running) and token_budget > 0:
            if _gfpp_cap is not None and len(scheduled_running_reqs) >= _gfpp_cap:
                break
            request = self.running[req_index]
"""

# ---------------------------------------------------------------------------------------------------- loader
LOADER_ANCHORS = (
    "\nfrom vllm.model_executor.model_loader.ep_weight_filter import (\n    should_skip_weight,\n)\n",
    "\nimport regex as re\n",
    "\nlogger = init_logger(__name__)\n",
)

LOADER_CODE = r"""

# --- [gaudi-fix:pipeline-parallel] -------------------------------------------------------------------------------
# With pipeline parallelism each stage skips the checkpoint tensors of decoder layers it does not own before reading
# them. The iterators above look up should_skip_weight in this module at call time.
_gfpp_orig_should_skip_weight = should_skip_weight
_GFPP_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_gfpp_load_state = {}


def _gfpp_load_range():
    if "r" in _gfpp_load_state:
        return _gfpp_load_state["r"]
    r = None
    try:
        from vllm.config import get_current_vllm_config
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.distributed.utils import get_pp_indices
        pp = get_pp_group()
        if pp.world_size > 1:
            cfg = get_current_vllm_config()
            mc = cfg.model_config
            if mc.hf_config is mc.hf_text_config:
                n = mc.hf_text_config.num_hidden_layers
                lo, hi = get_pp_indices(n, pp.rank_in_group, pp.world_size)
                keep_mtp = pp.is_last_rank and cfg.speculative_config is not None
                r = (lo, hi, n, keep_mtp)
                logger.info("[pipeline-parallel] loader: stage %d/%d reads decoder layers [%d, %d) of %d%s",
                            pp.rank_in_group, pp.world_size, lo, hi, n, " and the MTP layers" if keep_mtp else "")
            else:
                logger.info("[pipeline-parallel] loader: multimodal configuration, every layer is read")
    except Exception as e:  # never break loading: fall back to reading everything
        logger.warning("[pipeline-parallel] loader: layer skip off (%s: %s)", type(e).__name__, e)
        r = None
    _gfpp_load_state["r"] = r
    return r


def should_skip_weight(weight_name, local_expert_ids):
    if _gfpp_orig_should_skip_weight(weight_name, local_expert_ids):
        return True
    r = _gfpp_load_range()
    m = _GFPP_LAYER_RE.match(weight_name) if r is not None else None
    if m is None:
        return False
    lo, hi, n, keep_mtp = r
    i = int(m.group(1))
    if i >= n:
        return not keep_mtp
    return not lo <= i < hi
"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def require_once(src: str, needle: str, label: str) -> None:
    n = src.count(needle)
    if n != 1:
        raise Drift(f"{label}: {needle.strip()[:80]!r} found {n} times, expected 1")


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for needle in (*RUNNER_ANCHORS, EXECUTE_MODEL_GENERIC):
        require_once(src, needle, "hpu_model_runner.py")
    return src.rstrip("\n") + "\n" + RUNNER_CODE


def patch_worker(src: str, ctx: dict[str, str]) -> str:
    for needle in WORKER_ANCHORS:
        require_once(src, needle, "hpu_worker.py")
    return src.rstrip("\n") + "\n" + WORKER_CODE


def patch_scheduler(src: str, ctx: dict[str, str]) -> str:
    require_once(src, SCHED_OLD, "scheduler.py")
    require_once(src, "scheduled_running_reqs.append(request)", "scheduler.py")
    return src.replace(SCHED_OLD, SCHED_NEW, 1)


def patch_loader(src: str, ctx: dict[str, str]) -> str:
    for needle in LOADER_ANCHORS:
        require_once(src, needle, "weight_utils.py")
    return src.rstrip("\n") + "\n" + LOADER_CODE


# Text that shows the upstream fix is present (checked only where the marker is absent).
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (
    ("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),
    ("vllm_gaudi", "v1/worker/hpu_worker.py", patch_worker, "worker"),
    ("vllm", "v1/core/sched/scheduler.py", patch_scheduler, "scheduler"),
    ("vllm", "model_executor/model_loader/weight_utils.py", patch_loader, "loader"),
)


# ---- common driver (identical in every fix) --------------------------------------------------


def locate(pkg: str, rel: str, root: str | None) -> Path | None:
    """Find <pkg>/<rel> under --root, or in the installed package without importing it."""
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".gaudi-fix.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def run(check: bool, root: str | None) -> int:
    ctx: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for pkg, rel, _, key in TARGETS:
        path = locate(pkg, rel, root)
        if path is None:
            print(f"{MARKER} {pkg}/{rel} not found", file=sys.stderr)
            return 2
        paths[key] = path
        ctx[key] = path.read_text()
    if UPSTREAM_FIXED and all(MARKER not in ctx[k] and needle in ctx[k] for k, needle in UPSTREAM_FIXED):
        print(f"{MARKER} not needed: the upstream fix is present")
        return 0
    done, todo = [], []
    try:
        for _pkg, _rel, fn, key in TARGETS:
            if fn is None:
                continue
            src = ctx[key]
            if MARKER in src:
                done.append(paths[key])
                continue
            new = fn(src, ctx)
            if MARKER not in new:
                raise Drift("patched text lacks the marker")
            compile(new, str(paths[key]), "exec", dont_inherit=True)
            todo.append((paths[key], new))
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if done and todo:
        print(f"{MARKER} partly applied ({', '.join(map(str, done))}); restore the files and re-run", file=sys.stderr)
        return 2
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for path, new in todo:
        write_atomic(path, new)
        print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
