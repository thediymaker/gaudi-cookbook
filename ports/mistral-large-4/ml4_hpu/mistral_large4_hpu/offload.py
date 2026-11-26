"""KV offloading to host memory for ML4: vLLM's OffloadingConnector with vllm-gaudi's CPU <-> HPU
worker (vllm_gaudi/v1/kv_offload/worker/cpu_hpu.py), on ML4's own caches. Off unless the server runs with
--kv-transfer-config '{"kv_connector":"OffloadingConnector",...}'.

Three things differ from vllm-gaudi's path:
  - Registration after runner_patch 7b, which replaces each cache with a 2-D base tensor: vllm-gaudi registers inside
    initialize_kv_cache, before that (runner_patch suppresses it).
  - The caches keep their dtype. ML4 declares its caches bf16 (codec bytes packed inside), and vllm-gaudi views every
    cache as int8; on the HPU a dtype view is a copy (uint8 / bf16 .view(int8) gets its own
    storage, same-dtype views and slices alias), so the connector stored a snapshot and loaded into a copy while the
    model read the old rows. Here each cache is a same-dtype [blocks, page] view, the host
    copies have the same dtype, and the handler moves elements (index_select / index_put_ copy bits) through
    device buffers allocated at start-up, ML4_OFFLOAD_CHUNK blocks per copy: a store runs while other prompts
    prefill at the memory peak, where the card has no room for a new allocation.
  - Tags: a block loaded from host memory gets a fresh allocation number and its rows' tags (paged.py) are rewritten
    to (number, position) in every cache before the load is submitted; otherwise the rows keep the tags of the
    block's previous occupant and every read masks them.
"""

from __future__ import annotations

import logging
import os
from collections import deque

import habana_frameworks.torch.core as htcore
import numpy as np
import torch

logger = logging.getLogger("vllm.mistral_large4_hpu")


def ml4_caches(runner) -> dict:
    """ML4's own paged caches by layer name (the 63 latent and 44 index-K owners; KV-shared layers have none)."""
    ctx = runner.vllm_config.compilation_config.static_forward_context
    return {name: mod for name, mod in ctx.items() if type(mod).__name__ == "ML4KVCache" and mod.kv_cache is not None}


def register(runner, block_size: int) -> None:
    """Hand the connector's worker one same-dtype [blocks, page] view per cache (the pad block left out)."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor

    worker = get_kv_transfer_group().connector_worker
    nb = worker.kv_cache_config.num_blocks
    tensors, index = [], {}
    for name, mod in ml4_caches(runner).items():
        t = mod.kv_cache
        view = t.view(-1, block_size * t.shape[1])[:nb]
        assert view.data_ptr() == t.data_ptr() and view.dtype == t.dtype, name
        index[name] = len(tensors)
        tensors.append(CanonicalKVCacheTensor(tensor=view, page_size_bytes=view.shape[1] * view.element_size()))
    # layers without a cache of their own (KV-shared) have no tensor; the refs only size the transfer stats
    groups = [
        [
            CanonicalKVCacheRef(tensor_idx=index[n], page_size_bytes=tensors[index[n]].page_size_bytes)
            for n in g.layer_names
            if n in index
        ]
        for g in worker.kv_cache_config.kv_cache_groups
    ]
    worker._init_worker(CanonicalKVCaches(tensors=tensors, group_data_refs=groups))
    logger.warning(
        "ml4: KV offloading: %d caches registered with the connector (%s, %d blocks)",
        len(tensors),
        tensors[0].tensor.dtype,
        nb,
    )


def _handler_init(self, gpu_tensors, cpu_tensors, block_size_factor, kv_cache_groups_data_refs, gpu_to_cpu):
    """vllm-gaudi's SingleDirectionOffloadingHandler init for tensors of any one dtype (its own asserts int8)."""
    assert len(gpu_tensors) == len(cpu_tensors) > 0
    assert len(kv_cache_groups_data_refs) == 1
    for g, c in zip(gpu_tensors, cpu_tensors):
        assert g.ndim == 2 and c.ndim == 2 and c.device.type == "cpu" and g.dtype == c.dtype
        assert c.shape[1] == g.shape[1] * block_size_factor
    # transfer_async indexes the host side in GPU-block units (chunk * factor + i): one row per GPU block, the same
    # host memory (vllm-gaudi's [chunks, factor * page] rows fail the copy for any factor above 1)
    cpu_tensors = [c.view(-1, g.shape[1]) for g, c in zip(gpu_tensors, cpu_tensors)]
    self.src_tensors = gpu_tensors if gpu_to_cpu else cpu_tensors
    self.dst_tensors = cpu_tensors if gpu_to_cpu else gpu_tensors
    self.gpu_to_cpu = gpu_to_cpu
    self.src_block_size_factor = 1 if gpu_to_cpu else block_size_factor
    self.dst_block_size_factor = block_size_factor if gpu_to_cpu else 1
    self.tensor_block_size_in_bytes = [g.shape[1] * g.element_size() for g in gpu_tensors]
    self.group_block_size_in_bytes = [
        sum(self.tensor_block_size_in_bytes[r.tensor_idx] for r in refs) for refs in kv_cache_groups_data_refs
    ]
    self._transfer_events = {}
    self._transfers = deque()
    self._stream_pool = []
    self._event_pool = []
    for g in gpu_tensors:
        _stage(g.shape[1], g.dtype, g.device)


CHUNK = int(os.environ.get("ML4_OFFLOAD_CHUNK", "32"))  # GPU blocks per copy
_hpu = {}  # cpu_hpu's Transfer and expand_block_ids, filled by install()
_stages = {}  # (page, dtype) -> [CHUNK, page] device buffer; "idx" -> [CHUNK] int64 device index buffer


def _stage(page: int, dtype, device) -> torch.Tensor:
    """The device buffers every copy goes through, allocated once at start-up (one per page width, about 3 MB per card
    for ML4's two widths, shared by the store and load handlers)."""
    if "idx" not in _stages:
        _stages["idx"] = torch.zeros(CHUNK, dtype=torch.int64, device=device)
    key = (page, dtype)
    if key not in _stages:
        _stages[key] = torch.zeros((CHUNK, page), dtype=dtype, device=device)
    return _stages[key]


def _transfer_async(self, job_id, src_spec, dst_spec) -> bool:
    """vllm-gaudi's transfer_async without device allocations. Its own gathers every block of the transfer on the card in
    one index_select per cache: a 130K prompt's store took 81 MB per cache at once (PT_DEVMEM in the 2 x 130K step with
    3 GiB held), and copies of 32 blocks still failed there (2.6 MB): a store runs while another prompt
    prefills at the memory peak, where the card has no room for anything new. Every copy goes through the start-up
    buffers (_stage), CHUNK blocks at a time, the last chunk padded with repeats of its last pair (same source, same
    destination, same bytes) so every step has one shape and nothing compiles mid-run."""
    src_blocks, dst_blocks = src_spec.block_ids, dst_spec.block_ids
    assert src_blocks.ndim == 1 and dst_blocks.ndim == 1
    src_count = src_blocks.size * self.src_block_size_factor
    dst_count = dst_blocks.size * self.dst_block_size_factor
    skip = -dst_blocks.size % self.src_block_size_factor
    assert dst_count == src_count - skip
    padded = -(-dst_count // CHUNK) * CHUNK
    pairs = np.empty((padded, 2), dtype=np.int64)
    _hpu["expand"](src_blocks, self.src_block_size_factor, pairs[:dst_count, 0], skip_count=skip)
    _hpu["expand"](dst_blocks, self.dst_block_size_factor, pairs[:dst_count, 1])
    pairs[dst_count:] = pairs[dst_count - 1]
    pairs = torch.from_numpy(pairs)
    idx = _stages["idx"]

    stream = self._stream_pool.pop() if self._stream_pool else torch.hpu.Stream()
    start = self._event_pool.pop() if self._event_pool else torch.Event(enable_timing=True)
    end = self._event_pool.pop() if self._event_pool else torch.Event(enable_timing=True)
    if self.gpu_to_cpu:
        stream.wait_stream(torch.hpu.current_stream())  # the model's writes to these blocks first
    if self._transfers:
        stream.wait_event(self._transfers[-1].end_event)
    with torch.hpu.stream(stream):
        start.record(stream)
        for i in range(0, padded, CHUNK):
            s_idx, d_idx = pairs[i : i + CHUNK, 0], pairs[i : i + CHUNK, 1]
            if self.gpu_to_cpu:
                idx.copy_(s_idx)
                for src, dst in zip(self.src_tensors, self.dst_tensors):
                    buf = _stages[(src.shape[1], src.dtype)]
                    torch.index_select(src, 0, idx, out=buf)
                    dst.index_put_((d_idx,), buf.to("cpu"))  # the copy to the host runs the step
            else:
                idx.copy_(d_idx)
                for src, dst in zip(self.src_tensors, self.dst_tensors):
                    buf = _stages[(dst.shape[1], dst.dtype)]
                    buf.copy_(src.index_select(0, s_idx))
                    dst.index_put_((idx,), buf)
                    htcore.mark_step()
                    torch.hpu.synchronize()  # done with buf and idx before they are rewritten
        torch.hpu.synchronize()
        end.record(stream)
    self._transfer_events[job_id] = end
    self._transfers.append(
        _hpu["Transfer"](
            job_id=job_id,
            stream=stream,
            start_event=start,
            end_event=end,
            num_bytes=dst_count * self.group_block_size_in_bytes[0],
        )
    )
    return True


def _worker_init(self, kv_caches, blocks_per_chunk: int, num_cpu_chunks: int, mmap_region=None):
    """vllm-gaudi's CPUOffloadingWorker init without the int8 view: host tensors in the caches' own dtype."""
    from vllm.utils.platform_utils import is_pin_memory_available
    from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

    del mmap_region
    pin = is_pin_memory_available()
    gpu, cpu = [], []
    for kt in kv_caches.tensors:
        t = kt.tensor
        page = kt.page_size_bytes // t.element_size()
        g = t if t.dim() == 2 and t.shape[1] == page else t.reshape(-1, page)
        assert g.data_ptr() == t.data_ptr()
        gpu.append(g)
        cpu.append(torch.zeros((num_cpu_chunks, page * blocks_per_chunk), dtype=t.dtype, device="cpu", pin_memory=pin))
    logger.warning(
        "ml4: KV offloading: %d host tensors, %.1f GiB on this rank",
        len(cpu),
        sum(c.numel() * c.element_size() for c in cpu) / 2**30,
    )
    self._store_handler = SingleDirectionOffloadingHandler(
        gpu_tensors=gpu,
        cpu_tensors=cpu,
        block_size_factor=blocks_per_chunk,
        kv_cache_groups_data_refs=kv_caches.group_data_refs,
        gpu_to_cpu=True,
    )
    self._load_handler = SingleDirectionOffloadingHandler(
        gpu_tensors=gpu,
        cpu_tensors=cpu,
        block_size_factor=blocks_per_chunk,
        kv_cache_groups_data_refs=kv_caches.group_data_refs,
        gpu_to_cpu=False,
    )


def retag(runner, mark_fresh, epochs, block_size: int, block_ids, first_index: int) -> None:
    """Blocks loaded into the GPU blocks block_ids, holding the request's logical blocks first_index, ...: fresh
    allocation numbers (mark_fresh), then every cache's row tags set to (number, position)."""
    ids = np.asarray(block_ids, dtype=np.int64)
    if ids.size == 0:
        return
    mark_fresh(runner, ids.tolist())
    e = epochs(runner)
    slot = np.arange(block_size, dtype=np.int64)
    rows = (ids[:, None] * block_size + slot).reshape(-1)
    pos = ((first_index + np.arange(ids.size, dtype=np.int64))[:, None] * block_size + slot).reshape(-1)
    own = np.repeat(e[ids].astype(np.int64), block_size)
    caches = [m for m in ml4_caches(runner).values() if m.tags is not None]
    if not caches:
        return
    dev = caches[0].tags.device
    rows_t = torch.from_numpy(rows).to(dev)
    tag_t = torch.from_numpy(np.stack([own, pos], axis=-1).astype(np.int32)).to(dev)
    for mod in caches:
        mod.tags.index_copy_(0, rows_t, tag_t)


def _with_worker_meta(cls):
    """vllm-gaudi's execute_model builds its KVConnectorOutput without kv_connector_worker_meta, so the offloading
    scheduler heard of finished jobs only on steps with nothing to run (kv_connector_no_forward). Under load a request
    resumed after its load (finished_recving) with the load job still pending, and a preemption then failed the
    scheduler's `assert self._jobs[any_jid].is_store`; a preempted request with a pending store would
    have waited for ever. The output now carries the jobs this worker finished, read after get_finished."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

    def make(*args, **kw):
        out = cls(*args, **kw)
        if has_kv_transfer_group() and out.kv_connector_worker_meta is None:
            out.kv_connector_worker_meta = get_kv_transfer_group().build_connector_worker_meta()
        return out

    return make


def install(get_runner, mark_fresh, epochs, block_size) -> None:
    """Patch the HPU offload worker for same-dtype caches, retag loaded blocks at submission, and report finished
    jobs to the scheduler on every step."""
    try:
        from vllm_gaudi.v1.kv_offload.worker import cpu_hpu  # vllm-gaudi's patches first, ours on top
        from vllm_gaudi.v1.worker import hpu_model_runner
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker
        from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker, SingleDirectionOffloadingHandler
    except ImportError:
        return
    if getattr(OffloadingConnectorWorker, "_ml4", False):
        return
    hpu_model_runner.KVConnectorOutput = _with_worker_meta(hpu_model_runner.KVConnectorOutput)
    _hpu.update(Transfer=cpu_hpu.Transfer, expand=cpu_hpu.expand_block_ids)
    SingleDirectionOffloadingHandler.__init__ = _handler_init
    SingleDirectionOffloadingHandler.transfer_async = _transfer_async
    CPUOffloadingWorker.__init__ = _worker_init
    orig = OffloadingConnectorWorker.start_kv_transfers

    def start_kv_transfers(self, metadata):
        runner = get_runner()
        if runner is not None:
            for entry in metadata.load_jobs.values():
                spec = entry.dst_spec
                retag(runner, mark_fresh, epochs, block_size(), spec.block_ids, int(spec.block_indices[0]))
        return orig(self, metadata)

    OffloadingConnectorWorker.start_kv_transfers = start_kv_transfers
    OffloadingConnectorWorker._ml4 = True


__all__ = ["install", "register", "retag", "ml4_caches"]
