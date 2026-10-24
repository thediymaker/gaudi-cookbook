# SPDX-License-Identifier: Apache-2.0
"""Window-sized KV for Muse Spark (swa_kv.py) against vLLM core, on CPU (needs vLLM; skips without it).

Part A: Muse Spark's per-rank KV specs at TP 8 (16 full-attention NoPE layers 1, 5, ..., 61 + 46 sliding-window
(2048) layers, block 128, 2 KV heads x 64, bf16) through core's get_kv_cache_groups,
get_kv_cache_config_from_groups and get_kv_cache_configs, then swa_kv.SwaKvGroups.on_init_kv_cache on the result:
4 groups (full 16, sliding 16 / 15 / 15), 16 KV slots each aliasing one layer of every group,
num_blocks = mem // (16 x 64 KiB).

Part B: a real vLLM Scheduler (sync and async) on the 4-group config, driven through 4096-token chunked prefills,
decodes, prefix hits and preemption on a tiny pool. The worker's block tables are mirrored the way
hpu_model_runner._update_states keeps them, and every step's per-group slot mappings and window block lists are
built with swa_kv's builders, then checked against the KV cache manager:
  * sliding-window writes go to the request's own live block (never the null block);
  * window reads cover exactly (p - 2047 .. p] (decode, with the runner's head mask) / [c - 2047, c) (prefill
    context), and every unmasked slot sits in a live block the request holds;
  * full rows have no null blocks; no block id is live in two groups;
  * with async scheduling the same checks hold again after the next schedule().

  TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS=none python3 test_swa_kv_core.py     (or through pytest)
"""

import collections
import json
import os
import random
import sys
import tempfile
import time

try:
    import torch

    from muse_spark_hpu import swa_kv as swk
except ImportError:  # collected by pytest without torch: the test skips
    torch = swk = None

BS = 128
W = 2048
WIN_BLOCKS = W // BS + 1  # 17 = the runner's sliding_block_size for muse_spark
HEAD = 64
KVH = 2  # 16 KV heads / TP8
N_LAYERS = 62
FULL_LAYERS = tuple(i for i in range(N_LAYERS) if (61 - i) % 4 == 0)  # 1, 5, ..., 61
PAGE = BS * KVH * HEAD * 2 * 2  # k + v, bf16 = 65,536 B
TOK = 7
FAILS: list = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        if len(FAILS) <= 20:
            print("FAIL:", msg)


def lname(i):
    return f"model.layers.{i}.self_attn.attn"


def cdiv(a, b):
    return -(-a // b)


_FAKE_MODEL = None


def fake_model_dir():
    global _FAKE_MODEL
    if _FAKE_MODEL is None:
        _FAKE_MODEL = tempfile.mkdtemp(prefix="swakv-model-")
        cfg = {
            "architectures": ["OPTForCausalLM"],
            "model_type": "opt",
            "hidden_size": 64,
            "ffn_dim": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "vocab_size": 1024,
            "word_embed_proj_dim": 64,
            "max_position_embeddings": 1 << 21,
            "do_layer_norm_before": True,
            "torch_dtype": "bfloat16",
            "bos_token_id": 2,
            "eos_token_id": 2,
            "pad_token_id": 1,
        }
        with open(os.path.join(_FAKE_MODEL, "config.json"), "w") as f:
            json.dump(cfg, f)
    return _FAKE_MODEL


def make_vllm_config(
    max_model_len, max_num_seqs=4, max_batched=4096, async_sched=False, prefix=True, num_gpu_blocks_override=None
):
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig

    mc = ModelConfig(
        model=fake_model_dir(), dtype="bfloat16", seed=0, skip_tokenizer_init=True, max_model_len=max_model_len
    )
    sc = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        async_scheduling=async_sched,
        is_encoder_decoder=False,
        watermark=0.0,
    )
    cc = CacheConfig(
        block_size=BS,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=prefix,
        num_gpu_blocks_override=num_gpu_blocks_override,
    )
    vc = VllmConfig(
        model_config=mc,
        scheduler_config=sc,
        cache_config=cc,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    # The HPU platform reports support_hybrid_kv_cache() = True, so with no KV connector the hybrid
    # manager stays on (checked on the TP8 server log); the CPU test platform may report False.
    vc.scheduler_config.disable_hybrid_kv_cache_manager = False
    # The engine core resolves one layout from the workers' backends; vllm-gaudi's backends declare none,
    # so the default preference applies: LBNHC (layer-outermost).
    from vllm.v1.attention.backends.utils import record_kv_cache_layout

    record_kv_cache_layout(vc.cache_config, "LBNHC")
    return vc


def muse_full_specs():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    specs, windows = {}, {}
    for i in range(N_LAYERS):
        specs[lname(i)] = FullAttentionSpec(block_size=BS, num_kv_heads=KVH, head_size=HEAD, dtype=torch.bfloat16)
        windows[lname(i)] = None if i in FULL_LAYERS else W
    return specs, windows


def muse_specs():
    specs, windows = muse_full_specs()
    return swk.convert_spec_dict(specs, windows)


class _FakeRunner:
    """What on_init_kv_cache touches on the HPU runner."""

    def __init__(self):
        from types import SimpleNamespace

        self.attn_block_size = BS
        self.speculative_config = None
        self.use_merged_prefill = False
        self.use_contiguous_pa = False
        self.num_mamba_like_layers = 0
        self.model_config = SimpleNamespace(use_mla=False)
        self.vllm_config = SimpleNamespace(
            kv_transfer_config=None,
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        )
        self.reinit = None

    def may_reinitialize_input_batch(self, cfg, sizes):
        self.reinit = sizes


# --------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------
def part_a():
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_capacity,
        get_kv_cache_config_from_groups,
        get_kv_cache_configs,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    print("== Part A: groups and tensors (TP8)")
    vc = make_vllm_config(1048576)
    specs = muse_specs()
    groups = get_kv_cache_groups(vc, dict(specs))
    kinds = [type(g.kv_cache_spec).__name__ for g in groups]
    sizes = [len(g.layer_names) for g in groups]
    print("groups:", list(zip(kinds, sizes, strict=False)))
    check(len(groups) == 4, f"4 groups expected, got {len(groups)}")
    ga = [i for i, g in enumerate(groups) if type(g.kv_cache_spec) is FullAttentionSpec]
    sw = [i for i, g in enumerate(groups) if type(g.kv_cache_spec) is SlidingWindowSpec]
    check(len(ga) == 1 and len(sw) == 3, f"1 full + 3 sliding groups, got {kinds}")
    check(sorted(sizes) == [15, 15, 16, 16], f"group sizes {sizes}")
    check(groups[ga[0]].layer_names == [lname(i) for i in FULL_LAYERS], "full group layers")
    sw_layers = [lname(i) for i in range(N_LAYERS) if i not in FULL_LAYERS]
    for j, g in enumerate(sw):
        check(groups[g].layer_names == sw_layers[j::3], f"SW group {g} layers {groups[g].layer_names[:3]}...")
        s = groups[g].kv_cache_spec
        check(s.sliding_window == W and s.block_size == BS, f"SW spec {s}")
    check({g.kv_cache_spec.page_size_bytes for g in groups} == {PAGE}, "page size")

    mem = 15 * 2**30 + 512 * 2**20  # a 15.5 GiB KV budget
    cfg = get_kv_cache_config_from_groups(vc, groups, mem)
    check(cfg.num_blocks == mem // (PAGE * 16), f"num_blocks {cfg.num_blocks} != {mem // (PAGE * 16)}")
    print(
        "KV tensors:", [(len(t.layers), t.size, t.layer_stride, t.block_stride, t.offset) for t in cfg.kv_cache_tensors]
    )
    r = _FakeRunner()
    h = swk.SwaKvGroups(W, BS)
    h.on_init_kv_cache(r, cfg)
    check(h.n_slots == 16, f"KV slots {h.n_slots}")
    check(r.reinit == [BS] * 4, f"input batch block sizes {r.reinit}")
    per_slot = collections.defaultdict(list)
    for ln, s in h.layer_slot.items():
        per_slot[s].append(h.layer_gid[ln])
    check(all(len(set(v)) == len(v) for v in per_slot.values()), "a slot holds two layers of one group")
    # groups 16 / 15 / 15 sliding + 16 full: slots 0..14 hold a layer of every group, slot 15 only the
    # 16-layer groups' last layers (core's "2 padding layers")
    check(
        sorted(len(v) for v in per_slot.values()) == [2] + [4] * 15,
        f"layers per slot {sorted(len(v) for v in per_slot.values())}",
    )
    check(all(h.layer_gid[ln] == ga[0] for ln in [lname(i) for i in FULL_LAYERS]), "full layers in the full group")
    check(len({h.layer_slot[lname(i)] for i in FULL_LAYERS}) == 16, "each full layer owns a slot")

    full_specs, _ = muse_full_specs()
    g_off = get_kv_cache_groups(vc, dict(full_specs))
    cfg_off = get_kv_cache_config_from_groups(vc, g_off, mem)
    check(
        len(g_off) == 1 and cfg_off.num_blocks == mem // (PAGE * 62),
        f"full-length KV: {len(g_off)} groups, {cfg_off.num_blocks} blocks",
    )
    ratio = cfg.num_blocks / cfg_off.num_blocks
    print(
        f"pool {mem / 2**30:.2f} GiB: full-length KV {cfg_off.num_blocks} blocks = {cfg_off.num_blocks * BS:,} tokens; "
        f"groups {cfg.num_blocks} blocks = {cfg.num_blocks * BS:,} tokens of full-layer context ({ratio:.3f}x); "
        f"{16 * PAGE // BS} B/token/card vs {62 * PAGE // BS}"
    )
    check(abs(ratio - 62 / 16) < 0.01, f"ratio {ratio}")

    # --num-gpu-blocks-override: the pool is exactly that many block ids
    vco = make_vllm_config(1048576, num_gpu_blocks_override=12288)
    cfg_o = get_kv_cache_config_from_groups(vco, groups, mem)
    check(cfg_o.num_blocks == 12288, f"override -> {cfg_o.num_blocks}")

    for mml in (131072, 262144, 524288, 1048576):
        for async_sched in (False, True):
            vcx = make_vllm_config(mml, async_sched=async_sched)
            mems = [mem - rr * 3_000_000 for rr in range(8)]
            cfgs = get_kv_cache_configs(vcx, [dict(specs) for _ in range(8)], mems)
            nbs = {c.num_blocks for c in cfgs}
            check(nbs == {min(m // (PAGE * 16) for m in mems)}, f"min across ranks {nbs}")
            tokens, conc = get_kv_cache_capacity(vcx, cfgs[0])
            sw_cap = groups[sw[0]].kv_cache_spec.max_admission_blocks_per_request(vcx.max_in_flight_tokens, mml)
            print(
                f"max_model_len {mml} {'async' if async_sched else 'sync '}: {cfgs[0].num_blocks} blocks, core "
                f"capacity {tokens:,} tokens, max concurrency {conc:.2f}x; SW admission cap {sw_cap} "
                f"blocks/group/request (in-flight {vcx.max_in_flight_tokens})"
            )
    return groups


# --------------------------------------------------------------------------
# Part B: scheduler simulation
# --------------------------------------------------------------------------
def make_kv_cache_config(groups, num_blocks):
    from vllm.v1.kv_cache_interface import KVCacheConfig

    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def make_scheduler(groups, num_blocks, max_model_len, max_num_seqs, max_batched, async_sched, prefix=True):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.structured_output import StructuredOutputManager

    vc = make_vllm_config(max_model_len, max_num_seqs, max_batched, async_sched, prefix)
    vc.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vc)
    cls = AsyncScheduler if async_sched else Scheduler
    s = cls(
        vllm_config=vc,
        kv_cache_config=make_kv_cache_config(groups, num_blocks),
        block_size=BS,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vc),
    )
    s.use_v2_model_runner = False
    return s


_HASH_INIT = False


def make_request(req_id, prompt, max_tokens):
    global _HASH_INIT
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    if not _HASH_INIT:
        init_none_hash(sha256)
        _HASH_INIT = True
    sp = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
    sp.update_from_generation_config({}, 2)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


class Mirror:
    """Worker-side block tables, kept like hpu_model_runner._update_states keeps them."""

    def __init__(self, n_groups, max_reqs, max_model_len):
        from vllm.v1.worker.block_table import MultiGroupBlockTable

        self.G = n_groups
        self.bt = MultiGroupBlockTable(
            max_num_reqs=max_reqs,
            max_num_batched_tokens=8192,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[BS] * n_groups,
            kernel_block_sizes=[BS] * n_groups,
            max_num_blocks=[cdiv(max_model_len, BS)] * n_groups,
        )
        self.state = {}
        self.row = {}
        self.free_rows = list(range(max_reqs))[::-1]
        self.first_hits = {}

    def _remove(self, r):
        row = self.row.pop(r, None)
        if row is not None:
            self.bt.clear_row(row)
            self.free_rows.append(row)

    def apply(self, out):
        for r in out.finished_req_ids:
            self.state.pop(r, None)
            self._remove(r)
        for r in list(self.row):
            if r not in out.num_scheduled_tokens:
                self._remove(r)
        to_add = []
        for nr in out.scheduled_new_reqs:
            check(len(nr.block_ids) == self.G, f"new req {nr.req_id}: {len(nr.block_ids)} block lists")
            self.state[nr.req_id] = {
                "blocks": tuple(list(b) for b in nr.block_ids),
                "prompt": len(nr.prompt_token_ids),
                "computed": nr.num_computed_tokens,
            }
            self.first_hits.setdefault(nr.req_id, nr.num_computed_tokens)
            to_add.append(nr.req_id)
        cr = out.scheduled_cached_reqs
        for i, r in enumerate(cr.req_ids):
            st = self.state[r]
            st["computed"] = cr.num_computed_tokens[i]
            nb = cr.new_block_ids[i]
            if r in cr.resumed_req_ids:
                st["blocks"] = tuple(list(b) for b in nb)
            elif nb is not None:
                for ids, new in zip(st["blocks"], nb, strict=False):
                    ids.extend(new)
            if r not in self.row:
                to_add.append(r)
            elif nb is not None and r not in cr.resumed_req_ids:
                self.bt.append_row(nb, self.row[r])
        for r in to_add:
            row = self.free_rows.pop()
            self.row[r] = row
            self.bt.add_row(self.state[r]["blocks"], row)
        for r, row in self.row.items():
            for g in range(self.G):
                n = len(self.state[r]["blocks"][g])
                got = self.bt.block_tables[g].get_cpu_tensor()[row, :n].tolist()
                check(got == self.state[r]["blocks"][g], f"mirror row {r} g{g}")

    def view(self, out, ga, sw):
        prefills, decodes = [], []
        for r, q in out.num_scheduled_tokens.items():
            st = self.state[r]
            c = st["computed"]
            (prefills if (c < st["prompt"] or q > 1) else decodes).append((r, c, q))
        v = {"prefill": [], "decode": []}
        tabs = [self.bt.block_tables[g].get_cpu_tensor() for g in range(self.G)]
        for r, c, q in prefills:
            row = self.row[r]
            tseq = cdiv(q, BS) * BS
            item = {"req": r, "ctx": c, "q": q, "ga_row": tabs[ga][row, : cdiv(c + q, BS)].tolist(), "sw": {}}
            for g in sw:
                slots, win = swk.prefill_group_tensors(tabs[g], [row], [c], [q], 1, tseq, BS, WIN_BLOCKS, c > 0)
                check(
                    tuple(slots.shape) == (1, tseq) and (win is None or tuple(win.shape) == (1, WIN_BLOCKS)),
                    "prefill shapes",
                )
                item["sw"][g] = (slots[0, :q].clone(), None if win is None else win[0].tolist())
            v["prefill"].append(item)
        if decodes:
            nd = len(decodes)
            padded = nd + 2
            index = torch.tensor([[c] for _, c, _ in decodes], dtype=torch.int64)
            nblk = [cdiv(c + 1, BS) for _, c, _ in decodes]
            base = torch.full((padded, 1), -1, dtype=torch.int64)
            ga_tab = torch.stack([tabs[ga][self.row[r]] for r, _, _ in decodes])
            base, _ = swk.decode_group_tensors(ga_tab, nd, index, base, nblk, BS, WIN_BLOCKS)
            sm_list = base.tolist()
            items = [
                {"req": r, "p": c, "ga_row": tabs[ga][self.row[r], : cdiv(c + 1, BS)].tolist(), "sw": {}}
                for r, c, _ in decodes
            ]
            for g in sw:
                tab = torch.stack([tabs[g][self.row[r]] for r, _, _ in decodes])
                sm, tables = swk.decode_group_tensors(tab, nd, index, base, nblk, BS, WIN_BLOCKS)
                check(sm.shape == base.shape and torch.equal(sm[nd:], base[nd:]), "decode padding rows kept")
                tables, usage = swk.window_usage(tables, sm_list, BS, WIN_BLOCKS)
                for i, it in enumerate(items):
                    it["sw"][g] = (int(sm[i, 0]), tables[i], usage[i])
            v["decode"] = items
        return v


def manager_ids(mgr, r):
    blocks = mgr.req_to_blocks.get(r)
    return None if blocks is None else [b.block_id for b in blocks]


def check_view(sched, view, ga, sw, when, live_only=False):
    mgrs = sched.kv_cache_manager.coordinator.single_type_managers
    n_checked = 0
    for it in view["prefill"] + view["decode"]:
        r = it["req"]
        ids = {g: manager_ids(mgrs[g], r) for g in range(len(mgrs))}
        if ids[ga] is None:
            check(live_only, f"{when}: {r} not held by the manager")
            continue
        req = sched.requests.get(r)
        if live_only and (req is None or req.status.name != "RUNNING"):
            continue
        tag = f"{when} {r}"
        ga_row = it["ga_row"]
        check(0 not in ga_row, f"{tag}: null block in the full row")
        check(ga_row == ids[ga][: len(ga_row)], f"{tag}: full row != manager")
        for g in sw:
            held = ids[g]
            if "q" in it:  # prefill
                c, q = it["ctx"], it["q"]
                slots, win = it["sw"][g]
                pos = torch.arange(c, c + q)
                blk = (slots // BS).tolist()
                want = [held[p // BS] if p // BS < len(held) else -2 for p in pos.tolist()]
                check(0 not in blk, f"{tag} g{g}: prefill writes into the null block")
                check(blk == want, f"{tag} g{g}: prefill writes into blocks it does not hold")
                check(torch.equal(slots % BS, pos % BS), f"{tag} g{g}: slot offsets")
                if c > 0:
                    frame = max(0, cdiv(c, BS) * BS - WIN_BLOCKS * BS)
                    for p in range(max(0, c - (W - 1)), c):
                        j = (p - frame) // BS
                        b = win[j] if 0 <= j < len(win) else -1
                        check(
                            b > 0 and b == held[p // BS],
                            f"{tag} g{g}: prefill context pos {p} reads block {b}, "
                            f"held {held[p // BS] if p // BS < len(held) else None}",
                        )
            else:  # decode
                p = it["p"]
                slot, table, usage = it["sw"][g]
                check(
                    slot // BS != 0 and slot // BS == held[p // BS] and slot % BS == p % BS,
                    f"{tag} g{g}: decode write slot {slot}, held {held[p // BS]}",
                )
                n = cdiv(p + 1, BS)
                first = n - len(table)
                seen = set()
                for k, (b, u) in enumerate(zip(table, usage, strict=False)):
                    s_range = range(0, u) if u > 0 else range(-u, BS)
                    for s in s_range:
                        seen.add((first + k) * BS + s)
                    if len(s_range):
                        check(
                            b != 0 and b == held[first + k],
                            f"{tag} g{g}: decode window block {b} at {first + k}, held {held[first + k]}",
                        )
                check(
                    seen == set(range(max(0, p - (W - 1)), p + 1)),
                    f"{tag} g{g}: decode window covers {min(seen)}..{max(seen)} ({len(seen)}) for p={p}",
                )
        n_checked += 1
    where = {}
    for g, m in enumerate(mgrs):
        for r, blocks in m.req_to_blocks.items():
            for i, b in enumerate(blocks):
                if b.block_id == 0:
                    continue
                key = where.setdefault(b.block_id, (g, i))
                check(key == (g, i), f"{when}: block {b.block_id} live as {key} and ({g}, {i}) ({r})")
    return n_checked


def runner_output(out, partial):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(out.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[] if r in partial else [TOK] for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def simulate(
    name, groups, reqs, num_blocks, async_sched, max_model_len=32768, max_num_seqs=4, max_batched=4096, max_steps=40000
):
    t0 = time.time()
    G = len(groups)
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    ga = next(i for i, g in enumerate(groups) if type(g.kv_cache_spec) is FullAttentionSpec)
    sw = [i for i in range(G) if i != ga]
    sched = make_scheduler(groups, num_blocks, max_model_len, max_num_seqs, max_batched, async_sched)
    mirror = Mirror(G, max_num_seqs, max_model_len)
    todo = sorted(reqs, key=lambda x: x[0])
    added = {}
    depth = 2 if async_sched else 1
    pending = collections.deque()
    stats = collections.Counter()
    step = 0
    while todo or sched.has_unfinished_requests() or pending:
        while todo and todo[0][0] <= step:
            _, rid, prompt, mt = todo.pop(0)
            added[rid] = make_request(rid, prompt, mt)
            sched.add_request(added[rid])
        if sched.has_unfinished_requests() and len(pending) < depth:
            out = sched.schedule()
            step += 1
            check(step < max_steps, f"{name}: no progress")
            if step >= max_steps:
                break
            stats["steps"] += 1
            stats["preempted"] += len(out.preempted_req_ids or ())
            stats["resumed"] += len(out.scheduled_cached_reqs.resumed_req_ids)
            if out.num_scheduled_tokens:
                check(
                    all(out.num_common_prefix_blocks[g] == 0 for g in sw),
                    f"{name}: SW common prefix blocks {out.num_common_prefix_blocks}",
                )
            mirror.apply(out)
            view = mirror.view(out, ga, sw)
            stats["prefill_chunks"] += len(view["prefill"])
            stats["decodes"] += len(view["decode"])
            stats["checked"] += check_view(sched, view, ga, sw, f"{name} step {step}")
            for _, pview, _ in pending:
                stats["rechecked"] += check_view(
                    sched, pview, ga, sw, f"{name} step {step} (in flight)", live_only=True
                )
            partial = {
                r
                for r in out.num_scheduled_tokens
                if sched.requests[r].num_computed_tokens < sched.requests[r].num_tokens
            }
            pending.append((out, view, partial))
            if len(pending) < depth and sched.has_unfinished_requests():
                continue
        if pending:
            out, _, partial = pending.popleft()
            sched.update_from_output(out, runner_output(out, partial))
        elif not sched.has_unfinished_requests() and todo:
            step = max(step, todo[0][0])
    pool = sched.kv_cache_manager.block_pool
    check(
        pool.get_num_free_blocks() == num_blocks - 1,
        f"{name}: {pool.get_num_free_blocks()} free of {num_blocks - 1} at the end",
    )
    for rid, req in added.items():
        check(
            req.is_finished() and len(req.output_token_ids) == req.max_tokens,
            f"{name}: {rid} output {len(req.output_token_ids)}/{req.max_tokens}",
        )
    print(f"{name} [{'async' if async_sched else 'sync'}]: {dict(stats)}, {time.time() - t0:.1f}s")
    return stats, mirror


def min_pool(groups, max_model_len, budget, async_sched):
    """Blocks the startup check requires for one max_model_len request: full blocks plus each
    sliding group's admission cap."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    in_flight = budget * (2 if async_sched else 1)
    swg = [g for g in groups if type(g.kv_cache_spec) is SlidingWindowSpec]
    cap = swg[0].kv_cache_spec.max_admission_blocks_per_request(in_flight, max_model_len)
    return cdiv(max_model_len, BS) + len(swg) * cap


def part_b(groups):
    print("== Part B: scheduler simulation")
    rng = random.Random(0)

    def toks(n):
        return [rng.randrange(10, 1000) for _ in range(n)]

    for async_sched in (False, True):
        # 1. 4096-token chunked prefills next to decodes that cross block and window boundaries.
        reqs = [(0, "long", toks(20000), 2300), (0, "mid", toks(9000), 2200), (2, "short", toks(300), 2500)]
        st, _ = simulate("chunked", groups, reqs, num_blocks=4000, async_sched=async_sched)
        check(st["prefill_chunks"] >= 7 and st["decodes"] > 6000, f"chunked coverage {st}")

        # 2. Prefix hits (window 2048 = 16 contiguous cached blocks before the boundary in every SW group).
        P = toks(12000)
        a_prompt = P + toks(1000)
        reqs = [
            (0, "a", a_prompt, 40),
            (0, "a_twin", P[:9000] + toks(900), 30),
            (300, "b", P[:7037] + toks(2000), 30),
            (300, "c", list(a_prompt), 20),
            (600, "d", P[:12000], 10),
            (600, "e", P[:1500] + toks(100), 10),
        ]
        st, mirror = simulate("prefix", groups, reqs, num_blocks=4000, async_sched=async_sched)
        hits = mirror.first_hits
        print("  prefix hits (tokens):", hits)
        check(hits["b"] == 7037 // BS * BS, f"b should hit {7037 // BS * BS}, got {hits['b']}")
        check(hits["c"] == 12999 // BS * BS, f"c (identical prompt) should hit {12999 // BS * BS}, got {hits['c']}")
        check(hits["d"] == 11999 // BS * BS, f"d should hit {11999 // BS * BS}, got {hits['d']}")
        check(
            hits["e"] == 1500 // BS * BS,
            f"e (hit shorter than the window) should hit {1500 // BS * BS}, got {hits['e']}",
        )
        check(hits["a_twin"] in (0, 9000 // BS * BS), f"a_twin hit {hits['a_twin']}")

        # 3. Preemption on a tiny pool: 4 decoders outgrow it (2048-token budget, max_model_len 16384).
        mml, budget = 16384, 2048
        pool = min_pool(groups, mml, budget, async_sched) + 6
        reqs = [(0, f"p{i}", toks(3000 + 100 * i), 9000) for i in range(4)]
        st, _ = simulate(
            "preempt", groups, reqs, num_blocks=pool, async_sched=async_sched, max_model_len=mml, max_batched=budget
        )
        check(st["preempted"] >= 1 and st["resumed"] >= 1, f"preemption not exercised: {st}")

        # 4. Pool barely above the startup minimum: 4096-token chunks next to running decodes.
        pool = min_pool(groups, 32768, 4096, async_sched) + 20
        reqs = [(0, "x", toks(12000), 200), (1, "y", toks(12000), 200), (3, "z", toks(500), 2600)]
        simulate("tight", groups, reqs, num_blocks=pool, async_sched=async_sched, max_num_seqs=3)

    # 5. Capacity: one 1M-token request + three 128K on 15,872 blocks (15.5 GiB at 1 MiB per block).
    for async_sched in (False, True):
        mml = 1048576
        reqs = [(0, "L0", toks(mml - 400), 50)] + [(0, f"S{i}", toks(131072 - 300 - 7 * i), 50) for i in range(3)]
        st, _ = simulate("1Mx1+128Kx3", groups, reqs, num_blocks=15872, async_sched=async_sched, max_model_len=mml)
        check(st["preempted"] == 0, f"1M+3x128K preempted {st['preempted']}")


def main():
    t0 = time.time()
    groups = part_a()
    part_b(groups)
    import resource

    print(
        f"total {time.time() - t0:.1f}s, "
        f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB, "
        f"{len(FAILS)} failures"
    )
    if FAILS:
        print("FAIL")
        return 1
    print("PASS")
    return 0


def test_swa_kv_core():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
