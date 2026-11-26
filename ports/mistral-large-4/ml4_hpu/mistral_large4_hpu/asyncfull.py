"""ML4_ASYNC_SPEC=2: the device-side async + DSpark port, on top of asyncspec's mode-1 pieces.

Mode 1 (asyncspec) runs vLLM's AsyncScheduler with the HPU runner on its sync spec path and keeps the accept
sync inside sample_tokens: forward -> rejection (host loop, .cpu()) -> host postprocess -> DSpark policy + draft graph
launch -> return. Everything after that sync (sampler tail, draft launch, the next step's prep) leaves the card idle.

Mode 2 moves the sync to the latest point that still keeps every host decision exact (a "late sync"):

  step N, sample_tokens (fast path: decode-only, plain requests, no grammar / logprobs / LoRA / DP / warm-up):
    * forward, logits and the bonus-row sampler exactly as the stock path;
    * rejection ON THE CARD (verify_device: spec.verify_tokens' rule vectorised: greedy = cumulative product of
      argmax matches, sampling = the u < p(d) rule on the same random draws); out [Bp, T] int32, -1 past the emitted;
    * the emitted tokens go to the host on a copy stream (event after the rejection only, not after the draft graph);
    * DSpark inputs from the card: x_m = out[b, acc_b], m = p0_b + acc_b + 1 (p0 = the row-0 position, exact on the
      host); the draft graph for step N+1 is launched right away and its output stays on the card (asyncspec's
      device-draft scatter puts it into step N+1's token ids);
    * the drafting decision for N+1 (dspark_go) is made here, before step N's acceptance is known on the host: the
      tracker is ONE STEP LATE (exact for forced drafts, dspark_adapt=false; may change tokens on the adaptive path);
    * returns an AsyncModelRunnerOutput: the engine's output thread waits for the copy, the worker thread does not.
  step N+1, _update_states (outermost wrapper, before mode 1's correction):
    * resolve(): wait for step N's copy event (the late sync), then the host bookkeeping the stock path did after its
      sync (token_ids_cpu, num_tokens_no_spec, output_token_ids, acceptance tracker + stats). From here on the host
      state equals the sync path's, so positions, slots, table width and the 2048 regime are computed exactly as sync.

So the card runs [forward N][rejection N][draft graph N+1][forward N+1]; the host's post-sync work (resolve, scheduler
correction, _update_states, input prep) overlaps the draft graph instead of following it.

Any other step (prefill chunk in the batch, structured output, logprobs, penalties, seeds, ...) takes the stock sync
path (mode 1 behaviour) after resolve() has drained the pending step. Engine side (mode 2 only): the AsyncScheduler
gives placeholders only to requests that can draft (spec._plain), so structured-output requests never carry -1 drafts.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

logger = logging.getLogger("vllm.mistral_large4_hpu.asyncfull")
_f: dict = {
    "fifo": [],
    "fast": 0,
    "slow": 0,
    "resolved": 0,
    "drafts": 0,
    "stream": None,
    "why": {},
    "skip": 0,
    "row0": 0,
}
# ML4_ASYNC_NOSYNC=1 (mode 2 only, default off): after a step in which no request verified drafts, the next step
# does not wait for its tokens at all. The host positions are exact then (one token per request), the row-0 token of
# each request is copied from the previous step's device output into the token ids (row0_scatter), and the host
# bookkeeping of such steps is applied once their copy has landed (or when a later step needs it).
NOSYNC = os.environ.get("ML4_ASYNC_NOSYNC", "0") == "1"
FIFO_MAX = 3


def nosync_on() -> bool:
    """ML4_ASYNC_NOSYNC, or the control key async_nosync (bring-up A/B on one server; read every step)."""
    from .control import ctl

    return bool(ctl().get("async_nosync", NOSYNC))


# ML4_ASYNC_DEVPOS=1 (mode 2 only, default off; control key async_devpos): after a VERIFY step the next step does not
# wait either. The host builds the step from the scheduler's optimistic positions (all of the last step's drafts
# accepted: an upper bound, short by corr_b = ks_b - acc_b in [0, ks_b]); the card subtracts corr_b from every
# position row of request b (devpos_positions) and takes each request's row-0 token at out[b, acc_b] (row0 gather).
# The one host decision that reads positions, the block-table width (runner_patch._bt_width, which also fixes the
# decode graphs' dense-top-k shortcut, graphs.host_max_from_width), must be the same for the lower and the upper
# bound; when it is not (a request within ks tokens of a width step), the step waits as before.
DEVPOS = os.environ.get("ML4_ASYNC_DEVPOS", "0") == "1"


def devpos_on() -> bool:
    from .control import ctl

    return bool(ctl().get("async_devpos", DEVPOS))


# ------------------------------------------------------------------------------------------- the verify rule
def need_sort(top_p, top_k, V: int) -> bool:
    """spec._sample_rows' top-k / top-p switch, from the HOST copies (the device form must not sync)."""
    return bool(((top_k > 0) & (top_k < V)).any()) or bool((top_p < 1.0).any())


def sample_rows(logits, temp, top_p, top_k, sort: bool):
    """spec._sample_rows with the switch given (same ops, same order)."""
    from .spec import _EPS

    lf = logits.float()
    am = lf.argmax(-1)
    t = torch.where(temp < _EPS, torch.ones_like(temp), temp)
    lf = lf / t.unsqueeze(-1)
    V = lf.shape[-1]
    if sort:
        srt, idx = lf.sort(-1, descending=True)
        ps = torch.softmax(srt, -1)
        rank = torch.arange(V, device=lf.device).view(1, -1)
        k = torch.where((top_k > 0) & (top_k < V), top_k, torch.full_like(top_k, V)).view(-1, 1)
        drop = rank >= k
        cum = ps.cumsum(-1) - ps
        drop |= cum >= top_p.view(-1, 1)
        srt = srt.masked_fill(drop, float("-inf"))
        lf = torch.full_like(lf, float("-inf")).scatter(-1, idx, srt)
    return am, torch.softmax(lf, -1)


def verify_device(logits, drafts, n_draft, temp, top_p, top_k, gen=None):
    """spec.verify_tokens on the logits' device, no host sync: logits [Bp*T, V], drafts [Bp, T-1] on the device
    (or None at T = 1), n_draft host list [Bp], temp / top_p / top_k host tensors per row [Bp*T].
    -> out [Bp, T] int32 (-1 past the emitted tokens), emitted-1 per request [Bp] int32, both on the device.
    Same random draws as verify_tokens (exponential then uniform, same shapes, same generator).
    HPU lazy mode: int32 throughout, masks built on the host (an int64 form with an in-place slice store and a
    torch.tensor(list) upload returned wrong rows on the first run of a new shape: test/hpu_variants.out)."""
    from .spec import _EPS

    Bp = len(n_draft)
    R = logits.shape[0]
    T = R // Bp
    dev = logits.device
    jn = np.arange(T, dtype=np.int32)
    live_h = jn[None, :] < np.asarray([int(n) for n in n_draft], dtype=np.int32)[:, None]  # [Bp, T] j < n_draft
    jj_h = np.broadcast_to(jn, (Bp, T)).copy()
    live = torch.from_numpy(live_h).to(dev)
    jj = torch.from_numpy(jj_h).to(dev)
    z = torch.zeros(Bp, 1, dtype=torch.int32, device=dev)
    dfull = torch.cat([drafts.reshape(Bp, T - 1).to(torch.int32).clamp(min=0), z], 1) if T > 1 else z
    greedy_all = bool((temp < _EPS).all())
    if greedy_all:
        am = logits.argmax(-1).to(torch.int32).view(Bp, T)
        ok = am == dfull
        corr = am
    else:
        V = logits.shape[-1]
        am, probs = sample_rows(logits, temp.to(dev), top_p.to(dev), top_k.to(dev), need_sort(top_p, top_k, V))
        dd = dfull.view(-1).to(torch.long)
        pd = probs.gather(-1, dd.view(-1, 1)).view(-1)
        expo = torch.empty_like(probs).exponential_(1.0, generator=gen)
        samp = (probs / expo).argmax(-1)
        res = (probs.scatter(-1, dd.view(-1, 1), 0.0) / expo).argmax(-1)
        u = torch.rand(R, device=dev, generator=gen)
        g_h = np.broadcast_to((temp.view(Bp, T)[:, :1] < _EPS).numpy(), (Bp, T)).copy()
        g = torch.from_numpy(g_h).to(dev)
        am = am.to(torch.int32).view(Bp, T)
        samp = samp.to(torch.int32).view(Bp, T)
        res = res.to(torch.int32).view(Bp, T)
        ok = torch.where(g, am == dfull, u.view(Bp, T) < pd.view(Bp, T))
        corr = torch.where(g, am, torch.where(live, res, samp))
    ok = ok & live
    # accepted = the first row that is not ok (rows at or past n_draft never are): a min, not a cumprod
    acc = torch.where(ok, torch.full_like(jj, T), jj).min(1).values
    a = acc.view(Bp, 1)
    out = torch.where(jj < a, dfull, torch.where(jj == a, corr, torch.full_like(corr, -1)))
    _f["keep"] = (live_h, jj_h)  # host sources of the async uploads stay alive past the launch
    return out, acc


# ------------------------------------------------------------------------------------------- engine side
def install_scheduler() -> None:
    """AsyncScheduler: placeholder drafts only for requests the DSpark vehicle drafts for (spec._plain). A request
    that cannot draft (structured output, logprobs, penalties, seed, ...) is scheduled one token per step, as sync."""
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if getattr(AsyncScheduler._update_after_schedule, "_ml4f", False):
        return
    from .spec import _plain

    orig = AsyncScheduler._update_after_schedule

    def _update_after_schedule(self, scheduler_output):
        orig(self, scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            r = self.requests.get(req_id)
            if r is not None and r.spec_token_ids and not _plain(r.sampling_params):
                r.spec_token_ids = []

    _update_after_schedule._ml4f = True
    AsyncScheduler._update_after_schedule = _update_after_schedule


# ------------------------------------------------------------------------------------------- runner side
def _mod(runner):
    for c in type(runner).__mro__:
        m = sys.modules.get(c.__module__)
        if m is not None and hasattr(m, "trim_attn_metadata"):
            return m
    import vllm_gaudi.v1.worker.hpu_model_runner as m

    return m


def _why(k: str) -> bool:
    _f["why"][k] = _f["why"].get(k, 0) + 1
    return False


def fast_ok(self, grammar_output) -> bool:
    """The decode-only plain step the fast path covers; everything else takes the stock (mode-1) path."""
    from .spec import _plain

    so = self.scheduler_output
    if so is None or self.warmup_mode:
        return _why("warmup/none")
    if grammar_output is not None:
        return _why("grammar")
    if getattr(self, "is_pooling_model", False) or getattr(self, "lora_config", None):
        return _why("pool/lora")
    ib = self.input_batch
    n = ib.num_reqs
    if n == 0 or ib.num_prompt_logprobs:
        return _why("empty/plp")
    try:
        if self.get_dp_padding(n):
            return _why("dp")
    except Exception:  # noqa: BLE001
        pass
    for i in range(n):
        rid = ib.req_ids[i]
        if rid is None or self._is_prompt(i, so):
            return _why("prompt")
        if not _plain(getattr(self.requests.get(rid), "sampling_params", None)):
            return _why("not_plain")
    m = _mod(self)
    lmc = getattr(m, "LMCacheConnectorMetadata", None)
    if lmc is not None and isinstance(so.kv_connector_metadata, lmc):
        return _why("lmcache")
    return True


class _Pending:
    __slots__ = ("ev", "out_cpu", "out_dev", "reqs", "nd", "rows", "plain", "row_of", "p0_dev", "p0_keep", "landed")


def _copy_stream():
    s = _f.get("stream")
    if s is None:
        s = _f["stream"] = torch.hpu.Stream()
    return s


@torch.inference_mode()  # as the stock sample_tokens: the graphs' captured inputs are inference tensors
def fast_sample_tokens(self, grammar_output):
    """sample_tokens for a decode-only plain step without the accept sync (see the module docstring)."""
    from . import asyncspec, spec

    m = _mod(self)
    htorch = m.htorch
    so = self.scheduler_output
    batch_changed = self.batch_changed
    self.scheduler_output = None
    self.warmup_mode = False
    self._ensure_decodes_first(so)
    pd_info = self._get_prompts_and_decodes(so)
    num_decodes = len(pd_info.decode_req_ids)
    assert num_decodes > 0 and not pd_info.prompt_req_ids, "ml4 asyncfull: fast path got prompts"
    self.use_structured_output = False
    _, decode_input = self._prepare_inputs(so, 0, num_decodes, False)
    decode_data, _ = decode_input
    with m.set_forward_context(None, self.vllm_config):
        self.maybe_setup_kv_connector(so)
    lora_mask, lora_logits_mask = self._configure_lora(
        decode_data.token_ids, self.requests, pd_info.decode_req_ids, False
    )
    htorch.core.mark_step()
    _, _, _, logits = self._execute_model_generic(
        decode_data.token_ids,
        decode_data.position_ids,
        decode_data.attn_metadata,
        decode_data.logits_indices,
        self.kv_caches,
        lora_logits_mask,
        lora_mask,
        warmup_mode=False,
    )
    htorch.core.mark_step()
    md = decode_data.spec_decode_metadata
    ib = self.input_batch
    if md is None:
        sampler_output, _ = self._run_sampling(batch_changed, logits, pd_info.decode_req_ids, logits.shape[0])
        out = sampler_output.sampled_token_ids.view(-1, 1).to(torch.int32)
        Bp, T = out.shape[0], 1
        nd = [0] * Bp
    else:
        # the stock path samples the bonus rows too (sampling metadata state, generators): same call, output unused
        self._run_sampling(
            batch_changed, logits[md.bonus_logits_indices], pd_info.decode_req_ids, md.bonus_logits_indices.shape[0]
        )
        nd = list(md.num_draft_tokens)
        Bp = len(nd)
        T = logits.shape[0] // Bp
        cap = spec._spec.pop("verify_cap", None)
        if cap is not None and cap[0] == T and cap[1] < T:
            nd = [min(n, cap[1] - 1) for n in nd]
        temp = torch.tensor(
            [float(ib.temperature_cpu[b]) if b < ib.num_reqs else -1.0 for b in range(Bp)], dtype=torch.float32
        ).repeat_interleave(T)
        top_p = torch.tensor(
            [float(ib.top_p_cpu[b]) if b < ib.num_reqs else 1.0 for b in range(Bp)], dtype=torch.float32
        ).repeat_interleave(T)
        top_k = torch.tensor(
            [int(ib.top_k_cpu[b]) if b < ib.num_reqs else 0 for b in range(Bp)], dtype=torch.int64
        ).repeat_interleave(T)
        drafts = md.draft_token_ids.view(Bp, T - 1) if T > 1 else None
        out, _ = verify_device(logits, drafts, nd, temp, top_p, top_k)
    htorch.core.mark_step()
    # the late-sync copy: ordered after the rejection, before the draft graph
    ev = torch.hpu.Event()
    s = _copy_stream()
    with torch.hpu.stream(s):
        s.wait_stream(torch.hpu.current_stream())
        out_cpu = out.to("cpu", non_blocking=True)
        ev.record()
    reqs = list(ib.req_ids[:num_decodes])
    p = _Pending()
    p.ev, p.out_cpu, p.out_dev, p.reqs, p.nd = ev, out_cpu, out, reqs, [int(x) for x in nd[:num_decodes]]
    p.rows = logits.shape[0]
    p.plain = not any(p.nd)
    p.row_of = {r: b for b, r in enumerate(reqs)}
    _f["fifo"].append(p)
    # drafts for step N+1 from the card (decision one step late)
    asyncspec.clear_drafts()
    p0 = np.zeros(Bp, dtype=np.int64)
    p0[:num_decodes] = ib.num_computed_tokens_cpu[:num_decodes]  # devpos: an upper bound (gate width only)
    p0_dev = _f.pop("p0_dev", None)  # devpos: the card's row-0 positions
    _f["devpos"] = None
    if p0_dev is None:
        p.p0_keep = p0[:num_decodes].astype(np.int32)
        p0_dev = torch.from_numpy(p.p0_keep).to(out.device)
    p.p0_dev = p0_dev
    if self.speculative_config:
        propose_device(self, self.speculative_config.num_speculative_tokens, reqs, p.nd, p0, out, T, p0_dev)
    self._draft_token_ids = None
    finished_sending, finished_recving = set(), set()
    self.maybe_wait_for_kv_save()
    finished_sending, finished_recving = self.get_finished_kv_transfers(so)
    max_req_index = max(ib.req_id_to_index.values())
    r2i = dict(ib.req_id_to_index)
    mro = m.ModelRunnerOutput(
        req_ids=list(pd_info.decode_req_ids),
        req_id_to_index=r2i,
        sampled_token_ids=[[] for _ in range(max_req_index + 1)],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=m.KVConnectorOutput(finished_sending=finished_sending, finished_recving=finished_recving),
    )
    if m.has_kv_transfer_group():
        m.get_kv_transfer_group().clear_connector_metadata()
    _f["fast"] += 1
    return _AsyncOut(m.AsyncModelRunnerOutput, mro, p, r2i)


def _AsyncOut(base, mro, p, r2i):
    cls = _f.get("out_cls")
    if cls is None:

        class AsyncML4Output(base):
            def __init__(self, mro, p, r2i):
                self._mro, self._p, self._r2i = mro, p, r2i

            def get_output(self):
                self._p.ev.synchronize()  # output thread; releases the GIL while it waits
                self._p.landed = True  # drain() reads this flag: Event.query() blocked in the server
                rows = self._p.out_cpu.tolist()
                for b, rid in enumerate(self._p.reqs):
                    self._mro.sampled_token_ids[self._r2i[rid]] = [t for t in rows[b] if t >= 0]
                return self._mro

        cls = _f["out_cls"] = AsyncML4Output
    return cls(mro, p, r2i)


def resolve(self) -> None:
    """The late sync: wait for every pending fast step's tokens (oldest first), then the stock path's host bookkeeping
    for them."""
    while _f["fifo"]:
        _resolve_one(self, _f["fifo"].pop(0))


def drain(self) -> None:
    """Resolve the oldest pending steps past FIFO_MAX, and (late sync only) the ones whose copy already landed.
    With async_nosync / async_devpos on, the queue must not depend on landed: only the output rank runs get_output
    (vLLM MultiprocExecutor.handle_output), so only rank 0 ever sees landed; draining by it made rank 0 resolve and
    take the exact path while ranks 1-7 skipped the sync and took the one-step-late drafting decision -> different
    T per rank, mismatched collectives, garbage drafts and a device 'No progress' hang ."""
    q = _f["fifo"]
    early = not (nosync_on() or devpos_on())
    while q and (len(q) > FIFO_MAX or (early and _done(q[0]))):
        _resolve_one(self, q.pop(0))


def _done(p) -> bool:
    """Has the pending step's copy landed? A flag the output thread sets after its (GIL-free) event wait: on the
    serving engine torch.hpu.Event.query blocked until the card was idle (timelines: the late-sync wait sat
    in drain(), not in resolve()), which would make every step wait."""
    return bool(getattr(p, "landed", False))


def extra_tokens(rid) -> int:
    """Tokens of rid in pending PLAIN steps (one each): its num_tokens_no_spec once they are resolved = now + this."""
    return sum(1 for p in _f["fifo"] if rid in p.row_of)


def try_skip(self, so) -> bool:
    """ML4_ASYNC_NOSYNC: may this step run without waiting for the pending steps? Only if all of them are plain (no
    drafts verified: every request emitted exactly one token, so positions are known), the batch is exactly the last
    pending step's (no new, finished, resumed or dropped request, same order), every request is a plain decode."""
    q = _f["fifo"]
    if not (q and nosync_on()):
        return False
    if not all(p.plain for p in q):
        return False
    if not _same_batch(self, so, q[-1]):
        return False
    _f["skip"] += 1
    return True


def _same_batch(self, so, last) -> bool:
    from .spec import _plain

    if so.finished_req_ids or so.scheduled_new_reqs:
        return False
    cached = so.scheduled_cached_reqs
    if getattr(cached, "resumed_req_ids", None):
        return False
    ib = self.input_batch
    if list(ib.req_ids[: ib.num_reqs]) != last.reqs or set(so.num_scheduled_tokens) != set(last.reqs):
        return False
    spec = so.scheduled_spec_decode_tokens
    for rid in last.reqs:
        if so.num_scheduled_tokens[rid] != 1 + len(spec.get(rid, ())):
            return False
        if not _plain(getattr(self.requests.get(rid), "sampling_params", None)):
            return False
    return True


def try_devpos(self, so, prev_k: dict, rows: dict) -> bool:
    """ML4_ASYNC_DEVPOS: may this step run on the card's positions without waiting for the pending steps? rows =
    {req_id: (row, L)} of the drafts this step will verify (asyncspec's registry), prev_k = placeholders scheduled for
    each request in the last step (ks: the scheduler's optimism is ks - acc)."""
    from . import runner_patch

    q = _f["fifo"]
    if not (q and devpos_on()):
        return False
    last = q[-1]
    if not _same_batch(self, so, last):
        return False
    cached = so.scheduled_cached_reqs
    ks = {}
    for with_drafts in (False, True):  # the decision (decide) may still drop this step's drafts
        hi = lo = 0
        for i, rid in enumerate(cached.req_ids):
            k = int(prev_k.get(rid, 0))
            T = 1 + (rows[rid][1] if (with_drafts and rid in rows) else 0)
            base = int(cached.num_computed_tokens[i])
            hi, lo = max(hi, base + T - 1), max(lo, base - k + T - 1)
            ks[rid] = k
        if runner_patch._bt_width(np.array([[hi]])) != runner_patch._bt_width(np.array([[lo]])):
            _why("devpos_width")
            return False
    _f["devpos"] = {"last": last, "ks": ks}
    _f["devpos_n"] = _f.get("devpos_n", 0) + 1
    return True


def devpos_positions(self, out, num_decodes: int, num_scheduled_tokens) -> None:
    """After _create_decode_input_data on a devpos step: every position row of request b minus corr_b on the card,
    corr_b = sched_b - truth_b, sched_b the row-0 position the host built from (the scheduler's), truth_b = p0_b + acc_b
    + 1 of the last pending step (its row-0 position on the card and its accepted drafts): exact whether or not the
    scheduler had that step's output when it scheduled this one. Keeps the corrected row-0 positions (_f["p0_dev"])."""
    dp = _f.get("devpos")
    if dp is None:
        return
    last = dp["last"]
    ib = self.input_batch
    reqs = list(ib.req_ids[:num_decodes])
    T = max(int(n) for n in num_scheduled_tokens[:num_decodes])
    pos = out.position_ids
    R = pos.shape[0]
    dev = pos.device
    acc_all = (last.out_dev >= 0).to(torch.int32).sum(1) - 1  # [Bp_prev]
    prev = np.array([last.row_of[r] for r in reqs], dtype=np.int32)
    sched = np.array([int(ib.num_computed_tokens_cpu[b]) for b in range(num_decodes)], dtype=np.int32)
    pv = torch.from_numpy(prev).to(dev)
    truth = last.p0_dev.index_select(0, pv) + acc_all.index_select(0, pv) + 1
    corr = torch.from_numpy(sched).to(dev) - truth  # [num_decodes]
    corr = torch.cat([corr, torch.zeros(1, dtype=torch.int32, device=dev)])
    row_req = np.full(R, num_decodes, dtype=np.int32)
    for b in range(num_decodes):
        row_req[b * T : b * T + int(num_scheduled_tokens[b])] = b
    rr = torch.from_numpy(row_req).to(dev)
    c_rows = corr.index_select(0, rr).view(pos.shape).to(pos.dtype)
    out.position_ids = pos - c_rows
    first = np.array([b * T for b in range(num_decodes)], dtype=np.int32)
    _f["p0_dev"] = out.position_ids.view(-1).index_select(0, torch.from_numpy(first).to(dev)).to(torch.int32)
    _f["keep_d"] = (prev, sched, row_req, first)
    _f["devpos_rows"] = _f.get("devpos_rows", 0) + 1


def row0_sources(self, so):
    """The last pending step's record, for the row-0 token ids of a skipped-sync (or devpos) step."""
    q = _f["fifo"]
    if not q:
        return None
    last = q[-1]
    return last


def row0_scatter(self, last, token_ids_device, T0: int) -> None:
    """Row 0 of each request <- out[b_prev, acc_b] of the last pending step, on the card (acc = 0 after plain steps)."""
    ib = self.input_batch
    dev = token_ids_device.device
    Wp = int(last.out_dev.shape[1])
    acc_all = (last.out_dev >= 0).to(torch.int32).sum(1) - 1
    prev = np.array([last.row_of[rid] for rid in last.reqs], dtype=np.int32)
    base = torch.from_numpy(prev * Wp).to(dev)
    src = (base + acc_all.index_select(0, torch.from_numpy(prev).to(dev))).to(torch.long)
    dst = torch.from_numpy(np.array([ib.req_id_to_index[rid] * T0 for rid in last.reqs], dtype=np.int64)).to(dev)
    flat = token_ids_device.view(-1)
    flat.index_copy_(0, dst, last.out_dev.reshape(-1).index_select(0, src).to(flat.dtype))
    _f["keep_r"] = prev
    _f["row0"] += 1


def _resolve_one(self, p) -> None:
    from . import spec

    p.ev.synchronize()
    rows = p.out_cpu.tolist()
    ib = self.input_batch
    full = self.max_model_len + 1
    st, s2 = spec._spec["stats"], spec._stats()
    any_draft = any(p.nd)
    if any_draft:
        s2["verify_steps"] += 1
        s2["rows"] += p.rows
    for b, rid in enumerate(p.reqs):
        toks = [t for t in rows[b] if t >= 0]
        i = ib.req_id_to_index.get(rid)
        if i is None or not toks:
            raise RuntimeError(f"ml4 asyncfull: pending step lost {rid} (index {i}, tokens {toks})")
        start = int(ib.num_tokens_no_spec[i])
        end = start + len(toks)
        assert end <= full, f"ml4 asyncfull: {rid} past max_model_len ({end} > {full})"
        ib.token_ids_cpu[i, start:end] = toks
        ib.num_tokens_no_spec[i] = end
        ib.num_tokens[i] = end
        self.requests[rid].output_token_ids.extend(toks)
        n = p.nd[b]
        if n:
            a = len(toks) - 1
            st[0], st[1], st[2], st[3] = st[0] + 1, st[1] + n, st[2] + a, st[3] + a + 1
            s2["drafted"] += n
            s2["accepted"] += a
            s2["emitted"] += a + 1
            s2["hist"][min(a, spec.DEPTH)] += 1
            spec._acc_update(rid, n, a)
    if any_draft:
        spec._stats_write(s2)
    _f["resolved"] += 1


def propose_device(runner, K: int, reqs, nd, p0, out, T: int, p0_dev=None) -> None:
    """spec._dspark_propose with the inputs on the card: x_m = out[b, acc_b], m = p0_b + acc_b + 1. The draft graph is
    launched whenever the batch's draft length L > 0 (it only reads the rings: no side effect when unused); whether
    the next step verifies the drafts (the acceptance tracker, dspark_go) is decided in the next _update_states
    (decide): after the late sync it sees the same tracker and table width as the sync path; when that step skips
    the sync (async_nosync / async_devpos) it is one step late. Leaves the drafts on the card (asyncspec.set_drafts)."""
    from . import asyncspec, spec
    from .control import ctl

    n = len(reqs)
    model = spec._dspark_model(runner)
    ds = getattr(model, "dspark", None)
    if ds is None or not ds.capture or ctl().get("spec", "on") == "off":
        return
    table, costs, adapt = spec._policy()
    L = spec.dspark_len(n, K, table)
    rows = [i for i in range(n) if reqs[i]]
    if not (L > 0 and rows):
        st = spec._stats()
        st["propose_calls"] += 1
        st["skipped"] += 1
        return
    top_hi = max((int(p0[i]) + int(nd[i]) + 1 for i in rows), default=0) + L
    B = 1
    while B < len(rows):
        B *= 2
    dev = ds.mk_w1.device
    idx = np.zeros(B, dtype=np.int64)
    valid = np.zeros(B, dtype=bool)
    p0b = np.zeros(B, dtype=np.int64)
    slot = np.full(B, -1, dtype=np.int64)
    uid = np.full(B, -1, dtype=np.int32)
    slots, uids = spec.request_slots()
    for j, i in enumerate(rows):
        idx[j], valid[j], p0b[j] = i, True, int(p0[i])
        slot[j] = slots.get(reqs[i], -1)
        uid[j] = uids.get(reqs[i], -1)
    t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
    acc = (out >= 0).to(torch.int32).sum(1) - 1  # [Bp] emitted - 1 (= accepted drafts)
    tok_all = out.gather(1, acc.clamp(min=0).to(torch.long).view(-1, 1)).view(-1)
    idx32, p0b32 = idx.astype(np.int32), p0b.astype(np.int32)
    ii, vv = t(idx32), t(valid)
    zero = torch.zeros(B, dtype=torch.int32, device=dev)
    tok = torch.where(vv, tok_all.to(torch.int32).index_select(0, ii), zero).to(torch.long)
    p0src = t(p0b32) if p0_dev is None else p0_dev.index_select(0, ii)
    mm = torch.where(vv, p0src + acc.index_select(0, ii) + 1, zero).to(torch.long)
    keep = (idx32, valid, p0b32, slot, uid)
    dd = ds.graph(B)(tok, mm, t(slot), t(uid))
    _f["keep_p"] = keep
    asyncspec.set_drafts(
        dd, {reqs[i]: (j, L) for j, i in enumerate(rows)}, ([reqs[i] for i in rows], L, adapt, costs, top_hi)
    )
    _f["drafts"] += 1


def decide(self, dec, exact: bool) -> bool:
    """The drafting decision for this step (spec._dspark_propose's gate), taken in _update_states. exact: the pending
    steps are resolved, so the tracker holds the last step's acceptance and the table width comes from the true
    positions, as on the sync path; otherwise (sync skipped) the tracker is one step late and the width an upper
    bound."""
    from . import spec, runner_patch

    rids, L, adapt, costs, top_hi = dec
    st = spec._stats()
    st["propose_calls"] += 1
    ib = self.input_batch
    if exact:
        idx = [ib.req_id_to_index[r] for r in rids if r in ib.req_id_to_index]
        top = max((int(ib.num_tokens_no_spec[i]) - 1 for i in idx), default=0) + L
    else:
        top = top_hi
        _f["late_dec"] = _f.get("late_dec", 0) + 1
    lk = runner_patch._bt_width(np.array([top])) * (runner_patch._state["block_size"] or 128)
    go = spec.dspark_go(list(rids), L, adapt, costs, lk)
    st["gate"] = spec._spec.get("gate")
    if go:
        st["draft_calls"] += 1
    else:
        st["skipped"] += 1
    return go


def counters() -> dict:
    return {
        k: _f.get(k, 0)
        for k in ("fast", "slow", "resolved", "drafts", "skip", "row0", "devpos_n", "devpos_rows", "late_dec")
    } | {"why": dict(_f["why"]), "fifo": len(_f["fifo"])}
