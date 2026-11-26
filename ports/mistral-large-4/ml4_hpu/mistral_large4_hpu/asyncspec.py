"""ML4_ASYNC_SPEC=1: vLLM async scheduling together with DSpark speculative decoding on vllm-gaudi (prototype).

vllm-gaudi refuses async scheduling with a speculative config (platform check + runner assert). This module lets the
ENGINE run vLLM's AsyncScheduler (step N+1 is scheduled, and its RPC queued, while step N runs on the cards) and keeps
the HPU runner on its synchronous path, which already supports spec decode:

  * engine side (vLLM upstream, unchanged): placeholders [-1] * K for the next step's drafts, optimistic
    num_computed_tokens (all drafts accepted), rejections rolled back when step N's output arrives;
  * worker side (here): before _update_states, num_computed_tokens of a request that verified drafts in the previous
    step is corrected to the worker's truth (num_tokens_no_spec - 1), and the placeholders are cut to the draft
    length DSpark actually chose for this request (0..K; the scheduler then counts the rest as rejected);
  * DSpark drafts stay on the device: the draft graph is launched at the end of step N and NOT synced; step N+1's
    token ids get them by an index_copy_ on the device right after their H2D copy, before the spec-decode metadata
    gathers draft_token_ids from them. So the draft graph overlaps the return to the engine and step N+1's host prep.

The accept sync (rejection sampler -> host) stays: the host needs the accepted counts for the drafts' positions, the
acceptance tracker and the output. Drafting decisions are the same as the sync path's (same host state, same place).
Default off; nothing here runs unless ML4_ASYNC_SPEC=1.
"""

import os
import sys
import logging

MODE = os.environ.get("ML4_ASYNC_SPEC", "0").strip() or "0"  # 0 off (served), 1 minimal, 2 full device-side port
if MODE not in ("0", "1", "2"):
    raise ValueError(f"ML4_ASYNC_SPEC={MODE!r}: want 0, 1 or 2")
ON = MODE in ("1", "2")
FULL = MODE == "2"
# ML4_RESUME_FIX: the resume fix (_resume_prep) on the SYNC path (ML4_ASYNC_SPEC=0, served). vllm-gaudi's
# InputBatch.add_request counts a resumed request's earlier output as prompt only when nothing of it is cached; with a
# prefix-cache / offload hit every non-final chunk of the refill samples a token the scheduler appends (a wrong token
# per extra chunk). The async modes always run the fix. To ship it default-on, change RESUME_DEFAULT to "1"
# (the only edit needed; ML4_RESUME_FIX=0 then turns it off).
RESUME_DEFAULT = "1"
RESUME = os.environ.get("ML4_RESUME_FIX", RESUME_DEFAULT).strip() or RESUME_DEFAULT
if RESUME not in ("0", "1"):
    raise ValueError(f"ML4_RESUME_FIX={RESUME!r}: want 0 or 1")
RESUME_SYNC = RESUME == "1" and not ON  # the sync-path wrapper (install_resume_fix); ON modes have their own
# ML4_FULLHIT_FIX (default 0): the _is_prompt decode route for a new request cached up to its last token (
# bug 2). Split from ML4_RESUME_FIX because on hardware  it fixes that request only alone at 1-4 streams: a
# full-hit request with logprobs gets HTTP 500 (KeyError in the API server's logprobs: the token comes from the decode
# row, the logprobs from another), inside an 8-request step (DSpark K = 2 vs 3 placeholder drafts) the token is still
# wrong, and a whole-prompt CPU-tier hit at P = 0 mod 128 stays wrong (KV, not the sampling row).
FULLHIT = os.environ.get("ML4_FULLHIT_FIX", "0").strip() or "0"
if FULLHIT not in ("0", "1"):
    raise ValueError(f"ML4_FULLHIT_FIX={FULLHIT!r}: want 0 or 1")
logger = logging.getLogger("vllm.mistral_large4_hpu.asyncspec")
_x: dict = {
    "prev_k": {},
    "cur": None,
    "steps": 0,
    "corr": 0,
    "cut": 0,
    "scatter": 0,
    "warn": 0,
    "resume": (),
    "resume_np": 0,
    "resync": 0,
    "readd": 0,
    "nd": None,
    "nd_tagged": 0,
    "resume_steps": 0,
    "resume_dec": 0,
    "fullhit_dec": 0,
    "fullhit_seen": set(),
}


# ------------------------------------------------------------------------------------------- process side
def _patch_runner_init(hmr) -> None:
    cls = hmr.HPUModelRunner
    if getattr(cls.__init__, "_ml4x", False):
        return
    orig = cls.__init__

    def __init__(self, vllm_config, *a, **k):
        sc = vllm_config.scheduler_config
        was = sc.async_scheduling
        sc.async_scheduling = False  # the runner's own async path is plain-decode only: keep its sync path
        try:
            orig(self, vllm_config, *a, **k)
        finally:
            sc.async_scheduling = was
        logger.warning(
            "ml4 asyncspec: HPU runner on its sync path under engine async scheduling (async=%s, spec=%s)",
            was,
            getattr(vllm_config.speculative_config, "method", None),
        )

    __init__._ml4x = True
    cls.__init__ = __init__


class _Finder:
    """Post-import hook: patch HPUModelRunner.__init__ when vllm_gaudi's runner module is first imported (worker)."""

    name = "vllm_gaudi.v1.worker.hpu_model_runner"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        import importlib.machinery

        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_runner_init(module)

        loader.exec_module = exec_module
        return spec


def install_process() -> None:
    """From register() in every process. Config: let explicit --async-scheduling through with the n-gram vehicle."""
    if not ON or _x.get("proc"):
        return
    _x["proc"] = True
    from typing import Literal
    import vllm.config.vllm as cv

    cv.NgramGPUTypes = Literal["ngram_gpu", "ngram"]  # only read by VllmConfig's async-scheduling checks
    try:
        from vllm_gaudi.platform import HpuPlatform

        orig = HpuPlatform.check_and_update_config.__func__

        def check_and_update_config(cls, vllm_config):
            was = vllm_config.scheduler_config.async_scheduling
            orig(cls, vllm_config)
            if was and vllm_config.speculative_config is not None:
                vllm_config.scheduler_config.async_scheduling = True

        HpuPlatform.check_and_update_config = classmethod(check_and_update_config)
    except Exception as e:  # noqa: BLE001
        logger.warning("ml4 asyncspec: platform patch failed: %r", e)
    if FULL:
        try:
            from . import asyncfull

            asyncfull.install_scheduler()
        except Exception as e:  # noqa: BLE001
            logger.warning("ml4 asyncspec: scheduler patch (mode 2) failed: %r", e)
    try:
        install_spec_stats()
    except Exception as e:  # noqa: BLE001
        logger.warning("ml4 asyncspec: spec-decode stats patch failed: %r", e)
    mod = sys.modules.get(_Finder.name)
    if mod is not None:
        _patch_runner_init(mod)
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def install_spec_stats() -> None:
    """Engine side: vllm:spec_decode_* as the sync path counts them.

    The AsyncScheduler schedules K placeholder drafts for every decoding request and the worker cuts them to the
    draft length DSpark chose (0..K). update_from_output then counted all K as drafted and K - accepted as rejected:
    vLLM's acceptance read 0.063 against the true 0.433 (measured). The worker now reports the drafts
    it verified per request (ModelRunnerOutput.ml4_spec_nd, an extra attribute that the executor's pickle carries);
    here the cut ones become num_invalid_spec_tokens, which vLLM already subtracts from the drafted counts only
    (stats and per-request metrics), so the rejection rollback of num_computed_tokens and the placeholders still
    sees all K. A request that verified no drafts at all is not a draft (the sync path schedules none for it):
    make_spec_decoding_stats skips it and its per-request histogram entry is taken back."""
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if getattr(AsyncScheduler, "_ml4_stats", False):
        return
    orig_update = AsyncScheduler.update_from_output
    orig_make = AsyncScheduler.make_spec_decoding_stats

    def make_spec_decoding_stats(
        self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens, request_id
    ):
        if request_id in getattr(self, "_ml4_nodraft", ()):
            return spec_decoding_stats
        return orig_make(
            self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens, request_id
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        nd = getattr(model_runner_output, "ml4_spec_nd", None)
        if not nd:
            return orig_update(self, scheduler_output, model_runner_output)
        spec = scheduler_output.scheduled_spec_decode_tokens or {}
        prev = scheduler_output.num_invalid_spec_tokens
        inv = dict(prev or {})
        nodraft, undo = set(), []
        for rid, n in nd.items():
            ks = len(spec.get(rid) or ())
            if n >= ks:
                continue
            inv[rid] = max(inv.get(rid, 0), ks - n)
            if n == 0:
                nodraft.add(rid)
                r = self.requests.get(rid)
                m = getattr(r, "spec_decode_metrics", None) if r is not None else None
                if m is not None:
                    undo.append((m, sum(m.histogram), len(m.per_step_accepted)))
        scheduler_output.num_invalid_spec_tokens = inv
        self._ml4_nodraft = nodraft
        try:
            return orig_update(self, scheduler_output, model_runner_output)
        finally:
            scheduler_output.num_invalid_spec_tokens = prev
            self._ml4_nodraft = ()
            for m, h, n in undo:  # observe(0 drafts, 0 accepted) ran for it: take it back
                if sum(m.histogram) == h + 1 and m.histogram[0] > 0:
                    m.histogram[0] -= 1
                    del m.per_step_accepted[n:]
                    del m.per_step_drafted[n:]

    AsyncScheduler.make_spec_decoding_stats = make_spec_decoding_stats
    AsyncScheduler.update_from_output = update_from_output
    AsyncScheduler._ml4_stats = True


def _tag_output(out, nd) -> None:
    """The drafts this step verified per request ({req_id: n}), on the ModelRunnerOutput the engine receives."""
    target = getattr(out, "_mro", out)
    if target is None or not hasattr(target, "sampled_token_ids") or not getattr(target, "req_ids", None):
        return
    target.ml4_spec_nd = nd
    _x["nd_tagged"] += 1


def _resume_prep(runner, cached, resumed) -> None:
    """Requests resumed from preemption, before vllm-gaudi's _update_states re-adds them to the batch.

    1. Their catch-up prefill must emit exactly one token, at its last chunk. vllm-gaudi's InputBatch.add_request
       counts the tokens emitted before the preemption as prompt (num_prompt_tokens = P + O, the basis of
       _is_prompt and of a chunk's logits count) only when num_computed_tokens == 0; with a prefix-cache or offload
       hit (c > 0) it keeps P, so every non-final chunk of the catch-up asks for c + s - P + 1 > 0 logits
       (runner_patch trims them to one) and emits a token the scheduler appends: in the async modes the next
       steps' optimistic counts then run one short ('scheduler num_computed 121350, worker 121351').
       Here add_request applies the c == 0 rule to every resumed request (_x["resume"]) that has a refill, i.e.
       c < P + O - 1. One that comes back with everything but its last token cached (c == P + O - 1) has no refill:
       the scheduler runs it as a decode (from the waiting queue with K placeholder drafts), so it keeps vllm-gaudi's
       own count, which keeps it on the decode path (P + O there sent its 1 + K rows down the prompt path).
    2. The worker's output tokens must be the scheduler's. vllm-gaudi takes them from all_token_ids on a resume only
       under its own async scheduling, which asyncspec turns off in the runner; when the scheduler dropped a
       preempted request's in-flight output (drop_stale_output: reset_prefix_cache with running requests, or a
       connector that requires KV delivery) the worker kept those tokens. Same recovery as the stock async path.
    3. A request resumed in the step it was preempted (reset_prefix_cache with running requests clears
       prev_step_scheduled_req_ids for that) is still in the persistent batch, where vllm-gaudi would keep its old
       row (old tokens, old num_prompt_tokens) and only reset num_computed_tokens: take it out (remove + condense,
       as vllm-gaudi does for removed requests) so it is re-added like any resumed request."""
    ib = runner.input_batch
    holes = []
    for rid in resumed:
        if rid in ib.req_id_to_index:
            i = ib.remove_request(rid)
            if i is not None:
                holes.append(i)
    if holes:
        ib.condense(sorted(holes, reverse=True))
        _x["readd"] += len(holes)
    all_ids = getattr(cached, "all_token_ids", None) or {}
    for i, rid in enumerate(cached.req_ids):
        if rid not in resumed:
            continue
        st = runner.requests.get(rid)
        n_out = int(cached.num_output_tokens[i])
        if st is not None and n_out > 0 and rid in all_ids:
            want = list(all_ids[rid][-n_out:])
            if list(st.output_token_ids) != want:
                st.output_token_ids = want
                _x["resync"] += 1
    _x["resume"] = set(resumed)
    cls = type(runner.input_batch)
    if getattr(cls.add_request, "_ml4r", False):
        return
    orig = cls.add_request

    def add_request(self, request, req_index=None):
        idx = orig(self, request, req_index)
        if request.req_id in _x["resume"]:
            was = int(self.num_prompt_tokens[idx])
            n_all = len(request.prompt_token_ids or ()) + len(request.output_token_ids)
            if request.num_computed_tokens >= n_all - 1:
                # everything but the last token is cached (a hit up to num_tokens - 1, e.g. a block-aligned
                # num_tokens - 1): no refill, the scheduler runs it as a decode (with K placeholder drafts from the
                # waiting queue). vllm-gaudi's own count (P) keeps it on the decode path; P + O would send its
                # 1 + K rows down the prompt path and sample at a placeholder position.
                _x["resume_dec"] += 1
                return idx
            self.num_prompt_tokens[idx] = n_all
            _x["resume_np"] += 1
            if _x["resume_np"] <= 50:
                logger.warning(
                    "ml4 %s: resumed %s re-added: num_computed %d, prompt tokens %d -> %d (resyncs %d, re-adds %d)",
                    "asyncspec" if ON else "resume",
                    request.req_id,
                    request.num_computed_tokens,
                    was,
                    int(self.num_prompt_tokens[idx]),
                    _x["resync"],
                    _x["readd"],
                )
        return idx

    add_request._ml4r = True
    cls.add_request = add_request


def install_resume_fix(runner_cls) -> None:
    """ML4_RESUME_FIX=1 on the sync path: _resume_prep before vllm-gaudi's _update_states, on steps that
    resume a request; every other step calls the wrapped _update_states unchanged. Also _is_prompt: a row cached up
    to its last token and scheduled as 1 + K rows with drafts is a decode (new requests with such a hit). Called from runner_patch.install
    after spec.install_runner (outermost wrapper). No-op unless RESUME_SYNC (flag on, ML4_ASYNC_SPEC=0); the async
    modes run _resume_prep inside install_runner's wrapper."""
    if not RESUME_SYNC or _x.get("resume_runner"):
        return
    _x["resume_runner"] = True
    orig_is_prompt = runner_cls._is_prompt

    def _is_prompt(self, req_idx, scheduler_output):
        """a row whose every token but the last is cached and that the scheduler sent as 1 + K rows with
        drafts (vLLM pads a waiting request with num_computed = num_tokens - 1 to 1 + K placeholder drafts: a new
        request whose prompt hits up to P - 1, e.g. P = 1 mod block or a full CPU-tier hit) is a decode. vllm-gaudi
        calls it a prompt (num_computed P - 1 < num_prompt_tokens P) and samples at the last placeholder row: a wrong
        first token. Every other row: vllm-gaudi's answer."""
        r = orig_is_prompt(self, req_idx, scheduler_output)
        if not r:
            return r
        ib = self.input_batch
        rid = ib.req_ids[req_idx]
        spec = scheduler_output.scheduled_spec_decode_tokens.get(rid)
        if (
            spec
            and scheduler_output.num_scheduled_tokens[rid] == 1 + len(spec)
            and int(ib.num_computed_tokens_cpu[req_idx]) >= int(ib.num_tokens_no_spec[req_idx]) - 1
        ):
            if rid not in _x["fullhit_seen"]:
                _x["fullhit_seen"].add(rid)
                _x["fullhit_dec"] += 1
                if _x["fullhit_dec"] <= 20:
                    logger.warning(
                        "ml4 resume: %s: cached up to its last token, %d placeholder drafts: decode row "
                        "(num_computed %d, prompt tokens %d)",
                        rid,
                        len(spec),
                        int(ib.num_computed_tokens_cpu[req_idx]),
                        int(ib.num_prompt_tokens[req_idx]),
                    )
            return False
        return r

    if FULLHIT == "1":
        runner_cls._is_prompt = _is_prompt
    orig_update = runner_cls._update_states

    def _update_states(self, so):
        if _x["fullhit_seen"] and so.finished_req_ids:
            _x["fullhit_seen"].difference_update(so.finished_req_ids)
        cached = so.scheduled_cached_reqs
        res = getattr(cached, "resumed_req_ids", None)
        if not res:
            return orig_update(self, so)
        _x["resume_steps"] += 1
        _resume_prep(self, cached, res)
        try:
            return orig_update(self, so)
        finally:
            _x["resume"] = ()

    _update_states._ml4_resume = True
    runner_cls._update_states = _update_states
    logger.warning(
        "ml4 resume: ML4_RESUME_FIX=1: resumed requests re-added with their earlier output as prompt (sync path)"
    )


# ------------------------------------------------------------------------------------------- runner side
def set_drafts(d, rows, decide=None) -> None:
    """From _dspark_propose: d = the draft graph's device output [B, depth] (not synced), rows = {req_id: (j, L)};
    decide (mode 2): the inputs of the drafting decision, taken in the next _update_states (asyncfull.decide)."""
    _x["next"] = (d, rows) if decide is None else (d, rows, decide)


def clear_drafts() -> None:
    _x["next"] = None


def install_runner(runner_cls) -> None:
    if not ON or _x.get("runner"):
        return
    _x["runner"] = True
    import torch

    orig_update = runner_cls._update_states

    def _update_states(self, so):
        skip = devpos = False
        nxt = _x.pop("next", None)
        rows = nxt[1] if nxt else {}
        if FULL:
            from . import asyncfull

            asyncfull._f["devpos"] = None
            asyncfull.drain(self)  # pending steps whose copy already landed: no wait
            skip = asyncfull.try_skip(self, so)
            if not skip:
                devpos = asyncfull.try_devpos(self, so, _x["prev_k"], rows)
            if not (skip or devpos):
                asyncfull.resolve(self)  # the late sync: step N's tokens into the host state first
            if nxt is not None and len(nxt) > 2 and not asyncfull.decide(self, nxt[2], exact=not (skip or devpos)):
                nxt, rows = None, {}  # the tracker says no drafts this step: the scheduler's are cut to 0
        _x["row0"] = skip or devpos
        ib = self.input_batch
        cached = so.scheduled_cached_reqs
        prev_k = _x["prev_k"]
        # a resumed request restarts from the scheduler's count (no optimism to correct; mode 1 too)
        resumed = getattr(cached, "resumed_req_ids", ()) or ()
        # 1. optimistic num_computed_tokens (all of last step's drafts accepted) -> the worker's truth
        for i, rid in enumerate(cached.req_ids):
            k = prev_k.get(rid, 0)
            if not k or rid in resumed or devpos:  # devpos: the card corrects the positions (asyncfull)
                continue
            idx = ib.req_id_to_index.get(rid)
            if idx is None:
                if _x["warn"] < 20:
                    _x["warn"] += 1
                    logger.warning(
                        "ml4 asyncspec: %s left the batch after a verify step (resumed?): no correction", rid
                    )
                continue
            truth = int(ib.num_tokens_no_spec[idx]) - 1
            if skip:  # pending plain steps: one token each, not yet in the host state
                truth += asyncfull.extra_tokens(rid)
            corr = int(cached.num_computed_tokens[i]) - truth
            if not 0 <= corr <= k:
                raise RuntimeError(
                    f"ml4 asyncspec: {rid}: scheduler num_computed {cached.num_computed_tokens[i]}, "
                    f"worker {truth}, last drafts {k}"
                )
            if corr:
                cached.num_computed_tokens[i] = truth
                _x["corr"] += 1
        # 2. placeholders -> the draft length DSpark chose (0..K)
        new_k = {}
        nd = {}  # drafts verified per request, for the engine's spec-decode stats
        spec = so.scheduled_spec_decode_tokens
        for rid in list(spec):
            ks = len(spec[rid])
            L = min(rows[rid][1], ks) if rid in rows else 0
            new_k[rid] = ks
            nd[rid] = L
            if L < ks:
                _x["cut"] += 1
                so.num_scheduled_tokens[rid] -= ks - L
                so.total_num_scheduled_tokens -= ks - L
                if L:
                    spec[rid] = [-1] * L
                else:
                    del spec[rid]
        _x["prev_k"] = new_k
        _x["cur"] = (nxt[0], {r: rows[r] for r in spec if r in rows}) if (nxt and spec) else None
        _x["steps"] += 1
        _x["nd"] = (so, nd)
        res = getattr(cached, "resumed_req_ids", None)
        if res:
            _resume_prep(self, cached, res)
        try:
            return orig_update(self, so)
        finally:
            _x["resume"] = ()

    runner_cls._update_states = _update_states

    orig_specin = runner_cls._prepare_spec_decode_inputs

    def _prepare_spec_decode_inputs(self, so, logits_indices, token_ids_device, max_num_sampled_tokens):
        cur = _x.pop("cur", None)
        _x["cur"] = None
        if _x.pop("row0", False) and so is not None:
            # skipped sync: each request's row-0 token is the last pending step's output out[b, acc_b] (on the card)
            from . import asyncfull

            last = asyncfull.row0_sources(self, so)
            if last is not None:
                asyncfull.row0_scatter(self, last, token_ids_device, int(max_num_sampled_tokens))
        spec = so.scheduled_spec_decode_tokens if so is not None else None
        if cur is not None and spec:
            d, rows = cur
            T = int(max_num_sampled_tokens)
            W = int(d.shape[1])
            src, dst = [], []
            ib = self.input_batch
            for rid, toks in spec.items():
                if rid not in rows:
                    raise RuntimeError(f"ml4 asyncspec: {rid} scheduled {len(toks)} drafts that are not on the device")
                j, L = rows[rid]
                b = ib.req_id_to_index[rid]
                for q in range(len(toks)):
                    src.append(j * W + q)
                    dst.append(b * T + 1 + q)
            if src:
                dev = token_ids_device.device
                s = torch.tensor(src, dtype=torch.long).to(dev)
                t = torch.tensor(dst, dtype=torch.long).to(dev)
                flat = token_ids_device.view(-1)
                flat.index_copy_(0, t, d.reshape(-1).index_select(0, s).to(flat.dtype))
                _x["scatter"] += 1
        return orig_specin(self, so, logits_indices, token_ids_device, max_num_sampled_tokens)

    runner_cls._prepare_spec_decode_inputs = _prepare_spec_decode_inputs

    if FULL and hasattr(runner_cls, "_create_decode_input_data"):
        orig_cdi = runner_cls._create_decode_input_data

        def _create_decode_input_data(
            self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output=None
        ):
            out = orig_cdi(
                self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output
            )
            if scheduler_output is not None:
                from . import asyncfull

                asyncfull.devpos_positions(self, out, num_decodes, num_scheduled_tokens)
            return out

        runner_cls._create_decode_input_data = _create_decode_input_data

    # The executor serialises the output on its async-output thread while this thread runs the next step's
    # _update_states: the sync path hands out the live req_id_to_index dict, so give it a copy (as the async path).
    orig_sample = runner_cls.sample_tokens

    def sample_tokens(self, grammar_output):
        tag = _x.pop("nd", None)  # this step's verified drafts (_update_states), if it is this step's
        nd = tag[1] if tag is not None and tag[0] is getattr(self, "scheduler_output", None) else None
        if FULL:
            from . import asyncfull

            if asyncfull.fast_ok(self, grammar_output):
                out = asyncfull.fast_sample_tokens(self, grammar_output)
                if nd:
                    _tag_output(out, nd)
                return out
            asyncfull._f["slow"] += 1
            if asyncfull._f["fifo"]:  # not expected after try_skip's checks; the slow path needs the host state
                asyncfull._f["why"]["late_resolve"] = asyncfull._f["why"].get("late_resolve", 0) + 1
                asyncfull.resolve(self)
                asyncfull._f["devpos"] = None
                ib = self.input_batch  # devpos left the scheduler's optimistic positions: the truth now
                for i in range(ib.num_reqs):
                    rid = ib.req_ids[i]
                    if rid is not None and not self._is_prompt(i, self.scheduler_output):
                        ib.num_computed_tokens_cpu[i] = int(ib.num_tokens_no_spec[i]) - 1
                        self.requests[rid].num_computed_tokens = int(ib.num_tokens_no_spec[i]) - 1
        out = orig_sample(self, grammar_output)
        if out is not None and isinstance(getattr(out, "req_id_to_index", None), dict):
            out.req_id_to_index = dict(out.req_id_to_index)
            if isinstance(getattr(out, "req_ids", None), list):
                out.req_ids = list(out.req_ids)
        if nd and out is not None:
            _tag_output(out, nd)
        return out

    runner_cls.sample_tokens = sample_tokens
    logger.warning(
        "ml4 asyncspec: ML4_ASYNC_SPEC=%s: engine async scheduling + DSpark drafts kept on the device%s",
        MODE,
        " + device rejection, late sync, device draft inputs (asyncfull)" if FULL else "",
    )


def counters() -> dict:
    c = {k: _x[k] for k in ("steps", "corr", "cut", "scatter", "warn", "resume_np", "resync", "readd", "nd_tagged")}
    if RESUME_SYNC:
        c["resume_steps"] = _x["resume_steps"]
        c["resume_dec"] = _x["resume_dec"]
        c["fullhit_dec"] = _x["fullhit_dec"]
    if FULL:
        from . import asyncfull

        c["full"] = asyncfull.counters()
    return c
