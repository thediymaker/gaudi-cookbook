"""Speculative decoding for Mistral Large 4 on HPU: DSpark on the speculative driver of the
DeepSeek-V4.1 port.

ML4_SPEC=dspark (off by default; unset, nothing here is installed and serving does not change):

  vehicle    vLLM's `--speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 2}'`
             schedules drafts and verify rows; install_runner() replaces the n-gram proposals with DSpark's and
             vLLM's RejectionSampler.forward with rejection_forward() below. vllm-gaudi lays a verify step out as
             decode rows of one token each, request b's tokens in rows b*T .. b*T+T-1 (T = 1 + the batch's longest
             draft, at most 1 + 3: DSpark's block is 3 and its masks are position-specific).
  verify     the runner builds flat rows (_install_verify_layout): each of a request's T rows is a one-token decode
             row with its own position, the request's uid, block table and ring slot, and the step's T rides on the
             attention metadata (ml4_verify_rows). The model runs them as rows (ML4_VERIFY_ROWS, default):
             as_rows reshapes them to [B, T], one request per row, through the target's decode group graphs, and the
             output goes back to the flat order. Flat (ML4_VERIFY_ROWS=0): the decode graphs at B*T one-token rows
            , where every row re-reads and re-scores its request's whole index-K range.
             Every cache row is addressed by absolute position and tagged, so rejected drafts need no rollback
             (each layer writes all rows' entries before any row reads, reads are causal, and the next step rewrites
             a rejected position before any query reads it). MoE: the MXK decode op takes at most 16 tokens per
             call; moe.ML4MoE.routed runs larger decode calls in 16-token pieces.
  sampler    rejection_forward: exact for greedy and for temperature / top-p / top-k requests (Leviathan et al.
             with deterministic drafts: accept d with probability p(d), else sample p without d; all accepted:
             sample the bonus row). Only "plain" requests draft (_plain); the rest get no drafts.
  drafts     one DSpark draft graph call per step over the batch (dspark.ML4DSpark.graph(B)), greedy, from the
             rings the target's forwards wrote (taps -> TapBuffer -> write_context). Length by batch size
             (ML4_DSPARK_K) and the acceptance tracker (dspark_go) decide whether a step drafts at all.

Hooks in the target:
  * vllm_model: build() in __init__, load() at the end of load_weights, before_forward() every forward (the ring
    slot of each row into drafter.slot_buf; control key "spec": "off" turns taps, ring writes and drafts off);
  * taps (install_taps): forward hooks on target layers 1, 18, 36, 54 and 71 copy this rank's slice of the stream
    into the TapBuffer, and layer 71's hook projects the taps and writes the rings (write_context), inside whichever
    decode graph or eager stack runs the layer: prompt chunks, decode and verify steps alike;
  * runner_patch.install() calls install_runner(): request slots and uids are runner_patch's.
Control keys (bring-up, control.py): spec (on/off), dspark_k, dspark_cost, dspark_adapt, dspark_ctx_g,
dspark_ctx (the long-context gate), verify_rows, spec_epoch (resets the counters rank 0 writes
to ML4_SPEC_STATS).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm.mistral_large4_hpu")

SPEC = os.environ.get("ML4_SPEC", "")
DEPTH = 3  # DSpark block: drafts per call (num_speculative_tokens must be <= this)
TAP_LAYERS = (1, 18, 36, 54, 71)  # target layers whose OUTPUT stream is a tap (71: the final pre-norm stream)
# Draft length by batch size ("requests:length", the smallest entry at or above the batch). Default: 3 drafts up to
# 4 requests, 2 at 8.
DSPARK_K = sorted(
    (int(a), int(b))
    for a, b in (kv.split(":") for kv in os.environ.get("ML4_DSPARK_K", "1:3,2:3,4:3,8:2").split(",") if kv)
)
# Acceptance tracker (ML4_DSPARK_ADAPT, default on), DSV's: f = EMA of the fraction of a request's drafts accepted,
# a step with L drafts emits 1 + f L tokens; draft only when the batch mean reaches the verify step's cost relative
# to a plain step (ML4_DSPARK_COST by batch). Defaults MEASURED on the TP8 server with placement B and the v9g MoE
# kernel (ML4_PLACEMENT=B, ML4_MXK_V9G=1): the break-even cost seen from drafting runs
# (tokens per verify step / speed-up over plain at the same stream count) = 1.38 / 1.87 / 2.15 / 2.07 at 1 / 2 / 4 / 8
# requests (8: 2 drafts). Placement A with v8 measured 1.71 / 2.28 / 2.70 / 2.30: set ML4_DSPARK_COST to those
# there. The MXK decode kernel's cost grows with routes, so a verify step's MoE costs more than a plain step's.
ADAPT = os.environ.get("ML4_DSPARK_ADAPT", "1") == "1"
DSPARK_COST = sorted(
    (int(a), float(b))
    for a, b in (
        kv.split(":") for kv in os.environ.get("ML4_DSPARK_COST", "1:1.38,2:1.87,4:2.15,8:2.07").split(",") if kv
    )
)
ACC_PRIOR, ACC_EMA, ACC_RELAX = 0.6, 0.3, 0.02
# The gate at long context. ML4_DSPARK_COST is the verify / plain ratio measured on short prompts, where no
# indexer runs (every position <= 2047). Past that every row pays the indexer over the next step's table width Lk
# (keys): the ratio is ML4_DSPARK_COST x g(Lk), g from ML4_DSPARK_CTX_G ("keys:g_rows:g_flat" points, linear in Lk
# between them, the end points held outside; g_flat when the verify step runs the flat layout, ML4_VERIFY_ROWS=0).
# Measured on the serving config (synced graph step + the step's host part, 1 stream): with the rows layout
# the ratio stays the short one at every width (1.35 at 16K-64K and 1.28-1.37 at 128K against 1.355 dense; 4 streams
# 2.0 at 16K / 64K against 2.15), so g_rows is 1; the flat layout's verify step re-reads the index-K range per row and
# its ratio grows to 1.41 / 1.63 / 1.71 at 16K / 64K / 128K (g_flat 1.04 / 1.20 / 1.26). Short prompts keep the old
# ratio exactly. Control keys dspark_ctx_g / dspark_ctx ("off": the short ratio at every width).
DSPARK_CTX_G = os.environ.get("ML4_DSPARK_CTX_G", "4096:1:1,16384:1:1.04,65536:1:1.20,131072:1:1.26")
DSPARK_CTX = os.environ.get("ML4_DSPARK_CTX", "on") == "on"
# ML4_VERIFY_ROWS=1 (default; control key verify_rows): the model runs a verify step's flat rows as [B, T] (as_rows),
# one row of T tokens per request, through the decode graphs (graphs.DecodeGraphs keys on (B, T, table)): each full
# indexer reads and unpacks a request's index-K range once and scores its T queries in one dot, where the flat layout
# did that T times. "0": the flat layout (each of the T rows its own decode row).
VERIFY_ROWS = os.environ.get("ML4_VERIFY_ROWS", "1") == "1"
MOE_MAX_T = int(os.environ.get("ML4_SPEC_MOE_CHUNK", "16"))  # MXK decode op: <= 16 tokens (128 routes) per call
LOG_EVERY = int(os.environ.get("ML4_SPEC_LOG_EVERY", "2000"))
# ML4_SPEC_GRAMMAR=1 (default; r13): a structured-output request (a tool call's or response_format's grammar) drafts
# like a plain one. The scheduler truncates its drafts at the first token the grammar rejects (validate_tokens) and
# sends 1 + n mask rows for it (row i: the mask after drafts 0..i-1); a verify step applies them to the request's rows
# b*T .. b*T+n on the device before the verify sampler (so_verify), and the step's prompt rows take theirs as before
# (so_prompts). 0: today's behaviour (a grammar request is not plain, so no request in its batch drafts). Control key
# spec_grammar (0/1) switches it on a server started with it on (A/B; change it only while the server is idle).
SPEC_GRAMMAR = os.environ.get("ML4_SPEC_GRAMMAR", "1") == "1"
_EPS = 1e-5  # vLLM's _SAMPLING_EPS: temperatures below it are greedy (vllm-gaudi marks greedy as -1)
_spec: dict = {"runner": None, "stats": [0, 0, 0, 0]}  # steps with drafts, drafted, accepted, emitted


# ------------------------------------------------------------------------------------------- draft policy
def _table(v, cast):
    return sorted((int(a), cast(b)) for a, b in (kv.split(":") for kv in str(v).split(",") if kv))


def _policy():
    """(draft length table, cost table, adapt): the env defaults, or the control file's dspark_k / dspark_cost /
    dspark_adapt (bring-up A/B on one server; control.py). Same answer on every rank (they read one file)."""
    from .control import ctl

    c = ctl()
    k = _table(c["dspark_k"], int) if c.get("dspark_k") else DSPARK_K
    cost = _table(c["dspark_cost"], float) if c.get("dspark_cost") else DSPARK_COST
    return k, cost, bool(c.get("dspark_adapt", ADAPT))


def _ctx_points(v) -> list:
    return sorted(tuple(float(x) for x in kv.split(":")) for kv in str(v).split(",") if kv)


def _interp(pts, lk: float, col: int) -> float:
    if lk <= pts[0][0]:
        return pts[0][col]
    for a, b in zip(pts, pts[1:]):
        if lk <= b[0]:
            return a[col] + (b[col] - a[col]) * (lk - a[0]) / (b[0] - a[0])
    return pts[-1][col]


def verify_cost(n: int, L: int, lk: int, costs=None) -> float:
    """verify / plain step time for n drafting requests at the next step's table width lk keys: C(n) x g(lk) (the gate
    model above; lk <= 2048, the dense regime, or the context model off: C(n)). L (the drafts) is the batch's length
    by ML4_DSPARK_K, the one C(n) was measured at."""
    from .control import ctl
    from .indexer import TOPK

    costs = DSPARK_COST if costs is None else costs
    C = next((c for m, c in costs if m >= n), costs[-1][1])
    c = ctl()
    if lk <= TOPK or c.get("dspark_ctx", "on" if DSPARK_CTX else "off") == "off":
        return C
    pts = _ctx_points(c.get("dspark_ctx_g") or DSPARK_CTX_G)
    return C * _interp(pts, lk, 1 if verify_rows_on() else 2)


def dspark_len(n_req: int, K: int, table=None) -> int:
    table = DSPARK_K if table is None else table
    return min(K, DEPTH, next((L for n, L in table if n >= n_req), 0))


def _acc_update(rid, n: int, acc: int) -> None:
    f = _spec.setdefault("acc", {})
    f[rid] = f.get(rid, ACC_PRIOR) * (1 - ACC_EMA) + ACC_EMA * acc / n


def dspark_go(rids, L: int, adapt: bool = ADAPT, costs=None, lk: int = 0) -> bool:
    """Draft this step? Expected tokens per step with L drafts vs the verify step's relative cost at the next step's
    table width lk keys (verify_cost; 0: the short ratio)."""
    if not adapt or L == 0:
        return L > 0
    f = _spec.setdefault("acc", {})
    est = 1.0 + L * sum(f.get(r, ACC_PRIOR) for r in rids) / len(rids)
    cost = verify_cost(len(rids), L, lk, costs)
    _spec["gate"] = (len(rids), L, lk, round(cost, 3), round(est, 3))
    if est >= cost:
        return True
    for r in rids:  # no drafts this step: drift back toward the prior (probe again later)
        f[r] = f.get(r, ACC_PRIOR) + ACC_RELAX * (ACC_PRIOR - f.get(r, ACC_PRIOR))
    _spec["skipped"] = _spec.get("skipped", 0) + 1
    return False


def draft_lengths(n_req: int, K: int) -> list:
    return sorted({0, dspark_len(n_req, K)})


def verify_T_sets(bss, K: int) -> dict:
    """Rows per request (T = drafts + 1) of the verify graphs to warm for each decode batch bucket (DSV: a step's T
    comes from the previous step's batch; as_rows pads up to / cuts down to a warmed T)."""
    top = max([L for b in bss for L in draft_lengths(b, K)] + [0]) + 1
    return {B: sorted({L + 1 for L in draft_lengths(B, K) if L} or ({top} if top > 1 else set())) for B in bss}


def moe_chunks(n_tokens: int, max_t: int = MOE_MAX_T) -> list:
    """[start, stop) pieces of at most max_t tokens for the MXK decode op (verify rows B x T > 16).
    Note (moe.py): ML4MoE.forward runs a decode-sized call above max_t as these pieces (DSV41MoEHPU
    chunk_tokens); each extra piece costs ~16 us fixed per layer (04 sweep, 0 routes), the routes are linear anyway."""
    return [(s, min(n_tokens, s + max_t)) for s in range(0, n_tokens, max_t)]


# ------------------------------------------------------------------------------------------- target taps
def stream_of(out):
    """A layer's output as the full residual stream [B, S, D]: a tensor, or (hidden, residual) as vLLM layers
    return it (the stream is their sum)."""
    if isinstance(out, torch.Tensor):
        return out
    h, res = out[0], out[1] if len(out) > 1 else None
    return h if res is None else h + res


def install_taps(stack, drafter, layers_attr: str = "layers") -> list:
    """Forward hooks on the target layers in TAP_LAYERS: each copies this rank's slice of the stream LEAVING the
    layer into the drafter's TapBuffer (in place, so the copies replay inside the decode-group HPU graphs), and the
    last one (layer 71, the stream before the final norm) then projects the taps and writes the rings for this
    forward's rows (write_context): prompt chunks, decode and verify steps alike, inside whichever graph or eager
    stack runs the layer. A layer's forward args are (x, ctx, caches, shared, rope, allreduce); its output is the
    all-reduced stream [B, S, D]. The ring slot of each row comes from drafter.slot_buf (before_forward fills it
    every forward: the decode graphs' BatchCtx carries no slot). drafter.capture False
    makes the hooks no-ops for graphs captured afterwards (tests; serving leaves it on, before_forward).
    Returns the hook handles."""
    hs = []
    layers = getattr(stack, layers_attr)
    last_k = len(TAP_LAYERS) - 1
    for k, L in enumerate(TAP_LAYERS):

        def hook(mod, args, out, k=k):
            x = stream_of(out)
            N = x.shape[0] * x.shape[1]
            if not drafter.capture or N > drafter.taps.max_rows or x.shape[0] > drafter.slot_buf.shape[0]:
                return
            drafter.taps.tap(k, x)
            if k == last_k:
                ctx = args[1]
                pos = ctx.positions
                drafter.write_context(pos, drafter.slot_buf[: pos.shape[0]], ctx.req_uid)

        hs.append(layers[L].register_forward_hook(hook))
    return hs


def after_forward(drafter, positions, slot, uid) -> None:
    """Project the taps and write the rings (the last tap hook does this in serving; tests call it directly).
    positions [B, S] (-1 padding), slot [B] (-1: no ring), uid [B] int32."""
    drafter.write_context(positions, slot, uid)


# ------------------------------------------------------------------------------------------- verify rows
def verify_rows_on() -> bool:
    """Verify steps in the rows layout (ML4_VERIFY_ROWS; control key verify_rows)."""
    from .control import get as ctl_get

    return bool(ctl_get("verify_rows", VERIFY_ROWS))


def as_rows(input_ids, ctx, T: int):
    """vllm-gaudi's flattened verify rows [B*T, 1] -> [B, T] for the model (DSV41_SPEC_LAYOUT=rows): every per-row
    tensor of the BatchCtx is the request's own, so its first row stands for the request (a request's rows past its
    own tokens are padding rows, position -1 already: no cache write, every read masked). Rows are padded the same
    way up to the smallest T warmed for the batch, or cut to the largest (rejection_forward then verifies only the
    drafts that fit); serving warms nothing (warm_T unset), so T stays the step's own. Fresh copies, not views: the
    layer-group HPU graphs hash their inputs' view state (DSV: views re-captured every verify shape, +12 GiB)."""
    from .paged import BatchCtx

    R = input_ids.shape[0]
    B = R // T
    first = lambda t: None if t is None else t.view(B, T, *t.shape[1:])[:, 0].clone()  # noqa: E731
    c = BatchCtx(
        positions=ctx.positions.reshape(B, T).clone(),
        block_table=first(ctx.block_table),
        req_uid=first(ctx.req_uid),
        block_size=ctx.block_size,
        block_epoch=first(ctx.block_epoch),
        host_max_pos=ctx.host_max_pos,
        req_slot=first(ctx.req_slot),
    )
    ids = input_ids.reshape(B, T)
    ts = _spec.get("warm_T", {}).get(B, [])
    Tw = next((t for t in ts if t >= T), max(ts) if ts else T)
    _spec["verify_cap"] = (T, Tw)
    if Tw < T:
        ids, c.positions = ids[:, :Tw].clone(), c.positions[:, :Tw].clone()
    if Tw > T:
        pad = lambda x, v: torch.cat([x, torch.full((B, Tw - T), v, dtype=x.dtype, device=x.device)], 1)  # noqa
        ids, c.positions = pad(ids, 0), pad(c.positions, -1)
    return ids, c


# ------------------------------------------------------------------------------------------- the verify sampler
def _plain(sp, grammar: bool = False) -> bool:
    """Requests whose sampling the verify sampler reproduces exactly: temperature / top-p / top-k only, and with
    grammar=True (spec_grammar_on: the verify step applies the masks, so_verify) also a structured-output grammar."""
    return (
        sp is not None
        and sp.logprobs is None
        and sp.prompt_logprobs is None
        and not sp.presence_penalty
        and not sp.frequency_penalty
        and sp.repetition_penalty in (None, 1.0)
        and not sp.logit_bias
        and not sp.min_p
        and not sp.min_tokens
        and not sp.bad_words
        and not sp.allowed_token_ids
        and (grammar or sp.structured_outputs is None)
        and sp.seed is None
        and not getattr(sp, "logprob_token_ids", None)
        and getattr(sp, "thinking_token_budget", None) is None
        and getattr(sp, "repetition_detection", None) is None
    )


def draftable(sp) -> bool:
    """Requests the sync DSpark path drafts for, seen from any process (the engine's prefix-cache cap included):
    _plain, with a grammar allowed when ML4_SPEC_GRAMMAR and the control key spec_grammar say so (never in the async
    modes, ML4_ASYNC_SPEC)."""
    from . import asyncspec

    return _plain(sp, SPEC_GRAMMAR and not asyncspec.ON and spec_grammar_on())


def _sample_rows(logits, temp, top_p, top_k):
    """Per row: argmax, probs after temperature / top-k / top-p (fp32), on the logits' device."""
    lf = logits.float()
    am = lf.argmax(-1)
    t = torch.where(temp < _EPS, torch.ones_like(temp), temp)
    lf = lf / t.unsqueeze(-1)
    V = lf.shape[-1]
    if bool(((top_k > 0) & (top_k < V)).any()) or bool((top_p < 1.0).any()):
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


def verify_tokens(logits, drafts, n_draft, temp, top_p, top_k, gen=None):
    """The verify rule on one step, host loop over requests (also the unit the sampler test drives on the CPU):
    logits [Bp*T, V], drafts [Bp, T-1] (long), n_draft [Bp], temp/top_p/top_k per row [Bp*T] -> out [Bp, T] int32
    (-1 past the emitted tokens), accepted counts [Bp]."""
    Bp = len(n_draft)
    R = logits.shape[0]
    T = R // Bp
    dev = logits.device
    dfull = torch.zeros(Bp, T, dtype=torch.long)
    if T > 1:
        dfull[:, : T - 1] = drafts.clamp(min=0)
    if bool((temp < _EPS).all()):
        am = logits.argmax(-1).cpu()
        pd = samp = res = u = None
    else:
        am, probs = _sample_rows(logits, temp.to(dev), top_p.to(dev), top_k.to(dev))
        dd = dfull.view(-1).to(dev)
        pd = probs.gather(-1, dd.view(-1, 1)).view(-1)
        expo = torch.empty_like(probs).exponential_(1.0, generator=gen)
        samp = (probs / expo).argmax(-1)
        res = (probs.scatter(-1, dd.view(-1, 1), 0.0) / expo).argmax(-1)
        u = torch.rand(R, device=dev, generator=gen)
        am, pd, samp, res, u = torch.stack([am.float(), pd.float(), samp.float(), res.float(), u.float()]).cpu()
        am, samp, res = am.long(), samp.long(), res.long()
    out = torch.full((Bp, T), -1, dtype=torch.int32)
    acc = torch.zeros(Bp, dtype=torch.long)
    for b in range(Bp):
        greedy = float(temp[b * T]) < _EPS
        n, base = int(n_draft[b]), b * T
        for j in range(n):
            row, d = base + j, int(dfull[b, j])
            ok = int(am[row]) == d if greedy else float(u[row]) < float(pd[row])
            if ok:
                out[b, j] = d
            else:
                out[b, j] = int(am[row]) if greedy else int(res[row])
                break
        else:
            out[b, n] = int(am[base + n]) if greedy else int(samp[base + n])
        acc[b] = int((out[b, :n] == dfull[b, :n].to(torch.int32)).cumprod(0).sum()) if n else 0
    return out, acc


STATS_PATH = os.environ.get("ML4_SPEC_STATS", "/tmp/ml4_spec_stats.json")


def _stats() -> dict:
    """Counters since the control file's spec_epoch last changed (rank 0 writes them to STATS_PATH after every verify
    step, ~0.1 ms: the bring-up client reads acceptance per run from there)."""
    from .control import ctl

    ep = ctl().get("spec_epoch")
    st = _spec.get("st")
    if st is None or st["epoch"] != ep:
        st = _spec["st"] = {
            "epoch": ep,
            "verify_steps": 0,
            "rows": 0,
            "drafted": 0,
            "accepted": 0,
            "emitted": 0,
            "hist": [0] * (DEPTH + 1),
            "propose_calls": 0,
            "draft_calls": 0,
            "skipped": 0,
            "draft_ms": 0.0,
            "so_steps": 0,
            "so_rows": 0,
            "so_drafted": 0,
            "so_accepted": 0,
            "so_mask_ms": 0.0,
        }
    return st


def _stats_write(st) -> None:
    if not _spec.get("rank0"):
        return
    import json

    try:
        with open(STATS_PATH + ".tmp", "w") as f:
            json.dump(st, f)
        os.replace(STATS_PATH + ".tmp", STATS_PATH)
    except OSError:
        pass


def ragged_layout(nd, R: int, F: int):
    """r11: the step width Tm of a verify step whose draft counts are not uniform, from vllm-gaudi's own layout
    (_prepare_spec_decode_inputs): request b owns max(nd_b + 1, Tm) logits rows and max(nd_b, Tm - 1) draft entries,
    in order, so R = the sum of the first and F = the sum of the second. A request with nd_b + 1 > Tm overflowed: its
    rows past Tm are not its own (the forward ran Tm rows per request). The largest Tm that fits is the step's (every
    decode row has nd_b + 1 <= Tm; a smaller Tm that also fits gives the same offsets). -> (Tm, row offsets, draft
    offsets) or None."""
    top = max(nd, default=0) + 1
    for t in range(top, 0, -1):
        if sum(max(n + 1, t) for n in nd) == R and sum(max(n, t - 1) for n in nd) == F:
            ro, do, a, b = [], [], 0, 0
            for n in nd:
                ro.append(a)
                do.append(b)
                a += max(n + 1, t)
                b += max(n, t - 1)
            return t, ro, do
    return None


def _uniform_step(metadata, logits, nd):
    """r11: (logits [Bp*Tm, V], flat drafts [Bp*(Tm-1)] or None, nd capped at Tm - 1, Tm) for a ragged verify step:
    each request's own first Tm rows and first Tm - 1 drafts, the uniform layout verify_tokens expects. A request
    whose drafts did not fit the step (a non-decode row inside the decode bucket carrying the scheduler's 1 + K
    placeholder rows: view(8, 2) of 17 drafts) is verified on its first Tm - 1 drafts only."""
    Bp = len(nd)
    flat = metadata.draft_token_ids.view(-1)
    lay = ragged_layout(nd, int(logits.shape[0]), int(flat.numel()))
    if lay is None:
        raise RuntimeError(
            f"ml4 spec: verify step with {logits.shape[0]} logits rows and {flat.numel()} drafts "
            f"does not match draft counts {nd}"
        )
    Tm, ro, do = lay
    rows = torch.tensor([ro[b] + j for b in range(Bp) for j in range(Tm)], dtype=torch.long)
    drafts = None
    if Tm > 1:
        di = torch.tensor([do[b] + j for b in range(Bp) for j in range(Tm - 1)], dtype=torch.long)
        drafts = flat.index_select(0, di.to(flat.device))
    _spec["ragged"] = _spec.get("ragged", 0) + 1
    if _spec["ragged"] <= 20:
        logger.warning(
            "ml4 spec: ragged verify step (draft counts %s, %d logits rows, %d drafts): verified at width "
            "%d (%d so far)",
            nd,
            int(logits.shape[0]),
            int(flat.numel()),
            Tm,
            _spec["ragged"],
        )
    return logits.index_select(0, rows.to(logits.device)), drafts, [min(n, Tm - 1) for n in nd], Tm


def rejection_forward(self, metadata, draft_probs, logits, sampling_metadata):
    """RejectionSampler.forward for vllm-gaudi's flattened layout (installed by install_runner). The emitted tokens
    follow the target's own sampling distribution exactly (verify_tokens)."""
    from vllm.v1.outputs import SamplerOutput

    r = _spec["runner"]
    nd = list(metadata.num_draft_tokens)
    Bp = len(nd)
    T = logits.shape[0] // Bp
    flat = None
    if T * Bp != logits.shape[0] or metadata.draft_token_ids.numel() != Bp * (T - 1) or max(nd, default=0) > T - 1:
        # r11: not the uniform layout. Never once only decode rows carry drafts (_install_spec_rows); this keeps a
        # step that still gets here from killing the engine.
        logits, flat, nd, T = _uniform_step(metadata, logits, nd)
    cap = _spec.pop("verify_cap", None)
    if cap is not None and cap[0] == T and cap[1] < T:  # as_rows cut the step: drafts past it were not run
        nd = [min(n, cap[1] - 1) for n in nd]
    ib = r.input_batch
    temp = torch.tensor(
        [float(ib.temperature_cpu[b]) if b < ib.num_reqs else -1.0 for b in range(Bp)], dtype=torch.float32
    ).repeat_interleave(T)
    top_p = torch.tensor(
        [float(ib.top_p_cpu[b]) if b < ib.num_reqs else 1.0 for b in range(Bp)], dtype=torch.float32
    ).repeat_interleave(T)
    top_k = torch.tensor(
        [int(ib.top_k_cpu[b]) if b < ib.num_reqs else 0 for b in range(Bp)], dtype=torch.int64
    ).repeat_interleave(T)
    if flat is None and T > 1:
        flat = metadata.draft_token_ids.view(-1)
    drafts = flat.to(torch.long).cpu().view(Bp, T - 1) if T > 1 else None
    out, acc = verify_tokens(logits, drafts, nd, temp, top_p, top_k)
    st, s2 = _spec["stats"], _stats()
    s2["verify_steps"] += 1
    s2["rows"] += logits.shape[0]
    for b in range(min(Bp, ib.num_reqs)):
        if nd[b]:
            a = int(acc[b])
            st[0], st[1], st[2], st[3] = st[0] + 1, st[1] + nd[b], st[2] + a, st[3] + a + 1
            s2["drafted"] += nd[b]
            s2["accepted"] += a
            s2["emitted"] += a + 1
            s2["hist"][min(a, DEPTH)] += 1
            if ib.req_ids[b] in _spec.get("so_reqs", ()):
                s2["so_drafted"] += nd[b]
                s2["so_accepted"] += a
            _acc_update(ib.req_ids[b], nd[b], a)
    _stats_write(s2)
    if st[0] // LOG_EVERY > _spec.get("logged", 0):
        _spec["logged"] = st[0] // LOG_EVERY
        logger.warning(
            "ml4 spec: %d verify rows with drafts, %d drafted, %d accepted (%.2f), %.2f tokens per step; "
            "%d steps skipped by the acceptance tracker",
            st[0],
            st[1],
            st[2],
            st[2] / max(1, st[1]),
            st[3] / st[0],
            _spec.get("skipped", 0),
        )
        f = _spec.get("acc", {})
        if len(f) > 4 * max(1, ib.num_reqs) + 64:
            live = set(ib.req_ids[: ib.num_reqs])
            for k in [k for k in f if k not in live]:
                del f[k]
    return SamplerOutput(sampled_token_ids=out.to(logits.device), logprobs_tensors=None)


# ------------------------------------------------------------------------------------------- the runner
def _dspark_model(runner):
    m = runner.get_model()
    while getattr(m, "dspark", None) is None and hasattr(m, "model"):
        m = m.model
    return m


def _dspark_propose(runner, K: int, sampled_token_ids, reqs) -> list:
    """DSpark drafts for every request that sampled a token: one draft graph call over the batch padded to a
    power-of-two bucket (padding rows: slot -1, read nothing). Every TP rank runs it with the same host state, so the
    graph's collectives stay in step. x_m is the token just sampled at position m = num_tokens_no_spec - 1."""
    import time
    import numpy as np

    n = len(sampled_token_ids)
    out = [[] for _ in range(n)]
    model = _dspark_model(runner)
    ds = getattr(model, "dspark", None)
    from .control import ctl

    if ds is None or not ds.capture or ctl().get("spec", "on") == "off":
        return out
    st = _stats()
    st["propose_calls"] += 1
    table, costs, adapt = _policy()
    L = dspark_len(n, K, table)
    rows = [i for i in range(n) if reqs[i] and len(sampled_token_ids[i]) > 0]
    nts = runner.input_batch.num_tokens_no_spec
    # the next step's table width: the batch's highest position then (each row's last token + L drafts), as the
    # runner sizes it (runner_patch._bt_width); every running request's row pays it, not only the drafting ones
    from . import runner_patch

    top = max((int(nts[i]) - 1 for i in range(n) if reqs[i]), default=0) + L
    lk = runner_patch._bt_width(np.array([top])) * (runner_patch._state["block_size"] or 128)
    go = L > 0 and bool(rows) and dspark_go([reqs[i] for i in rows], L, adapt, costs, lk)
    st["gate"] = _spec.get("gate")  # (requests, drafts, table keys, cost, expected tokens) of the last call
    if not go:
        st["skipped"] += 1
        return out
    t0 = time.perf_counter()
    B = 1
    while B < len(rows):
        B *= 2
    tok = np.zeros(B, dtype=np.int64)
    m = np.zeros(B, dtype=np.int64)
    slot = np.full(B, -1, dtype=np.int64)
    uid = np.full(B, -1, dtype=np.int32)
    slots, uids = request_slots()
    for j, i in enumerate(rows):
        rid = reqs[i]
        m[j] = int(nts[i]) - 1
        tok[j] = int(sampled_token_ids[i][-1])
        slot[j] = slots.get(rid, -1)
        uid[j] = uids.get(rid, -1)
    if not _spec.get("tok_checked"):
        _spec["tok_checked"] = True
        i = rows[0]
        have = int(runner.input_batch.token_ids_cpu[i, m[0]])
        if have != tok[0]:
            logger.warning("ml4 dspark: token_ids_cpu[%d, %d] = %d, sampled %d: m is off", i, m[0], have, tok[0])
    dev = ds.mk_w1.device
    t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
    from . import asyncspec

    if asyncspec.ON:  # ML4_ASYNC_SPEC: the drafts stay on the device; the next step's token ids take them there
        dd = ds.graph(B)(t(tok), t(m), t(slot), t(uid))
        asyncspec.set_drafts(dd, {reqs[i]: (j, L) for j, i in enumerate(rows)})
        for j, i in enumerate(rows):
            out[i] = [-1] * L
    else:
        d = ds.graph(B)(t(tok), t(m), t(slot), t(uid)).cpu().tolist()
        for j, i in enumerate(rows):
            out[i] = d[j][:L]
    st["draft_calls"] += 1
    st["draft_ms"] += 1e3 * (time.perf_counter() - t0)
    return out


def request_slots():
    """({req_id: ring slot}, {req_id: uid}) for the running requests: runner_patch's request slots (one per running
    request, max_num_seqs of them) and uids (never reused). A slot is freed with its request; its ring rows are
    invalidated by the next owner's uid (tags), never cleared."""
    from . import runner_patch

    st = runner_patch._state
    return st["slots"], st["uids"]


# ------------------------------------------------------------------------------------------- verify layout
def verify_layout(context_lens, n_tok, rows: int):
    """vllm-gaudi's flattened verify rows (request b's tokens in rows b*T .. b*T + n_tok[b] - 1, T = the longest)
    -> (positions [rows, 1] int64, request index of each row, T). Rows past a request's own tokens are padding:
    request -1 (uid -1, so they write nothing and read nothing), position -1. DSV41 spec.verify_layout."""
    import numpy as np

    nd = len(n_tok)
    T = max(n_tok) if nd else 1
    assert rows % T == 0, (rows, T)
    pos = np.full((rows, 1), -1, dtype=np.int64)
    req = np.full((rows,), -1, dtype=np.int64)
    for b in range(nd):
        req[b * T : b * T + int(n_tok[b])] = b
        pos[b * T : b * T + int(n_tok[b]), 0] = int(context_lens[b]) + np.arange(int(n_tok[b]))
    return pos, req, T


def _install_verify_layout(runner_cls) -> None:
    """Verify steps as flat decode rows: every one of a request's T rows is a one-token decode row with its own
    position, the request's uid, block table and ring slot ("flat" in DSV 18: the decode graphs run B*T rows; each
    layer writes all rows' cache entries before any row reads, so a draft row reads the rows before it in the same
    step). Wraps runner_patch's decode-input hook: a step with more than one token per request is re-attached with
    one request per row."""
    from . import runner_patch as rp

    orig = runner_cls._create_decode_input_data

    def _create_decode_input_data(
        self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output=None
    ):
        n_tok = [int(n) for n in num_scheduled_tokens[:num_decodes]]
        _spec["n_dec"] = num_decodes  # r11: the decode rows, for _prepare_spec_decode_inputs (_install_spec_rows)
        try:
            out = orig(self, num_decodes, num_scheduled_tokens, context_lens, block_table_cpu_tensor, scheduler_output)
        finally:
            _spec["n_dec"] = None
        if max(n_tok, default=1) <= 1:
            out.attn_metadata.ml4_verify_rows = None
            return out
        B = out.token_ids.shape[0]
        real = list(self.input_batch.req_ids[:num_decodes]) if scheduler_output is not None else []
        bt = block_table_cpu_tensor[:num_decodes].tolist()
        blocks = [[b for b in row if b >= 0] for row in bt]
        pos, req, T = verify_layout(context_lens, n_tok, B)
        rids = [real[q] if 0 <= q < len(real) else None for q in req]
        blocks = [blocks[q] if 0 <= q < len(blocks) else [] for q in req]
        rp._attach(self, out.attn_metadata, rids, blocks, pos)
        out.attn_metadata.ml4_verify_rows = T  # the model runs the step as [B, T] rows (verify_rows_on)
        _spec["verify_T"] = T
        return out

    runner_cls._create_decode_input_data = _create_decode_input_data


# ------------------------------------------------------------------------------------------- sample_tokens fixes
def _first_rows(logits, decode_data, num_decodes: int):
    """Structured-output steps sample the decode rows together with the prompts, one row per request. In a verify
    step a request's rows are b*T .. b*T+T-1 (padded layout): its first row is the "every draft rejected" sample."""
    md = getattr(decode_data, "spec_decode_metadata", None)
    if md is None:
        return logits[:num_decodes]
    nd = torch.tensor(md.num_draft_tokens, dtype=md.bonus_logits_indices.dtype, device=md.bonus_logits_indices.device)
    return logits[(md.bonus_logits_indices - nd)[:num_decodes].long()]


class _NoSpecTokens:
    """scheduler_output as apply_grammar_bitmask should see it after _first_rows: one logits row per request."""

    def __init__(self, so, grammar_output):
        spec = so.scheduled_spec_decode_tokens or {}
        assert not any(spec.get(r) for r in grammar_output.structured_output_request_ids), (
            "ml4 spec: a structured-output request carries drafts"
        )
        self._so = so
        self.scheduled_spec_decode_tokens = {}

    def __getattr__(self, name):
        return getattr(self._so, name)


def _patch_sample_tokens(runner_cls, grammar: bool = False) -> None:
    """DSV41's four fixes of vllm-gaudi's HPUModelRunner.sample_tokens with a speculative config (DSV 18 phase 2),
    vendored: the method is recompiled from its own source with anchored edits and the engine refuses to start if an
    anchor moved. sampling_metadata unbound on steps that sample nothing; structured-output rows of a verify step;
    top-p / top-k metadata padded to B*T rows instead of the B bonus rows; grammar bitmask offsets. grammar (r13,
    ML4_SPEC_GRAMMAR): a verify step with a grammar runs its decode rows through the verify sampler with the masks
    applied (so_verify) and samples only its prompt rows as a structured-output step (so_prompts)."""
    import inspect
    import sys
    import textwrap

    raw = inspect.unwrap(runner_cls.sample_tokens)
    if getattr(raw, "_ml4_spec", False):
        return
    lines, first = inspect.getsourcelines(raw)
    src = "\n" * (first - 1) + textwrap.dedent("".join(lines))
    edits = [
        ("    spec_decode_num_tokens = None\n", "    spec_decode_num_tokens = None; sampling_metadata = None\n"),
        (
            "logits_decode.append(logits_device[:num_decodes])",
            "logits_decode.append(_ml4_first_rows(logits_device, decode_data, num_decodes))",
        ),
        (
            "pd_info.decode_req_ids, logits_device.shape[0])",
            "pd_info.decode_req_ids, logits_device.shape[0] if spec_decode_metadata is None "
            "else spec_decode_metadata.bonus_logits_indices.shape[0])",
        ),
        (
            "self.apply_grammar_bitmask(scheduler_output, grammar_output, logits)",
            "self.apply_grammar_bitmask(_ml4_NoSpecTokens(scheduler_output, grammar_output), grammar_output, logits)",
        ),
    ]
    if grammar:  # ML4_SPEC_GRAMMAR: verify steps with grammar masks (so_verify / so_prompts)
        edits[0] = (
            "    spec_decode_num_tokens = None\n",
            "    spec_decode_num_tokens = None; sampling_metadata = None; _ml4_dec_verified = False\n",
        )
        edits += [
            (
                "        if self.use_structured_output:\n            logits_decode.append(_ml4_first_rows(",
                "        _ml4_dec_verified = self.use_structured_output and _ml4_so_verify(\n"
                "            self, scheduler_output, grammar_output, decode_data, logits_device, num_decodes)\n"
                "        if self.use_structured_output and not _ml4_dec_verified:\n"
                "            logits_decode.append(_ml4_first_rows(",
            ),
            (
                "    if self.use_structured_output:\n        # Scheduler places cached before prompt\n",
                "    if self.use_structured_output and _ml4_dec_verified:\n"
                "        _ml4_so_prompts(self, batch_changed, scheduler_output, grammar_output, logits_prompt,\n"
                "                        prefill_sampled_requests, prefill_sampled_token_ids, logprobs_segments)\n"
                "    elif self.use_structured_output:\n        # Scheduler places cached before prompt\n",
            ),
            (
                "    if self.use_async_scheduling or self.use_structured_output:\n",
                "    if self.use_async_scheduling or (self.use_structured_output and not _ml4_dec_verified):\n",
            ),
        ]
    for old, new in edits:
        assert src.count(old) == 1, f"ml4 spec: sample_tokens anchor not found once: {old.strip()!r}"
        src = src.replace(old, new)
    mod = sys.modules[raw.__module__]
    mod.__dict__["_ml4_first_rows"] = _first_rows
    mod.__dict__["_ml4_NoSpecTokens"] = _NoSpecTokens
    mod.__dict__["_ml4_so_verify"] = so_verify
    mod.__dict__["_ml4_so_prompts"] = so_prompts
    ns: dict = {}
    exec(compile(src, inspect.getsourcefile(raw) or "<sample_tokens>", "exec"), mod.__dict__, ns)
    ns["sample_tokens"]._ml4_spec = True
    runner_cls.sample_tokens = ns["sample_tokens"]


# ------------------------------------------------------------------------------------------- r13: grammar requests draft
def spec_grammar_on() -> bool:
    """Grammar requests draft (ML4_SPEC_GRAMMAR; control key spec_grammar)."""
    from .control import get as ctl_get

    return bool(ctl_get("spec_grammar", int(SPEC_GRAMMAR)))


def mask_rows(logits, bitmask, rows, src) -> None:
    """logits[rows[j]] gets the grammar mask bitmask[src[j]] in place: tokens whose bit is 0 become -inf, as
    xgrammar's apply_token_bitmask_inplace_cpu (bit t % 32 of word t // 32; tokens past the mask's 32 W columns are left
    alone). The packed rows are unpacked on the host (np.unpackbits, little bit order) into one [R, V] boolean that
    goes to the device in one copy, so the logits stay where they are (no fp32 round trip through the host)."""
    import numpy as np

    if not rows:
        return
    R, V = logits.shape
    W = bitmask.shape[1]
    full = np.full((R, W), -1, dtype=np.int32)
    full[rows] = bitmask[src]
    allow = np.unpackbits(full.view(np.uint8), axis=1, bitorder="little")
    n = min(V, W * 32)
    if n < V:
        allow = np.concatenate([allow[:, :n], np.ones((R, V - n), dtype=np.uint8)], 1)
    elif allow.shape[1] > V:
        allow = allow[:, :V]
    m = torch.from_numpy(np.ascontiguousarray(allow)).to(logits.device).bool()
    logits.masked_fill_(~m, float("-inf"))


def _verify_offsets(md, R: int):
    """(T, first logits row of each request) of a verify step: b*T (the uniform layout), or ragged_layout's."""
    nd = list(md.num_draft_tokens)
    Bp = len(nd)
    F = int(md.draft_token_ids.numel())
    T = R // Bp if Bp else 1
    if T * Bp == R and F == Bp * (T - 1) and max(nd, default=0) <= T - 1:
        return T, [b * T for b in range(Bp)]
    lay = ragged_layout(nd, R, F)
    if lay is None:
        raise RuntimeError(
            f"ml4 spec: grammar verify step with {R} logits rows and {F} drafts does not match draft counts {nd}"
        )
    return lay[0], lay[1]


def so_verify(runner, so, grammar_output, decode_data, logits, num_decodes: int) -> bool:
    """r13, sample_tokens of a structured-output step: True when the decode rows are a verify step (drafts) and
    ML4_SPEC_GRAMMAR is on; then every structured-output decode request's 1 + n mask rows (the scheduler's: row i is
    the mask for the token after drafts 0..i-1, the last one the bonus row's) are applied to its logits rows
    b*T .. b*T+n in place, and the caller verifies the decode rows with the verify sampler (rejection_forward) on the
    masked logits, so its tokens follow the masked target distribution exactly. False: today's path (_first_rows)."""
    import time

    md = getattr(decode_data, "spec_decode_metadata", None) if decode_data is not None else None
    if md is None or grammar_output is None or not spec_grammar_on():
        return False
    t0 = time.perf_counter()
    nd = list(md.num_draft_tokens)
    _, offs = _verify_offsets(md, int(logits.shape[0]))
    spec = so.scheduled_spec_decode_tokens or {}
    r2i = runner.input_batch.req_id_to_index
    rows, src, cum = [], [], 0
    for rid in grammar_output.structured_output_request_ids:
        k = len(spec.get(rid, ()))
        b = r2i.get(rid)
        if b is not None and b < num_decodes:
            if b >= len(nd) or nd[b] != k:
                raise RuntimeError(
                    f"ml4 spec: grammar request {rid} at decode row {b} has {k} drafts, the verify "
                    f"step {nd[b] if b < len(nd) else None}"
                )
            rows.extend(offs[b] + i for i in range(k + 1))
            src.extend(cum + i for i in range(k + 1))
        cum += 1 + k
    bm = grammar_output.grammar_bitmask
    if cum > bm.shape[0]:
        raise RuntimeError(f"ml4 spec: grammar bitmask has {bm.shape[0]} rows, the step needs {cum}")
    mask_rows(logits, bm, rows, src)
    st = _stats()
    st["so_steps"] += 1
    st["so_rows"] += len(rows)
    st["so_mask_ms"] += 1e3 * (time.perf_counter() - t0)
    return True


def so_prompts(runner, batch_changed, so, grammar_output, logits_prompt, reqs, prefill_tokens, segments) -> None:
    """r13: the prompt rows of a step whose decode rows so_verify handed to the verify sampler, sampled as vllm-gaudi's
    structured-output path samples them (one row per request that produced logits, its first mask row applied, one
    _run_sampling call over them), appending to the caller's prefill tokens and logprobs segments."""
    reqs = list(reqs)
    if not logits_prompt or not reqs:
        # no prompt row ends its prompt this step (chunks only; vllm-gaudi's plain prefill path skips them the same
        # way): nothing to sample. The sampler over a [0, V] tensor faulted the cards (HMMU write page fault, r13).
        if logits_prompt and sum(int(x.shape[0]) for x in logits_prompt):
            raise RuntimeError(
                f"ml4 spec: prompt logits rows {[int(x.shape[0]) for x in logits_prompt]} for no request"
            )
        return
    logits = torch.cat(logits_prompt, dim=0)
    if len(reqs) != logits.shape[0] or len(set(reqs)) != len(reqs):
        raise RuntimeError(f"ml4 spec: {logits.shape[0]} prompt logits rows for requests {reqs}")
    spec = so.scheduled_spec_decode_tokens or {}
    row_of = {r: i for i, r in enumerate(reqs)}
    rows, src, cum = [], [], 0
    for rid in grammar_output.structured_output_request_ids:
        if rid in row_of:
            rows.append(row_of[rid])
            src.append(cum)
        cum += 1 + len(spec.get(rid, ()))
    mask_rows(logits, grammar_output.grammar_bitmask, rows, src)
    out, _ = runner._run_sampling(batch_changed, logits, reqs, logits.shape[0])
    for i in range(logits.shape[0]):
        prefill_tokens.append(out.sampled_token_ids[i].flatten())
    segments.append((reqs, out.logprobs_tensors))


# ------------------------------------------------------------------------------------------- r11: drafts on decode rows
def strip_placeholders(so):
    """r11: vLLM 0.30's scheduler pads a request it takes from the waiting queue with every token but the last one
    computed (a new request whose prompt is cached to P - 1: an exact re-send with P = 1 mod 128 still in the HBM
    cache, a whole-prompt CPU-tier hit; or a resumed request cached to P + O - 1) to 1 + K rows with K placeholder
    drafts (-1), for a uniform verify width on GPUs. Nothing drafted them, so on the sync path they only do harm:
      (a) vllm-gaudi runs a new request's 1 + K rows as a prompt and samples at the last placeholder: a wrong first
          token (bug 2);
      (b) at 5-7 decodes drafting K(8) = 2, that prompt row sits inside the padded decode bucket and its 3 drafts
          enter the decode verify metadata: 7 x 2 + 3 = 17 drafts for a [8, 2] view ('s engine death);
      (c) a structured-output request with placeholders trips _NoSpecTokens' assert (engine death);
      (d) a step that drafts only because of placeholders runs the verify sampler, which returns no logprobs ('s
          KeyError, HTTP 500).
    The step is handed to vllm-gaudi without them: the row runs as the one-token row it is (a new request: its last
    prompt token as a prompt chunk, sampled there; a resumed one: a one-token decode). The scheduler keeps its own
    SchedulerOutput and counts the K placeholders as rejected (num_computed P - 1 + (1 + K) - K = P), as for the
    async modes' cut (asyncspec). -> (the SchedulerOutput the runner should run: a copy when something was cut, never
    the caller's object mutated; {req_id: placeholders cut} or None)."""
    spec = so.scheduled_spec_decode_tokens if so is not None else None
    if not spec:
        return so, None
    waiting = cut = None
    for rid, toks in spec.items():
        if not toks or toks[0] != -1 or any(t != -1 for t in toks):
            continue
        if waiting is None:
            waiting = {r.req_id for r in so.scheduled_new_reqs}
            waiting.update(getattr(so.scheduled_cached_reqs, "resumed_req_ids", None) or ())
        if rid in waiting:
            cut = cut or {}
            cut[rid] = len(toks)
    if not cut:
        return so, None
    import copy

    so = copy.copy(so)
    so.scheduled_spec_decode_tokens = {r: t for r, t in spec.items() if r not in cut}
    so.num_scheduled_tokens = dict(so.num_scheduled_tokens)
    for rid, k in cut.items():
        so.num_scheduled_tokens[rid] -= k
    so.total_num_scheduled_tokens -= sum(cut.values())
    n = _spec["placeholders_cut"] = _spec.get("placeholders_cut", 0) + len(cut)
    if n <= 20 or n % 1000 == 0:
        logger.warning(
            "ml4 spec: %s from the waiting queue with every token but the last one computed: placeholder "
            "drafts dropped, one-token row (%d so far)",
            ", ".join(f"{r} ({k})" for r, k in cut.items()),
            n,
        )
    return so, cut


def compact_grammar(grammar_output, cut):
    """The scheduler's grammar bitmask without the K extra rows of each request strip_placeholders cut (it built
    1 + K rows for it: the first is the mask for the token sampled now, the placeholder rows are unconstrained), so the
    bitmask has one row per structured-output request, as _NoSpecTokens' apply_grammar_bitmask reads it."""
    if grammar_output is None or not cut or not any(r in cut for r in grammar_output.structured_output_request_ids):
        return grammar_output
    import dataclasses
    import numpy as np

    keep, i = [], 0
    for rid in grammar_output.structured_output_request_ids:
        keep.append(i)
        i += 1 + cut.get(rid, 0)
    bm = grammar_output.grammar_bitmask
    if i > bm.shape[0]:
        raise RuntimeError(f"ml4 spec: grammar bitmask has {bm.shape[0]} rows, the step needs {i}")
    return dataclasses.replace(grammar_output, grammar_bitmask=np.ascontiguousarray(bm[keep]))


class _DecodeSpec:
    """scheduler_output as _prepare_spec_decode_inputs should see it: drafts of decode rows only."""

    def __init__(self, so, spec):
        self._so = so
        self.scheduled_spec_decode_tokens = spec

    def __getattr__(self, name):
        return getattr(self._so, name)


def install_spec_rows(runner_cls) -> None:
    """r11, always on with ML4_SPEC=dspark: only decode rows carry drafts.
      * execute_model: strip_placeholders on the sync path (ML4_ASYNC_SPEC=0; the async modes cut placeholders in
        asyncspec's _update_states already), sample_tokens: compact_grammar for the requests it cut.
      * _prepare_spec_decode_inputs (defence): vllm-gaudi counts the drafts of every request whose batch index falls
        inside the padded decode bucket, prompts included; only decode rows (index < num_decodes) are counted here.
      * rejection_forward (defence): a verify step that is still ragged is laid out uniformly (_uniform_step).
    A step in which no request is cut and no prompt row carries drafts runs exactly as before."""
    if getattr(runner_cls, "_ml4_spec_rows", False):
        return
    runner_cls._ml4_spec_rows = True
    from . import asyncspec

    orig_exec = runner_cls.execute_model

    def execute_model(self, scheduler_output, *a, **k):
        _spec["cut"] = None
        if not asyncspec.ON and scheduler_output is not None:
            scheduler_output, _spec["cut"] = strip_placeholders(scheduler_output)
        return orig_exec(self, scheduler_output, *a, **k)

    runner_cls.execute_model = execute_model
    orig_sample = runner_cls.sample_tokens

    def sample_tokens(self, grammar_output):
        cut, _spec["cut"] = _spec.get("cut"), None
        if cut:
            grammar_output = compact_grammar(grammar_output, cut)
        return orig_sample(self, grammar_output)

    runner_cls.sample_tokens = sample_tokens
    orig_specin = runner_cls._prepare_spec_decode_inputs

    def _prepare_spec_decode_inputs(self, so, logits_indices, token_ids_device, max_num_sampled_tokens):
        n_dec = _spec.get("n_dec")
        spec = so.scheduled_spec_decode_tokens if so is not None else None
        if spec and n_dec is not None:
            r2i = self.input_batch.req_id_to_index
            drop = [r for r in spec if r2i.get(r, n_dec) >= n_dec]
            if drop:
                n = _spec["nondecode_drafts"] = _spec.get("nondecode_drafts", 0) + len(drop)
                if n <= 20:
                    logger.warning(
                        "ml4 spec: drafts of non-decode rows %s left out of the verify step (%d so far)", drop, n
                    )
                so = _DecodeSpec(so, {r: t for r, t in spec.items() if r not in drop})
        return orig_specin(self, so, logits_indices, token_ids_device, max_num_sampled_tokens)

    runner_cls._prepare_spec_decode_inputs = _prepare_spec_decode_inputs


def install_runner(runner_cls) -> None:
    """ML4_SPEC=dspark: n-gram vehicle with DSpark's drafts, guarded to plain requests, the verify sampler, the verify
    rows' layout and DSV's sample_tokens fixes. Called once from runner_patch.install (after its own hooks)."""
    if SPEC != "dspark" or _spec.get("runner_installed"):
        return
    from . import asyncspec

    grammar = SPEC_GRAMMAR and not asyncspec.ON  # the async modes keep refusing grammar requests (spec._plain)
    _patch_sample_tokens(runner_cls, grammar)  # first: it fails closed (anchors) before anything else is patched
    _spec["runner_installed"] = True
    _spec["grammar_installed"] = grammar
    from vllm.v1.sample.rejection_sampler import RejectionSampler

    orig_update = runner_cls._update_states

    def _update_states(self, scheduler_output):
        _spec["runner"] = self
        return orig_update(self, scheduler_output)

    runner_cls._update_states = _update_states

    def propose_ngram_draft_token_ids(self, sampled_token_ids):
        from . import asyncspec

        asyncspec.clear_drafts()  # ML4_ASYNC_SPEC: no drafts unless _dspark_propose launches them below
        K = self.speculative_config.num_speculative_tokens
        n = len(sampled_token_ids)
        reqs = list(self.input_batch.req_ids[:n])
        g = _spec.get("grammar_installed", False) and spec_grammar_on()
        sps = {rid: getattr(self.requests.get(rid), "sampling_params", None) for rid in reqs if rid}
        if not all(_plain(sp, g) for sp in sps.values()):
            return [[] for _ in sampled_token_ids]
        _spec["so_reqs"] = {r for r, sp in sps.items() if sp.structured_outputs is not None} if g else set()
        return _dspark_propose(self, K, sampled_token_ids, reqs)

    runner_cls.propose_ngram_draft_token_ids = propose_ngram_draft_token_ids
    RejectionSampler.forward = rejection_forward
    _install_verify_layout(runner_cls)
    install_spec_rows(runner_cls)  # placeholder drafts dropped, drafts on decode rows only
    from . import asyncspec

    asyncspec.install_runner(runner_cls)  # ML4_ASYNC_SPEC=1 only (outermost _update_states wrapper)
    logger.warning(
        "ml4: ML4_SPEC=dspark: DSpark drafts + verify steps (%s layout, draft length by batch %s, cost %s, "
        "long-context gate %s %s; plain requests%s)",
        "rows" if VERIFY_ROWS else "flat",
        DSPARK_K,
        DSPARK_COST,
        "on" if DSPARK_CTX else "off",
        DSPARK_CTX_G,
        " and grammar requests (ML4_SPEC_GRAMMAR)" if grammar else " only",
    )


# ------------------------------------------------------------------------------------------- the model side
TAP_ROWS = int(os.environ.get("ML4_DSPARK_TAP_ROWS", "8192"))  # >= the largest forward's B * S


def build(model) -> None:
    """ML4ForCausalLM.__init__ (ML4_SPEC=dspark): the drafter, parameters on the host until load(). It is kept out
    of the model's module tree (no vLLM weight or quantization pass touches it); model.dspark is a plain attribute."""
    object.__setattr__(model, "dspark", None)
    if SPEC != "dspark":
        return
    from .dspark import ML4DSpark
    from .ref.dspark import DSparkSpec

    s = DSparkSpec.from_params(model.model_dir)
    assert s.depth == DEPTH and tuple(s.target_layers) == TAP_LAYERS, (s.depth, s.target_layers)
    mx = model.vllm_config.model_config.max_model_len + 2 * DEPTH + 64
    object.__setattr__(model, "dspark", ML4DSpark(s, model.tp_rank, model.tp_size, max_pos=mx, device="cpu"))
    _spec["rank0"] = model.tp_rank == 0


def load(model) -> None:
    """After the target's load: the draft weights (mtp_dspark.*, shard 54, read whole on every rank and sliced by
    ML4DSpark.shard_plan), moved to the device, the target's lm_head / embedding slices bound, rings for
    max_num_seqs request slots, the tap buffer, the slot buffer and the tap hooks."""

    ds = getattr(model, "dspark", None)
    if ds is None:
        return
    try:
        _load(model, ds)
    except Exception:  # noqa: BLE001  (the target still serves, without drafts; logged on every rank)
        logger.exception("ml4 dspark: rank %d drafter load failed: serving without DSpark", model.tp_rank)
        object.__setattr__(model, "dspark", None)


def _load(model, ds) -> None:
    import time
    from .ref.dspark import PREFIX, load_weights

    t0 = time.time()
    W = load_weights(model.model_dir, ds.spec)
    ds.load(lambda n: W[n.removeprefix(PREFIX)])
    del W
    dev = model.stack.norm.device
    ds.to(dev)
    ds.bind_target(model.stack.lm_head, model.stack.embed)
    sc = model.vllm_config.scheduler_config
    ds.make_buffers(slots=sc.max_num_seqs, max_rows=TAP_ROWS, device=dev)
    ds.slot_buf = torch.full((max(1024, sc.max_num_seqs * (DEPTH + 1)),), -1, dtype=torch.long, device=dev)
    _spec["hooks"] = install_taps(model.stack, ds)
    logger.warning(
        "ml4 dspark: rank %d drafter loaded in %.0f s: weights %.3f GiB, rings + taps %.3f GiB (%d slots, %d tap rows)",
        model.tp_rank,
        time.time() - t0,
        ds.weight_bytes() / 2**30,
        ds.buffer_bytes() / 2**30,
        sc.max_num_seqs,
        TAP_ROWS,
    )


def before_forward(model, ctx) -> None:
    """Every forward, before the stack or the decode graphs run (vllm_model.forward): this forward's ring slot per row
    into the drafter's slot buffer (the tap hook inside the graphs reads it). The taps and ring writes always run once
    the drafter is loaded: the control key "spec": "off" only stops the drafts (_dspark_propose). Switching the hooks
    themselves would need a re-capture of the decode graphs with one all-reduce fewer per step, and a server that did
    that (on -> off) hung in its next collective."""
    ds = getattr(model, "dspark", None)
    if ds is None:
        return
    B = ctx.positions.shape[0]
    if B <= ds.slot_buf.shape[0]:
        s = getattr(ctx, "req_slot", None)
        if s is None:
            ds.slot_buf[:B].fill_(-1)
        else:
            ds.slot_buf[:B].copy_(s.to(ds.slot_buf.dtype))
