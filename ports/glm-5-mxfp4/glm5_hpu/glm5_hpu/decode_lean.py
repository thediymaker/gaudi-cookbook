# SPDX-License-Identifier: Apache-2.0
"""Decode-step levers for GLM-5.x with MTP (``patches/decode_lean.py`` wires them in). Each has its own switch and
is off by default; with a switch off the stock code path runs. Greedy outputs are bit-identical with the switches
on and off: they remove host work from the step, not device math.

``GLM_ROUTER_FAST=1`` (vLLM ``fused_moe/router/cpu_router.py``, ``_grouped_topk``)
    The MXFP4 MoE method routes through vLLM's torch grouped top-k. When every group is selected (GLM-5.x:
    ``n_group`` 1, ``topk_group`` 1) the group stage (group scores, group top-k, mask, masked fill) leaves the scores
    unchanged but costs about 25 us per MoE layer on the critical path. The flag skips it: the same biased scores
    go to the same top-k, the weights are gathered from the unbiased scores and renormalised as before.

``GLM_SPEC_DEVICE_REJECT=1`` (vllm-gaudi ``v1/sample/hpu_rejection_sampler.py``)
    The HPU rejection sampler copies the draft ids, target argmax, draft probabilities, recovered tokens and bonus
    tokens to the host and walks the requests in Python. That copy is a device sync, so the host cannot enqueue the
    next work until the target forward has finished. The flag keeps the same computation on the device: the same
    tensors, the same host uniform draw (rounded up to the next float32, so the float32 comparison equals the
    float64 one), and the accept/reject walk as prefix logic over the [requests, k] grid.

``GLM_EARLY_DRAFT=1`` (vllm-gaudi ``v1/worker/hpu_model_runner.py``; needs ``GLM_SPEC_DEVICE_REJECT=1`` to help)
    In a decode-only step, the MTP draft is enqueued right after the rejection sampler, with its row indices
    computed on the device from the sampler output, instead of after the host has copied the sampled tokens back
    and rebuilt the indices (``HpuEagleProposer.prepare_inputs``). The device then runs target forward, sampler and
    draft back to back. Only for one draft token, MTP or EAGLE without auxiliary hidden states, outside warm-up;
    every other step takes the stock path.
"""

import os

import torch

ROUTER_FAST = os.environ.get("GLM_ROUTER_FAST", "0") == "1"
EARLY_DRAFT = os.environ.get("GLM_EARLY_DRAFT", "0") == "1"
DEVICE_REJECT = os.environ.get("GLM_SPEC_DEVICE_REJECT", "0") == "1"

# Host sources of non-blocking H2D copies stay referenced until the device is past them.
_keep = []


def _h2d(t, dev):
    _keep.append(t)
    del _keep[:-64]
    return t.to(dev, non_blocking=True)


# ---------------------------------------------------------------------------------------------------------------------
# GLM_ROUTER_FAST


def grouped_topk_all_groups(scores, topk, renormalize, routed_scaling_factor, e_score_correction_bias):
    """vLLM's grouped top-k when every group is selected: the same code with the identity group mask removed."""
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int32)


# ---------------------------------------------------------------------------------------------------------------------
# GLM_EARLY_DRAFT


def early_draft(runner, out, md, hidden_states, num_decodes, num_prefills, decode_data, warmup_mode):
    """Device-side twin of HpuEagleProposer.prepare_inputs + propose for one draft token.

    out: rejection sampler output [B, 2] (-1 for rejected slots); md.num_draft_tokens per padded row. prepare_inputs
    gives, per row i (base 2i, n drafts, L accepted tokens): last_token_indices[i] = base + (L - 1 if n > 0 else 0),
    hidden_states_indices[2i + j] = base + j if j < (L if n > 0 else 1) else -1.
    """
    runner._glm_early_draft = None
    sc = runner.speculative_config
    if (
        warmup_mode
        or num_prefills > 0
        or decode_data is None
        or md is None
        or num_decodes <= 0
        or sc is None
        or sc.num_speculative_tokens != 1
        or not sc.use_eagle()
        or runner.use_aux_hidden_state_outputs
        or decode_data.position_ids is None
    ):
        return
    nd = list(md.num_draft_tokens)
    b = len(nd)
    if out.dim() != 2 or out.shape[0] < b or out.shape[1] != 2 or num_decodes > b:
        return
    dev = out.device
    o = out[:b]
    length = ((o != -1) & (o < runner.input_batch.vocab_size)).sum(dim=1).to(torch.int64)
    has = _h2d(torch.tensor([1 if n > 0 else 0 for n in nd], dtype=torch.int64), dev)
    eff = torch.where(has > 0, length, torch.ones_like(length))
    base = torch.arange(b, dtype=torch.int64, device=dev) * 2
    last = base + eff - 1
    j = torch.arange(2, dtype=torch.int64, device=dev)
    picked = torch.where(
        j.unsqueeze(0) < eff.unsqueeze(1),
        base.unsqueeze(1) + j.unsqueeze(0),
        torch.full((b, 2), -1, dtype=torch.int64, device=dev),
    ).reshape(-1)
    target_hidden_states = hidden_states[picked]
    if target_hidden_states.dim() == 2:
        target_hidden_states = target_hidden_states.unsqueeze(1)
    block_table = runner.input_batch.block_table[0].get_cpu_tensor()[:num_decodes]
    draft = runner.drafter.propose(
        out.reshape(-1, 1)[picked],
        decode_data.position_ids[picked],
        target_hidden_states,
        last,
        decode_data.attn_metadata,
        block_table,
        runner,
    )
    runner._glm_early_draft = draft[:num_decodes]


def take_early_draft(runner, num_decodes):
    """The draft enqueued by early_draft for this step, or None."""
    draft = getattr(runner, "_glm_early_draft", None)
    runner._glm_early_draft = None
    return None if draft is None else draft[:num_decodes]


# ---------------------------------------------------------------------------------------------------------------------
# GLM_SPEC_DEVICE_REJECT


def _accept_grid(n_list, k, accept, tok, bonus, dev, placeholder):
    """Accept/reject walk over [n_req, k] as prefix logic. accept / tok: [n_req, k], row j valid for j < n_i. A slot is
    written while every earlier valid slot was accepted; the bonus goes to column n_i when all n_i were."""
    n_req = len(n_list)
    nd = _h2d(torch.tensor(n_list, dtype=torch.int32), dev)
    cols = torch.arange(k, dtype=torch.int32, device=dev).unsqueeze(0)
    valid = cols < nd.unsqueeze(1)
    rej = (~accept) & valid
    rej_before = (torch.cumsum(rej.to(torch.int32), dim=1) - rej.to(torch.int32)) > 0
    write = valid & ~rej_before
    head = torch.where(write, tok.to(torch.int32), torch.full_like(tok, placeholder, dtype=torch.int32))
    all_ok = rej.to(torch.int32).sum(dim=1) == 0
    out = torch.cat([head, torch.full((n_req, 1), placeholder, dtype=torch.int32, device=dev)], dim=1)
    cols1 = torch.arange(k + 1, dtype=torch.int32, device=dev).unsqueeze(0)
    at_bonus = (cols1 == nd.unsqueeze(1)) & all_ok.unsqueeze(1)
    return torch.where(at_bonus, bonus.reshape(-1, 1).to(torch.int32).expand(-1, k + 1), out)


def make_device_reject(host_reject, placeholder, greedy_eps):
    """rejection_sample replacement with the walk on the device; falls back to host_reject where it does not apply."""

    def rejection_sample(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_probs,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode=False,
        synthetic_conditional_rates=None,
        use_fp64_gumbel=False,
    ):
        n_list = [int(n) for n in num_draft_tokens]
        n_rows = int(target_probs.shape[0])
        n_req = len(n_list)
        k = n_rows // n_req if n_req else 0
        # The host greedy walk also counts mismatches in padded draft slots (j >= n_i), which only exist with k > 1
        # and partial drafts: keep those batches on the host path.
        partial = sampling_metadata.all_greedy and k > 1 and any(n != k for n in n_list)
        if synthetic_mode or draft_probs is not None or n_req == 0 or n_rows % n_req or max(n_list) > k or partial:
            return host_reject(
                draft_token_ids,
                num_draft_tokens,
                max_spec_len,
                cu_num_draft_tokens,
                draft_probs,
                target_probs,
                bonus_token_ids,
                sampling_metadata,
                synthetic_mode,
                synthetic_conditional_rates,
                use_fp64_gumbel,
            )
        dev = target_probs.device
        bonus = bonus_token_ids.reshape(-1)
        if sampling_metadata.all_greedy:
            arg = target_probs.argmax(dim=-1).to(torch.int32).reshape(n_req, k)
            draft = draft_token_ids.reshape(-1)[:n_rows].to(torch.int32).reshape(n_req, k)
            return _accept_grid(n_list, k, arg == draft, arg, bonus, dev, placeholder)
        logits = target_probs.float()
        draft = draft_token_ids.reshape(-1).to(torch.long)[:n_rows]
        probs = logits.softmax(dim=-1)
        p_draft = probs.gather(1, draft.unsqueeze(1)).squeeze(1)
        argmax = logits.argmax(dim=-1)
        # Gumbel-max recovered token, the same draw as the host path
        e = torch.rand((n_req, probs.shape[1]), device=probs.device, dtype=torch.float32)
        e = e.clamp_min_(1e-20).log_().neg_()
        masked = probs.scatter(1, draft.unsqueeze(1), 0.0)
        recovered = (masked / e.repeat_interleave(k, dim=0)[:n_rows]).argmax(dim=-1)
        temp = sampling_metadata.temperature
        if temp is not None and not sampling_metadata.all_random:
            t = temp[:n_req]
            if t.shape[0] < n_req:
                t = torch.cat([t, torch.full((n_req - t.shape[0],), 1.0, dtype=t.dtype, device=t.device)])
            greedy = t < greedy_eps
        else:
            greedy = torch.zeros(n_req, dtype=torch.bool, device=dev)
        u = torch.rand(n_rows, dtype=torch.float64)  # the host path's draw
        u32 = u.to(torch.float32)
        u32 = torch.where(u32.to(torch.float64) < u, torch.nextafter(u32, torch.full_like(u32, 2.0)), u32)
        u_dev = _h2d(u32, dev)  # p32 >= u32 (rounded up) <=> p32 >= u (float64)
        g = greedy.reshape(-1, 1)
        arg_k, draft_k = argmax.reshape(n_req, k), draft.reshape(n_req, k)
        rnd_ok = ((p_draft >= u_dev) & (p_draft > 0.0)).reshape(n_req, k)
        accept = torch.where(g, arg_k == draft_k, rnd_ok)
        tok = torch.where(g, arg_k, torch.where(rnd_ok, draft_k, recovered.reshape(n_req, k)))
        return _accept_grid(n_list, k, accept, tok, bonus, dev, placeholder)

    return rejection_sample
