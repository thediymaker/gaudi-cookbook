# topk-subset-sampler

Top-k / top-p sampling on HPU without sorting the whole vocabulary on every decode step.
Performance fix, opt-in.

| | |
| --- | --- |
| Affects | every model, requests with `top_k` (with or without `top_p`), including defaults from `generation_config.json` |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/v1/sample/ops/topk_topp_sampler.py` (vLLM core), `vllm_gaudi/v1/worker/hpu_input_batch.py` |
| Upstream | not filed |
| Test | `test_topk_subset_sampler.py` (CPU, needs torch and numpy) |

## Problem

On HPU, `TopKTopPSampler` uses `forward_native`, and without Triton it falls to
`apply_top_k_top_p_pytorch`. That function sorts the full `[batch, vocab]` logits on every decode
step whenever any row has `top_k` or `top_p`; the top-k-only shortcut is enabled only on CPU. A
model whose generation config sets `top_k` / `top_p` puts every request that does not override
them on this path.

## Symptom

Aggregate decode throughput stops growing with `--max-num-seqs` and per-stream speed drops as the
batch grows. Requests that send `top_k: -1, top_p: 1` are much faster than the defaults. The
effect is largest with large vocabularies and large batches.

## Who is affected

Models with large vocabularies served at high batch sizes with `top_k` sampling. Greedy requests
and requests without `top_k` are not affected.

## The fix

When every row in the batch has a `top_k` of at most 1024, the sampler takes `torch.topk(K)` once,
with `K` the smallest of 64, 128, 256, 512 and 1024 covering the batch's largest `top_k` (so only a
few graph shapes compile). It applies each row's `top_k` and `top_p` inside those `K` values and
draws the token with the stock exponential race. Tokens outside a row's top-k have probability 0
in the stock code, so the distribution is the same; only ties at the k-th logit are treated
differently, which does not occur in practice. The batch's largest `top_k` comes from the input
batch's CPU copy, so no device sync is added.

A batch with a row without `top_k`, with per-request generators (`seed`), with fp64 Gumbel noise,
or with a processed-logits logprobs mode keeps the stock path unchanged. The tokens drawn for a
given random state differ from the stock sampler, so the fix is not in the default profiles.

Apply after [seeded-sampling](../seeded-sampling/README.md) when both are used; they edit the same
file in different places.

## Verify

```bash
python3 fixes/topk-subset-sampler/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/topk-subset-sampler                     # GAUDI_FIXES_ROOT=<dir with vllm/ and vllm_gaudi/> if not installed
```

End to end: measure decode throughput at your production batch size with the model's default
sampling parameters, with and without the fix.
