# seeded-sampling

Source: [`fixes/seeded-sampling/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/seeded-sampling)

Seeded random sampling on HPU without `torch.Generator` ops on the device, so a request with a
`seed` cannot crash a worker, seeded output does not depend on the batch, and in compile mode a
seeded request does not recompile the sampler.

| | |
| --- | --- |
| Affects | every model, requests with `seed` and temperature > 0 |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/v1/sample/ops/topk_topp_sampler.py` (vLLM core), `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | not filed |
| Test | `test_seeded_sampling.py` (CPU, needs torch) |

## Problem

A request with `seed` gets its own `torch.Generator` on the HPU, and vLLM's `random_sample` draws
that request's noise one row at a time with `q[i].exponential_(generator=g)`. In lazy mode with
lazy collectives (the vllm-gaudi default) this races in the Habana bridge under load, and a
tensor-parallel worker segfaults in `habana::exponential_`. Requests without a seed draw noise for
the whole batch at once and are not affected.

Separately, the runner pads a batch by repeating its first request. A seeded first request's
generator then advances once per padded row, so its tokens depend on the batch bucket.

In compile mode (`PT_HPU_LAZY_MODE=0`) the runner compiles the sampler with `torch.compile`, and
dynamo guards on each request's generator object. Every new seeded request is a new object, so the
compiled sampler recompiles for it, stalling the whole batch, until dynamo reaches its recompile
limit.

## Symptom

Under sustained load with seeded requests, a worker dies with a segfault trace through
`GeneratorToSeed` and `exponential_`, and every in-flight request fails with `EngineDeadError`.
Seeded requests that do run can return different text for the same seed and prompt depending on
what else is in the batch.

In compile mode, each new seeded request adds a recompile to the step that first samples it, and
the server log eventually shows dynamo's `recompile_limit` warning for the sampler. Measured on
MiMo-V2.6-Flash in compile mode on a `releases/v0.30.0` base with the deployment's own model patches
(not the cookbook's v0.26.0 MiMo port), warm, one request at a time: time to first token 0.42 s for a
seeded request against 0.19 s unseeded, and 0.102 s against 0.101 s with the fix.

## Who is affected

Any deployment where clients send `seed` with sampling (some SDKs and evaluation harnesses do
this by default). Greedy requests and unseeded sampling are not affected. The recompile part matters in compile
mode only; it was measured on v0.30.0. Two compile-mode models on v0.26.0 showed no seeded time to
first token penalty, and the change is the same no-op there when no recompile happens.

## The fix

- Seeded requests get an `HpuSeededGenerator`: a CPU `torch.Generator` subclass seeded with the
  request seed that also carries the seed and a per-request step counter. No generator is created
  on the HPU.
- In `random_sample`, on an HPU tensor, rows with such a generator get counter-based Exp(1) noise:
  `-log(u)` with `u` from a 32-bit integer hash of (seed, step, token id), computed on the device
  in one batched op and written with one `index_copy_`. The noise is a pure function of
  (seed, step, token id), so the same seed and prompt give the same tokens whatever else is in the
  batch. Padded copies of a row reuse its key, and the generator advances once per call.
- Unseeded rows keep `q.exponential_()`, other generator types keep the stock per-row loop, and a
  batch with no seeded rows runs the stock code unchanged. The seeded subset is padded to a power
  of two to bound the number of graph shapes.
- With async scheduling, a discarded partial-prefill sample gives its step back (as the GPU runner
  rewinds its generator), so tokens do not depend on how a prompt was chunked.
- When the batch has per-request generators and the sampler is a `torch.compile` wrapper, the runner
  calls the uncompiled module (`_orig_mod`) for that step. Past the recompile limit dynamo ran those
  batches eagerly anyway, at the same decode speed, so eager costs nothing measurable. In lazy mode
  the sampler is not compiled and the call is unchanged.

Seeded output stays deterministic but differs from the stock noise stream. The fix refuses to
apply if vllm-gaudi stops replacing the stock rejection sampler, whose generator loops would meet
a CPU generator.

Speculative decoding is outside this fix. On both releases the HPU rejection sampler supports greedy
requests only, so per-request seeds never reach it. Ports that add random rejection sampling must
take seeded rows' accept draws and recovery noise from the same (seed, step) keys, or seeds are not
honoured on those models.

## Verify

```bash
python3 fixes/seeded-sampling/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/seeded-sampling                     # GAUDI_FIXES_ROOT=<dir with vllm/ and vllm_gaudi/> if not installed
```

End to end: run sustained concurrent load where every request sets a `seed` and temperature > 0,
with tensor parallelism. Without the fix a worker eventually segfaults. With it, repeating a
request with the same seed returns the same text at any concurrency. In compile mode, compare the
time to first token of seeded and unseeded requests on a warm server; they should match (needs a
device).
