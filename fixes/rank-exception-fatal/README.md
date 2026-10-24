# rank-exception-fatal

Stop the engine when a tensor-parallel worker fails a call whose result is read from another
rank, instead of serving with the ranks out of step.

| | |
| --- | --- |
| Affects | every multi-card deployment that uses vLLM's multiprocessing executor (TP > 1, the default executor on one node) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/v1/executor/multiproc_executor.py` |
| Upstream | not filed yet (the same code is on vLLM main) |
| Test | `test_rank_exception_fatal.py` (stdlib only; needs a vLLM source tree, see below) |

## Problem

Each worker process of `MultiprocExecutor` runs RPCs from a broadcast queue. When a call raises,
the worker logs `WorkerProc hit an exception.` and sends the exception back only if it is the
call's output rank. For `execute_model` and `sample_tokens` that is rank 0, so on every other rank
the exception is logged and dropped. The engine keeps going with rank 0's result while the failed
rank skipped the rest of that call.

## Symptom

After one rank fails (for example a device memory allocation failure during a long prefill), the
ranks are out of step and every later response is wrong, often one repeated token from NaN
logits, even for short prompts. `/health` keeps returning 200 and nothing restarts. The only trace
is `WorkerProc hit an exception.` in the worker log.

## Who is affected

Anyone running vLLM with tensor parallelism through the multiprocessing executor. Nothing in the
failing path is specific to Gaudi. Single-card servers and the Ray executor are not affected.

## The fix

When a call fails on a rank that is not the call's output rank, the worker logs an error and
exits with status 1. The executor's worker monitor sees the dead process and shuts the engine
down, so the server fails visibly and an orchestrator restarts it. Failures on the output rank
and on calls that every rank answers are still returned to the caller as before.

## Verify

```bash
python3 fixes/rank-exception-fatal/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<dir holding vllm/> pytest fixes/rank-exception-fatal
```

The test extracts the worker loop from the patched file and drives it with a fake queue: a
failing non-output rank must exit, the output rank must reply and continue. Without
`GAUDI_FIXES_ROOT` it uses the installed vLLM, and it is skipped when neither is available.
