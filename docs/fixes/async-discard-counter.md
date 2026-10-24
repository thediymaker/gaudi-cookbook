# async-discard-counter

Source: [`fixes/async-discard-counter/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/async-discard-counter)

Make the HPU async scheduler use vLLM's `async_tokens_to_discard` counter, so a preempted request
no longer stops the engine with an `AttributeError`.

| | |
| --- | --- |
| Affects | every model served with async scheduling (the default) when a request is preempted |
| Releases | v0.26.0 (v0.30.0 already contains the upstream fix; the script reports "not needed") |
| Patches | `vllm_gaudi/v1/core/sched/hpu_async_scheduler.py` |
| Upstream | [vllm-gaudi#1794](https://github.com/vllm-project/vllm-gaudi/pull/1794), [vllm-gaudi#1803](https://github.com/vllm-project/vllm-gaudi/pull/1803) (merged, in v0.30.0) |
| Test | none (one-attribute change; covered by the end-to-end check below) |

## Problem

The vLLM commit that v0.26.0 tracks replaced the `Request.discard_latest_async_tokens` boolean
with `async_tokens_to_discard`, a counter of stale in-flight output frames to drop after a forced
preemption. `HPUAsyncScheduler._update_request_with_output` still reads the old attribute in the
branch that drops tokens produced while a request has no output placeholders: a partial prefill
chunk, or a request resumed after preemption.

## Symptom

The engine stops with

```text
AttributeError: 'Request' object has no attribute 'discard_latest_async_tokens'
```

raised from `hpu_async_scheduler.py`, the first time a request reaches that branch. It shows up
under KV-cache pressure, when long prompts force a preemption.

## Who is affected

Anyone on v0.26.0 with async scheduling (the default) whose load can preempt a request. A large
KV cache with a small `--max-num-seqs` rarely preempts, so the bug can stay hidden until load
grows.

## The fix

As in vLLM's own `AsyncScheduler`, the dropped frame consumes one pending discard: if
`async_tokens_to_discard` is positive it is decremented, and the frame is dropped as before. On a
release whose scheduler already reads the old attribute with `getattr` (the upstream fix),
`apply.py` reports "not needed" and writes nothing.

## Verify

```bash
python3 fixes/async-discard-counter/apply.py --check   # 0 = applied or not needed, 1 = not applied, 2 = source drifted
```

End to end: start the server with a small KV cache (for example a low
`--num-gpu-blocks-override`) and send several long prompts at once so that a request is
preempted. Without the fix the engine stops with the error above; with it the requests finish.
