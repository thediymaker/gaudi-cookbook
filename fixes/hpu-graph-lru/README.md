# hpu-graph-lru

A bounded HPU graph cache for lazy mode that evicts the least-recently-used graph instead of growing
without limit or freezing on the first shapes it saw. Opt-in, inert until `VLLM_HPU_MAX_GRAPHS` is set.

| | |
| --- | --- |
| Affects | lazy-mode servers that see many distinct batch shapes, typically with `VLLM_SKIP_WARMUP=true` and long uptimes |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | not filed yet |
| Test | `test_hpu_graph_lru.py` (stdlib; the source tests need a release's sources) |

## Problem

In lazy mode the runner wraps the model with Habana's `wrap_in_hpu_graph(..., disable_tensor_cache=True)`
and passes no cap. Every new shape (batch size, query length and block count bucket) is captured and its
graph is kept for the life of the process. With warm-up, the shapes are the bucket set and the count is
known. With `VLLM_SKIP_WARMUP=true`, or with traffic outside the warmed buckets, the cache keeps growing and
device memory with it.

Habana's own cap (`max_graphs`) does not help. It is first come, first kept: once the cache holds N
graphs, every shape that is not among them runs uncaptured for the life of the process:

```python
cached = cache.get(h)
if cached is None:
    if max_graphs is not None and len(cache) == max_graphs:
        return orig_fwd(*args, **kwargs)
```

An uncaptured lazy-mode step is recorded op by op from the host while the cards sit idle. When live
traffic moves to a decode shape outside the kept set, every decode step of that shape is several times
slower, until the process restarts.

## Symptom

Without a cap: device memory grows with uptime as new shapes arrive, until an allocation fails with
`PT_DEVMEM Allocation failed`.

With a first-come cap: after hours or days of uptime, decode on some replicas drops to a few tokens per
second at low load while other replicas of the same deployment stay fast. The engine process is busy, the
cards are mostly idle, and a Python stack sample of a worker shows the step in `graphs.py` at the
`len(cache) == max_graphs` branch (the uncaptured `orig_fwd` call) instead of the graph replay.

## Who is affected

Lazy-mode deployments (the default, `PT_HPU_LAZY_MODE=1`) that see more distinct shapes than they can
keep graphs for: `VLLM_SKIP_WARMUP=true`, wide bucket ranges, large `--max-cudagraph-capture-size`, or
multimodal inputs with varied sizes. Fully warmed servers with a bounded bucket set and enough device
memory for all of its graphs are not affected. Compile mode (`PT_HPU_LAZY_MODE=0`) does not use this
wrapper and is unchanged.

## The fix

With `VLLM_HPU_MAX_GRAPHS=N`, `_maybe_wrap_in_hpu_graph` hands Habana's `wrapped_hpugraph_forward` a cache
of at most N graphs that makes room before the cap check:

- a hit moves the shape to the most-recently-used end;
- a miss with room left captures as before;
- a miss with the cache full is counted; on the shape's `VLLM_HPU_GRAPH_ADMIT_AFTER`-th miss (default 2)
  the least-recently-used graph is evicted and the shape is captured. Prompt graphs are evicted before
  decode graphs, and a prompt shape never evicts a decode graph, so one-off prompt shapes stay uncaptured
  instead of churning the cache;
- an evicted graph is released (`HPUGraph.reset()`) after a device synchronize, so device memory stays
  bounded by N graphs.

Every tensor-parallel rank sees the same shape sequence and the policy uses nothing else (no clocks), so
all ranks capture and evict on the same steps. Rank 0 prints a stats line at most every
`VLLM_HPU_GRAPH_STATS_INTERVAL` seconds (default 300, `0` turns it off) when the counters changed:

```text
[hpu-graph-lru] graphs <held>/<N> (prompt <n>, decode <n>); hit prompt <n> decode <n>; captured prompt <n> decode <n>; evicted prompt <n> decode <n>; uncaptured prompt <n> decode <n>
```

A steady rise of `uncaptured decode` means the cap is too small for the live decode shapes.

| Variable | Meaning | Default |
| --- | --- | --- |
| `VLLM_HPU_MAX_GRAPHS` | graphs kept per wrapped model (the drafter of a speculative model gets its own cache of the same size) | unset: the stock unbounded wrapper |
| `VLLM_HPU_GRAPH_ADMIT_AFTER` | misses while full before a shape may evict a graph | `2` |
| `VLLM_HPU_GRAPH_STATS_INTERVAL` | seconds between stats lines on rank 0 | `300` |

The wrapper calls Habana's `graphs.wrapped_hpugraph_forward` with the arguments of Gaudi software 1.24.1.
It checks that signature when the model is wrapped and stops start-up with a clear error if it differs.

## Measured

GLM-5.3-Flash (the `glm-5-3-flash` port) on Gaudi 2, vllm-gaudi v0.26.0 with the port's patches, ten
replicas with `VLLM_SKIP_WARMUP=true`, `--max-cudagraph-capture-size 2048` and Habana's first-come cap
of 64 graphs, after 27 days of uptime at low load. Single runs, stack samples of the rank-0 worker:

| | Decode step |
| --- | --- |
| first-come cap, 6 replicas whose kept set missed the live decode shape | 675 to 923 ms |
| first-come cap, 4 replicas whose kept set held it | 79 to 83 ms |
| this fix, same cap of 64, all 10 replicas after restart | 45 to 55 ms inter-token latency, 0 uncaptured decode steps |

On one replica with the cap forced down to 8 under mixed traffic: 41 evictions, 99.6% of decode steps
replayed from a graph, device memory flat, and greedy output identical to the uncapped server.

## Verify

```bash
python3 fixes/hpu-graph-lru/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/hpu-graph-lru
```

On hardware (the cache behaviour needs a device): start with `VLLM_HPU_MAX_GRAPHS` set well below the
number of buckets and `VLLM_HPU_GRAPH_STATS_INTERVAL=30`, drive mixed prompt lengths and concurrency, and
check that the start-up line `[hpu-graph-lru] max_graphs N` appears, that the stats line shows evictions
while `uncaptured decode` stays near zero, that device memory stays flat, and that greedy answers match a
server without the cap.
