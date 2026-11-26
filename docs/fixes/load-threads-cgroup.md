# load-threads-cgroup

Source: [`fixes/load-threads-cgroup/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/load-threads-cgroup)

Size the CPU thread pool used while the model loads by the container's CPU quota instead of the host's core
count, so FP8 checkpoints do not load for an hour under CPU throttling. Opt-in.

| | |
| --- | --- |
| Affects | containers with a CPU limit (Kubernetes `resources.limits.cpu`, `docker run --cpus`) on hosts with many more cores, loading FP8 checkpoints with many weight tensors (performance) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py` |
| Upstream | not filed yet |
| Test | `test_load_threads_cgroup.py` (stdlib; the source tests need a release's sources) |

## Problem

While the model loads, the HPU runner sets `OMP_NUM_THREADS` and `torch.set_num_threads` from
`len(psutil.Process().cpu_affinity())`, divided by the world size, and restores them afterwards. A container
without CPU pinning sees every host core there (160 on a Gaudi 2 server), while its CFS quota may be 16 or
32 CPUs. The loader's CPU work, for example the Gaudi 2 halving of every FP8 weight and doubling of its scale,
then runs with five to ten times more threads than the quota allows. The container is throttled in nearly
every scheduling period and each small tensor operation waits for its next time slice.

## Symptom

An FP8 checkpoint with many tensors (per-expert MoE weights) takes tens of minutes to over an hour to load,
much longer than reading it from disk, and the time is the same on a warm page cache. The load log shows

```text
Setting OMP_NUM_THREADS to 160 and torch.set_num_threads to 160 for 160 available CPU cores and world size 1
```

with a host-sized count, `/sys/fs/cgroup/cpu.stat` shows `nr_throttled` rising in almost every period, and a
Python stack sample of the worker sits on the FP8 rescale in `vllm_gaudi/extension/ops.py`.

## Who is affected

Containers whose CPU limit is well below the host's core count and that are not pinned to cores, loading
checkpoints whose load does per-tensor CPU work (FP8 per-expert MoE weights on Gaudi 2). Containers without a
CPU limit, or pinned with a matching cpuset, already get the right count. Serving speed is not affected: the
runner restores the thread settings after the load.

## The fix

After the stock count is taken, it is capped at the cgroup v2 quota from `/sys/fs/cgroup/cpu.max` (quota
divided by period, rounded up). A quota of `max`, a missing file (cgroup v1) or an unreadable value keeps the
stock count. The loaded weights are the same: the rescale is elementwise and gave bit-identical tensors at 160
and at 32 threads on 800 real expert tensors.

## Measured

Qwen3.6-35B-A3B-FP8 on one Gaudi 2 card, vllm-gaudi v0.26.0 with the deployment's other patches, Kubernetes
pods with a 32-CPU limit on 160-core hosts. Single runs:

| | Container start to ready | Weight load |
| --- | --- | --- |
| stock thread count (160), six replicas | 78.2 to 78.7 min | about 77 min |
| this fix (32 threads), weights not in the page cache, empty recipe cache | 109 s | 49 s |

In a CPU-only pod with the same 32-CPU limit, one FP8 weight and scale pair took 146 ms to rescale with 160
threads and 0.32 ms with 32.

## Verify

```bash
python3 fixes/load-threads-cgroup/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/load-threads-cgroup
```

On a device: in a container with a CPU limit, the load log line `Setting OMP_NUM_THREADS to N ...` shows the
limit (divided by the world size) instead of the host's core count, and the load time drops accordingly.
