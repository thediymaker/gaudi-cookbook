# Health and failure

The failures that cost the most are the ones every layer reports as healthy. This page
covers how to detect them and how to treat faulty nodes.

## Stall-aware readiness

vLLM's `/health` only says the API server is up. When the engine hangs, for example a
multi-card collective that never completes, `/health` keeps returning 200, the pod stays
Ready, and the router keeps sending it requests that never finish. On one model this
pattern repeated every few days, each time holding requests for about ten minutes before
the runtime gave up.

[`vllm_stall_probe.py`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/tools/vllm_stall_probe.py)
is an exec readiness probe that fails when:

- `/health` fails, or
- requests are running or waiting, and the engine's step counter
  (`vllm:iteration_tokens_total` sample count) has not moved for `STALL_PROBE_SECONDS`.

An idle engine resets the clock, so only time spent busy without progress counts. If
`/metrics` cannot be read, `/health` decides.

- **Readiness only.** The pod leaves the Service and the router, and returns by itself when
  steps resume. Nothing restarts. Leave liveness on `/health`.
- **Pick the threshold from data.** It must be longer than the longest legitimate busy gap
  with no finished step (on a cold recipe cache, the first compile of a new shape can take
  tens of seconds; very long prefills on large models can take longer) and shorter than the
  hangs you want to cut. Scan your metrics for busy-without-progress gaps first. On the
  model above, the longest legitimate gap in 30 days was about two minutes, so 180 s
  separated them cleanly.
- **Test it** on an unrouted copy: send a few long streams, `kill -STOP` the engine
  process, and watch the pod go not Ready after the threshold; `kill -CONT` and it comes back
  and the streams finish.

## Fatal device errors

`synStatus 31 [Synapse is being terminated due to fatality]` is terminal for the
**process**, not the request: every later HPU call in that process fails the same way. In
vLLM the engine normally dies with it, `/health` fails and liveness restarts the pod.

Services that drive a Gaudi card directly (image, speech or custom model servers) need the
same outcome explicitly. A broad `except Exception` in a request handler turns a fatal
device error into one 500 per request, while every probe still passes. One such replica
served nothing but errors for weeks while looking healthy to the kubelet, the Service and
the router.

| Piece | What it does |
| --- | --- |
| Narrow fatal markers | Match only process-fatal strings (`synapse is being terminated`, `fatal error :: module`). A bad request or a transient device OOM must never take a replica offline |
| Latch | The first fatal error sets a flag; later ones do not overwrite it |
| `/ready` fails, quickly | Leave routing within seconds |
| `/health` fails, more slowly | Liveness restarts the pod after readiness has already shed traffic |
| 503, not 500 | "This replica is going away, retry elsewhere" instead of "your request is bad" |
| Watchdog | Compare the age of the in-flight device call with a deadline. A wedged card raises nothing, so no exception handler can catch it; only a timestamp can |
| Quiet probe logs | Drop successful probe and scrape lines from access logs, so the fault is still in the log when someone looks |

Set the watchdog deadline well above the slowest legitimate request; it separates "slow"
from "never returning", it does not police latency. For a server that cannot fail a health
endpoint (for example a UI framework that always answers `/`), exit the process instead:
process death is the one signal the kubelet always acts on.

**Block SIGPIPE in custom HPU services.** The Habana runtime installs its own SIGPIPE
handler, and its failure analysis treats the signal as fatal: one write to a closed pipe or
socket (a subprocess that exits early, a client that hangs up) can end the device context
with the same `synStatus 31` error. Block the signal, rather than ignoring it, in the main
thread before importing torch, so threads and children inherit the mask and the write fails
with `BrokenPipeError` instead:

```python
import signal

signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGPIPE})
```

## Rank failures

With tensor parallelism, each rank is a separate worker process. In the vLLM multiprocess
executor of the 0.26 era, an exception on a rank that is not the output rank was logged
(`WorkerProc hit an exception.`) and then dropped: rank 0 kept serving, the ranks fell out
of step, and output became garbage (for example every response the same token from NaN
logits) while `/health` stayed 200. One trigger was a device memory allocation failure on a
single rank during a very long prefill.

- Make a worker exception fatal: the worker exits, the executor notices, the engine fails,
  `/health` fails, and the pod restarts. A fail-closed runtime patch does this on affected
  releases.
- Alert on `WorkerProc hit an exception` in the logs.
- Run a cheap output canary per model (a fixed prompt with a known answer) in addition to
  the health checks.

Device memory that fails only after days of uptime (a small allocation failing during graph
replay of a common shape) points at the graph memory pool. Raising
`VLLM_GRAPH_RESERVED_MEM` gives graphs more room at the cost of KV cache; log
`torch.hpu.memory_stats()` periodically to see the trend before changing it.

## Host memory fragmentation

Symptom: a pod dies at start with `Device acquire failed`, and the node's kernel log shows
`habanalabs ...: Failed to create context -12` next to `page allocation failure: order:9`.

Opening a card needs a physically contiguous 2 MiB block of host memory. Heavy page cache
from weight loads breaks free memory into small pieces, so the allocation fails even with
hundreds of GB free. `free` will not show it; `/proc/buddyinfo` will (the right-hand
columns are the high-order free blocks):

```bash
grep Normal /proc/buddyinfo
```

The card is fine. Do not cordon the node or replace the card for this. Mitigations, at the
node level: trigger compaction before starts (`echo 1 > /proc/sys/vm/compact_memory`),
raise `vm.compaction_proactiveness`, and let the pod restart (a retry usually succeeds once
compaction catches up). Bin-packing makes this more likely, because it puts more loads on
the busiest nodes. A rollout guard that pauses on the first restart will trip on it.

## Node fault patterns

[`hpu-fleet-sweep.sh`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/tools/hpu-fleet-sweep.sh)
compares what Kubernetes believes with what each node sees, and sorts faults into these
buckets:

| Pattern | How it shows | What fixes it |
| --- | --- | --- |
| Card missing from the PCIe bus | Fewer Gaudi devices on the bus than expected | A cold power cycle (full power off) often brings it back; a warm reboot does not re-enumerate it and can drop another. Otherwise reseat or swap |
| Card on the bus, dropped by the driver | Bus count right, fewer cards bound or in `hl-smi` | Cold power cycle; if it repeats, swap |
| Dead at driver init | `Fatal error during habanalabs device init` in the kernel log | Swap. Power cycling does not help |
| Withheld by the device plugin | Allocatable lower than capacity on the Node | Look at the plugin and driver logs for that card |
| No cards at all | Zero Gaudi devices on the bus | Node-level fault (baseboard, PCIe complex, power), not eight card failures. Count it once when ordering spares |
| Stale capacity | The Node advertises cards it no longer has | Removing a device plugin does not clear an extended resource; patch the Node status to zero it |
| Leaked allocations | Pods in `Failed` or `Unknown` on the node with card requests | Delete them after confirming ownership; [`hpu-acquire-check.sh`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/tools/hpu-acquire-check.sh) lists them |
| Degraded under concurrency | Single-process tests pass, multi-process acquire fails | `hpu-acquire-check.sh -n <node> -p 8`; rule out fragmentation first |
| Scale-out links dead but "up" | Multi-node replicas fail or hang; ports report link up with carrier | Ping every scale-out port to its peers on a timer; bounce only ports that reach no peer |

A node that has lost one card is still useful: keep it in service on the remaining cards
and record which slot needs attention at the next hardware visit. A node with a partial
bind (the driver count does not match the bus count) is unsafe; keep it out of service.

## Cordon policy

- **Cordon only faulty nodes, or nodes held for a recorded reason** (a maintenance window,
  a hardware test). Write the reason down where the people fixing hardware will see it.
- **Uncordon healthy serving nodes.** A cordon with no fault behind it hides capacity and
  makes the fault list noisy.
- **Do not cordon to reserve nodes for a model.** Use scheduling (bin-packing, autoscaler
  floors) instead. A cordon as one step of a defrag procedure is fine if it is lifted at the
  end.
- **Missing a card is not a reason to cordon.** Log the slot and keep the node serving.
- Keep Node objects of offline nodes you plan to repair. Labels, taints and history come
  back with them.

## Diagnosing a model that "looks fine"

`kubectl get pods` is not evidence: replicas that serve nothing can read `1/1 Running` with
zero restarts. Ask each pod directly, by pod IP, because a Service hides which replica
answered, which is exactly the question:

```bash
kubectl -n llm get pods -l app.kubernetes.io/name=<deployment> -o wide
kubectl -n llm exec <some-pod> -- curl -s http://<pod-ip>:8000/metrics | grep -E 'num_requests_(running|waiting)|iteration_tokens_total_count'
```

Requests waiting with a step counter that does not move means stalled, not busy.

Keep the evidence: mount `HABANA_LOGS` to a per-pod directory on the node
(`subPathExpr: $(POD_NAME)`, as in the template) so device fault dumps survive the
container, and prune old directories on the nodes.
