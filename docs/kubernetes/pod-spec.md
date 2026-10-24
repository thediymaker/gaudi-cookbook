# Pod spec

What a vLLM pod on Gaudi needs, and why. The full template is at the end of this page.

## Cluster prerequisites

- **Device plugin.** The Intel Gaudi device plugin DaemonSet runs on every Gaudi node and
  advertises the cards as the extended resource `habana.ai/gaudi`. Run it with
  `priorityClassName: system-node-critical` so it is never evicted.
- **Container runtime.** The Habana container runtime sets up the devices the plugin
  allocates. Register it in containerd on each Gaudi node and create a RuntimeClass whose
  handler matches it
  ([`runtimeclass.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/runtimeclass.yaml)).
- **Hugepages.** The Gaudi runtime uses 2 MiB hugepages for host buffers. Preallocate them
  on each node (kernel command line or sysctl) so the kubelet advertises
  `hugepages-2Mi`. 4 GiB per card (32 GiB per 8-card node) has served models from 1 to 8
  cards.
- **Driver match.** The Gaudi software in the image must match the host driver release
  (see [Getting started](../getting-started.md)). A fleet with mixed driver releases needs
  node labels and images per release.

## The settings, one by one

| Setting | Why | What happens without it |
| --- | --- | --- |
| `runtimeClassName: habana` | Runs the container under the Habana runtime, which sets up the allocated devices | The container starts without usable devices; the first HPU call fails with `Device acquire failed` |
| `hostIPC: true` | Habana's documented container setup uses the host IPC namespace; tensor-parallel ranks are separate processes | Multi-card starts can fail at device acquire or collective init, while 1-card pods work |
| `habana.ai/gaudi` in requests and limits | Allocates whole cards; requests must equal limits | The pod gets no cards, or the API rejects it |
| `hugepages-2Mi`, requests equal to limits | Host buffers for the runtime; Kubernetes requires requests equal to limits for hugepages | Pending with `Insufficient hugepages-2Mi`, or runtime allocation failures |
| `/dev/shm` as a memory `emptyDir` | PyTorch shares tensors between processes through `/dev/shm`; the container default is 64 MiB | Crashes or hangs when worker processes exchange data |
| Toleration for the Gaudi taint | If Gaudi nodes are tainted to keep other work off | Pending |
| `shareProcessNamespace: true` | Makes the pause container PID 1, so the preStop hook can stop a pod that is still starting | A pod deleted during start-up holds its cards for the whole grace period |
| `enableServiceLinks: false` | A Service named `vllm` injects `VLLM_PORT=tcp://...`, which vLLM reads as its own setting | vLLM fails to start with a confusing port error |
| No `HABANA_VISIBLE_DEVICES` in `env` | The device plugin sets it to the cards this pod was given | Copying `all` from a `docker run` example can point the runtime at cards other pods own |

The image normally runs as root; the Gaudi runtime needs access to the device nodes.

## Environment

| Variable | Value | Why |
| --- | --- | --- |
| `PT_HPU_LAZY_MODE` | `1` | Lazy mode with HPU graphs. `0` (eager) and `--enforce-eager` collapse throughput; debug compile failures in graph mode instead |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES` | `true` | Needed for tensor parallelism with HPU graphs in lazy mode |
| `PT_HPU_RECIPE_CACHE_CONFIG` | `<dir>,<delete_on_start>,<max_MB>` | Compiled graph cache on node-local disk; see [Start-up and warm-up](startup-and-warmup.md#recipe-cache) |
| `VLLM_SKIP_WARMUP` | `false` or `true` | Trade-off between start time and live compiles; see [Start-up and warm-up](startup-and-warmup.md#warm-up) |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | e.g. `1800` | A first compile of a new shape inside one step can exceed the multi-card executor's default deadline |
| `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1` | the shared store | Serving pods read weights from the store and never call the hub |
| `HABANA_LOGS` | a mounted node directory | Device fault dumps outlive the container |

vLLM logs `Unknown vLLM environment variable detected` for many Gaudi settings
(`VLLM_SKIP_WARMUP`, the bucketing and graph-memory variables). The vllm-gaudi plugin reads
them directly from the environment, so they do take effect. Do not delete them to silence
the warning.

## Resources

Size requests per card, at or below the node's per-card share:

```text
per-card request <= (node allocatable - DaemonSet requests) / cards per node
```

If a 1-card pod asks for a quarter of a node's memory, that node can never hold eight
1-card pods and its spare cards are stranded. Proportional requests let any mix of 1-, 2-,
4- and 8-card pods pack onto any node.
[`hpu-resource-audit.py`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/tools/hpu-resource-audit.py)
checks this for a directory of manifests.

- **Memory limits above requests** give headroom for compile spikes, but make the pod
  Burstable: under node memory pressure it can be evicted. For models where that is not
  acceptable, set requests equal to limits and size the requests honestly.
- **No CPU limit.** Weight loading and graph compilation are CPU heavy. A CPU limit shows up
  as CFS throttling and very slow starts, which is easy to misread as slow storage.
- **`/dev/shm` counts toward the memory limit**, because a memory `emptyDir` is RAM.

## Probes

| Probe | Check | Settings in the template | Purpose |
| --- | --- | --- | --- |
| startup | `GET /health` | every 30 s, 240 failures (2 h) | Cover weight load and warm-up; size it to the slowest measured cold start |
| readiness | [stall probe](health-and-failure.md#stall-aware-readiness) | every 10 s, 3 failures | Take a pod out of routing when it fails `/health` or stops making progress |
| liveness | `GET /health` | every 30 s, 5 failures | Restart only when the server is gone |

Keep liveness looser than readiness: shed traffic first, restart second. Never put the
stall check on liveness; a restart costs a full cold start, and a stall often clears.

## Runtime patches

Fixes that must be applied to a released image before `vllm serve` (see
[Fixes](../fixes/index.md)) belong in a ConfigMap, mounted read-only and run from the
container command with `|| exit 1`, never as `sed` lines inline in the manifest. A patch that
does not match its anchors must stop the container instead of serving a half-patched engine.

Editing a ConfigMap that is mounted as a volume does not restart anything: the file on disk
changes within a minute, but the running process keeps the code it loaded at start. Bump an
annotation on the pod template in the same change (for example `config-rev: "2"`) so the
Deployment rolls, or name ConfigMaps by version and change the reference.

## Common start-up errors

| Error | Usual causes | First check |
| --- | --- | --- |
| `RuntimeError: synStatus=8 [Device not found] Device acquire failed` | No `runtimeClassName`; no `hostIPC` on a multi-card pod; host memory fragmentation; pods in `Failed` or `Unknown` still holding devices; a degraded card | Pod spec first; then `dmesg` on the node for `Failed to create context -12` and `page allocation failure: order:9` ([fragmentation](health-and-failure.md#host-memory-fragmentation)); then [`hpu-acquire-check.sh`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/tools/hpu-acquire-check.sh) |
| `synStatus 31 [Synapse is being terminated due to fatality]` | A fatal device error earlier in the process; every later HPU call fails | The first error in the log, and the Habana logs; see [Health and failure](health-and-failure.md#fatal-device-errors) |
| Pending with `Insufficient habana.ai/gaudi` while cards are free | Fragmentation across nodes | [Scheduling](scheduling.md) |
| Pending with `Insufficient hugepages-2Mi` | Node has no or too few preallocated hugepages | `kubectl describe node` allocatable |
| Exit code 137 | Memory limit hit (remember `/dev/shm`), or SIGKILL at the end of the grace period | `kubectl describe pod` last state; container memory graph |
| Exit code 1 with a Python traceback | Configuration or software | `kubectl logs --previous` |
| `HfHubHTTPError: 429 Too Many Requests` at start | Hub calls during start-up, shared rate limit across the fleet | Serve from the shared store offline ([Start-up and warm-up](startup-and-warmup.md#model-store)) |
| `Running`, never `Ready`, no restarts | Start-up still in progress, or the startup probe budget is too small for this model | Container log progress; startup probe settings |

Diagnose before changing the image: read `kubectl logs --previous` and the last termination
reason and exit code. Most failed bring-ups are configuration, not the image.

## Template

[`kubernetes/manifests/vllm-deployment.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/vllm-deployment.yaml):

```yaml
--8<-- "kubernetes/manifests/vllm-deployment.yaml"
```
