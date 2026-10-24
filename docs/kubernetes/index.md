# Kubernetes

This section covers running vLLM on Gaudi at fleet scale on Kubernetes: what a Gaudi pod
needs, how to place pods so cards do not fragment, how to start, drain, roll and autoscale
model servers that take minutes to load, and how to notice when one stops working while
still looking healthy.

The practices come from operating a fleet of over a thousand Gaudi 2 cards (8 cards per
node) serving dozens of models at once, from 1-card models with many replicas to models
that need a whole node, all autoscaled. Each page states the rule, the reason for it and,
where it helps, what went wrong without it.

The manifests and tools live in
[`kubernetes/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/kubernetes) in the
repository. They use placeholders (`<registry>/<image>:<tag>`, `<storage-class>`) and the
namespace `llm`; replace them for your cluster.

## What a fleet needs, in order

| Step | What | Page |
| --- | --- | --- |
| 1 | Gaudi device plugin on every Gaudi node, the Habana container runtime, and a `habana` RuntimeClass | [Pod spec](pod-spec.md) |
| 2 | A model-server pod template with the Gaudi settings, probes and drain | [Pod spec](pod-spec.md) |
| 3 | A shared model store and a prefetch Job, so pods never download weights at start-up | [Start-up and warm-up](startup-and-warmup.md) |
| 4 | A bin-packing scheduler profile, so small models do not strand cards that big models need | [Scheduling](scheduling.md) |
| 5 | The stall-aware readiness probe, and a rule for when to cordon a node | [Health and failure](health-and-failure.md) |
| 6 | Graceful drain, one-pod-at-a-time rollouts, an autoscaler on the right metric | [Rollouts and scaling](rollouts-and-scaling.md) |
| 7 | A model-aware router and one consistent timeout chain | [Routing](routing.md) |
| 8 | Per-pod scraping and the alerts that catch silent failures | [Observability](observability.md) |

Steps 1 to 3 are enough to serve one model. Steps 4 to 8 start to matter once several
models share the fleet, replicas come and go, and users hold long streams open.

## Decision guide

| Symptom | Likely cause | Where to look |
| --- | --- | --- |
| `synStatus=8 [Device not found] Device acquire failed` at start | Missing RuntimeClass or `hostIPC`, host memory fragmentation, leaked allocations, a degraded card | [Pod spec](pod-spec.md#common-start-up-errors), [Health and failure](health-and-failure.md#host-memory-fragmentation) |
| An 8-card model stays Pending while many cards are free | Small pods spread over empty nodes | [Scheduling](scheduling.md) |
| A pod is Ready, `/health` is 200, but requests never finish | Engine stall | [Health and failure](health-and-failure.md#stall-aware-readiness) |
| Streams stop mid-response during scale-down or rollouts | No drain: vLLM aborts in-flight requests on SIGTERM by default | [Rollouts and scaling](rollouts-and-scaling.md#graceful-drain) |
| A single-replica model goes fully down during a rollout | `maxUnavailable: 1` with one replica | [Rollouts and scaling](rollouts-and-scaling.md#rollout-strategy) |
| The autoscaler swings between floor and ceiling | Loading pods in the metric, wrong metric or target | [Rollouts and scaling](rollouts-and-scaling.md#autoscaler-pitfalls) |
| Cold starts take an hour | Per-node downloads, CPU throttling, a cold recipe cache | [Start-up and warm-up](startup-and-warmup.md) |
| Long non-streamed requests fail at exactly 300 s or 600 s | A proxy or gateway timeout shorter than the rest of the chain | [Routing](routing.md#the-timeout-chain) |
| One pod queues while its siblings idle | Clients that bypass the router, or KV-heavy prompts on one pod | [Routing](routing.md) |
| Prometheus runs out of memory after autoscaler churn | Per-pod series from pods that no longer exist | [Observability](observability.md#keep-prometheus-healthy) |

## Files

| File | What it is |
| --- | --- |
| `manifests/runtimeclass.yaml` | The `habana` RuntimeClass |
| `manifests/vllm-deployment.yaml` | Deployment and Service template for one model |
| `manifests/model-store.yaml` | Shared model store PVC and a prefetch Job |
| `manifests/gaudi-binpack-scheduler.yaml` | A second kube-scheduler that bin-packs Gaudi cards |
| `manifests/hpa.yaml` | HorizontalPodAutoscaler on `vllm:num_requests_running` |
| `manifests/aibrix-podautoscaler.yaml` | Optional AIBrix PodAutoscaler equivalent |
| `manifests/pdb.yaml` | PodDisruptionBudget for a model |
| `manifests/monitoring.yaml` | PodMonitor and alert rules (Prometheus Operator) |
| `tools/vllm_stall_probe.py` | Stall-aware readiness probe |
| `tools/hpu-acquire-check.sh` | Leaked-allocation audit and multi-process device acquire test |
| `tools/hpu-fleet-sweep.sh` | Read-only per-card health census over SSH |
| `tools/hpu-resource-audit.py` | Checks manifests for per-card proportional requests and Gaudi pod basics |
