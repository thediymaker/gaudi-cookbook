# Kubernetes

Manifests and tools for running vLLM on Gaudi at fleet scale. The reasoning behind every
setting is in the documentation under [`docs/kubernetes/`](../docs/kubernetes/index.md).

| Path | What it is |
| --- | --- |
| `manifests/runtimeclass.yaml` | The `habana` RuntimeClass |
| `manifests/vllm-deployment.yaml` | Deployment and Service template for one model (runtime class, hostIPC, hugepages, shm, recipe cache, probes, drain) |
| `manifests/model-store.yaml` | Shared model store PVC and a prefetch Job |
| `manifests/gaudi-binpack-scheduler.yaml` | A second kube-scheduler that bin-packs Gaudi cards |
| `manifests/hpa.yaml` | HorizontalPodAutoscaler on `vllm:num_requests_running` |
| `manifests/aibrix-podautoscaler.yaml` | Optional AIBrix PodAutoscaler equivalent |
| `manifests/pdb.yaml` | PodDisruptionBudget for a model |
| `manifests/monitoring.yaml` | PodMonitor and alert rules (Prometheus Operator) |
| `tools/vllm_stall_probe.py` | Stall-aware readiness probe (exec probe, standard library only) |
| `tools/hpu-acquire-check.sh` | Leaked-allocation audit and multi-process device acquire test |
| `tools/hpu-fleet-sweep.sh` | Read-only per-card health census over SSH |
| `tools/hpu-resource-audit.py` | Checks manifests for per-card proportional requests and Gaudi pod basics |

Placeholders to replace: `<registry>/<image>:<tag>`, `<storage-class>`, and the namespace
`llm`. Each script prints its usage with `-h`.
