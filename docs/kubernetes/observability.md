# Observability

What to scrape, what to alert on, and how to keep logs and Prometheus useful on a large
fleet.

Scrape every model pod directly (a PodMonitor, or pod annotations), not through its
Service: a Service scrape samples one endpoint at a time, and most questions here are per
pod.

## vLLM metrics that matter

| Metric | Type | Use |
| --- | --- | --- |
| `vllm:num_requests_running` | gauge | Occupancy; autoscaling metric; capped at `--max-num-seqs` |
| `vllm:num_requests_waiting` | gauge | Queue on the pod; zero most of the time, so never the only scaling signal |
| `vllm:kv_cache_usage_perc` | gauge (0 to 1) | KV pressure; near 1 means preemptions. Older releases: `vllm:gpu_cache_usage_perc` |
| `vllm:num_preemptions_total` | counter | Requests pushed out of the KV cache |
| `vllm:iteration_tokens_total` | histogram | Its `_count` grows once per engine step: the progress signal for stall detection |
| `vllm:request_queue_time_seconds` | histogram | Time from arrival to first schedule; the user-facing queue |
| `vllm:time_to_first_token_seconds` | histogram | TTFT: queue plus prefill |
| `vllm:inter_token_latency_seconds` | histogram | Decode speed per stream (older releases: `vllm:time_per_output_token_seconds`) |
| `vllm:e2e_request_latency_seconds` | histogram | Request duration; input for choosing the drain timeout |
| `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` | counters | Throughput |
| `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` | counters | Prefix cache hit rate |
| `vllm:request_success_total` | counter, by `finished_reason` | Completions; `length` versus `stop` |
| `vllm:cache_config_info` | info | Confirms the running cache settings (for example `enable_prefix_caching`) |

Useful derived numbers:

```text
capacity slots        = count(ready endpoints) x max-num-seqs
occupancy             = sum(num_requests_running) / capacity slots
prefix hit rate       = rate(prefix_cache_hits_total) / rate(prefix_cache_queries_total)
queued minutes (7d)   = sum_over_time((avg by (model_name)(vllm:num_requests_waiting) > bool 0)[7d:1m])
```

Queued minutes over a week is a better input for autoscaler floors than peak queue depth.

## Fleet and card metrics

- **Capacity versus allocatable** per node from kube-state-metrics
  (`kube_node_status_capacity{resource="habana_ai_gaudi"}` and
  `kube_node_status_allocatable`): the difference is cards the device plugin withholds.
- **Card requests by pod and phase**
  (`kube_pod_container_resource_requests{resource="habana_ai_gaudi"}` with
  `kube_pod_status_phase`): Pending pods with free cards in the fleet mean fragmentation.
- **Per-card utilization, memory and temperature** need a Gaudi metrics exporter on each
  node. Verify that it actually emits card series after installing it: an exporter that
  cannot read the kubelet's pod-resources socket (`/var/lib/kubelet/pod-resources`) can
  fail every collection without anyone noticing for months.

## Alerts

[`monitoring.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/monitoring.yaml)
has these as Prometheus Operator rules:

| Alert | Condition | Catches |
| --- | --- | --- |
| Engine stalled | Requests in flight, no engine step for 5 minutes | Hangs with `/health` 200 |
| Queueing | 90th percentile queue time above a threshold for 15 minutes | Under-provisioning; an autoscaler at its ceiling that reports "stable" |
| KV cache full | KV use above 95% for 10 minutes | Long-prompt pressure, preemptions |
| Cards withheld | Allocatable below capacity for 30 minutes | Unhealthy cards |
| Pods pending on cards | A Gaudi pod Pending for 30 minutes | Fragmentation, faulty nodes, quota |
| Restarting | More than two restarts in an hour | Crash loops, fatal device errors |
| Replica flapping | More than ten replica changes in two hours | Autoscaler instability |

Also worth alerting on, from other components: the gateway's count of upstream resets after
the response started (cut streams, usually from missing drain), idle-timeout cuts, and any
fatal-error gauge exported by custom HPU services.

## Logs

**Habana log volume.** With `ENABLE_CONSOLE=true` and `LOG_LEVEL_ALL=3`, the runtime prints
every graph-compiler warning to stdout: tens of thousands of lines per minute during
compilation, which rotates the container log within minutes and takes the evidence of any
real failure with it. Either leave console logging off and keep the file logs (mounted from
the node, as in the template), or raise the level for the compiler and profiler loggers
only, which cut the volume by about 99% while keeping runtime warnings:

```yaml
- {name: LOG_LEVEL_ALL, value: "3"}
- {name: LOG_LEVEL_SHARED_LAYER_API, value: "4"}
- {name: LOG_LEVEL_TPC_NODE, value: "4"}
- {name: LOG_LEVEL_SRAM_SLICE, value: "4"}
- {name: LOG_LEVEL_GC, value: "4"}
- {name: LOG_LEVEL_ROI_SPLITTER, value: "4"}
- {name: LOG_LEVEL_HW_TRACE, value: "4"}
- {name: LOG_LEVEL_PROF_SDK, value: "4"}
- {name: LOG_LEVEL_MEM_COHERENCE, value: "4"}
- {name: LOG_LEVEL_GC_TPC_FUSER, value: "4"}
```

`LOG_LEVEL_ALL=4` silences all warnings; errors still print.

**Probe and scrape noise.** A router or autoscaler that scrapes `/metrics` on every pod
every few seconds, plus probes, can dominate a server's access log. In custom servers, drop
successful probe and scrape lines (match the exact path, not a prefix), and keep failures:
a readiness endpoint flipping to 503 is the line worth having.

**Keep device logs past the container.** Mount `HABANA_LOGS` to a per-pod directory on the
node. Prune those directories on the node, for example with a `tmpfiles.d` age rule.

## Keep Prometheus healthy

- **Per-pod series from pods that are gone.** Components that re-export every model pod's
  histograms with a `pod` label (some gateways do) and never forget deleted pods grow
  without bound under autoscaler churn. One such job reached millions of series and ran
  Prometheus out of memory. Drop the per-pod series you do not query with
  `metric_relabel_configs`, and restart or fix the exporter.
- **Node exporter collectors** you do not use (hwmon, softnet, per-CPU frequency, virtual
  network interfaces) add up quickly on large nodes; drop them at scrape time.
- **Watch the watcher.** A crash-loop check that reads kube-state-metrics through
  Prometheus cannot see Prometheus itself crash looping; check it from outside.
