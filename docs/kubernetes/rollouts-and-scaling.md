# Rollouts and scaling

Model servers take minutes to start and hold long streams open. Rollouts, autoscaler
scale-downs and node drains all delete pods, and each deletion is either a clean drain or a
burst of cut streams. This page covers both sides: replacing pods safely, and deciding how
many there should be.

## Rollout strategy

Roll one pod at a time, and keep the model serving throughout.

| Situation | `maxSurge` | `maxUnavailable` | Why |
| --- | --- | --- | --- |
| Spare cards available | 1 | 0 | The new pod starts before an old one leaves; no capacity dip |
| Cards tight, or a large model | 0 | 1 | No spare cards needed; one replica short during each step |
| **Single replica** | 1 | 0 | Required. With one replica, `maxUnavailable: 1` deletes the only pod at the start of the rollout, whatever `maxSurge` says, and the model is down for a whole cold start |
| `Recreate` | | | Avoid. Every pod drains, then every new pod loads: the outage is drain plus cold start |

Before any template change to a single-replica model, check both fields: `maxSurge >= 1`
**and** `maxUnavailable == 0`.

- **Stuck surge rollouts hold cards.** If new pods cannot pass start-up, the old ReplicaSet
  is never scaled down, and the model holds old plus new cards indefinitely. Watch
  `kubectl get rs -l <selector>` during a rollout, and set `progressDeadlineSeconds` so a
  stuck rollout is reported.
- **One model at a time.** Concurrent rollouts of several models can compete for the same
  spare cards. Large loads on a capable shared store are not a reason to serialize, but
  capacity is.
- **One restart per model.** Bundle pending template changes (a fix, a scheduler switch, a
  storage change) into one rollout instead of three.

### Applying changes

- **Leave `replicas` out of manifests whose replicas an autoscaler owns.** Applying a
  manifest that sets `replicas` resets the autoscaler's count. Removing the field later
  from a manifest that was applied with it can also reset it, so leave it out from the
  start.
- **Canaries get their own labels.** A canary that carries the production model label
  joins the production Service and router pool. Some gateway controllers create a model's
  route when a labeled Deployment is created and delete it by name when any labeled
  Deployment is deleted, so deleting a canary that shares the label can remove the
  production route.
- **Verify the new code is live,** not that `kubectl apply` succeeded: check a marker the
  change adds (a log line, a metric, a file in the running container).

## Graceful drain

By default, nothing drains. vLLM's shutdown timeout defaults to 0, which means **abort**:
on SIGTERM the engine aborts every in-flight request at once. Kubernetes' default grace
period is 30 s. The result, measured: every stream stops mid-response with no
`finish_reason`, no `[DONE]` and no error frame, the connection hangs until the kubelet
SIGKILLs the container at 30 s (exit 137), and the gateway logs a reset after the response
started. A proxy cannot retry once bytes were sent, so the client sees a failed request.

It is worse than it sounds because autoscaler scale-down deletes the **newest** pods
first, and a least-request router has just filled those.

### The fix, per Deployment

1. **preStop hook, start-up aware.** If vLLM is listening on its port, sleep 20 s so the
   router and the Service drop the pod before SIGTERM. If it is still starting, kill
   everything in the pod at once (`kill -KILL -1`): there is nothing to drain, and a SIGTERM
   during start-up is either ignored (the shell is PID 1 before `exec`) or lost (engine
   init), which would hold the cards for the whole grace period.
2. **`--shutdown-timeout N`** on `vllm serve`. On SIGTERM the engine stops accepting new
   requests and keeps stepping in-flight ones for up to N seconds, then exits. Anything
   still running at N is cut.
3. **`terminationGracePeriodSeconds = 20 + N + 30`.** The preStop counts inside the grace
   period.
4. **`shareProcessNamespace: true`,** so the pause container is PID 1 and the preStop can
   stop a pod stuck in start-up. Without it a pod deleted during engine init stayed for the
   full grace period. It requires that no container relies on being PID 1.

With this, measured on several models and releases: idle pods leave about 25 s after
deletion, pods deleted during start-up leave within seconds, and streams shorter than N
complete.

### Choosing N

Pick N per model from its own request durations. A request that is in flight at a random
moment is biased toward long requests, so weight by time:

```text
cut(N) = sum(max(d - N, 0)) / sum(d)      over recent successful requests, d = duration
```

Take the smallest N (in steps of 30 s) where `cut(N)` is at most 1%, clamped to a range
such as 60 to 900 s. N moves with the traffic mix: a few users running very long generations
can push a model to the ceiling within a day. A high N costs nothing while a pod is idle or
its requests are short.

What a long grace period costs:

- On scale-down the replica count drops at once, but a busy pod keeps its cards for up to
  `20 + N` s.
- With `maxSurge: 0`, a replacement that needs the old pod's cards (a whole-node model on a
  full node) waits for the old pod to drain at each step.
- With `Recreate`, the outage grows by the drain time.
- Node drains and descheduler evictions wait up to `20 + N` s per busy pod.

### Idle mode, when native drain is not available

Older vLLM releases have no `--shutdown-timeout`, and an unknown flag makes the pod crash
loop. With several API server processes (`--api-server-count > 1`), each API process closes
its engine client on SIGTERM and native drain freezes every stream. In both cases drain in
the preStop instead: after the 20 s, poll `/metrics` until nothing is running or waiting,
for at most N s. The default abort then finds nothing to abort.

```bash
if grep -qE '^ *[0-9]+: [0-9A-F]+:1F40 [0-9A-F]+:0000 0A' /proc/net/tcp /proc/net/tcp6 2>/dev/null; then
  sleep 20
  end=$((SECONDS + 120))                     # N
  while (( SECONDS < end )); do
    busy=$(curl -sf localhost:8000/metrics \
      | awk '/^vllm:num_requests_(running|waiting)/ {s += $NF} END {print s + 0}') || break
    [[ "$busy" == 0 ]] && break
    sleep 2
  done
else
  kill -KILL -1
fi
```

### Check the router side

- Kubernetes marks a deleted pod's endpoint `ready=false, terminating=true` within a fraction
  of a second; kube-proxy stops sending new connections to it, except when a Service has
  **only** terminating endpoints (then it keeps using them).
- A model-aware router should drop pods with a deletion timestamp immediately; check yours.
- Newer vLLM releases keep the listener open while draining: new requests get an immediate
  HTTP 500 instead of a refused connection, and `/health` keeps returning 200 until the pod
  is gone. Routers that drop terminating pods are unaffected; anything that sends straight
  to pod IPs is not.
- After rolling the drain out, confirm on the next pod termination that the log shows the
  drain starting and in-flight requests completing, and that the gateway's
  reset-after-response-started count for that model stops growing.

## Capacity is pods times max-num-seqs

A model's concurrent capacity is `ready pods x --max-num-seqs`. Everything that limits or
counts concurrency must follow that number:

- **Autoscaler target.** A target on `num_requests_running` above `--max-num-seqs` can never
  be reached, so the model never scales up. Move the target whenever max-num-seqs changes.
- **Gateway concurrency limits.** A gateway that admits a fixed number of in-flight
  requests per replica must use the live max-num-seqs. If it uses a lower value, it queues
  users at the gateway while pods sit half empty, and an autoscaler watching per-pod
  occupancy never sees the load, so it never scales up either.
- **Capacity dashboards.** Derive slots from ready endpoints times max-num-seqs; they move
  every time the autoscaler does.

## Autoscaling

The limiting factor is cold start, not policy. A burst that lasts five minutes is over long
before a pod that takes twenty minutes to load is Ready. Make starts fast first
([Start-up and warm-up](startup-and-warmup.md)); only then can floors come down.

**Metric.** Scale on `vllm:num_requests_running` per pod, at about 0.6 x max-num-seqs.

- `num_requests_running` is occupancy. It stays up while requests execute, which holds
  replicas through a burst. Its limit: it cannot exceed max-num-seqs, so once pods are full
  it carries no information about whether 10 or 1000 requests are queued. An autoscaler
  pinned at its ceiling reports "stable" either way, so alert on queue time separately.
- `num_requests_waiting` alone reads 0 whenever the queue is empty, including right after
  new pods drained it. A waiting-only autoscaler scales down, refills the queue and
  oscillates.
- With a high floor, load spreads thin and per-pod occupancy may never reach 60% before the
  queue grows. Bucket measured per-pod occupancy by queue depth and set the target where
  real backlog begins.

**Floors** come from queue **duration** over a week or more, not peak queue depth: a model
can show a large peak from one brief spike and be idle all week. Keep floors higher than
demand suggests where a single long prefill would otherwise make concurrent users time out.

**Ceilings** are the brake on overshoot, because the autoscaler cannot see backlog depth.

[`hpa.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/hpa.yaml)
is a starting point. An HPA's scale-down stabilization window takes the **highest**
recommendation in the window, so a long window protects against cutting on a dip.
[`aibrix-podautoscaler.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/aibrix-podautoscaler.yaml)
is the equivalent for AIBrix, whose behaviour differs (see the file header).

### Autoscaler pitfalls

- **Loading pods poison the metric.** Pods that are still loading have no `/metrics`. If the
  controller counts a failed scrape as zero, or lets slow fetches from loading pods use up
  its per-cycle deadline so that ready pods also read zero, a scale-up under load reads as
  idle and the autoscaler deletes the pods it just started. Use a controller that scrapes
  only Ready pods and leaves failed fetches out of the average (the HPA sets missing
  metrics aside).
- **Starting pods counted as capacity.** If the replica count used in the calculation
  includes pods that are still starting while ready pods are saturated, each cycle
  multiplies again and the deployment runs to its ceiling, then overshoots down to its
  floor. Damp scale-down, and do not let pending replicas compound scale-up.
- **Defaults built for fast-starting web services.** Cutting on a 10% dip, or halving
  replicas per cycle, thrashes models that take minutes to load.
- **Stabilization semantics differ.** Read your controller's code: in some, a longer
  scale-down window cuts deeper instead of later.
- **Flapping is expensive beyond the cold starts.** One model flapping between floor and
  ceiling created thousands of pods in a day; every new pod is a new series in every
  per-pod exporter, which helped run Prometheus out of memory. Alert on replica changes.
- **Do not scale by hand.** `kubectl scale` is reverted by the autoscaler within seconds;
  change its floor or ceiling instead.

## Disruption budgets

[`pdb.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/pdb.yaml)
sets `maxUnavailable: 1` per model. A PDB limits voluntary evictions (node drains, the
descheduler), not rollouts or autoscaler scale-down. With a single replica, `minAvailable: 1`
allows zero disruptions and blocks every node drain; prefer `maxUnavailable: 1` and decide
consciously whether a single-replica model may go down during maintenance.
