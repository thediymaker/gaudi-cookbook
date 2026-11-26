# Scheduling

How to place Gaudi pods so the fleet does not fragment, and why model pods should never be
pinned to nodes.

## The problem: cards fragment

A Gaudi 2 node has 8 cards. A large model needs a whole node: 8 cards and most of the
node's memory. Many other models use 1, 2 or 4 cards.

The default kube-scheduler scores nodes by free CPU and memory (`LeastAllocated`, plus
`NodeResourcesBalancedAllocation`) and ignores extended resources such as
`habana.ai/gaudi`. Each new small pod therefore goes to the emptiest node. Over a few days
of autoscaling, 1-card pods sit on many otherwise empty nodes, and an 8-card model stays
Pending while hundreds of cards are free in fragments. This has happened on a real fleet
with several hundred free cards and not one node with 8 free.

## Part 1: pack new pods

Score nodes by how full their **cards** already are, and fill the fullest node that fits.
Empty nodes are used last, so they stay free for whole-node models.

The profile, by weight:

| Tier | Plugins | Weight | Role |
| --- | --- | --- | --- |
| 1 | TaintToleration, NodeAffinity, InterPodAffinity | 50 | Explicit wishes in the pod spec win (for example preferred anti-affinity that spreads one model's replicas) |
| 2 | NodeResourcesFit, `MostAllocated` on `habana.ai/gaudi` (weight 100) and `hugepages-2Mi` (weight 5) | 25 | Pack cards |
| 3 | ImageLocality, PodTopologySpread | 1 | Tie-breakers only; together they never outweigh one card |
| removed | NodeResourcesBalancedAllocation | | It favours empty nodes and works against packing |

Two details matter:

- **Leave cpu and memory out of the `MostAllocated` list.** With them, every pod without
  cards would also be pulled to the busiest Gaudi nodes. Only cards should decide.
- **Set `percentageOfNodesToScore: 100`.** On large clusters the scheduler otherwise stops
  after a sample of feasible nodes and may never see the fullest one. Scoring a few hundred
  nodes costs well under a millisecond.

This only changes where **new** pods go. Running pods are not moved.

### A second scheduler, or the default one

[`gaudi-binpack-scheduler.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/gaudi-binpack-scheduler.yaml)
runs a stock kube-scheduler as a Deployment with this profile under the name
`gaudi-binpack`. A model opts in with `schedulerName: gaudi-binpack` in its pod template.
Nothing in the control plane changes, rollback is removing the field, and it can be adopted
one model at a time (each adoption is one rolling restart, which also repacks that model).

Preemption is disabled and its RBAC has no pod delete, so it can never evict anything. If
both replicas are down, only opted-in pods wait in Pending.

The alternative is to give the default scheduler the same profile through its `--config`
file. That needs control-plane access and a restart of each server, one at a time. On
distributions that embed the scheduler, passing `--config` can make kube-scheduler ignore
the kubeconfig and profiling flags the distribution passes, so the config file itself must
set `clientConnection.kubeconfig`. Without it the scheduler cannot reach the API server and
exits, which on some distributions stops the whole server process. Test the exact file as a
throwaway secondary scheduler first.

### One scheduler for every Gaudi pod

While some Gaudi pods use the default scheduler and others use `gaudi-binpack`, the two
race for the same cards. Each scheduler keeps its own cache of what is free. When a pod
ends and frees cards, both can bind a waiting pod to that node in the same second. The
kubelet admits the first and rejects the second:

```text
UnexpectedAdmissionError ... habana.ai/gaudi. Requested: 4, Available: 1
```

The rejected pod goes `Failed` (it is not retried on another node), its controller creates
a new one, and that one waits for the next freed node, where it can lose again. Find the
losers with:

```bash
kubectl get pods -A --field-selector=status.phase=Failed \
  -o custom-columns=NS:.metadata.namespace,POD:.metadata.name,REASON:.status.reason,SCHED:.spec.schedulerName \
  | grep UnexpectedAdmission
```

What it cost on a fleet that ran both schedulers for a few weeks:

- One whole-node model left on the default scheduler lost 15 pods in 5 days, each time to a
  pod of another whole-node model on `gaudi-binpack` bound in the same second. Its
  autoscaler wanted 7 replicas; 4 were Ready.
- The canary pod of a rollout lost the race 6 times in a row over about 4 hours while
  another model had 7 pods Pending, so the rollout could not advance.
- A user's batch Job with `backoffLimit: 0` failed outright on its first lost race.

The race is not a capacity problem and does not go away with more nodes: it happens
whenever cards are freed while pods of both schedulers are Pending.

**Rule: every pod that requests `habana.ai/gaudi` uses the same scheduler.** That includes
canaries, test pods, multi-node replicas and image or speech servers, not only the main
model Deployments. Add `schedulerName` to each template (each change is one rolling
restart). Delete the `Failed` pods afterwards: they keep no cards, but they clutter every
listing.

Pods that users create themselves (Jobs, notebooks) will not carry the field. Set it at
admission instead: a mutating admission policy that sets `schedulerName` on every new pod
that requests Gaudi cards and still has the default scheduler. On API servers that serve
`MutatingAdmissionPolicy` (beta in Kubernetes 1.34, not enabled everywhere) this needs no
extra component; otherwise use a policy engine such as Kyverno or Kubewarden. A sketch,
not run on a cluster here (validate it with `kubectl create --dry-run=server` on a test
pod):

```yaml
apiVersion: admissionregistration.k8s.io/v1beta1
kind: MutatingAdmissionPolicy
metadata:
  name: gaudi-scheduler
spec:
  matchConstraints:
    resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE"]
        resources: ["pods"]
  matchConditions:
    - name: requests-gaudi
      expression: >-
        object.spec.containers.exists(c, has(c.resources.limits) &&
        'habana.ai/gaudi' in c.resources.limits)
    - name: on-default-scheduler
      expression: >-
        !has(object.spec.schedulerName) || object.spec.schedulerName == 'default-scheduler'
  failurePolicy: Ignore          # a policy error must not block pod creation
  reinvocationPolicy: Never
  mutations:
    - patchType: ApplyConfiguration
      applyConfiguration:
        expression: >-
          Object{spec: Object.spec{schedulerName: "gaudi-binpack"}}
---
apiVersion: admissionregistration.k8s.io/v1beta1
kind: MutatingAdmissionPolicyBinding
metadata:
  name: gaudi-scheduler
spec:
  policyName: gaudi-scheduler
```

With one scheduler for every Gaudi pod, that scheduler is on the critical path: run two
replicas with leader election, as the manifest does, and alert when Gaudi pods stay
Pending.

### Caveats

- A model's anti-affinity outranks packing, so a replica can still go to an emptier node
  when every fuller node already runs that model. Use `preferred`, never `required`,
  anti-affinity: a hard rule can leave replicas Pending on a fragmented fleet.
- Workloads without anti-affinity will stack replicas on the same full nodes. Add the usual
  preferred anti-affinity where losing one node should not take out every replica.
- Packing puts more pods and more page cache on the busiest nodes, which raises the chance
  of [host memory fragmentation](health-and-failure.md#host-memory-fragmentation) at card
  open.
- Requests must be [proportional per card](pod-spec.md#resources), or packing fails on
  memory before cards run out.

## Part 2: repack running pods

Packing only affects new pods. To free whole nodes that are already fragmented:

**Descheduler, cautiously.** The descheduler's `HighNodeUtilization` plugin with a
threshold on `habana.ai/gaudi` (for example 40%, i.e. 3 or fewer of 8 cards used) evicts
pods from nearly empty nodes so the bin-packing scheduler places them on fuller ones. Keep
it conservative, because each eviction is a cold start:

- run it only after bin-packing is live for those pods, or the evicted pods go straight
  back to empty nodes and the two fight forever;
- evict at most one pod per node and a handful per run, in a quiet window;
- opt models in explicitly: only small (1-2 card), fast-loading models with at least two
  replicas, weights on the shared store, covered by a PodDisruptionBudget with
  `maxUnavailable: 1`, so no model ever loses more than one replica and none while a
  replica is not Ready;
- never 4- or 8-card pods, never slow loaders, never single-replica models.

```yaml
apiVersion: descheduler/v1alpha2
kind: DeschedulerPolicy
maxNoOfPodsToEvictPerNode: 1
maxNoOfPodsToEvictTotal: 4
profiles:
  - name: gaudi-repack
    pluginConfig:
      - name: DefaultEvictor
        args:
          labelSelector:
            matchLabels:
              repack: allowed          # opt-in label on the pod template
          minReplicas: 2
          minPodAge: 1h
          nodeFit: true
          podProtections:
            defaultDisabled: [PodsWithLocalStorage]   # every model pod has emptyDir /dev/shm
            extraEnabled: [PodsWithoutPDB]
      - name: HighNodeUtilization
        args:
          thresholds:
            habana.ai/gaudi: 40
          evictionModes: [OnlyThresholdingResources]
    plugins:
      balance:
        enabled: [HighNodeUtilization]
```

**By hand, for an urgent whole-node model.** Cordon the nodes you want to empty, delete the
small pods on them one model at a time (waiting for each replacement to be Ready), then
uncordon as soon as the large pods have bound. The cordon is a temporary step of the
procedure, not a hold; see [cordon policy](health-and-failure.md#cordon-policy).

Slow-loading small models are what block whole nodes the longest: the descheduler
should leave them alone, and making them load faster frees more nodes than any policy.

## Why no node pins

Do not pin model pods to hosts (`nodeName`, a `kubernetes.io/hostname` selector, required
node affinity on a host list), and do not keep nodes reserved for a model by cordoning them.

- **Pins go stale.** On a large fleet nodes fail and come back every week. A pinned replica
  waits in Pending for a node that is gone, and someone has to remember the list.
- **Pins defeat autoscaling.** The autoscaler can only add replicas where the pin allows.
- **Pins interact badly with rollouts.** An autoscaler scale-up during a rollout splits
  replicas between the old and new ReplicaSets. Old pods that carry a pin stay Pending, and
  with `maxUnavailable: 0` the rollout stalls on them.
- **Reserved nodes hide capacity.** A cordoned healthy node is invisible to everything else.

To add capacity for a model, raise its autoscaler floor and ceiling. Handle node health
needs fleet-wide (fix or cordon the faulty nodes) instead of steering one model around them.
One manifest per model, scaled by its autoscaler; no side Deployments to dodge a restart.

If a pod needs something a node may lack (for example the shared model store mount), make
the pod check for it at start and fail with a clear message. A pod that lands on a
misconfigured node then fails fast and visibly, instead of everyone maintaining an exclude
list. Delete the pod to reschedule it while the node is fixed.
