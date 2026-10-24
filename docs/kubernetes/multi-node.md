# Multi-node replicas

How to run one model replica across two 8-card nodes, and what changes on Kubernetes when a
replica is a group of pods instead of one pod.

Most models here fit one node; prefer that whenever a recipe offers it. Two nodes are for
checkpoints that do not fit 8 cards in the format you want to serve (for example FP8 experts
of a 700B to 1T model), or for a KV pool larger than one node holds.

## Two ways to split a model

| | Pipeline parallel (PP 2 x TP 8) | Tensor parallel 16 |
| --- | --- | --- |
| What crosses the nodes | one hidden-state row per token, between stages | every layer's all-reduce |
| Link | the ordinary pod or host network (gloo) | RoCE scale-out ports on both nodes, configured for HCCL |
| Software | the [pipeline-parallel](../fixes/pipeline-parallel.md) fix, `--no-async-scheduling` | stock collectives; an executor that spans nodes |
| Placement | any two whole nodes | only node pairs whose scale-out links reach each other |
| Verified here | yes, on `v0.26.0` and `v0.30.0` (below) | no |

Pipeline parallelism is the one to start with: it needs no fabric, and any two free nodes
form a replica. Its cost is a pipeline bubble at low concurrency and the loss of async
scheduling. Tensor parallelism across nodes is faster per stream when the fabric exists, but
it ties each replica to a cabled pair.

## The replica is the group

Two pods that form one engine behave like one pod with two halves:

- **Restart together.** A stage cannot rejoin a running pipeline (or a running tensor-parallel
  group). If either pod dies, the other hangs on its next transfer. Recreate the whole group.
- **Route to the leader only.** The node with rank 0 runs the API server; the other node runs
  headless and has no HTTP port. The Service selects the leader.
- **Probe the leader.** The leader's `/health` reflects the whole engine: a lost stage stops
  its steps. The worker gets no HTTP probes; group restart covers it.
- **Scale and roll by group.** An autoscaler or rollout adds or removes whole groups. With
  `maxSurge 0`, a rollout of a single group drops the model to zero replicas while it
  restarts.
- **Schedule whole nodes.** Each pod requests all 8 cards, which keeps the two stages on
  different nodes without an anti-affinity rule. Both pods must be placed before either can
  start; a group with one pod Pending holds 8 cards idle.

[LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) expresses these rules directly
(`size: 2`, `RecreateGroupOnPodRestart`, `LWS_LEADER_ADDRESS` for the master address). An
[AIBrix](https://github.com/vllm-project/aibrix) StormService with a pod group of two, or a
Ray cluster per replica for builds that use Ray, are alternatives with the same rules.

## Example

[`kubernetes/manifests/vllm-two-node-pipeline.yaml`](https://github.com/thediymaker/gaudi-cookbook/tree/main/kubernetes/manifests/vllm-two-node-pipeline.yaml)
is a LeaderWorkerSet for PP 2 x TP 8. The two `vllm serve` commands inside are the verified
ones; the LeaderWorkerSet around them was not applied on a cluster, so treat it as a starting
point. The essentials:

```bash
python3 fixes/apply_all.py --profile v0.30.0 pipeline-parallel || exit 1
export GLOO_SOCKET_IFNAME=eth0 VLLM_HOST_IP=<this pod's IP>
# leader
vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2 --no-async-scheduling \
  --distributed-executor-backend mp --nnodes 2 --node-rank 0 --master-addr <leader IP> --master-port 29555 ...
# worker: the same with --node-rank 1 --headless
```

## Measured

Qwen3-30B-A3B-Instruct-2507 at TP 2 on each of two nodes (PP 2), against TP 2 on one node,
pods on the pod network, `--no-async-scheduling`, Gaudi software 1.24.1.

| Check | Result |
| --- | --- |
| Greedy answers, 12 prompts | identical to one node, both releases |
| Teacher-forced top-1 tokens, 8,441 positions | identical (NLL 1.4425 both) |
| Needles at 12K | 3 of 3 |

Decode per stream, and in brackets the aggregate (all tokens over the wall time, prefill
included), in tok/s, 2K cached prefix, 128 new tokens. Measured as: same two pods, separate
server starts for one node and two nodes, one run each.

| Release | Streams | One node, TP 2 | Two nodes, PP 2 x TP 2 |
| --- | --- | --- | --- |
| `v0.30.0` | 1 | 36.2 (31.5) | 30.4 (26.4) |
| `v0.30.0` | 4 | 34.1 (119.3) | 28.2 (109.2) |
| `v0.30.0` | 8 | 31.5 (235.0) | 25.1 (179.3) |
| `v0.26.0` | 1 | 36.1 (31.6) | 30.2 (26.2) |
| `v0.26.0` | 4 | 34.1 (121.2) | 27.3 (105.6) |
| `v0.26.0` | 8 | 31.8 (237.3) | 23.1 (166.1) |

A model this small loses about 17% per stream to the stage hop and the pipeline bubble; the
point of two nodes is fitting what one node cannot hold. For a large model on this path, see
[Two-node pipeline parallel](../results/two-node-pipeline.md).
