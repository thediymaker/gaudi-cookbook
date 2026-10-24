# Start-up and warm-up

A model server's cold start decides how fast the fleet can react: how quickly an autoscaler
can follow a burst, how long a rollout takes, and how much a node failure costs. On Gaudi a
cold start has five parts, and each needs a different fix.

| Phase | What happens | Typical fix when slow |
| --- | --- | --- |
| Image pull | Multi-GB image to the node | `IfNotPresent`, few image tags, optional pre-pull DaemonSet |
| Weight fetch | Download from the model hub | Shared store, prefetch Job, offline serving pods |
| Weight load | Read shards, copy to the cards | Storage throughput, CPU (no CPU limit), fewer and larger tensors |
| Graph compile and warm-up | Build HPU graphs for each bucket shape | Recipe cache keyed to the config |
| First requests | Shapes not compiled yet compile live | Warm-up before the pod is Ready |

Read the container log timestamps for each phase before changing anything.

## Model store

**Use one shared copy of each model, not one copy per node.** A per-node cache downloads
the model again on every node a pod lands on. With autoscaling, bin-packing and node
failures, pods move often, so the same weights end up on many nodes: one fleet measured
about 8 times the unique bytes on disk. Each copy is also another trip through the hub's
CDN and its rate limits, and node disks fill as models come and go.

A ReadWriteMany volume (NFS, CephFS, a parallel file system) in the standard Hugging Face
cache layout serves every node. Pods mount it read-only, set `HF_HUB_CACHE` to it and run
with `HF_HUB_OFFLINE=1`, so a serving pod can never stall on, or be rate limited by, the
hub. A well-provisioned NFS server has served weights directly, with no local copy, at
several hundred MB/s per node.

The trade-offs to accept: the store is shared infrastructure, its throughput matters, and
it must be mounted on every node.

**Prefetch as its own Job.**
[`model-store.yaml`](https://github.com/thediymaker/gaudi-cookbook/blob/main/kubernetes/manifests/model-store.yaml)
downloads a model, then verifies it offline. Doing this before the Deployment separates "the
download failed" from "the engine failed".

**Verify completeness, not existence.**

- An interrupted download leaves `*.incomplete` files or zero-byte placeholders behind.
  Check that every shard named in `model.safetensors.index.json` is present and non-empty.
- Copying a node-local cache into the store can copy a partial snapshot, because a pod
  that ran online fetched only the files it needed. Newer `huggingface_hub` releases refuse
  incomplete snapshots offline (`IncompleteSnapshotError`), so the copy works today and
  breaks on the next image bump. Complete it with `snapshot_download(repo)` online, then
  check `snapshot_download(repo, local_files_only=True)`.
- Completing a snapshot can move `refs/main` to a newer commit. Pin a revision where a
  model must not change.

**If pods download on start, lock and stay out of the hub's way.**

- When many replicas start together, exactly one should download and the rest wait. Use an
  atomic `mkdir` lock with a heartbeat file, not `flock`: over NFSv3, `flock` can silently
  become a node-local lock and let every node download at once.
- Even with a warm cache, vLLM's start-up config resolution can probe the hub for a file
  that does not exist in the repo (an optional config file). A missing file can never be
  cached, so the probe goes to the network on every start. The hub's rate limit is shared
  across every pod behind the same egress address and token, and a short retry budget turns
  a routine 429 into a crash loop that makes more calls. Serving offline from the store
  removes this; otherwise widen the retry budget with backoff and jitter.
- `HF_HUB_OFFLINE=1` with a per-node cache hard-fails any pod that lands on a node that does
  not have the model yet. Offline mode belongs with a shared store.

**Fail fast when the store is not there.** A `hostPath` mount of the store's mount point
cannot tell the real mount from an empty local directory with the same path. A pod that
lands on a node where the mount failed would silently build a private copy on local disk.
Check the mount (filesystem type, or a sentinel file) and the model's completeness at the
top of the container command, and exit with a clear message. The template's
`snapshot_download(..., local_files_only=True)` check does the second part.

## Concurrent loads

A capable shared store handles many large loads at once. Do not serialize rollouts or hold
one model's start for another's by default. When a load is slow, measure where the time goes
before blaming storage:

```bash
# Bytes each rank has actually read, compared with its share of the weights
cat /proc/<pid>/io | grep read_bytes
# A host thread pegged at 100% (per-tensor dispatch to the device)
top -H -p <pid>
# CPU throttling of the pod's cgroup
cat /sys/fs/cgroup/cpu.stat        # nr_throttled, throttled_usec
# Raw read speed of one shard from the store, bypassing the page cache
dd if=<shard>.safetensors of=/dev/null bs=16M iflag=direct status=progress
```

A real example: a slow multi-card load looked like storage, but each rank read only its own
share at a fraction of the store's raw speed. The Habana host thread of every worker was
pegged at 100% dispatching tens of thousands of small per-expert tensors, and the pod's CPU
quota was being throttled. Storage was fine; CPU and tensor count were the bottleneck.

Models with many small tensors (large MoE checkpoints) load faster when the checkpoint is
repacked into fewer, larger tensors, and they need CPU headroom while loading.

## Recipe cache

`PT_HPU_RECIPE_CACHE_CONFIG=<dir>,<delete_on_start>,<max_MB>` keeps compiled HPU graphs
("recipes") on disk so the next start on that node skips compilation. The third field is a
size cap in MB, not the model length.

- **Key the directory to the exact configuration**: model, image, tensor parallel size,
  `--max-model-len`, `--max-num-seqs`, block size and bucket settings. Recipes compiled for
  a different bucket geometry can load as valid and crash the engine, even on a short
  prompt. The template builds the path from a `RECIPE_KEY` variable for this reason.
- With a correctly keyed directory, `delete_on_start=False` reuses the cache across
  restarts. `True` clears it on every start: safe, but every start compiles from scratch.
- Keep it on **node-local** disk (`hostPath` or a local volume). Compiled graphs are
  specific to the node's software stack, and the network-filesystem mode has failed in
  testing.
- The cache is per node, so a pod on a fresh node compiles again. Its first long prompts
  are slow until the shapes it sees are compiled.

## Warm-up

`VLLM_SKIP_WARMUP=false` makes vLLM compile the configured bucket shapes before the server
reports healthy. `true` starts faster, but every shape not yet in the recipe cache compiles
on live traffic, and while one request compiles, every other stream on that pod waits.

- Keep warm-up on where start time allows it, or where the recipe cache is usually warm.
- Skip it where start time blocks autoscaling, and accept slow first requests per shape.
- Some models fail during warm-up on some releases; then skipping it is the only option
  until that is fixed.

**Warm a pod before it takes traffic, never after.** Requests sent from inside a pod
(`kubectl exec ... curl localhost`) are invisible to the router. A least-request router
counts only its own in-flight requests, so it keeps sending users to the pod that is busy
compiling, and they stall. Freshly started pods look idle, so such routers prefer them;
adding replicas just before an event makes this worse unless the new pods are warmed before
they are Ready.

To warm with real requests before readiness, set `STALL_PROBE_READY_FILE` for the
[stall probe](health-and-failure.md#stall-aware-readiness) and create that file after the
warm-up requests:

```bash
set -euo pipefail
vllm serve "${MODEL}" --port 8000 <flags> &
pid=$!
trap 'kill -TERM "$pid"' TERM          # forward SIGTERM so the drain still works
until curl -sf localhost:8000/health >/dev/null; do sleep 5; done
python3 /opt/warmup/warmup.py           # your representative prompts, to localhost
touch /tmp/warm                         # STALL_PROBE_READY_FILE=/tmp/warm
wait "$pid"
```

## Timeouts during start-up

- **startupProbe**: size `failureThreshold x periodSeconds` to the slowest cold start you
  have measured for that model, with margin. Large models can take well over an hour on a
  cold node.
- **`progressDeadlineSeconds`** on the Deployment: above the slowest start, so a healthy
  rollout is not marked failed mid-load.
- **Engine deadlines**: a first compile inside one step can exceed vLLM's multi-card
  execute deadline; raise `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`.
- **Do not raise the runtime's no-progress timeout** to hide hangs. With a much higher
  value, a hung engine stayed wedged indefinitely with `/health` returning 200, which is
  worse than crashing and restarting.
