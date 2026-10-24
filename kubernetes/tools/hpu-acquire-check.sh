#!/usr/bin/env bash
# hpu-acquire-check.sh: find Gaudi nodes that fail multi-process device acquire.
#
# Why: a node can advertise 8 allocatable habana.ai/gaudi, show every card idle and healthy
# in hl-smi, pass a single-process `torch.ones(1).to("hpu")`, and still fail every
# multi-rank vLLM start with
#
#     RuntimeError: synStatus=8 [Device not found] Device acquire failed.
#
# raised from the worker's device init, before any weight is read. Because it comes before
# model load it looks like a model or image bug. The discriminator is concurrency: vLLM
# starts one process per rank and they open their cards at the same moment. This script
# does the same, N processes at once in one pod.
#
# Two parts:
#   1. Leak audit (always, read-only, uses no cards): pods in phase Failed or Unknown that
#      still sit on a node with habana.ai/gaudi requests. The scheduler counts only
#      Running and Pending pods, so a node whose devices are still held by such pods can
#      look free. Confirm ownership before deleting them.
#   2. Live test (only on nodes you name): one pod per node requesting -p cards, which
#      starts that many processes that each open a card. Consumes the cards for up to a few
#      minutes.
#
# A live failure does not always mean a bad card. Check the node's kernel log first:
#   "Failed to create context -12" next to "page allocation failure: order:9" means host
#   memory fragmentation (opening a card needs a contiguous 2 MiB block); a retry after
#   memory compaction usually succeeds. See docs/kubernetes/health-and-failure.md.
#
# Usage:
#   IMAGE=<registry>/<image>:<tag> kubernetes/tools/hpu-acquire-check.sh              # leak audit only
#   IMAGE=<registry>/<image>:<tag> kubernetes/tools/hpu-acquire-check.sh -n NODE -p 4
#   IMAGE=<registry>/<image>:<tag> kubernetes/tools/hpu-acquire-check.sh -n NODE1,NODE2 -p 8
#
# Environment:
#   IMAGE          a vllm-gaudi (or any Habana PyTorch) image; required for the live test
#   NS             namespace for the test pods (default: llm)
#   RUNTIME_CLASS  RuntimeClass for Gaudi pods (default: habana)
#   HUGEPAGES_PER_CARD  hugepages-2Mi per process, in Gi (default: 4)
#   TIMEOUT        seconds to wait for each test pod (default: 300)
#
# Needs kubectl and python3 on the machine you run it from.
set -euo pipefail

NS="${NS:-llm}"
IMAGE="${IMAGE:-}"
RUNTIME_CLASS="${RUNTIME_CLASS:-habana}"
HUGEPAGES_PER_CARD="${HUGEPAGES_PER_CARD:-4}"
TIMEOUT="${TIMEOUT:-300}"
NODES=""
PROCS=4

usage() { sed -n '2,45p' "$0"; }

while getopts "n:p:i:h" opt; do
  case $opt in
    n) NODES="$OPTARG" ;;
    p) PROCS="$OPTARG" ;;
    i) IMAGE="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
  esac
done

echo "=== Leaked habana.ai/gaudi allocations (Failed/Unknown pods still bound to a node) ==="
kubectl get pods -A -o json | python3 -c '
import json, sys
from collections import defaultdict

leak = defaultdict(list)
for p in json.load(sys.stdin)["items"]:
    node = p["spec"].get("nodeName")
    if not node or p["status"].get("phase") not in ("Failed", "Unknown"):
        continue
    cards = 0
    for c in p["spec"]["containers"]:
        req = (c.get("resources") or {}).get("requests") or {}
        cards += int(req.get("habana.ai/gaudi", 0))
    if cards:
        leak[node].append((p["metadata"]["namespace"] + "/" + p["metadata"]["name"], cards))
if not leak:
    print("  none")
    raise SystemExit
for node in sorted(leak, key=lambda k: -sum(c for _, c in leak[k])):
    print(f"  {node}: {sum(c for _, c in leak[node])} card(s) across {len(leak[node])} pod(s)")
    for name, cards in leak[node][:3]:
        print(f"      {name} ({cards})")
    if len(leak[node]) > 3:
        print(f"      ... and {len(leak[node]) - 3} more")
print(f"\n  nodes affected: {len(leak)}")
'

if [[ -z "$NODES" ]]; then
  echo
  echo "No -n NODE given; skipping the live acquire test."
  exit 0
fi
[[ -n "$IMAGE" ]] || { echo "IMAGE is required for the live test" >&2; exit 1; }

echo
echo "=== Live multi-process acquire test (${PROCS} concurrent processes per node) ==="
for N in ${NODES//,/ }; do
  POD="hpu-acq-$(printf '%s' "$N" | md5sum | cut -c1-8)-$$"
  HP="$((PROCS * HUGEPAGES_PER_CARD))Gi"
  MEM="$((PROCS * 8))Gi"
  kubectl apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $POD
  namespace: $NS
  labels: {app.kubernetes.io/name: hpu-acquire-check}
spec:
  restartPolicy: Never
  runtimeClassName: $RUNTIME_CLASS
  hostIPC: true
  tolerations: [{key: habana, operator: Exists, effect: NoSchedule}]
  nodeSelector: {kubernetes.io/hostname: "$N"}
  containers:
    - name: t
      image: $IMAGE
      command:
        - /bin/bash
        - -c
        - |
          for i in \$(seq 1 $PROCS); do
            (python3 -c 'import torch, habana_frameworks.torch; torch.ones(1).to("hpu"); print("OK")' 2>&1 | tail -1) &
          done
          wait
      resources:
        requests: {habana.ai/gaudi: "$PROCS", hugepages-2Mi: $HP, memory: $MEM, cpu: "4"}
        limits: {habana.ai/gaudi: "$PROCS", hugepages-2Mi: $HP, memory: $MEM}
      volumeMounts: [{mountPath: /dev/shm, name: shm}]
  volumes:
    - name: shm
      emptyDir: {medium: Memory, sizeLimit: 4Gi}
EOF

  phase=""
  for _ in $(seq 1 $((TIMEOUT / 5))); do
    phase=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)
    [[ "$phase" == "Succeeded" || "$phase" == "Failed" ]] && break
    sleep 5
  done

  LOG=$(kubectl logs -n "$NS" "$POD" 2>/dev/null || true)
  OK=$(printf '%s\n' "$LOG" | grep -c '^OK$' || true)
  BAD=$(printf '%s\n' "$LOG" | grep -c 'synStatus' || true)
  if [[ "$phase" == "Pending" ]]; then
    VERDICT="NOT RUN (pod Pending: not enough free cards on the node?)"
  elif [[ "$OK" == "$PROCS" ]]; then
    VERDICT="HEALTHY"
  else
    VERDICT="** FAILED ** (check dmesg for ctx_init -12 / order:9 before blaming a card)"
  fi
  printf '  %-24s ok=%s/%s synStatus=%s  %s\n' "$N" "$OK" "$PROCS" "$BAD" "$VERDICT"
  kubectl delete pod -n "$NS" "$POD" --ignore-not-found --wait=false >/dev/null 2>&1 || true
done
