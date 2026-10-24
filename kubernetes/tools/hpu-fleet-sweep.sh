#!/usr/bin/env bash
# hpu-fleet-sweep.sh: per-card health census of a Gaudi fleet, read-only.
#
# Answers "which cards are bad, on which nodes, and what kind of bad" so a hardware visit
# brings the right parts. For every Gaudi node it compares what Kubernetes believes with
# what the node itself sees:
#
#   k8s capacity / allocatable   from the Node object (lags reality; can be stale)
#   pci        Gaudi devices on the PCIe bus (vendor 0x1da3 in /sys/bus/pci/devices)
#   bound      devices bound to the habanalabs driver
#   hlsmi      devices hl-smi can use
#   dead       PCI addresses that logged "Fatal error during habanalabs device init"
#
# and sorts the faults into buckets that need different fixes:
#
#   MISSING    fewer devices on the bus than EXPECTED_CARDS. A cold power cycle (full power
#              off, not a warm reboot) often brings a card back; otherwise reseat or swap.
#   DEAD       on the bus but failed driver init. These do not recover by power cycling.
#   UNHEALTHY  enumerated and bound, but the device plugin withholds them
#              (allocatable < capacity).
#   ZERO       no Gaudi device on the bus at all. Usually a node-level fault (baseboard,
#              PCIe complex, power), not EXPECTED_CARDS separate card failures; counting it
#              as that many dead cards overstates the spares you need.
#   STALE      the node shows Gaudi capacity in Kubernetes but has no cards. Removing a
#              device plugin does not clear an extended resource from the Node status.
#
# Read-only: no cordon, drain, reboot or write on any node.
#
# Access: SSH to each node's InternalIP (from the Node object). Needs passwordless SSH as
# SSH_USER, and kubectl on the machine you run it from.
#
# Usage:
#   kubernetes/tools/hpu-fleet-sweep.sh                       # all nodes matching NODE_SELECTOR
#   kubernetes/tools/hpu-fleet-sweep.sh --nodes node-a,node-b
#   kubernetes/tools/hpu-fleet-sweep.sh --jobs 32 --out sweep.tsv
#
# Environment:
#   NODE_SELECTOR   label selector for Gaudi nodes (default: habana.ai/gaudi=present)
#   EXPECTED_CARDS  cards per healthy node (default: 8)
#   SSH_USER        default: root
#   JOBS            parallel SSH sessions (default: 16)
set -uo pipefail

NODE_SELECTOR="${NODE_SELECTOR:-habana.ai/gaudi=present}"
EXPECTED_CARDS="${EXPECTED_CARDS:-8}"
SSH_USER="${SSH_USER:-root}"
JOBS="${JOBS:-16}"
OUT=""
NODES=""

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes) NODES="${2//,/ }"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$OUT" ]] || OUT="hpu-fleet-sweep-$(date +%Y%m%d-%H%M%S).tsv"

# Everything needed from one SSH round trip. Runs on the node.
# shellcheck disable=SC2016
PROBE='
pci=0
for d in /sys/bus/pci/devices/*; do
  [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1da3" ] || continue
  pci=$((pci + 1))
done
bound=$(ls /sys/bus/pci/drivers/habanalabs/ 2>/dev/null | grep -cE "^[0-9a-f]{4}:")
smi=$(hl-smi -Q index -f csv,noheader 2>/dev/null | grep -c .)
dead=$(dmesg 2>/dev/null | grep "Fatal error during habanalabs device init" \
  | grep -oE "[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]" | sort -u | tr "\n" " ")
up=$(awk "{printf \"%.1f\", \$1/86400}" /proc/uptime)
echo "PCI=${pci}|BOUND=${bound}|SMI=${smi}|DEAD=${dead}|UP=${up}"
'

field() { sed -n "s/.*$1=\([^|]*\).*/\1/p" <<<"$2"; }

probe_node() {
  local n="$1" ip cap alloc cordoned out pci bound smi dead up
  ip=$(kubectl get node "$n" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)
  cap=$(kubectl get node "$n" -o jsonpath='{.status.capacity.habana\.ai/gaudi}' 2>/dev/null)
  alloc=$(kubectl get node "$n" -o jsonpath='{.status.allocatable.habana\.ai/gaudi}' 2>/dev/null)
  cordoned=$(kubectl get node "$n" -o jsonpath='{.spec.unschedulable}' 2>/dev/null)
  out=""
  [[ -n "$ip" ]] && out=$(timeout 30 ssh -o ConnectTimeout=6 -o BatchMode=yes -o LogLevel=ERROR \
    "${SSH_USER}@${ip}" "$PROBE" 2>/dev/null)
  if [[ -z "$out" ]]; then
    printf '%s\t%s\t%s\t%s\tUNREACHABLE\t\t\t\t\t\t\n' "$n" "${cap:-0}" "${alloc:-0}" "${cordoned:-false}"
    return
  fi
  pci=$(field PCI "$out"); bound=$(field BOUND "$out"); smi=$(field SMI "$out")
  dead=$(field DEAD "$out"); up=$(field UP "$out")
  local missing=$((EXPECTED_CARDS - pci)); ((missing < 0)) && missing=0
  printf '%s\t%s\t%s\t%s\tOK\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "${cap:-0}" "${alloc:-0}" "${cordoned:-false}" \
    "$pci" "$bound" "$smi" "$missing" "${dead:--}" "$up"
}
export -f probe_node field
export PROBE EXPECTED_CARDS SSH_USER

if [[ -z "$NODES" ]]; then
  NODES=$(kubectl get nodes -l "$NODE_SELECTOR" -o name 2>/dev/null | sed 's|^node/||')
fi
[[ -n "$NODES" ]] || die "no nodes found (NODE_SELECTOR=$NODE_SELECTOR)"
COUNT=$(wc -w <<<"$NODES")
echo "Gaudi fleet sweep: ${COUNT} nodes, ${JOBS} parallel, read-only"

printf 'node\tk8s_cap\tk8s_alloc\tcordoned\tstatus\tpci\tbound\thlsmi\tmissing\tdead_bdf\tuptime_d\n' >"$OUT"
# shellcheck disable=SC2086
printf '%s\n' $NODES | xargs -P "$JOBS" -I{} bash -c 'probe_node "$1"' _ {} >>"$OUT"
echo "raw TSV: $OUT"
echo

awk -F'\t' -v exp="$EXPECTED_CARDS" '
NR == 1 { next }
{
  node = $1; cap = $2 + 0; alloc = $3 + 0; st = $5; pci = $6 + 0; smi = $8 + 0; miss = $9 + 0; dead = $10
  total++
  if (st == "UNREACHABLE") { unreach = unreach " " node; nu++; next }
  if (pci == 0) {
    if (cap > 0) { stale = stale " " node; ns++ } else { zero = zero " " node; nz++ }
    next
  }
  if (miss > 0) { mcards += miss; mlist = mlist sprintf("  %-28s %d missing of %d\n", node, miss, exp); nm++ }
  if (dead != "-" && dead != "") {
    nd++; dcards += split(dead, tmp, " "); dlist = dlist sprintf("  %-28s %s\n", node, dead)
  }
  if (cap > alloc) { ucards += cap - alloc; ulist = ulist sprintf("  %-28s %d withheld\n", node, cap - alloc); nuh++ }
  if (pci == exp && smi == exp && cap == alloc) healthy++
  cards += alloc
}
END {
  printf "SUMMARY  %d nodes | %d fully healthy | %d allocatable cards\n", total, healthy, cards
  printf "  MISSING from PCIe bus : %d cards on %d nodes (cold power cycle, then reseat or swap)\n", mcards, nm
  printf "  DEAD at driver init   : %d cards on %d nodes (swap; power cycling does not help)\n", dcards, nd
  printf "  UNHEALTHY (withheld)  : %d cards on %d nodes\n", ucards, nuh
  printf "  ZERO cards on bus     : %d nodes (node-level fault, not a card swap)\n", nz
  printf "  STALE k8s capacity    : %d nodes (no cards, but the Node still advertises them)\n", ns
  printf "  UNREACHABLE over SSH  : %d nodes\n", nu
  if (nd) printf "\nDEAD cards (PCI address):\n%s", dlist
  if (nm) printf "\nMISSING cards:\n%s", mlist
  if (nuh) printf "\nUNHEALTHY cards:\n%s", ulist
  if (nz) printf "\nZERO cards:%s\n", zero
  if (ns) printf "\nSTALE capacity:%s\n", stale
  if (nu) printf "\nUNREACHABLE:%s\n", unreach
}' "$OUT"
