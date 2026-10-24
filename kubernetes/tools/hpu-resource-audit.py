#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit Gaudi workload manifests for per-card proportional requests and pod-spec basics.

Why proportional requests matter: a node has a fixed number of cards (8 on Gaudi 2) and a
fixed amount of CPU, memory and hugepages. If a 1-card pod requests a quarter of the node's
memory, the node can never hold eight 1-card pods, and its spare cards are stranded. Keep
every pod's requests at or below its share of the node, per card, so any mix of pods packs.

Default per-card request budget, sized for a Gaudi 2 node with about 160 CPUs, at least
512 GiB of memory and 32 GiB of 2 MiB hugepages. Derive your own from the node's allocatable
resources (`kubectl describe node`), leaving room for DaemonSets:

  cpu 16, memory 56Gi, hugepages-2Mi 4Gi per card
  a whole-node pod may request up to 400Gi of memory

Errors (exit 1):
  - cpu, memory or hugepages requests above the per-card budget
  - hugepages or habana.ai/gaudi requests that differ from limits (Kubernetes requires
    equal values for both)

Warnings (printed, exit 0 unless --strict):
  - no runtimeClassName, or hostIPC not true
  - no /dev/shm mount
  - HABANA_VISIBLE_DEVICES set in the spec (the device plugin sets it)
  - PT_HPU_LAZY_MODE=0 or --enforce-eager (both turn off HPU graphs)
  - a hostname pin (nodeName, or a nodeSelector/required node affinity on the hostname)
  - no preStop hook, or the default 30 s termination grace period

Usage:
  kubernetes/tools/hpu-resource-audit.py kubernetes/manifests
  kubernetes/tools/hpu-resource-audit.py --cpu 12 --memory 56Gi --whole-node-memory 400Gi path/ more.yaml
  kubernetes/tools/hpu-resource-audit.py --strict path/

Needs PyYAML.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

UNITS = {
    "": 1,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
}
GI = 2**30
POD_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "Job", "Pod", "ReplicaSet"}


def qty(value) -> float | None:
    """Parse a Kubernetes quantity into a plain number (bytes, or cores for cpu)."""
    if value is None:
        return None
    m = re.fullmatch(r"([0-9.]+)(m|k|M|G|T|Ki|Mi|Gi|Ti)?", str(value).strip())
    if not m:
        return None
    number, unit = float(m.group(1)), m.group(2) or ""
    if unit == "m":
        return number / 1000
    return number * UNITS[unit]


def pod_spec(doc: dict) -> dict | None:
    kind = doc.get("kind")
    if kind == "Pod":
        return doc.get("spec") or {}
    if kind == "CronJob":
        return ((doc.get("spec") or {}).get("jobTemplate") or {}).get("spec", {}).get("template", {}).get("spec")
    if kind in POD_KINDS:
        return ((doc.get("spec") or {}).get("template") or {}).get("spec")
    return None


def hostname_pinned(spec: dict) -> bool:
    if spec.get("nodeName"):
        return True
    if "kubernetes.io/hostname" in (spec.get("nodeSelector") or {}):
        return True
    required = (
        ((spec.get("affinity") or {}).get("nodeAffinity") or {})
        .get("requiredDuringSchedulingIgnoredDuringExecution", {})
        .get("nodeSelectorTerms", [])
    )
    for term in required:
        for expr in term.get("matchExpressions", []) + term.get("matchFields", []):
            if expr.get("key") in ("kubernetes.io/hostname", "metadata.name") and expr.get("operator") == "In":
                return True
    return False


def audit_container(c: dict, args) -> tuple[int, list[str], list[str]]:
    res = c.get("resources") or {}
    req, lim = res.get("requests") or {}, res.get("limits") or {}
    cards = int(lim.get("habana.ai/gaudi", req.get("habana.ai/gaudi", 0)) or 0)
    errors: list[str] = []
    warnings: list[str] = []
    if not cards:
        return 0, errors, warnings
    if str(req.get("habana.ai/gaudi", cards)) != str(lim.get("habana.ai/gaudi", cards)):
        errors.append("habana.ai/gaudi requests != limits")
    cpu, mem = qty(req.get("cpu")), qty(req.get("memory"))
    hp_req, hp_lim = qty(req.get("hugepages-2Mi")), qty(lim.get("hugepages-2Mi"))
    mem_cap = args.whole_node_memory if cards >= args.cards_per_node else args.memory * cards
    if cpu is not None and cpu > args.cpu * cards:
        errors.append(f"cpu request {cpu:g} > {args.cpu * cards:g}")
    if mem is not None and mem > mem_cap:
        errors.append(f"memory request {mem / GI:g}Gi > {mem_cap / GI:g}Gi")
    if hp_req is not None and hp_req > args.hugepages * cards:
        errors.append(f"hugepages-2Mi request {hp_req / GI:g}Gi > {args.hugepages * cards / GI:g}Gi")
    if (hp_req or hp_lim) and hp_req != hp_lim:
        errors.append("hugepages-2Mi requests != limits")
    if hp_req is None and hp_lim is None:
        warnings.append("no hugepages-2Mi request")
    env = {e.get("name"): str(e.get("value", "")) for e in c.get("env") or []}
    if "HABANA_VISIBLE_DEVICES" in env:
        warnings.append("HABANA_VISIBLE_DEVICES is set; leave it to the device plugin")
    if env.get("PT_HPU_LAZY_MODE") == "0":
        warnings.append("PT_HPU_LAZY_MODE=0 turns off lazy mode and HPU graphs")
    if "--enforce-eager" in " ".join(str(x) for x in (c.get("command") or []) + (c.get("args") or [])):
        warnings.append("--enforce-eager turns off HPU graphs")
    mounts = {m.get("mountPath") for m in c.get("volumeMounts") or []}
    if "/dev/shm" not in mounts:
        warnings.append("no /dev/shm mount")
    if not ((c.get("lifecycle") or {}).get("preStop")):
        warnings.append("no preStop hook (scale-down and rollouts will cut in-flight streams)")
    return cards, errors, warnings


def audit_file(path: Path, args) -> tuple[int, int]:
    n_err = n_warn = 0
    try:
        docs = list(yaml.safe_load_all(path.read_text()))
    except yaml.YAMLError as exc:
        print(f"{path}: parse error: {exc}")
        return 1, 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        spec = pod_spec(doc)
        if not spec:
            continue
        name = f"{doc.get('kind')}/{(doc.get('metadata') or {}).get('name', '?')}"
        pod_warnings: list[str] = []
        any_cards = False
        for c in spec.get("containers") or []:
            cards, errors, warnings = audit_container(c, args)
            if not cards:
                continue
            any_cards = True
            for msg in errors:
                print(f"ERROR {path} {name} [{c.get('name')}] {cards} card(s): {msg}")
            for msg in warnings:
                print(f"warn  {path} {name} [{c.get('name')}]: {msg}")
            n_err += len(errors)
            n_warn += len(warnings)
        if not any_cards:
            continue
        if not spec.get("runtimeClassName"):
            pod_warnings.append("no runtimeClassName (Gaudi pods normally run under the habana RuntimeClass)")
        if spec.get("hostIPC") is not True:
            pod_warnings.append("hostIPC is not true (multi-card starts can fail at device acquire)")
        if hostname_pinned(spec):
            pod_warnings.append("pinned to a hostname; pins go stale when nodes fail and block autoscaling")
        if spec.get("terminationGracePeriodSeconds", 30) <= 30:
            pod_warnings.append("default 30 s termination grace period leaves no time to drain")
        for msg in pod_warnings:
            print(f"warn  {path} {name}: {msg}")
        n_warn += len(pod_warnings)
    return n_err, n_warn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="YAML files or directories (searched recursively)")
    ap.add_argument("--cpu", type=float, default=16, help="cpu request budget per card (cores)")
    ap.add_argument("--memory", default="56Gi", help="memory request budget per card")
    ap.add_argument("--hugepages", default="4Gi", help="hugepages-2Mi request budget per card")
    ap.add_argument("--whole-node-memory", default="400Gi", help="memory request cap for a whole-node pod")
    ap.add_argument("--cards-per-node", type=int, default=8)
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    args = ap.parse_args()
    args.memory = qty(args.memory)
    args.hugepages = qty(args.hugepages)
    args.whole_node_memory = qty(args.whole_node_memory)

    files: list[Path] = []
    for p in map(Path, args.paths):
        files += sorted(p.rglob("*.y*ml")) if p.is_dir() else [p]
    n_err = n_warn = 0
    for f in files:
        e, w = audit_file(f, args)
        n_err += e
        n_warn += w
    print(f"{len(files)} file(s): {n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err or (args.strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
