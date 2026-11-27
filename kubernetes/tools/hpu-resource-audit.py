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
  - a hostname pin (nodeName, or a nodeSelector/required node affinity on the hostname)
  - no preStop hook on a container that serves HTTP (declares a port or an HTTP probe), or the
    default 30 s termination grace period

Info (printed, never fails a run):
  - PT_HPU_LAZY_MODE=0 (compile mode) or --enforce-eager: right when the recipe's verified mode is
    one of them, wrong otherwise; only the recipe knows, so check the manifest against it

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


def pod_specs(doc: dict) -> list[tuple[str, dict]]:
    """(label, pod spec) for every pod template in the document; the label is empty when there is only one."""
    kind, spec = doc.get("kind"), doc.get("spec") or {}
    if kind == "Pod":
        found = [("", spec)]
    elif kind == "CronJob":
        found = [("", ((spec.get("jobTemplate") or {}).get("spec") or {}).get("template", {}).get("spec"))]
    elif kind == "LeaderWorkerSet":
        lwt = spec.get("leaderWorkerTemplate") or {}
        found = [(role, (lwt.get(f"{role}Template") or {}).get("spec")) for role in ("leader", "worker")]
    elif kind in POD_KINDS:
        found = [("", (spec.get("template") or {}).get("spec"))]
    else:
        found = []
    return [(label, s) for label, s in found if s]


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


def audit_container(c: dict, args) -> tuple[int, list[str], list[str], list[str]]:
    res = c.get("resources") or {}
    req, lim = res.get("requests") or {}, res.get("limits") or {}
    cards = int(lim.get("habana.ai/gaudi", req.get("habana.ai/gaudi", 0)) or 0)
    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []
    if not cards:
        return 0, errors, warnings, infos
    if str(req.get("habana.ai/gaudi", cards)) != str(lim.get("habana.ai/gaudi", cards)):
        errors.append("habana.ai/gaudi requests != limits")
    # Kubernetes sets a missing request to the limit, so a container with limits only still requests that much.
    cpu, mem = qty(req.get("cpu", lim.get("cpu"))), qty(req.get("memory", lim.get("memory")))
    hp_lim = qty(lim.get("hugepages-2Mi"))
    hp_req = qty(req.get("hugepages-2Mi", lim.get("hugepages-2Mi")))
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
    # Some recipes are verified in compile mode or eager; the manifest alone cannot tell, so these are notes.
    if env.get("PT_HPU_LAZY_MODE") == "0":
        infos.append("PT_HPU_LAZY_MODE=0 (compile mode); check it is the recipe's verified mode")
    if "--enforce-eager" in " ".join(str(x) for x in (c.get("command") or []) + (c.get("args") or [])):
        infos.append("--enforce-eager (no HPU graphs); check it is the recipe's verified mode")
    mounts = {m.get("mountPath") for m in c.get("volumeMounts") or []}
    if "/dev/shm" not in mounts:
        warnings.append("no /dev/shm mount")
    # A container with no port and no HTTP probe (a headless pipeline stage) takes no requests to drain.
    probes = ("startupProbe", "readinessProbe", "livenessProbe")
    serves_http = bool(c.get("ports")) or any((c.get(p) or {}).get("httpGet") for p in probes)
    if serves_http and not ((c.get("lifecycle") or {}).get("preStop")):
        warnings.append("no preStop hook (scale-down and rollouts will cut in-flight streams)")
    return cards, errors, warnings, infos


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
        for label, spec in pod_specs(doc):
            name = f"{doc.get('kind')}/{(doc.get('metadata') or {}).get('name', '?')}" + (f" {label}" if label else "")
            pod_warnings: list[str] = []
            any_cards = False
            for c in spec.get("containers") or []:
                cards, errors, warnings, infos = audit_container(c, args)
                if not cards:
                    continue
                any_cards = True
                for msg in errors:
                    print(f"ERROR {path} {name} [{c.get('name')}] {cards} card(s): {msg}")
                for msg in warnings:
                    print(f"warn  {path} {name} [{c.get('name')}]: {msg}")
                for msg in infos:
                    print(f"info  {path} {name} [{c.get('name')}]: {msg}")
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
        if not p.exists():
            ap.error(f"no such file or directory: {p}")
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
