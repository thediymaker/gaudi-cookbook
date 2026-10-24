#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kernel off vs on: compare two run_gates.sh --out directories, one from a server with DSV41_MXK=0 and one with
DSV41_MXK=1 (same image, node and recipe otherwise).

PASS: every gate passed in both runs, the perplexities agree within --tol (default 0.01), and 1-stream decode is
faster with the kernels on.

  python3 kernel_ab.py OFF_DIR ON_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def lines(d: str, gate: str) -> list[str]:
    try:
        with open(os.path.join(d, f"{gate}.log")) as f:
            return f.read().splitlines()
    except OSError:
        return []


def passed(d: str) -> bool:
    gates = ("speed", "ppl", "decode-tf", "needles", "isolation", "structured")
    return all(any(ln.endswith(" PASS") or " PASS " in ln for ln in lines(d, g)) for g in gates)


def first_json(d: str, gate: str, key: str) -> dict:
    for ln in lines(d, gate):
        if ln.startswith("{") and key in ln:
            return json.loads(ln)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args()
    out = {}
    for name, d in (("off", a.off), ("on", a.on)):
        sp = {}
        for ln in lines(d, "speed"):
            if ln.startswith("{"):
                j = json.loads(ln)
                sp[j["streams"]] = j
        out[name] = {
            "all_gates_pass": passed(d),
            "ppl": first_json(d, "ppl", "server_ppl").get("server_ppl"),
            "tps_1": (sp.get(1) or {}).get("mean_per_stream"),
            "tps_4_aggregate": (sp.get(4) or {}).get("aggregate_decode_tps"),
        }
    off, on = out["off"], out["on"]
    ok = bool(off["all_gates_pass"] and on["all_gates_pass"])
    ok = ok and None not in (off["ppl"], on["ppl"], off["tps_1"], on["tps_1"])
    ok = ok and abs(on["ppl"] - off["ppl"]) <= a.tol and on["tps_1"] > off["tps_1"]
    print(json.dumps(out))
    ratio = f"{on['tps_1'] / off['tps_1']:.2f}x" if off["tps_1"] and on["tps_1"] else "n/a"
    print(f"KERNEL-AB {'PASS' if ok else 'FAIL'} 1-stream {off['tps_1']} -> {on['tps_1']} tok/s ({ratio})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
