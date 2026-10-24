# SPDX-License-Identifier: Apache-2.0
"""Safetensors reading, shard writing and checking shared by the two GLM-5.x converters."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MTP_LAYER = 78
AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "LICENSE",
)
_EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$")


def parse_layers(spec: str) -> set[int]:
    """'3-21,78' -> {3, ..., 21, 78}"""
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


class Checkpoint:
    """Lazy tensor reader over every *.safetensors file of a directory."""

    def __init__(self, path: str):
        self.path = path
        self.shard_of: dict[str, str] = {}
        self._open: dict[str, object] = {}
        for sp in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(sp, framework="pt") as f:
                for n in f.keys():  # noqa: SIM118 (safe_open is not a dict)
                    self.shard_of[n] = sp
        if not self.shard_of:
            raise SystemExit(f"no safetensors under {path}")

    def _h(self, name: str):
        sp = self.shard_of[name]
        if sp not in self._open:
            self._open[sp] = safe_open(sp, framework="pt")
        return self._open[sp]

    def __contains__(self, name: str) -> bool:
        return name in self.shard_of

    def get(self, name: str) -> torch.Tensor:
        return self._h(name).get_tensor(name)

    def dtype(self, name: str) -> str:
        return str(self._h(name).get_slice(name).get_dtype())

    def expert_layers(self) -> dict[int, int]:
        """{layer: number of routed experts} from the per-expert tensor names."""
        n_exp: dict[int, int] = defaultdict(int)
        for n in self.shard_of:
            m = _EXPERT.match(n)
            if m:
                n_exp[int(m.group(1))] = max(n_exp[int(m.group(1))], int(m.group(2)) + 1)
        return dict(n_exp)


def fused_names(layer: int) -> tuple[str, str, str, str]:
    # The full name, routed_experts included: deepseek_v2's loader looks fused names up directly and skips a short
    # name without a warning (the layer would load with zero experts).
    p = f"model.layers.{layer}.mlp.experts.routed_experts"
    return f"{p}.w13_weight", f"{p}.w13_weight_scale", f"{p}.w2_weight", f"{p}.w2_weight_scale"


class Sink:
    """Where converted tensors go: shards in a directory, or a comparison against an existing checkpoint."""

    def __init__(self, dst: str | None, tag: str, shard_gb: float, verify: str | None):
        self.dst, self.tag = dst, tag
        self.limit = int(shard_gb * 2**30)
        self.ref = Checkpoint(verify) if verify else None
        self.buf: dict[str, torch.Tensor] = {}
        self.bytes = 0
        self.idx = 0
        self.written: dict[str, str] = {}
        self.same = 0
        self.diff: list[str] = []
        self.seen: set[str] = set()

    def emit(self, name: str, t: torch.Tensor) -> None:
        t = t.contiguous()
        self.seen.add(name)
        if self.ref is not None:
            ok = name in self.ref
            if ok:
                r = self.ref.get(name)
                same_bytes = torch.equal(r.reshape(-1).view(torch.uint8), t.reshape(-1).view(torch.uint8))
                ok = r.dtype == t.dtype and r.shape == t.shape and same_bytes
            if ok:
                self.same += 1
            else:
                self.diff.append(name)
                print(f"[verify] DIFFERS: {name}", flush=True)
            return
        self.buf[name] = t
        self.bytes += t.numel() * t.element_size()
        if self.bytes >= self.limit:
            self.flush()

    def flush(self) -> None:
        if self.ref is not None or not self.buf:
            return
        fn = f"model-{self.tag}-{self.idx:05d}.safetensors"
        save_file(self.buf, os.path.join(self.dst, fn), metadata={"format": "pt"})
        self.written.update(dict.fromkeys(self.buf, fn))
        print(f"[convert] wrote {fn} ({self.bytes / 1e9:.1f} GB, {len(self.buf)} tensors)", flush=True)
        self.buf, self.bytes = {}, 0
        self.idx += 1

    def close(self) -> int:
        """Flush and write this worker's manifest; in verify mode print the verdict. Returns an exit code."""
        if self.ref is not None:
            verdict = "IDENTICAL" if not self.diff else f"{len(self.diff)} DIFFER"
            print(f"[verify] {self.tag}: {self.same} tensors byte-identical, {verdict}", flush=True)
            return 0 if not self.diff else 1
        self.flush()
        with open(os.path.join(self.dst, f"_manifest-{self.tag}.json"), "w") as f:
            json.dump(self.written, f)
        return 0


def finalize(src: str, dst: str, expected_layers: set[int], mtp_layers: int) -> None:
    """Merge the workers' manifests into the index, write config.json and copy the tokenizer files."""
    written: dict[str, str] = {}
    for mf in sorted(glob.glob(os.path.join(dst, "_manifest-*.json"))):
        with open(mf) as f:
            written.update(json.load(f))
    if not os.path.exists(os.path.join(dst, "_manifest-rest.json")):
        raise SystemExit("the --rest worker has not finished: the non-expert tensors are missing")
    got = {int(k.split(".")[2]) for k in written if k.endswith(".w13_weight")}
    if got != expected_layers:
        raise SystemExit(f"expert layers missing from the output: {sorted(expected_layers - got)}")
    total = sum(os.path.getsize(os.path.join(dst, f)) for f in set(written.values()))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": written}, f, indent=1)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    # vllm-gaudi accepts "gpt_oss_mxfp4" (not "mxfp4"); its MXFP4 MoE method also serves non-gpt-oss MoE layers
    cfg["quantization_config"] = {"quant_method": "gpt_oss_mxfp4"}
    cfg["num_nextn_predict_layers"] = mtp_layers
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for aux in AUX_FILES:
        if os.path.exists(os.path.join(src, aux)):
            shutil.copy2(os.path.join(src, aux), os.path.join(dst, aux))
    print(f"[convert] finalized {dst}: {total / 1e9:.1f} GB, {len(written)} tensors, {len(got)} expert layers")
