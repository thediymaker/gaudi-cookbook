#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""hpu-graph-lru: a bounded HPU graph cache that evicts least-recently-used graphs.

In lazy mode the runner wraps the model with Habana's wrap_in_hpu_graph and passes no cap, so every
new (batch, blocks) shape keeps its captured graph for the life of the process and device memory
grows with the number of shapes seen. Habana's own cap (max_graphs) is first come, first kept: the
first N shapes stay forever and every later shape runs uncaptured, op by op from the host, which is
several times slower at decode.

With VLLM_HPU_MAX_GRAPHS=N the runner's graph wrapper gets a cache of at most N graphs that evicts
instead of freezing. A miss with room captures as before. A miss when the cache is full is counted;
on the shape's VLLM_HPU_GRAPH_ADMIT_AFTER-th miss (default 2) the least-recently-used graph is
evicted (prompt graphs before decode graphs, and a prompt shape never evicts a decode graph) and
the shape is captured. One-off prompt shapes stay uncaptured instead of churning the cache. An
evicted graph is released after a device synchronize. Every tensor-parallel rank sees the same
shape sequence and the policy uses nothing else, so all ranks capture and evict on the same steps.
Rank 0 prints a "[hpu-graph-lru]" stats line at most every VLLM_HPU_GRAPH_STATS_INTERVAL seconds
(default 300, 0 = off) when the counters changed.

VLLM_HPU_MAX_GRAPHS unset: the stock unbounded wrapper, so applying the fix alone changes nothing.

Usage: apply.py [--check] [--root SITE_PACKAGES]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

FIX_ID = "hpu-graph-lru"
MARKER = f"[gaudi-fix:{FIX_ID}]"

# Module-level code inserted above the wrapper. Kept free of the runner's own imports so the test
# can run it on its own.
LRU_CODE = r'''
import collections as _hgl_collections
import functools as _hgl_functools
import os as _hgl_os
import sys as _hgl_sys
import time as _hgl_time

_HGL_PHASES = ("prompt", "decode")
_HGL_MISS_TABLE_MAX = 4096


class HpuGraphLRU(_hgl_collections.OrderedDict):
    """The `cache` mapping Habana's wrapped_hpugraph_forward reads and writes.

    That function only calls get(), len(), __setitem__ and values() on it. Room is made in get()
    on a miss, before it compares len() with max_graphs.
    """

    def __init__(self, max_graphs, admit_after=2, destroy=None):
        super().__init__()
        if max_graphs < 1:
            raise ValueError("VLLM_HPU_MAX_GRAPHS must be >= 1")
        self.max_graphs = max_graphs
        self.admit_after = max(1, admit_after)
        self.current_phase = "decode"
        self._phase_of = {}
        self._misses = _hgl_collections.Counter()
        self._destroy = destroy or (lambda entry: None)
        self.stats = _hgl_collections.Counter()

    def get(self, key, default=None):
        entry = super().get(key)
        if entry is not None:
            self.move_to_end(key)
            self.stats["hit", self.current_phase] += 1
            return entry
        if len(self) >= self.max_graphs and not self._admit(key):
            self.stats["uncaptured", self.current_phase] += 1
        return default

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._phase_of[key] = self.current_phase
        self.stats["captured", self.current_phase] += 1

    def clear(self):
        for key in list(self):
            self._evict(key)

    def _admit(self, key):
        """A new shape while full: evict a victim once the shape has missed admit_after times."""
        self._misses[key] += 1
        if self._misses[key] < self.admit_after:
            return False
        victim = self._victim()
        if victim is None:
            return False
        self._evict(victim)
        del self._misses[key]
        if len(self._misses) > _HGL_MISS_TABLE_MAX:
            self._misses.clear()
        return True

    def _victim(self):
        """Least-recently-used prompt graph, else (for a decode shape only) the least-recently-used graph."""
        for key in self:  # least recently used first
            if self._phase_of.get(key) == "prompt":
                return key
        if self.current_phase == "decode":
            return next(iter(self), None)
        return None

    def _evict(self, key):
        entry = super().pop(key)
        phase = self._phase_of.pop(key, "decode")
        self._destroy(entry)
        self.stats["evicted", phase] += 1

    def held(self):
        counts = _hgl_collections.Counter(self._phase_of.values())
        return {p: counts.get(p, 0) for p in _HGL_PHASES}

    def summary(self):
        held = self.held()
        parts = ["graphs %d/%d (prompt %d, decode %d)" % (len(self), self.max_graphs, held["prompt"], held["decode"])]
        for what in ("hit", "captured", "evicted", "uncaptured"):
            parts.append("%s prompt %d decode %d" % (what, self.stats[what, "prompt"], self.stats[what, "decode"]))
        return "; ".join(parts)


def _hgl_phase(kwargs):
    meta = kwargs.get("attn_metadata")
    is_prompt = getattr(meta, "is_prompt", None)
    if is_prompt is None:
        ids = kwargs.get("input_ids")
        is_prompt = ids is not None and ids.dim() > 1 and ids.shape[-1] > 1
    return "prompt" if is_prompt else "decode"


def _hgl_is_rank0():
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True


class _HglStatsLog:
    """Prints the cache counters on rank 0 at most every `interval` seconds, only when they changed."""

    def __init__(self, cache, interval):
        self.cache = cache
        self.interval = interval
        self.next_at = _hgl_time.monotonic() + interval
        self.last = None

    def tick(self):
        if self.interval <= 0 or _hgl_time.monotonic() < self.next_at:
            return
        self.next_at = _hgl_time.monotonic() + self.interval
        line = self.cache.summary()
        if line != self.last and _hgl_is_rank0():
            print("[hpu-graph-lru] %s" % line, file=_hgl_sys.stdout, flush=True)
        self.last = line


def _hgl_destroy(entry):
    import torch

    torch.hpu.synchronize()  # an earlier asynchronous replay may still use the graph
    entry.graph.reset()
    entry.graph_inputs = entry.graph_outputs = None


def _hgl_wrap(module):
    """wrap_in_hpu_graph(module, disable_tensor_cache=True) with an evicting cache of VLLM_HPU_MAX_GRAPHS."""
    import inspect

    import habana_frameworks.torch as _htorch
    from habana_frameworks.torch.hpu import graphs as _graphs

    max_graphs = _hgl_os.environ.get("VLLM_HPU_MAX_GRAPHS", "").strip()
    if not max_graphs:
        return _htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)
    cache = HpuGraphLRU(int(max_graphs), int(_hgl_os.environ.get("VLLM_HPU_GRAPH_ADMIT_AFTER", "2")),
                        destroy=_hgl_destroy)
    log = _HglStatsLog(cache, float(_hgl_os.environ.get("VLLM_HPU_GRAPH_STATS_INTERVAL", "300")))
    stream = _htorch.hpu.default_stream()
    orig_fwd = module.forward
    # Positional arguments after (cache, stream, orig_fwd, args, kwargs): disable_tensor_cache,
    # asynchronous, dry_run, max_graphs, and the last three as wrap_in_hpu_graph passes them.
    tail = (True, False, None, cache.max_graphs, True, False, 100)
    try:
        inspect.signature(_graphs.wrapped_hpugraph_forward).bind(cache, stream, orig_fwd, (), {}, *tail)
    except (AttributeError, TypeError, ValueError) as e:
        raise RuntimeError("[hpu-graph-lru] habana_frameworks graphs.wrapped_hpugraph_forward does not have the "
                           "signature this fix was written for (Gaudi software 1.24.1); unset VLLM_HPU_MAX_GRAPHS "
                           "(%s)" % e) from e
    print("[hpu-graph-lru] max_graphs %d, admit after %d misses" % (cache.max_graphs, cache.admit_after), flush=True)

    @_hgl_functools.wraps(orig_fwd)
    def forward(*args, **kwargs):
        cache.current_phase = _hgl_phase(kwargs)
        out = _graphs.wrapped_hpugraph_forward(cache, stream, orig_fwd, args, kwargs, *tail)
        log.tick()
        return out

    def clear_inputs():
        for entry in cache.values():
            entry.graph.clear_inputs()

    module.forward = forward
    module.clear_cache = cache.clear
    module.clear_inputs = clear_inputs
    module.graph_cache = cache
    return module


'''

WRAP_OLD = """def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    return htorch.hpu.wrap_in_hpu_graph(HpuModelAdapter(
        *args, **kwargs), disable_tensor_cache=True) if htorch.utils.internal.is_lazy() else HpuModelAdapter(
            *args, **kwargs)
"""
WRAP_NEW = f"""# ---- {MARKER} bounded HPU graph cache (VLLM_HPU_MAX_GRAPHS), least recently used out ----
{LRU_CODE.strip()}


def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    # {MARKER} with VLLM_HPU_MAX_GRAPHS set the graph cache evicts instead of growing without bound.
    adapter = HpuModelAdapter(*args, **kwargs)
    return _hgl_wrap(adapter) if htorch.utils.internal.is_lazy() else adapter
"""


class Drift(Exception):
    """The installed source does not match what this fix expects."""


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise Drift(f"anchor {label!r} found {n} times, expected 1")
    return src.replace(old, new, 1)


def patch_runner(src: str, ctx: dict[str, str]) -> str:
    for name in ("HpuGraphLRU", "_hgl_wrap"):
        if name in src:
            raise Drift(f"{name} already defined in hpu_model_runner.py")
    return replace_once(src, WRAP_OLD, WRAP_NEW, "_maybe_wrap_in_hpu_graph")


# Not fixed upstream.
UPSTREAM_FIXED = ()

# (package, path inside the package, patch function or None for read-only context, context key)
TARGETS = (("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch_runner, "runner"),)


# ---- common driver (identical in every fix) --------------------------------------------------


def locate(pkg: str, rel: str, root: str | None) -> Path | None:
    """Find <pkg>/<rel> under --root, or in the installed package without importing it."""
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".gaudi-fix.tmp")
    tmp.write_text(text)
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def run(check: bool, root: str | None) -> int:
    ctx: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for pkg, rel, _, key in TARGETS:
        path = locate(pkg, rel, root)
        if path is None:
            print(f"{MARKER} {pkg}/{rel} not found", file=sys.stderr)
            return 2
        paths[key] = path
        ctx[key] = path.read_text()
    if UPSTREAM_FIXED and all(MARKER not in ctx[k] and needle in ctx[k] for k, needle in UPSTREAM_FIXED):
        print(f"{MARKER} not needed: the upstream fix is present")
        return 0
    done, todo = [], []
    try:
        for _pkg, _rel, fn, key in TARGETS:
            if fn is None:
                continue
            src = ctx[key]
            if MARKER in src:
                done.append(paths[key])
                continue
            new = fn(src, ctx)
            if MARKER not in new:
                raise Drift("patched text lacks the marker")
            compile(new, str(paths[key]), "exec", dont_inherit=True)
            todo.append((paths[key], new))
    except (Drift, SyntaxError) as e:
        print(f"{MARKER} {e}; source drifted, nothing written", file=sys.stderr)
        return 2
    if done and todo:
        print(f"{MARKER} partly applied ({', '.join(map(str, done))}); restore the files and re-run", file=sys.stderr)
        return 2
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied (anchors match)")
        return 1
    for path, new in todo:
        write_atomic(path, new)
        print(f"{MARKER} patched {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report the state without writing (0 applied, 1 not applied)")
    ap.add_argument("--root", help="directory holding vllm/ and vllm_gaudi/ (default: the installed packages)")
    args = ap.parse_args()
    return run(args.check, args.root)


if __name__ == "__main__":
    sys.exit(main())
