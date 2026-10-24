# SPDX-License-Identifier: Apache-2.0
"""CPU test for rank-exception-fatal: drive the real (patched) worker RPC loop with a fake queue.

The busy loop and its helpers are taken from vLLM's multiproc_executor.py (patched in a temporary
copy) and run with stand-ins for the queue, the worker and os._exit, so no torch is needed.
The source comes from GAUDI_FIXES_ROOT (a directory holding vllm/) or the installed vllm package;
the test is skipped when neither is available.

Run with: pytest fixes/rank-exception-fatal
"""

import ast
import contextlib
import functools
import importlib.util
import os
import shutil
import subprocess
import sys
import traceback
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REL = Path("vllm/v1/executor/multiproc_executor.py")


def _source() -> Path:
    root = os.environ.get("GAUDI_FIXES_ROOT")
    if root and (Path(root) / REL).is_file():
        return Path(root) / REL
    spec = importlib.util.find_spec("vllm")
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / REL.relative_to("vllm")
        if p.is_file():
            return p
    pytest.skip("no vLLM source: set GAUDI_FIXES_ROOT to a directory holding vllm/")


class Exited(Exception):
    pass


class Stop(Exception):
    pass


def _load(text: str):
    """Build a WorkerProc stand-in from the busy-loop methods of the given source."""
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerProc")
    wanted = {"worker_busy_loop", "_execute_worker_rpc"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_gaudi_fix_")]

    def fake_exit(code):
        raise Exited(code)

    watchdog = types.SimpleNamespace(disable=contextlib.nullcontext, feed=lambda: None)
    logger = types.SimpleNamespace(exception=lambda *a, **k: None, error=lambda *a, **k: None, handlers=[])
    ns = {
        "Any": object,
        "partial": functools.partial,
        "cloudpickle": None,
        "traceback": traceback,
        "logger": logger,
        "os": types.SimpleNamespace(_exit=fake_exit),
        "get_watch_dog": lambda: watchdog,
    }
    module = ast.Module(body=[*methods, *helpers], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "multiproc_executor_extract", "exec"), ns)
    return {n.name: ns[n.name] for n in methods}


class Worker:
    def ok(self):
        return "out"

    def boom(self):
        raise RuntimeError("device allocation failed")


def _run(funcs, rank, calls):
    """calls: [(method, output_rank)] -> (replies, exit code or None)"""
    queue = list(calls)
    replies = []

    class MQ:
        def dequeue(self, indefinite=True):
            if not queue:
                raise Stop()
            meth, out_rank = queue.pop(0)
            return meth, (), {}, out_rank

    self = types.SimpleNamespace(rank=rank, worker=Worker(), rpc_broadcast_mq=MQ(), handle_output=replies.append)
    if "_execute_worker_rpc" in funcs:
        self._execute_worker_rpc = types.MethodType(funcs["_execute_worker_rpc"], self)
    try:
        funcs["worker_busy_loop"](self)
    except Stop:
        return replies, None
    except Exited as e:
        return replies, e.args[0]
    return replies, None


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    src = _source()
    root = tmp_path_factory.mktemp("site")
    dst = root / REL
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    stock = dst.read_text()
    rc = subprocess.run([sys.executable, str(HERE / "apply.py"), "--root", str(root)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    return stock, dst.read_text()


def test_non_output_rank_failure_exits(sources):
    funcs = _load(sources[1])
    replies, code = _run(funcs, 1, [("ok", 0), ("boom", 0), ("ok", 0)])
    assert code == 1
    assert replies == []


def test_output_rank_failure_replies_and_continues(sources):
    funcs = _load(sources[1])
    replies, code = _run(funcs, 0, [("boom", 0), ("ok", 0)])
    assert code is None
    assert isinstance(replies[0], RuntimeError) and replies[1] == "out"


def test_broadcast_call_failure_replies_and_continues(sources):
    funcs = _load(sources[1])
    replies, code = _run(funcs, 1, [("boom", None), ("ok", None)])
    assert code is None
    assert isinstance(replies[0], RuntimeError) and replies[1] == "out"


def test_successful_calls_unchanged(sources):
    funcs = _load(sources[1])
    assert _run(funcs, 3, [("ok", 0), ("ok", 0)]) == ([], None)


def test_stock_source_drops_the_exception(sources):
    """The bug: unpatched, a failing non-output rank neither replies nor stops."""
    if "[gaudi-fix:rank-exception-fatal]" in sources[0]:
        pytest.skip("the source tree is already patched")
    funcs = _load(sources[0])
    replies, code = _run(funcs, 1, [("boom", 0), ("ok", 0)])
    assert code is None and replies == []
