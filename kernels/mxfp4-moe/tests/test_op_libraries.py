# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the two op libraries' sources and of build.sh's --pt2 option (no device, no toolchain needed).

The lazy-mode library (src/mxk_op_lazy.cpp) and the torch.compile one (src/mxk_op_pt2.cpp) must register the same
ops under the same schemas and the same -DMXK_HAS_<guid> guards, so a port calls either one the same way.
"""

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
BUILD = HERE.parent / "build.sh"
PT2_OPS = {"mxk_w13x", "mxk_w13xi", "mxk_w13xt", "mxk_w2x", "mxk_w2xr", "mxk_deq"}  # gpt-oss ops stay lazy-only


def schemas(path):
    text = path.read_text()
    block = text[text.index("TORCH_LIBRARY_FRAGMENT") :]
    block = block[: block.index("\n}\n")]
    out, guard = {}, None
    pending = ""
    for line in block.splitlines():
        g = re.match(r"\s*#(?:if|ifdef)\s+(.*)", line)
        if g:
            guard = " ".join(sorted(re.findall(r"MXK_HAS_(\w+)", g.group(1))))
            continue
        pending += "".join(re.findall(r'"([^"]*)"', line))
        if line.rstrip().endswith(");") and pending:
            name = pending.split("(", 1)[0]
            out[name] = (pending, guard)
            pending = ""
    return out


def impls(path, key):
    text = path.read_text()
    block = text[text.index(f"TORCH_LIBRARY_IMPL(custom_op, {key}, m)") :]
    block = block[: block.index("\n}\n")]
    return set(re.findall(r'm\.impl\("(\w+)"', block))


def test_pt2_schemas_match_lazy():
    lazy, pt2 = schemas(SRC / "mxk_op_lazy.cpp"), schemas(SRC / "mxk_op_pt2.cpp")
    assert set(pt2) == PT2_OPS
    for name, (schema, guard) in pt2.items():
        assert name in lazy, name
        assert schema == lazy[name][0], name
        assert guard == lazy[name][1], (name, guard, lazy[name][1])


def test_pt2_has_hpu_and_meta_impls():
    hpu, meta = impls(SRC / "mxk_op_pt2.cpp", "HPU"), impls(SRC / "mxk_op_pt2.cpp", "Meta")
    assert hpu == meta == impls(SRC / "mxk_op_lazy.cpp", "HPU") & PT2_OPS


def run_build(*args):
    return subprocess.run(["bash", str(BUILD), *args], capture_output=True, text=True, timeout=60)


def test_pt2_refuses_lazy_only_builds(tmp_path):
    r = run_build("--variant", "gptoss", "--pt2", str(tmp_path / "out"))
    assert r.returncode == 2 and "--pt2" in r.stderr
    extra = tmp_path / "k"
    extra.mkdir()
    for f in ("my_kernel.c", "glue.inc", "op.cpp"):
        (extra / f).write_text("")
    r = run_build("--variant", "glm", "--extra-kernel", str(extra), "--pt2", str(tmp_path / "out"))
    assert r.returncode == 2 and "--extra-kernel" in r.stderr
    assert not (tmp_path / "out").exists()
