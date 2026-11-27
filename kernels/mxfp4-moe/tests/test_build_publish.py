# SPDX-License-Identifier: Apache-2.0
"""CPU check that build.sh changes OUT_DIR only after both gates pass (no device, no toolchain needed).

The compilers, strip, objcopy and the two gates are stubs on PATH; the Gaudi header directory is pointed at an empty
stub. A gate that fails must leave an earlier build in OUT_DIR as it was.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parent.parent / "build.sh"
OUT_FILES = ("libmxk_tpc.so", "libmxk_op_lazy.so", "libmxk_op_pt2.so", "mxk_lib_gate", "MANIFEST")

STUBS = {
    "tpc-clang": """#!/bin/bash
[ "${1:-}" = --version ] && { echo stub; exit 0; }
while [ $# -gt 0 ]; do [ "$1" = -o ] && echo elf > "$2"; shift; done
""",
    "objcopy": '#!/bin/bash\necho obj > "${@: -1}"\n',
    "strip": "#!/bin/bash\nexit 0\n",
    # The lib gate binary is "compiled" as a script that exits with STUB_LIB_GATE_RC.
    "g++": """#!/bin/bash
o=""; gate=0; prev=""
for a in "$@"; do [ "$prev" = -o ] && o=$a; case $a in *mxk_lib_gate.cpp) gate=1 ;; esac; prev=$a; done
if [ $gate = 1 ]; then printf '#!/bin/bash\\nexit ${STUB_LIB_GATE_RC:-0}\\n' > "$o"; chmod +x "$o"
else echo new > "$o"; fi
""",
    "python3": """#!/bin/bash
case "${1:-}" in
  -c) case "$2" in *_GLIBCXX*) echo 1 ;; *include_paths*) echo -I/x ;; *) echo /x ;; esac ;;
  *mxk_op_gate.py) exit ${STUB_OP_GATE_RC:-0} ;;
  *tpc_elf_code.py) echo code ;;
  *) exit 1 ;;
esac
""",
}


def run_build(tmp_path, out, **env):
    if not shutil.which("bash") or not shutil.which("install"):
        pytest.skip("needs bash and coreutils")
    bin_dir, inc = tmp_path / "bin", tmp_path / "inc"
    for d in (bin_dir, inc):
        d.mkdir(exist_ok=True)
    (inc / "tpc_kernel_lib_interface.h").write_text("")
    for name, text in STUBS.items():
        p = bin_dir / name
        p.write_text(text)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    script = BUILD.read_text().replace("/usr/include/habanalabs", str(inc))
    script = script.replace('HERE=$(cd "$(dirname "$0")" && pwd)', f"HERE={BUILD.parent}")
    (tmp_path / "build.sh").write_text(script)
    full_env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", TPC_CLANG=str(bin_dir / "tpc-clang"))
    full_env.update(MXK_BUILD_DIR=str(tmp_path / "b"), MXK_STOCK_LIB=os.devnull, **env)
    cmd = ["bash", str(tmp_path / "build.sh"), "--variant", "glm", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=full_env)


def earlier_build(out):
    out.mkdir()
    for f in OUT_FILES:
        (out / f).write_text("earlier\n")
    return {p.name: p.read_text() for p in out.iterdir()}


@pytest.mark.parametrize("gate", ["STUB_LIB_GATE_RC", "STUB_OP_GATE_RC"])
def test_failed_gate_leaves_out_as_it_was(tmp_path, gate):
    out = tmp_path / "out"
    before = earlier_build(out)
    r = run_build(tmp_path, out, **{gate: "1"})
    assert r.returncode != 0, r.stdout + r.stderr
    assert {p.name: p.read_text() for p in out.iterdir()} == before


def test_passing_gates_publish_the_build(tmp_path):
    out = tmp_path / "out"
    earlier_build(out)
    r = run_build(tmp_path, out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in out.iterdir()) == sorted(set(OUT_FILES) - {"libmxk_op_pt2.so"})
    assert (out / "libmxk_tpc.so").read_text() == "new\n"
    assert "version: glm-v10-" in (out / "MANIFEST").read_text()
