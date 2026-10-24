#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the MXFP4 MoE decode kernels for Gaudi 2 inside a vllm-gaudi image (or any container on the same Gaudi
# software base). Needs, all shipped in the Habana pytorch-installer base image:
#   - tpc-clang, the TPC C/C++ compiler (1.24.1 base: "TPC C/C++ compiler version 2.7.33", /opt/habanalabs/bin)
#   - tpc_kernel_lib_interface.h (/usr/include/habanalabs)
#   - hpu_custom_op.h and libhabana_pytorch_plugin.so (from the installed habana_frameworks package)
#   - g++, objcopy, strip, python3 with torch
#
#   ./build.sh [OUT_DIR]          default OUT_DIR: ./dist
#
# Env: TPC_CLANG (compiler path), MXK_BUILD_DIR (scratch dir, default /tmp/mxk-mxfp4-moe-build; keep it fixed:
# tpc-clang records it in the ELF), MXK_STOCK_LIB (stock kernel library for the gate, default
# /usr/lib/habanalabs/libtpc_kernels.so), MXK8_FLAGS (extra tpc-clang -D flags for experiments; they become part of
# the version string).
#
# Output in OUT_DIR:
#   libmxk_tpc.so     kernel library for GC_KERNEL_PATH: mxk_w13xib + mxk_w2xb, every other GUID forwarded to the
#                     stock library, so GC_KERNEL_PATH stays a single file and the recipe cache keeps working
#   libmxk_op_lazy.so torch.ops.custom_op.mxk_w13xib / mxk_w2xb (lazy mode)
#   mxk_lib_gate      the library gate (run at the end of this script)
#   MANIFEST          version, toolchain versions, sha256 of every output
# Compiled files are build outputs; do not commit them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/src
OUT=${1:-$HERE/dist}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
B=${MXK_BUILD_DIR:-/tmp/mxk-mxfp4-moe-build}
mkdir -p "$B"
STOCK_LIB=${MXK_STOCK_LIB:-/usr/lib/habanalabs/libtpc_kernels.so}

TPCCLANG=${TPC_CLANG:-}
if [ -z "$TPCCLANG" ]; then
  if [ -x /opt/habanalabs/bin/tpc-clang ]; then TPCCLANG=/opt/habanalabs/bin/tpc-clang
  else TPCCLANG=$(command -v tpc-clang || true); fi
fi
[ -n "$TPCCLANG" ] && [ -x "$TPCCLANG" ] || {
  echo "tpc-clang not found: set TPC_CLANG, or build inside the Habana pytorch-installer based image" >&2; exit 1; }
[ -f /usr/include/habanalabs/tpc_kernel_lib_interface.h ] || {
  echo "/usr/include/habanalabs/tpc_kernel_lib_interface.h not found (Gaudi software headers)" >&2; exit 1; }

KERNELS=(mxk_w13xib mxk_w2xb)
declare -A KSRC=([mxk_w13xib]=mxk8_w13xib [mxk_w2xb]=mxk8_w2xb)
SRCS=("$SRC/mxk8_core.h" "$SRC/mxk8_w13.h" "$SRC/mxk8_w2.h" "$SRC/mxk8_w13xib.c" "$SRC/mxk8_w2xb.c"
      "$SRC/mxk_tpc_lib.cpp" "$SRC/mxk_tpc.map" "$SRC/mxk_op_lazy.cpp" "$HERE/build.sh")
SRCHASH=$(cat "${SRCS[@]}" | sha256sum | cut -c1-12)
MXK8_FLAGS=${MXK8_FLAGS:-}
[ -n "$MXK8_FLAGS" ] && SRCHASH=$(echo "$SRCHASH $MXK8_FLAGS" | sha256sum | cut -c1-12)
VERSION=mxfp4-moe-v8-$SRCHASH

# 1. TPC kernels -> ELF -> x86 object holding the ELF as a binary blob
for n in "${KERNELS[@]}"; do
  cp "$SRC/${KSRC[$n]}.c" "$B/$n.c"
  # shellcheck disable=SC2086
  (cd "$B" && "$TPCCLANG" -march=gaudi2 -O2 $MXK8_FLAGS -x c++ -I"$SRC" "$n.c" -c -o "$n.o")
  # assembly listing for inspection (optional)
  # shellcheck disable=SC2086
  (cd "$B" && "$TPCCLANG" -march=gaudi2 -O2 $MXK8_FLAGS -x c++ -I"$SRC" "$n.c" -S -o "$n.s" 2>/dev/null || true)
  (cd "$B" && objcopy -I binary -O elf64-x86-64 -B i386:x86-64 "$n.o" "${n}_x86.o")
done

# 2. kernel library (single file, forwards stock GUIDs)
g++ -O2 -fPIC -shared -std=c++17 -fvisibility=hidden -DMXK_VERSION="\"$VERSION\"" -I/usr/include/habanalabs \
  "$SRC/mxk_tpc_lib.cpp" "$B/mxk_w13xib_x86.o" "$B/mxk_w2xb_x86.o" -ldl -Wl,-soname,libmxk_tpc.so \
  -Wl,--version-script="$SRC/mxk_tpc.map" -Wl,-z,noexecstack -o "$B/libmxk_tpc.so"

# 3. torch op library (lazy mode, legacy custom-op API), against the image's torch and habana-torch-plugin
TI=$(python3 -c "from habana_frameworks.torch.utils.lib_utils import get_include_dir;print(get_include_dir())")
TL=$(python3 -c "from habana_frameworks.torch.utils.lib_utils import get_lib_dir;print(get_lib_dir())")
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
TORCHINC=$(python3 -c "import torch.utils.cpp_extension as c;print(' '.join('-I'+p for p in c.include_paths()))")
TORCHLIB=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
ABI=$(python3 -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
# shellcheck disable=SC2086
g++ -O2 -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI="$ABI" -DTORCH_EXTENSION_NAME=mxk_op_lazy \
  -I"$TI" -I/usr/include/habanalabs -I"$PYINC" $TORCHINC "$SRC/mxk_op_lazy.cpp" \
  -L"$TL" -lhabana_pytorch_plugin -L"$TORCHLIB" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TL" \
  -Wl,-soname,libmxk_op_lazy.so -o "$B/libmxk_op_lazy.so"
strip --strip-unneeded "$B/libmxk_tpc.so" "$B/libmxk_op_lazy.so"

# 4. library gate (host only, no device needed)
g++ -O2 -std=c++17 -I/usr/include/habanalabs "$SRC/mxk_lib_gate.cpp" -ldl -o "$B/mxk_lib_gate"
install -m 0644 "$B/libmxk_tpc.so" "$B/libmxk_op_lazy.so" "$OUT/"
install -m 0755 "$B/mxk_lib_gate" "$OUT/"
"$OUT/mxk_lib_gate" "$OUT/libmxk_tpc.so" "$STOCK_LIB"

TORCHV=$(python3 -c "import torch;print(torch.__version__)")
HTPV=$(python3 -c "import importlib.metadata as m;print(m.version('habana-torch-plugin'))" 2>/dev/null || echo unknown)
{
  echo "version: $VERSION"
  echo "kernels: ${KERNELS[*]}"
  echo "source_sha256_12: $SRCHASH"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "torch: $TORCHV"
  echo "habana_torch_plugin: $HTPV"
  echo "tpc_clang: $("$TPCCLANG" --version 2>/dev/null | head -1)"
  for n in "${KERNELS[@]}"; do echo "elf_sha256 $n: $(sha256sum "$B/$n.o" | cut -c1-64)"; done
  (cd "$OUT" && sha256sum libmxk_tpc.so libmxk_op_lazy.so | sed 's/^/sha256 /')
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"
