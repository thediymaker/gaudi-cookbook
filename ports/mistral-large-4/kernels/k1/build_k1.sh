#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the Mistral Large 4 decode-step kernel library (model-only TPC kernels: fused decode indexer, sparse-attention
# gather, exact node cuts) inside the stock vllm-gaudi image:
#
#   ports/mistral-large-4/kernels/k1/build_k1.sh OUT_DIR
#
# OUT_DIR gets
#   libk1_tpc.so      kernel library: these GUIDs + forwarding of every other GUID to the next library
#                     (K1_TPC_FWD_LIB, default the stock libtpc_kernels.so); mistral_large4_hpu.mxk.setup_env chains
#                     it behind the MXFP4 MoE library: GC_KERNEL_PATH=libmxk_tpc.so -> libk1_tpc.so -> stock
#   libk1_op_lazy.so  torch.ops.custom_op.k1_* / k2_* / r5_* / r6_* (lazy-mode legacy custom ops)
#   MANIFEST          version (k1-<source hash>), kernels, toolchain, sha256 and code sha256 of every kernel ELF
# Needs tpc-clang, tpc_kernel_lib_interface.h and the habana torch plugin headers, all in the stock image.
# Env: TPC_CLANG (compiler path); K1_BUILD_DIR (scratch dir, default /tmp/k1_ml4_build: tpc-clang records it in the
# ELF, so a fixed path keeps the ELFs reproducible). Compiled files are build outputs; do not commit them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: build_k1.sh OUT_DIR}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
B=${K1_BUILD_DIR:-/tmp/k1_ml4_build}
mkdir -p "$B"
CODE=$HERE/../../../../kernels/mxfp4-moe/python/tpc_elf_code.py
read -r -a R5 < "$HERE/r5_kernels.txt"
read -r -a R6 < "$HERE/r6_kernels.txt"
KS=(k1_kg k1_sc k1_p2 k1_p3 k1_p4 k1_fin k1_cm1 k1_cm2 k2_g2 "${R5[@]}" "${R6[@]}")
SRCHASH=$(cat "$HERE"/kernels/* "$HERE"/k1_tpc.cpp "$HERE"/k1_inst.h "$HERE"/k1_tpc.map "$HERE"/k1_op_legacy.cpp \
  "$HERE"/k1_ops.inc "$HERE"/r5_* "$HERE"/r6_* "$HERE"/build_k1.sh | sha256sum | cut -c1-12)
VERSION=k1-$SRCHASH
TPCCLANG=${TPC_CLANG:-/opt/habanalabs/bin/tpc-clang}
[ -x "$TPCCLANG" ] || { echo "tpc-clang not found (set TPC_CLANG)" >&2; exit 1; }

# 1. TPC kernels -> ELF -> x86 object holding the ELF as a binary blob
OBJS=()
for n in "${KS[@]}"; do
  (cd "$HERE/kernels" && "$TPCCLANG" -march=gaudi2 -O2 -x c++ "$n.c" -c -o "$B/$n.o")
  (cd "$B" && objcopy -I binary -O elf64-x86-64 -B i386:x86-64 "$n.o" "${n}_x86.o")
  OBJS+=("$B/${n}_x86.o")
done

# 2. kernel library (single file, forwards every other GUID)
g++ -O2 -fPIC -shared -std=c++17 -fvisibility=hidden -DK1_VERSION="\"$VERSION\"" -I/usr/include/habanalabs -I"$HERE" \
  "$HERE/k1_tpc.cpp" "${OBJS[@]}" -ldl -Wl,-soname,libk1_tpc.so -Wl,--version-script="$HERE/k1_tpc.map" \
  -Wl,-z,noexecstack -o "$B/libk1_tpc.so"

# 3. torch op library (lazy mode, legacy custom-op API), against the image's torch and habana-torch-plugin
TI=$(python3 -c "from habana_frameworks.torch.utils.lib_utils import get_include_dir;print(get_include_dir())")
TL=$(python3 -c "from habana_frameworks.torch.utils.lib_utils import get_lib_dir;print(get_lib_dir())")
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
TORCHINC=$(python3 -c "import torch.utils.cpp_extension as c;print(' '.join('-I'+p for p in c.include_paths()))")
TORCHLIB=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
ABI=$(python3 -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
# shellcheck disable=SC2086
g++ -O2 -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI="$ABI" -DTORCH_EXTENSION_NAME=k1_op_lazy \
  -I"$TI" -I/usr/include/habanalabs -I"$PYINC" -I"$HERE" $TORCHINC "$HERE/k1_op_legacy.cpp" \
  -L"$TL" -lhabana_pytorch_plugin -L"$TORCHLIB" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TL" \
  -Wl,-soname,libk1_op_lazy.so -o "$B/libk1_op_lazy.so"
strip --strip-unneeded "$B/libk1_tpc.so" "$B/libk1_op_lazy.so"
install -m 0644 "$B/libk1_tpc.so" "$B/libk1_op_lazy.so" "$OUT/"

# 4. MANIFEST
{
  echo "version: $VERSION"
  echo "kernels: ${KS[*]}"
  echo "source_sha256_12: $SRCHASH"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "torch: $(python3 -c 'import torch;print(torch.__version__)')"
  echo "tpc_clang: $("$TPCCLANG" --version 2>/dev/null | head -1)"
  for n in "${KS[@]}"; do echo "elf_sha256 $n: $(sha256sum "$B/$n.o" | cut -c1-64)"; done
  if [ -f "$CODE" ]; then
    for n in "${KS[@]}"; do echo "code_sha256 $n: $(python3 "$CODE" "$B/$n.o")"; done
  fi
  (cd "$OUT" && sha256sum libk1_tpc.so libk1_op_lazy.so | sed 's/^/sha256 /')
} > "$OUT/MANIFEST"
grep -q "k1_tpc $VERSION" "$OUT/libk1_tpc.so" || { echo "version string missing from libk1_tpc.so" >&2; exit 1; }
cat "$OUT/MANIFEST"
