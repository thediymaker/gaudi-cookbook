#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the MXFP4 MoE kernels for Gaudi 2 inside a vllm-gaudi image (or any container on the same Gaudi software
# base). Needs, all shipped in the Habana pytorch-installer base image:
#   - tpc-clang, the TPC C/C++ compiler (1.24.1 base: "TPC C/C++ compiler version 2.7.33", /opt/habanalabs/bin)
#   - tpc_kernel_lib_interface.h (/usr/include/habanalabs)
#   - hpu_custom_op.h and libhabana_pytorch_plugin.so (from the installed habana_frameworks package)
#   - g++, objcopy, strip, python3 with torch
#
#   ./build.sh [--variant NAME] [--kernels v8|v10] [--extra-kernel DIR]... [--pt2] [OUT_DIR]
#   defaults: --variant gptoss, --kernels v10 for a variant that has a v10 set (all but muse) unless --pt2, v8 otherwise,
#   OUT_DIR ./dist
#
# Variants (the kernels each model's integration calls; one variant per serving image):
#   gptoss   mxk_w13xib mxk_w2xb          gpt-oss: interleaved w13 + bias + SwiGLU-OAI, w2 + bias, <= 32 tokens
#            v10 adds mxk_w13xibn mxk_w2gn mxk_w2cb
#   glm      mxk_w13x mxk_w2x mxk_deq     GLM-5.x: [gate; up] w13 + SiLU, w2 sum (<= 16 tokens), prefill unpack
#            v10 adds mxk_w13xn mxk_w2xc mxk_w2gp mxk_w2c (and the mxk_w2g ELF, mxk_w2gp's fallback, no GUID)
#   dsv41    mxk_w13xi mxk_w2x            DeepSeek-V4.1-Flash: interleaved w13 + clamped SwiGLU, w2 sum
#            v10 adds mxk_w13xin mxk_w2gn mxk_w2c (and the mxk_w2gp ELF, mxk_w2gn's nch-2 kernel, no GUID)
#   muse     mxk_w13x mxk_w2xr            Muse Spark Open: [gate; up] w13 + SiLU, w2 per route
#   kimi     mxk_w13xt mxk_w2x            Kimi-K2.7: [gate; up] w13 + SiLU for K up to 8192, w2 sum
#            v10 adds mxk_w13xk (K <= 7168) mxk_w2xc mxk_w2gp mxk_w2c (and the mxk_w2g ELF)
#   none     (no MoE kernels)             a port that only needs its own --extra-kernel (GLM-5.3-Flash)
#
# --kernels v10 builds the variant's v8 kernels (the same machine code as a v8 build) PLUS its v10 kernels into one
# library, so the port picks the v10 path or keeps the v8 path at run time from the ops the library registers.
# The v10 kernels give the v8 kernels' outputs bit for bit. --kernels v8 builds the v8 library only. Variants
# without a v10 set build v8 and refuse --kernels v10. v10 has no torch.compile ops: --pt2 builds v8 and refuses --kernels v10.
#
# --extra-kernel DIR adds a model-only TPC kernel supplied by a port. DIR holds:
#   <guid>.c   the TPC-C source; its file name is the GUID. Self-contained (compiled without include paths).
#   glue.inc   C++ included into libmxk_tpc.so after the shared helpers. Defines
#              static GlueCodeReturn inst_<guid>(HabanaKernelParams*, HabanaKernelInstantiation*)
#              and may use DECL_ELF/ELF(<guid>) (declared for it), copyElf, checkTypes, allRequired, tpcPoints,
#              setIndexSpace (src/mxk_tpc_lib.cpp).
#   op.cpp     a translation unit linked into libmxk_op_lazy.so that registers torch.ops.custom_op.<guid> with
#              TORCH_LIBRARY_FRAGMENT(custom_op, m) / TORCH_LIBRARY_IMPL(custom_op, HPU, m) (hpu_custom_op.h API).
# The extra GUIDs follow the variant's GUIDs in the library, the gates expect them, and they are part of the version.
#
# --pt2 also builds libmxk_op_pt2.so (src/mxk_op_pt2.cpp): the same ops on the PT2 custom-op API with Meta
# implementations, for models served in torch.compile mode (PT_HPU_LAZY_MODE=0). Not for gptoss (lazy mode) and not
# with --extra-kernel (an extra kernel's op.cpp uses the lazy API).
#
# Env: TPC_CLANG (compiler path); MXK_BUILD_DIR (scratch dir, default /tmp/mxk_<variant>_v8_build for both kernel sets:
# tpc-clang records it in the ELF, so a fixed path keeps the kernel ELFs reproducible); MXK_STOCK_LIB (stock kernel library for the
# gate, default /usr/lib/habanalabs/libtpc_kernels.so).
#
# Output in OUT_DIR:
#   libmxk_tpc.so     kernel library for GC_KERNEL_PATH: the variant's kernels, every other GUID forwarded to the
#                     stock library, so GC_KERNEL_PATH stays a single file and the recipe cache keeps working
#   libmxk_op_lazy.so the variant's torch.ops.custom_op.* ops (lazy mode)
#   libmxk_op_pt2.so  with --pt2: the same ops for torch.compile mode
#   mxk_lib_gate      the library gate (run here, and again by an image build if it likes)
#   MANIFEST          version, variant, toolchain versions, sha256 of every kernel ELF, kernel code and library
# Compiled files are build outputs; do not commit them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/src

VARIANT=gptoss
KV=""
EXTRA_DIRS=()
OUT=""
PT2=0
while [ $# -gt 0 ]; do
  case $1 in
    --variant) VARIANT=$2; shift 2 ;;
    --variant=*) VARIANT=${1#*=}; shift ;;
    --kernels) KV=$2; shift 2 ;;
    --kernels=*) KV=${1#*=}; shift ;;
    --extra-kernel) EXTRA_DIRS+=("$(cd "$2" && pwd)"); shift 2 ;;
    --extra-kernel=*) EXTRA_DIRS+=("$(cd "${1#*=}" && pwd)"); shift ;;
    --pt2) PT2=1; shift ;;
    -h|--help) sed -n '2,57p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) [ -z "$OUT" ] || { echo "one OUT_DIR only" >&2; exit 2; }; OUT=$1; shift ;;
  esac
done

case $VARIANT in
  gptoss) KERNELS=(mxk_w13xib mxk_w2xb) ;;
  glm)    KERNELS=(mxk_w13x mxk_w2x mxk_deq) ;;
  dsv41)  KERNELS=(mxk_w13xi mxk_w2x) ;;
  muse)   KERNELS=(mxk_w13x mxk_w2xr) ;;
  kimi)   KERNELS=(mxk_w13xt mxk_w2x) ;;
  none)   KERNELS=() ;;
  *) echo "unknown variant '$VARIANT' (gptoss, glm, dsv41, muse, kimi, none)" >&2; exit 2 ;;
esac
# v10 kernels per variant: GUIDs, and ELFs built into the library without a GUID of their own
declare -A V10=([glm]="mxk_w13xn mxk_w2xc mxk_w2gp mxk_w2c" [kimi]="mxk_w13xk mxk_w2xc mxk_w2gp mxk_w2c"
                [dsv41]="mxk_w13xin mxk_w2gn mxk_w2c" [gptoss]="mxk_w13xibn mxk_w2gn mxk_w2cb")
declare -A V10_NOGUID=([glm]="mxk_w2g" [kimi]="mxk_w2g" [dsv41]="mxk_w2gp")
if [ -z "$KV" ]; then
  if [ -n "${V10[$VARIANT]:-}" ] && [ "$PT2" = 0 ]; then KV=v10; else KV=v8; fi
fi
case $KV in
  v8) NOGUID=() ;;
  v10)
    [ -n "${V10[$VARIANT]:-}" ] || { echo "--kernels v10: variant $VARIANT has no v10 kernels (build it with v8)" >&2; exit 2; }
    read -r -a NEW <<< "${V10[$VARIANT]}"
    read -r -a NOGUID <<< "${V10_NOGUID[$VARIANT]:-}"
    KERNELS+=("${NEW[@]}") ;;
  *) echo "unknown --kernels '$KV' (v8, v10)" >&2; exit 2 ;;
esac
# GUID -> source (the ELF symbol follows the GUID: each source is copied to <GUID>.c)
declare -A KSRC=([mxk_w13x]=mxk8_w13x [mxk_w13xi]=mxk8_w13xi [mxk_w13xib]=mxk8_w13xib [mxk_w13xt]=mxk8_w13xt
                 [mxk_w2x]=mxk8_w2x
                 [mxk_w2xr]=mxk8_w2xr [mxk_w2xb]=mxk8_w2xb [mxk_deq]=mxk_deq
                 [mxk_w13xn]=mxk10_w13xn [mxk_w13xk]=mxk10_w13xk [mxk_w13xin]=mxk10_w13xin
                 [mxk_w13xibn]=mxk10_w13xibn [mxk_w2gn]=mxk9_w2gn [mxk_w2cb]=mxk9_w2cb [mxk_w2xc]=mxk10_w2xc [mxk_w2gp]=mxk9_w2gp [mxk_w2c]=mxk9_w2c
                 [mxk_w2g]=mxk9_w2g)

EXTRA=()
for d in "${EXTRA_DIRS[@]}"; do
  srcs=("$d"/*.c)
  [ ${#srcs[@]} -eq 1 ] && [ -f "${srcs[0]}" ] || { echo "$d: need exactly one <guid>.c" >&2; exit 2; }
  [ -f "$d/glue.inc" ] && [ -f "$d/op.cpp" ] || { echo "$d: need glue.inc and op.cpp" >&2; exit 2; }
  g=$(basename "${srcs[0]}" .c)
  [[ $g =~ ^[a-z][a-z0-9_]*$ ]] || { echo "$d: GUID '$g' must be [a-z][a-z0-9_]*" >&2; exit 2; }
  EXTRA+=("$g")
done
[ "$VARIANT" != none ] || [ ${#EXTRA[@]} -gt 0 ] || { echo "--variant none needs at least one --extra-kernel" >&2; exit 2; }
if [ "$PT2" = 1 ]; then
  [ "$VARIANT" != gptoss ] && [ "$VARIANT" != none ] || { echo "--pt2: no torch.compile ops for variant $VARIANT" >&2; exit 2; }
  [ ${#EXTRA[@]} -eq 0 ] || { echo "--pt2 cannot be combined with --extra-kernel (lazy-mode op.cpp)" >&2; exit 2; }
  [ "$KV" = v8 ] || { echo "--pt2: the v10 kernels have no torch.compile ops; build with --kernels v8 --pt2" >&2; exit 2; }
fi

OUT=${OUT:-$HERE/dist}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
B=${MXK_BUILD_DIR:-/tmp/mxk_${VARIANT}_v8_build}
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

# Version: variant + hash of everything that goes into the libraries
SRCS=("$SRC/mxk8_core.h" "$SRC/mxk8_w13.h" "$SRC/mxk8_w2.h")
[ "$KV" = v8 ] || SRCS+=("$SRC"/mxk10_*.h "$SRC"/mxk9_*.h)
for n in "${KERNELS[@]}" "${NOGUID[@]}"; do SRCS+=("$SRC/${KSRC[$n]}.c"); done
SRCS+=("$SRC/mxk_tpc_lib.cpp" "$SRC/mxk_tpc.map" "$SRC/mxk_op_lazy.cpp" "$HERE/build.sh")
for d in "${EXTRA_DIRS[@]}"; do SRCS+=("$d"/*.c "$d/glue.inc" "$d/op.cpp"); done
[ "$PT2" = 0 ] || SRCS+=("$SRC/mxk_op_pt2.cpp")
SRCHASH=$(cat "${SRCS[@]}" | sha256sum | cut -c1-12)
VERSION=$VARIANT-$KV-$SRCHASH
HAS=()
for n in "${KERNELS[@]}"; do HAS+=("-DMXK_HAS_$n"); done
for n in "${NOGUID[@]}"; do HAS+=("-DMXK_ELF_$n"); done

# 1. TPC kernels -> ELF -> x86 object holding the ELF as a binary blob
tpc() {  # tpc <guid> <source> [include dir]
  cp "$2" "$B/$1.c"
  (cd "$B" && "$TPCCLANG" -march=gaudi2 -O2 -x c++ ${3:+-I"$3"} "$1.c" -c -o "$1.o")
  (cd "$B" && objcopy -I binary -O elf64-x86-64 -B i386:x86-64 "$1.o" "${1}_x86.o")
}
for n in "${KERNELS[@]}" "${NOGUID[@]}"; do tpc "$n" "$SRC/${KSRC[$n]}.c" "$SRC"; done
EXTRA_GLUE=()
EXTRA_OPS=()
if [ ${#EXTRA[@]} -gt 0 ]; then
  : > "$B/mxk_extra.inc"
  ents=""
  for i in "${!EXTRA[@]}"; do
    g=${EXTRA[$i]} d=${EXTRA_DIRS[$i]}
    tpc "$g" "$d/$g.c"
    printf 'DECL_ELF(%s)\n#include "%s/glue.inc"\n' "$g" "$d" >> "$B/mxk_extra.inc"
    ents="$ents {\"$g\", inst_$g},"
    EXTRA_OPS+=("$d/op.cpp")
  done
  echo "#define MXK_EXTRA_KERNELS$ents" >> "$B/mxk_extra.inc"
  EXTRA_GLUE=("-DMXK_EXTRA_INC=\"$B/mxk_extra.inc\"")
fi
ALL=("${KERNELS[@]}" "${EXTRA[@]}")
OBJS=()
for n in "${ALL[@]}" "${NOGUID[@]}"; do OBJS+=("$B/${n}_x86.o"); done

# 2. kernel library (single file, forwards stock GUIDs)
g++ -O2 -fPIC -shared -std=c++17 -fvisibility=hidden -DMXK_VERSION="\"$VERSION\"" "${HAS[@]}" "${EXTRA_GLUE[@]}" \
  -I/usr/include/habanalabs "$SRC/mxk_tpc_lib.cpp" "${OBJS[@]}" -ldl -Wl,-soname,libmxk_tpc.so \
  -Wl,--version-script="$SRC/mxk_tpc.map" -Wl,-z,noexecstack -o "$B/libmxk_tpc.so"

# 3. torch op library (lazy mode, legacy custom-op API), against the image's torch and habana-torch-plugin
TI=$(python3 -c "from habana_frameworks.torch.utils.lib_utils import get_include_dir;print(get_include_dir())")
TL=$(python3 -c "from habana_frameworks.torch.utils.lib_utils import get_lib_dir;print(get_lib_dir())")
PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
TORCHINC=$(python3 -c "import torch.utils.cpp_extension as c;print(' '.join('-I'+p for p in c.include_paths()))")
TORCHLIB=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
ABI=$(python3 -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
# shellcheck disable=SC2086
g++ -O2 -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI="$ABI" -DTORCH_EXTENSION_NAME=mxk_op_lazy "${HAS[@]}" \
  -I"$TI" -I/usr/include/habanalabs -I"$PYINC" $TORCHINC "$SRC/mxk_op_lazy.cpp" "${EXTRA_OPS[@]}" \
  -L"$TL" -lhabana_pytorch_plugin -L"$TORCHLIB" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TL" \
  -Wl,-soname,libmxk_op_lazy.so -o "$B/libmxk_op_lazy.so"
strip --strip-unneeded "$B/libmxk_tpc.so" "$B/libmxk_op_lazy.so"
OPLIBS=(libmxk_op_lazy.so)
if [ "$PT2" = 1 ]; then
  # 3b. the same ops on the PT2 custom-op API (torch.compile mode), with Meta implementations
  # shellcheck disable=SC2086
  g++ -O2 -fPIC -shared -std=c++17 -D_GLIBCXX_USE_CXX11_ABI="$ABI" -DTORCH_EXTENSION_NAME=mxk_op_pt2 "${HAS[@]}" \
    -I"$TI" -I/usr/include/habanalabs -I"$PYINC" $TORCHINC "$SRC/mxk_op_pt2.cpp" \
    -L"$TL" -lhabana_pytorch2_plugin -L"$TORCHLIB" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TL" \
    -Wl,-soname,libmxk_op_pt2.so -o "$B/libmxk_op_pt2.so"
  strip --strip-unneeded "$B/libmxk_op_pt2.so"
  OPLIBS+=(libmxk_op_pt2.so)
fi

# 4. library gate (host only, no device needed): stock GUIDs unchanged, then exactly this build's GUIDs
g++ -O2 -std=c++17 -I/usr/include/habanalabs "$SRC/mxk_lib_gate.cpp" -ldl -o "$B/mxk_lib_gate"
rm -f "$OUT/libmxk_op_pt2.so"   # a library left from an earlier --pt2 build would not match this one
for f in libmxk_tpc.so "${OPLIBS[@]}"; do install -m 0644 "$B/$f" "$OUT/"; done
install -m 0755 "$B/mxk_lib_gate" "$OUT/"
"$OUT/mxk_lib_gate" "$OUT/libmxk_tpc.so" "$STOCK_LIB" "${ALL[@]}"

TORCHV=$(python3 -c "import torch;print(torch.__version__)")
HTPV=$(python3 -c "import importlib.metadata as m;print(m.version('habana-torch-plugin'))" 2>/dev/null || echo unknown)
{
  echo "version: $VERSION"
  echo "variant: $VARIANT"
  echo "kernel_set: $KV"
  echo "kernels: ${ALL[*]}"
  echo "fallback_elfs: ${NOGUID[*]}"
  echo "extra_kernels: ${EXTRA[*]}"
  echo "op_libraries: ${OPLIBS[*]}"
  echo "source_sha256_12: $SRCHASH"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "torch: $TORCHV"
  echo "habana_torch_plugin: $HTPV"
  echo "tpc_clang: $("$TPCCLANG" --version 2>/dev/null | head -1)"
  # The ELF embeds the source text and the command line, so its hash changes with any comment or path. The code
  # hash (.text and .tpc_metadata sections) does not: equal code hashes mean the same kernel machine code.
  for n in "${ALL[@]}" "${NOGUID[@]}"; do echo "elf_sha256 $n: $(sha256sum "$B/$n.o" | cut -c1-64)"; done
  for n in "${ALL[@]}" "${NOGUID[@]}"; do echo "code_sha256 $n: $(python3 "$HERE/python/tpc_elf_code.py" "$B/$n.o")"; done
  (cd "$OUT" && sha256sum libmxk_tpc.so "${OPLIBS[@]}" | sed 's/^/sha256 /')
} > "$OUT/MANIFEST"

# 5. op gate: the op library loads, registers this variant's ops and no op of another variant (lazy mode would hand
# an op with the same input signature the other op's cached result)
for f in "${OPLIBS[@]}"; do
  python3 "$HERE/python/mxk_op_gate.py" --lib "$OUT" --op-lib "$f" --guids "$(IFS=,; echo "${ALL[*]}")"
done
cat "$OUT/MANIFEST"
