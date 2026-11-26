// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13xin (v10, DeepSeek-V4.1-Flash): mxk_w13xi (interleaved gate/up rows, clamped SwiGLU) on the
// activation x in NATURAL order, with the packed reduction tree carried into the next pass (MXG_W13C 2) and the
// one-branch flush (MXG_FB 2). Same contract and bit-identical output as mxk_w13xi. See mxk10_w13.h.
#define MXK8_IL 1
#define MXKX_NATX 1
#define MXG_W13C 2
#define MXG_FB 2
#include "mxk10_w13.h"
