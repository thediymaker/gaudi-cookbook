// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13xn (v10, GLM-5.x): mxk_w13x on the activation x in NATURAL order (the per-512 permute is done while
// staging the row in VLM, so the host permute node goes away), with the packed reduction tree carried into the next
// pass (MXG_W13C 2) and a one-branch flush test (MXG_FB 2). Same contract and bit-identical output as mxk_w13x.
// See mxk10_w13.h and MXKX_NATX in mxk10_core.h.
#define MXK8_IL 0
#define MXKX_NATX 1
#define MXG_W13C 2
#define MXG_FB 2
#include "mxk10_w13.h"
