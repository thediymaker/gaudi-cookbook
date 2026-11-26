// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13xibn (v10, gpt-oss): mxk_w13xib (interleaved rows + bias + SwiGLU-OAI) on the activation x in
// NATURAL order and unpadded ([T, K], K <= 3072: no host pad or permute), with the packed reduction tree carried into
// the next pass (MXG_W13C 2) and the one-branch flush (MXG_FB 2). Same output as mxk_w13xib, bit for bit. See
// mxk10_w13.h.
#define MXK8_IL 1
#define MXK8_BIAS 1
#define MXK8_XCH 6
#define MXG_W13C 2
#define MXG_FB 2
#define MXKX_NATX 1
#include "mxk10_w13.h"
