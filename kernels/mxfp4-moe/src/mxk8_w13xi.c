// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13xi (DeepSeek-V4.1-Flash): interleaved gate/up rows [g0, u0, g1, u1, ...], no bias, clamped SwiGLU
// through the activation parameters. See mxk8_w13.h. K <= 6144 (x row staged in VLM).
#define MXK8_IL 1
#include "mxk8_w13.h"
