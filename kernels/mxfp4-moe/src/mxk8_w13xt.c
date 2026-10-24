// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13xt (Kimi-K2.7): w13 rows [gate (I); up (I)], no bias, for activation rows longer than the VLM stage
// (K = 7168). The first 12 chunks of the route's x row are staged in VLM, the rest are read from the tensor.
// See mxk8_w13.h and MXK8_XTAIL in mxk8_core.h.
#define MXK8_IL 0
#define MXK8_XTAIL 1
#include "mxk8_w13.h"
