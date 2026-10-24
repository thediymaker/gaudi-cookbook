// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13x (GLM-5.x, Muse Spark Open): w13 rows [gate (I); up (I)], no bias. The activation parameters come
// from the caller (SiLU: alpha 1, limit +inf, beta 0, ascale 1). See mxk8_w13.h. K <= 6144 (x row staged in VLM).
#define MXK8_IL 0
#include "mxk8_w13.h"
