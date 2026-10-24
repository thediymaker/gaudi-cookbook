// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2xr (Muse Spark Open): w2 GEMV per route, out [Q, K], zero rows for routes whose expert is not local
// (the model normalises every route's output before the token sum). See mxk8_w2.h. I <= 3072 (6 chunks of VLM).
#define MXK8_ROUTE 1
#define MXK8_XCH 6
#include "mxk8_w2.h"
