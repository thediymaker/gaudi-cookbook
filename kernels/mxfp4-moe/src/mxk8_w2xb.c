// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2xb (gpt-oss): w2 GEMV, per-expert output bias and routing-weight sum per token, up to 32 tokens.
// See mxk8_w2.h. The activation row is staged in VLM for I <= 3072 (6 chunks of 512; gpt-oss I = 2880).
#define MXK8_ROUTE 0
#define MXK8_BIAS 1
#define MXK8_TMAX 32
#define MXK8_XCH 6
#include "mxk8_w2.h"
