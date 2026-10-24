// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2x (GLM-5.x, DeepSeek-V4.1-Flash): w2 GEMV and routing-weight sum per token, up to 16 tokens.
// See mxk8_w2.h. The activation row is staged in VLM for I <= 3072 (6 chunks of 512).
#define MXK8_ROUTE 0
#define MXK8_XCH 6
#include "mxk8_w2.h"
