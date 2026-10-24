// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13xib (gpt-oss): interleaved gate/up rows, per-row bias, SwiGLU-OAI epilogue. See mxk8_w13.h.
// The activation row is staged in VLM for K <= 3072 (6 chunks of 512; gpt-oss K = 2880).
#define MXK8_IL 1
#define MXK8_BIAS 1
#define MXK8_XCH 6
#include "mxk8_w13.h"
