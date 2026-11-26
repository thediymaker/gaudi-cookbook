// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2xc (v10, GLM-5.x): mxk_w2x for nch == 2 (I_slice <= 1024): one basic block per pass, packed
// reduction tree carried into the next pass, accumulator by dual group. Same contract and bit-identical output as
// mxk_w2x; the glue serves any other I with the mxk_w2x ELF. See mxk10_w2.h.
#define MXK8_ROUTE 0
#define MXK8_XCH 6
#define MXT_N2 1
#define MXT_RED 2
#define MXG_DG 1
#include "mxk10_w2.h"
