// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2gp (v10): mxk_w2g with the packed reduction tree and the tail carried past the next pass's dot, on
// the one-route passes (MXKP_MODE 3) and on the two-route half passes (MXKQ_MODE 3). Serves nch == 2 calls; the glue
// gives other I the mxk_w2g ELF. Output = mxk_w2g's partials bit for bit (pair it with mxk_w2c). See mxk9_w2q.h.
#define MXK9_XCH 4
#define MXKP_MODE 3
#define MXKQ_MODE 3
#include "mxk9_w2q.h"
