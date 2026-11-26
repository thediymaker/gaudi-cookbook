// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2g (grouped routes, per-route fp32 w2 sums): see mxk9_w2.h. Two activation rows in VLM: I <= 3072.
// Built into the library as mxk_w2gp's ELF for nch != 2; it has no GUID of its own.
#define MXK9_W2_COMBINE 0
#include "mxk9_w2.h"
