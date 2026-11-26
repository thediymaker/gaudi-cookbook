// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2gn (v10, DeepSeek-V4.1-Flash, gpt-oss): mxk_w2g with the packed reduction tree and the carried tail
// at general even nch (4 / 6: 1024 < I <= 3072). Output = mxk_w2g's partials bit for bit (pair it with mxk_w2c or
// mxk_w2cb). See mxk9_w2n.h.
#define MXKN_MODE 3
#include "mxk9_w2n.h"
