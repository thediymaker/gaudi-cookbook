// r6 shared helpers (Gaudi2 TPC-C), on top of r5_common.h
#pragma once
#include "r5_common.h"
// the e4m3fn grid value of y (|y| <= 448), RNE: r5_ff8's r5_e4m3
inline float64 r6_e4m3(float64 y)
{
    const float64 a = v_f32_abs_b(y, 0, 0, 1, 0);
    const float64 p = as_float64(as_int64(a) & 0x7F800000);
    const float64 step = v_f32_max_b(p, 0.015625f) * 0.125f;
    const float64 r = v_f32_nearbyint_b(a * v_reciprocal_f32(step), SW_RHNE, 0, 1, 0) * step;
    return as_float64((as_int64(r) & 0x7FFFFFFF) | (as_int64(y) & (int64)0x80000000));
}
// the e4m3fn byte of q (|q| <= 240), RNE incl. subnormals (step 2^-9 below 2^-6), sign kept (-0 -> 0x80): the device
// cast .to(float8_e4m3fn); value in the low 8 bits
inline int64 r6_e4m3_code(float64 q)
{
    const float64 r = v_f32_abs_b(r6_e4m3(q), 0, 0, 1, 0);
    const int64 rb = as_int64(r);
    const int64 nrm = ((((rb >> 23) & 0xFF) - 120) << 3) | ((rb >> 20) & 7);
    const int64 sub = v_convert_f32_to_i32_b(r * 512.0f, SW_RHNE, 0, 1, 0);
    const int64 c = SEL_LT(rb, (int64)0x3C800000, sub, nrm);
    return (c & 0x7F) | ((as_int64(q) >> 24) & 0x80);
}
