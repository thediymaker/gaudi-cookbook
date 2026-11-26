// r5_ff8: numerics.fake_fp8_ue8m0 (ML4_NUMERICS_RW forms) on groups of 128: per group amax = max|x| (>= 1e-10),
// s = pow2_ceil(amax * fp32(1/448)), y = clamp(x * (1/s), +-448), the e4m3fn grid value of y (step = max(2^floor
// log2|y|, 2^-6) / 8, |y| * reciprocal(step) (the native division), RNE, * step, sign of y), times s, to bf16.
//   x bf16 [M][128] (r5_ff8) or zb bf16 [1] (unused, sets the output dtype under capture), x f32 [M][128] (r5_ff8f) -> y bf16 [M][128]; index {M}
// Natural element order: unpack loads (bf16) / plain loads (f32), one 32->16 packing store per group.
#include "r5_common.h"
inline float64 r5_e4m3(float64 y)
{
    const float64 a = v_f32_abs_b(y, 0, 0, 1, 0);
    const float64 p = as_float64(as_int64(a) & 0x7F800000);
    const float64 step = v_f32_max_b(p, 0.015625f) * 0.125f;
    const float64 r = v_f32_nearbyint_b(a * v_reciprocal_f32(step), SW_RHNE, 0, 1, 0) * step;
    return as_float64((as_int64(r) & 0x7FFFFFFF) | (as_int64(y) & (int64)0x80000000));
}
#ifdef FF8_F32
void main(tensor zb, tensor x, tensor yo)
#else
void main(tensor x, tensor yo)
#endif
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    for (int m = is[0]; m < ie[0]; m++) {
#ifdef FF8_F32
        const float64 f0 = v_f32_ld_tnsr_b((int5){0, m, 0, 0, 0}, x);
        const float64 f1 = v_f32_ld_tnsr_b((int5){64, m, 0, 0, 0}, x);
#else
        const float64 f0 = as_float64(v_i32_ld_tnsr_b((int5){0, m, 0, 0, 0}, x, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
        const float64 f1 = as_float64(v_i32_ld_tnsr_b((int5){64, m, 0, 0, 0}, x, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
#endif
        float64 amax = v_f32_reduce_max(v_f32_max_b(v_f32_abs_b(f0, 0, 0, 1, 0), v_f32_abs_b(f1, 0, 0, 1, 0)));
        amax = v_f32_max_b(amax, 1e-10f);
        const int64 sb = (as_int64(amax * (float)(1.0 / 448.0)) + 0x7FFFFF) & 0x7F800000;
        const float64 s = as_float64(sb);
        const float64 inv = as_float64(0x7F000000 - sb);
        const float64 y0 = v_f32_min_b(v_f32_max_b(f0 * inv, -448.0f), 448.0f);
        const float64 y1 = v_f32_min_b(v_f32_max_b(f1 * inv, -448.0f), 448.0f);
        const int64 o0 = RNE16(as_int64(r5_e4m3(y0) * s)), o1 = RNE16(as_int64(r5_e4m3(y1) * s));
        v_i32_st_tnsr((int5){0, m, 0, 0, 0}, yo, ((o0 >> 16) & 0xFFFF) | o1, SW_PACK | SW_PCK_32_TO_16, 1, 0);
    }
}
