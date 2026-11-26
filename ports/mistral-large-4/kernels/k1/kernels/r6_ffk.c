// r6_ffk: the indexer k straight to its index-K cache row (ML4_INDEX_U8): numerics.fake_fp8_ue8m0 (r5_ff8's forms)
// then paged.index_u8_pack: amax = max|v|, s = amax > 0 ? 2 * pow2_ceil(amax / 448) : 2, codes = e4m3fn(v / s) (RNE,
// |v / s| <= 224), row = 128 codes | the fp32 s. Saves the bf16 k and the pack's ~6 dependent nodes per indexer layer.
//   zi i8 [1] (unused: input 0 sets the outputs' dtype under capture), x bf16 [M][128] (r6_ffk) or f32 [M][128]
//   (r6_ffkf: the fp32 k the compiler keeps ahead of fake_fp8 on NoPE layers) -> row i8 [M][132]; index {M}
//   mode bits 0-1: amax / 448 as amax * fp32(1/448) (0), * reciprocal_f32(448) (1), * reciprocal_cephes_f32(448) (2);
//   bit 2: v / s as v * reciprocal_cephes_f32(s) instead of the exact power-of-two inverse.
#include "r6_common.h"
void main(tensor zi, tensor x, tensor row, int mode)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 L = LANE();
    float64 c448 = (float)(1.0 / 448.0);
    if ((mode & 3) == 1) c448 = v_reciprocal_f32((float64)448.0f);
    if ((mode & 3) == 2) c448 = reciprocal_cephes_f32((float64)448.0f);
    for (int m = is[0]; m < ie[0]; m++) {
#ifdef FFK_F32
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
        const float64 v0 = as_float64(RNE16(as_int64(r6_e4m3(y0) * s)));       // the bf16 k, as fp32
        const float64 v1 = as_float64(RNE16(as_int64(r6_e4m3(y1) * s)));
        const float64 av = v_f32_reduce_max(v_f32_max_b(v_f32_abs_b(v0, 0, 0, 1, 0), v_f32_abs_b(v1, 0, 0, 1, 0)));
        const int64 ab = as_int64(av);
        const int64 pc = (as_int64(av * c448) + 0x7FFFFF) & 0x7F800000;
        const int64 s2 = SEL_LT((int64)0, ab, pc + 0x00800000, (int64)0x40000000);    // 2 * pow2_ceil | 2.0
        const float64 i2 = (mode & 4) ? reciprocal_cephes_f32(as_float64(s2)) : as_float64(0x7F000000 - s2);
        const int64 c0 = r6_e4m3_code(v0 * i2), c1 = r6_e4m3_code(v1 * i2);
        // PCK_32_TO_8: byte j of lane l -> row byte 64 j + l; bytes 128..131 (j 2, lanes 0..3) = s, little endian
        int64 b2 = SEL_EQ(L, (int64)0, s2 & 0xFF, (int64)0);
        b2 = SEL_EQ(L, (int64)1, (s2 >> 8) & 0xFF, b2);
        b2 = SEL_EQ(L, (int64)2, (s2 >> 16) & 0xFF, b2);
        b2 = SEL_EQ(L, (int64)3, (s2 >> 24) & 0xFF, b2);
        v_i32_st_tnsr((int5){0, m, 0, 0, 0}, row, c0 | (c1 << 8) | (b2 << 16), SW_PACK | SW_PCK_32_TO_8, 1, 0);
    }
}
