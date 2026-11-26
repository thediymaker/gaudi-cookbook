// k2_g2: the exact gather of k2_g with the addressing hoisted out (memoised per forward, as the plugin's own
// ctx.rows memo): per key the physical row and the expected owner come in as int32 streams, read sequentially; the
// tag pair is checked with a vector load (no dependent random scalar loads).
//   tags i32 [nrows*2]; data bf16 [nrows][328]; idx i32 [R][K] clamped key positions; prow i32 [R][K] physical row
//   (0 when not addressable); pown i32 [R][K] expected owner, INT_MIN when the key is not wanted / not addressable
//   out rows i32 [R][K][576] (fp32 bits), valid i32 [R][K]
//   index space {K/64, R}; scalar tagcheck
#include "k1_common.h"
#define DEC(J, U, BYTE)                                                                                             \
    {                                                                                                               \
        int64 qi = BYTE == 3 ? (U) : ((U) << (24 - 8 * BYTE));                                                      \
        qi = qi >> 24;                                                                                              \
        float64 f = v_convert_i32_to_f32_b(qi, SW_RHNE) * sc##J;                                                    \
        int64 v = as_int64(f);                                                                                      \
        v = v + 0x7FFF + ((v >> 16) & 1);                                                                           \
        v = v & (int64)0xFFFF0000;                                                                                  \
        v_i32_st_tnsr((int5){64 * J, k, r, 0, 0}, rows, v);                                                         \
    }
#define BCL(X, T) v_i32_mov_dual_group_all_b(v_i32_shuffle_b(X, PAT32((int64)(T)), 0, zero, 1, 0), 0xFFFFFFFF, \
                         0, 0, 0, 0, MkWrA(0b11, 0b11, 0b11, 0b11), zero, 1, 0)
void main(tensor tags, tensor data, tensor idx, tensor prow, tensor pown, tensor rows, tensor valid, int tagcheck)
{
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 zero = 0, one = 1;
    const int64 L = LANE();
    for (int r = is[1]; r < ie[1]; r++) {
        for (int g = is[0]; g < ie[0]; g++) {
            int64 acc = 0;
            for (int kk = 0; kk < 64; kk++) {
                const int k = g * 64 + kk;
                const int row = K1_LD_I(prow, k, r, 0);
                const int own = K1_LD_I(pown, k, r, 0);
                const int xi = K1_LD_I(idx, k, r, 0);
                const int64 U0 = v_i32_ld_tnsr_b((int5){0, row, 0, 0, 0}, data, SW_UNPACK | SW_UNPCK_8_TO_32, 0, 1, 0);
                const int64 U1 = v_i32_ld_tnsr_b((int5){128, row, 0, 0, 0}, data, SW_UNPACK | SW_UNPCK_8_TO_32, 0, 1, 0);
                const int64 SC = v_i32_ld_tnsr_b((int5){256, row, 0, 0, 0}, data);
                const int64 KP = v_i32_ld_tnsr_b((int5){264, row, 0, 0, 0}, data, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0);
                int64 v;
                if (tagcheck) {
                    const int64 TG = v_i32_ld_tnsr_b((int5){2 * row, 0, 0, 0, 0}, tags);
                    const int64 A = BCL(TG, 0), Bv = BCL(TG, 1);
                    v = SEL_EQ(A, (int64)own, SEL_EQ(Bv, (int64)xi, one, zero), zero);
                } else {
                    v = one;
                }
                if (own == (int)0x80000000) v = zero;
                acc = SEL_EQ(L, (int64)kk, v, acc);
                const float64 sc0 = as_float64(BCL(SC, 0)), sc1 = sc0, sc2 = as_float64(BCL(SC, 1)), sc3 = sc2;
                const float64 sc4 = as_float64(BCL(SC, 2)), sc5 = sc4, sc6 = as_float64(BCL(SC, 3)), sc7 = sc6;
                DEC(0, U0, 0) DEC(1, U0, 1) DEC(2, U0, 2) DEC(3, U0, 3)
                DEC(4, U1, 0) DEC(5, U1, 1) DEC(6, U1, 2) DEC(7, U1, 3)
                v_i32_st_tnsr((int5){512, k, r, 0, 0}, rows, KP << 16);
            }
            v_i32_st_tnsr((int5){g * 64, r, 0, 0, 0}, valid, acc);
        }
    }
}
