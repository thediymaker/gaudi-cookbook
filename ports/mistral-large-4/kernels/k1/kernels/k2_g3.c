// k2_g3 (ML4_R5_K2FOLD): k2_g2 with the per-key addressing folded in (BatchCtx._rows with the power-of-two
// block size + k2attn._addr): x = max(t, 0), blk = bt[b][x >> 7] when x >> 7 < NB, row = blk * 128 + (x & 127) when
// blk >= 0 else 0, owner = epoch[b][x >> 7] when t >= 0 and addressable else INT_MIN; b = r / S.
//   tags i32 [nrows*2]; data bf16 [nrows][328]; tk i32 [R][K] top-k positions (-1 none); bt i32 [B][NB];
//   ep i32 [B][NB] (block epochs) -> rows i32 [R][K][576] (fp32 bits), valid i32 [R][K]
//   index space {K/64, R}; scalars tagcheck, S
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
void main(tensor tags, tensor data, tensor tk, tensor bt, tensor ep, tensor rows, tensor valid, int tagcheck, int S)
{
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 zero = 0, one = 1;
    const int64 L = LANE();
    const int NB = get_dim_size(bt, 0);
    for (int r = is[1]; r < ie[1]; r++) {
        const int b = r / S;
        for (int g = is[0]; g < ie[0]; g++) {
            int64 acc = 0;
            for (int kk = 0; kk < 64; kk++) {
                const int k = g * 64 + kk;
                const int t = K1_LD_I(tk, k, r, 0);
                const int xi = t < 0 ? 0 : t;
                const int bi0 = xi >> 7;
                const int bi = bi0 < NB ? bi0 : NB - 1;
                const int blk = K1_LD_I(bt, bi, b, 0);
                const int epo = K1_LD_I(ep, bi, b, 0);
                const bool okr = bi0 < NB && blk >= 0;
                const int row = okr ? blk * 128 + (xi & 127) : 0;
                const int own = (okr && t >= 0) ? epo : (int)0x80000000;
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
