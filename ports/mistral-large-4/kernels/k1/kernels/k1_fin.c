// k1_fin: from the 4 passes' count vectors, per row: v (the k-th largest key), and per chunk c the output offset
// and how many of its keys equal to v it takes (ties to the lower index: whole earlier chunks first).
//   out fin i32 [B][S][256 + NCH*MW]: [0,64) lane c = offset, [64,128) lane c = quota, [128,192) = v, [192,256) = need;
//   the rest is k1_cm's per-chunk scratch (mask words + sentinel; k1_cm writes it). index {S, B}
#include "k1_common.h"
void main(tensor c1, tensor c2, tensor c3, tensor c4, tensor fin, int NCH, int K)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 L = LANE();
    const int64 zero = 0;
    for (int b = is[1]; b < ie[1]; b++)
    for (int t = is[0]; t < ie[0]; t++) {
        int64 base = k1_digit(k1_tot(c1, NCH, t, b), K) << 12;
        base += k1_digit(k1_tot(c2, NCH, t, b), K) << 8;
        base += k1_digit(k1_tot(c3, NCH, t, b), K) << 4;
        const int64 tot4 = k1_tot(c4, NCH, t, b);
        const int64 j4 = k1_digit(tot4, K);
        const int64 v = base + j4;
        const int64 gtot = v_i32_reduce_add(SEL_EQ(L, j4 + 1, tot4, zero));
        const int64 need = (int64)K - gtot;
        int64 gtb = 0, eqb = 0, OFF = 0, Q = 0;
        for (int c = 0; c < NCH; c++) {
            const int64 cv = v_i32_ld_tnsr_b((int5){0, c, t, b, 0}, c4);
            const int64 ge = v_i32_reduce_add(SEL_EQ(L, j4, cv, zero));
            const int64 gt = v_i32_reduce_add(SEL_EQ(L, j4 + 1, cv, zero));
            const int64 eq = ge - gt;
            const int64 off = gtb + v_i32_min_b(eqb, need, 0, zero, 1, 0);
            int64 q = need - eqb;
            q = v_i32_max_b(v_i32_min_b(q, eq, 0, zero, 1, 0), zero, 0, zero, 1, 0);
            OFF = SEL_EQ(L, c, off, OFF);
            Q = SEL_EQ(L, c, q, Q);
            gtb += gt;
            eqb += eq;
        }
        v_i32_st_tnsr((int5){0, t, b, 0, 0}, fin, OFF);
        v_i32_st_tnsr((int5){64, t, b, 0, 0}, fin, Q);
        v_i32_st_tnsr((int5){128, t, b, 0, 0}, fin, v);
        v_i32_st_tnsr((int5){192, t, b, 0, 0}, fin, need);
    }
}
