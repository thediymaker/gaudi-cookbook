// Shared TPC helpers of the k1 kernels (Gaudi2).
#pragma once
#define LANE() ((int64)read_lane_id_4b_b())
// 32-bit-lane shuffle pattern: dest lane (within its dual group of 16) takes source lane srcl (int64, 0..15)
// Gaudi2 i32 shuffle (measured): pattern byte 0 of lane l: bits 0-2 = source lane in group, 0x80 enable; the
// group (0x20) bit that applies to lane l is the one in lane l^1's pattern.
#define PAT32P(srcl, srcp) as_uchar256((((srcl) & 7) | (((srcp) & 8) << 2)) | 0x80)
#define PAT32(srcl) PAT32P(srcl, srcl)
// broadcast lane 15 of each dual group to the whole dual group
// inclusive prefix sum of x over 64 lanes (x small ints). li = lane & 15.
// inclusive prefix sum of x over 64 lanes (x small ints); li = lane & 15. Only shuffle forms measured to work:
// sources inside the destination's own 8-lane group, or every lane from the other group of its dual group.
#define K1_GSRC(li, s) v_i32_sel_geq_i32_b((li) & 7, s, (li) - (s), li, 0, 0, 1, 0)
inline int64 k1_prefix64(int64 x, int64 li)
{
    const int64 z = 0;
    int64 t;
    t = v_i32_shuffle_b(x, PAT32(K1_GSRC(li, 1)), 0, z, 1, 0);
    x = x + v_i32_sel_geq_i32_b(li & 7, 1, t, z, 0, 0, 1, 0);
    t = v_i32_shuffle_b(x, PAT32(K1_GSRC(li, 2)), 0, z, 1, 0);
    x = x + v_i32_sel_geq_i32_b(li & 7, 2, t, z, 0, 0, 1, 0);
    t = v_i32_shuffle_b(x, PAT32(K1_GSRC(li, 4)), 0, z, 1, 0);
    x = x + v_i32_sel_geq_i32_b(li & 7, 4, t, z, 0, 0, 1, 0);
    // lower group's total (its lane 7) into the upper group
    t = v_i32_shuffle_b(x, PAT32((int64)7), 0, z, 1, 0);          // every lane <- lane 7 (lower group)
    x = x + v_i32_sel_geq_i32_b(li, 8, t, z, 0, 0, 1, 0);
    int64 T = v_i32_shuffle_b(x, PAT32((int64)15), 0, z, 1, 0);   // dual-group total
    int64 A = v_i32_mov_dual_group_all_b(T, 0xFFFFFFFF, 0, 0, 1, 2, MkWrA(0b00, 0b11, 0b11, 0b11), z, 1, 0);
    int64 Bv = v_i32_mov_dual_group_all_b(T, 0xFFFFFFFF, 0, 0, 0, 1, MkWrA(0b00, 0b00, 0b11, 0b11), z, 1, 0);
    int64 C = v_i32_mov_dual_group_all_b(T, 0xFFFFFFFF, 0, 0, 0, 0, MkWrA(0b00, 0b00, 0b00, 0b11), z, 1, 0);
    return x + A + Bv + C;
}

// bool256 predicate (per byte) from an i32 0/-1 lane mask, for st_tnsr_sqz
#define K1_P256(m) v_u8_cmp_neq_b(as_uchar256(m), (unsigned char)0, 0, (bool256)0, 1, 0)
#define K1_LD_I(t, c0, c1, c2) s_i32_ld_g((__global void*)gen_addr((int5){c0, c1, c2, 0, 0}, t), 0, 0, 1, 0)
#define K1_LD_I4(t, c0, c1, c2, c3) s_i32_ld_g((__global void*)gen_addr((int5){c0, c1, c2, c3, 0}, t), 0, 0, 1, 0)
#define K1_LD_F(t, c0, c1, c2) s_f32_ld_g((__global void*)gen_addr((int5){c0, c1, c2, 0, 0}, t), 0, 0, 1, 0)
// i32 lane select helpers: a OP b ? x : y
#define SEL_EQ(a, b, x, y) v_i32_sel_eq_i32_b(a, b, x, y, 0, 0, 1, 0)
#define SEL_GE(a, b, x, y) v_i32_sel_geq_i32_b(a, b, x, y, 0, 0, 1, 0)
#define SEL_GT(a, b, x, y) v_i32_sel_grt_i32_b(a, b, x, y, 0, 0, 1, 0)
#define SEL_LE(a, b, x, y) v_i32_sel_leq_i32_b(a, b, x, y, 0, 0, 1, 0)

// count vector of a radix pass: lane j (0..16) = #(n >= j) summed over the 17 per-lane counters
#define K1_CNT_DECL int64 k1c0 = 0, k1c1 = 0, k1c2 = 0, k1c3 = 0, k1c4 = 0, k1c5 = 0, k1c6 = 0, k1c7 = 0, k1c8 = 0, \
    k1c9 = 0, k1c10 = 0, k1c11 = 0, k1c12 = 0, k1c13 = 0, k1c14 = 0, k1c15 = 0, k1c16 = 0;
#define K1_C1(j) k1c##j += SEL_GE(n, j, one, zero);
#define K1_CNT_ADD(n_) { int64 n = (n_); K1_C1(0) K1_C1(1) K1_C1(2) K1_C1(3) K1_C1(4) K1_C1(5) K1_C1(6) K1_C1(7) \
    K1_C1(8) K1_C1(9) K1_C1(10) K1_C1(11) K1_C1(12) K1_C1(13) K1_C1(14) K1_C1(15) K1_C1(16) }
#define K1_R1(j) out = SEL_EQ(L, j, v_i32_reduce_add(k1c##j), out);
#define K1_CNT_OUT(out) { out = 0; K1_R1(0) K1_R1(1) K1_R1(2) K1_R1(3) K1_R1(4) K1_R1(5) K1_R1(6) K1_R1(7) K1_R1(8) \
    K1_R1(9) K1_R1(10) K1_R1(11) K1_R1(12) K1_R1(13) K1_R1(14) K1_R1(15) K1_R1(16) }

// total over chunks of one pass's count vectors, then the pass's digit: jstar = #(lanes with tot >= K) - 1
inline int64 k1_tot(tensor cn, int nch, int t, int b)
{
    int64 tot = 0;
    for (int c = 0; c < nch; c++) tot += v_i32_ld_tnsr_b((int5){0, c, t, b, 0}, cn);
    return tot;
}
inline int64 k1_digit(int64 tot, int K)
{
    const int64 one = 1, zero = 0;
    return v_i32_reduce_add(SEL_GE(tot, K, one, zero)) - 1;
}
