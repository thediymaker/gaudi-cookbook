// k1_kg: gather + dequantize the index-K range of every request from the paged U8 cache, with the tag check.
//   tags i32 [nrows*2] (the flat (owner, position) pairs), data bf16 [nrows][66] holding 132 bytes (128 e4m3 + fp32
//   scale), bt i32 [B][NB], own i32 [B][NB]
//   out K i32 [B][Lk][128]: the fp32 bits of index_u8_unpack(row).float() for a valid key, NaN for a key whose row is
//   unallocated or fails the tag check (the dot then gives NaN in every head: k1_sc reads that as "unseen").
//   index space {Lk/64, B}; scalars BS, tagcheck. tags is input 0 and K is declared int32: under HPU graph capture a
//   custom op's outputs take the dtype of input 0.
// Per key: one 256-byte unpacking load (lane i = bytes i, 64+i, 128+i, 192+i), the hardware e4m3 -> f32 convert of
// byte 0 (d = i) and byte 1 (d = 64+i), times the row's power-of-two scale (exact). Tags: 32 (owner, pos) pairs per
// load, compared on all lanes, pair-ANDed; each key's flag is broadcast with a uniform shuffle + a dual-group move.
#include "k1_common.h"
#ifdef KG_NOST2
#define KG_ST2(x)
#else
#define KG_ST2(x) x
#endif
#ifdef KG_ORDER_HALVES
#define KG_STORES ST0(0) ST0(1) ST0(2) ST0(3) ST0(4) ST0(5) ST0(6) ST0(7) ST1(0) ST1(1) ST1(2) ST1(3) ST1(4) ST1(5) ST1(6) ST1(7)
#else
#define KG_STORES ST0(0) ST1(0) ST0(1) ST1(1) ST0(2) ST1(2) ST0(3) ST1(3) ST0(4) ST1(4) ST0(5) ST1(5) ST0(6) ST1(6) ST0(7) ST1(7)
#endif
#ifdef KG_NOLOAD
#define LDU(JJ) const int64 U##JJ = L * (JJ + 3);
#else
#define LDU(JJ) const int64 U##JJ = v_i32_ld_tnsr_b((int5){0, blk >= 0 ? r0 + (DGJ) * 8 + (JJ) + jb : 0, 0, 0, 0}, \
                                              data, SW_UNPACK | SW_UNPCK_8_TO_32, 0, 1, 0);
#endif
#define KEY(D, JJ)                                                                                                  \
    {                                                                                                               \
        const int j = (DGJ) * 8 + (JJ) + jb;                                                                        \
        const int64 okv = v_i32_shuffle_b(D, PAT32((int64)(2 * (JJ))), 0, zero, 1, 0);                              \
        int64 S = as_int64(v_u8_shuffle_b(as_uchar256(U##JJ), pscale, 0, as_uchar256(zero), 1, 0));                 \
        S = v_i32_mov_dual_group_all_b(S, 0xFFFFFFFF, 0, 0, 0, 0, MkWrA(0b11, 0b11, 0b11, 0b11), S, 1, 0);           \
        const float64 sc = as_float64(S);                                                                           \
        const int64 f0 = as_int64(v_convert_f8_to_f32_b((minifloat256)U##JJ, SW_LANE_0, 0.0f, 1, 0) * sc);          \
        const int64 f1 = as_int64(v_convert_f8_to_f32_b((minifloat256)(U##JJ >> 8), SW_LANE_0, 0.0f, 1, 0) * sc);   \
        F0##JJ = SEL_EQ(okv, 0, qnan, f0);                                                                          \
        F1##JJ = SEL_EQ(okv, 0, qnan, f1);                                                                          \
        (void)j;                                                                                                    \
    }
#define ST0(JJ) v_i32_st_tnsr((int5){0, s0 + (DGJ) * 8 + (JJ) + jb, b, 0, 0}, K, F0##JJ);
#define ST1(JJ) v_i32_st_tnsr((int5){64, s0 + (DGJ) * 8 + (JJ) + jb, b, 0, 0}, K, F1##JJ);
// 8 keys of one dual group: their 8 row loads first (independent of the stores), then decode + store
#define DG8(PV, DG)                                                                                                 \
    {                                                                                                               \
        enum { DGJ = DG };                                                                                          \
        LDU(0) LDU(1) LDU(2) LDU(3) LDU(4) LDU(5) LDU(6) LDU(7)                                                     \
        const int64 D = v_i32_mov_dual_group_all_b(PV, 0xFFFFFFFF, DG, DG, DG, DG, MkWrA(0b11, 0b11, 0b11, 0b11),   \
                                                   PV, 1, 0);                                                       \
        int64 F00, F01, F02, F03, F04, F05, F06, F07, F10, F11, F12, F13, F14, F15, F16, F17;                        \
        KEY(D, 0) KEY(D, 1) KEY(D, 2) KEY(D, 3) KEY(D, 4) KEY(D, 5) KEY(D, 6) KEY(D, 7)                             \
        KG_STORES                                                                                                   \
    }
void main(tensor tags, tensor data, tensor bt, tensor own, tensor K, int BS, int tagcheck)
{
    // scalar loads below read tensors written by earlier kernels; the scalar data cache can hold lines of an earlier
    // launch on this core at the same addresses (seen: one stale k1_cm offset/mask in a 252-case run). Drop them.
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 L = LANE();
    const int64 li = L & 15;
    const int64 zero = 0, one = 1, qnan = 0x7FC00000;
    const uchar256 pscale = as_uchar256((int64)(0x8E8A8682));     // every lane <- bytes (2,6,10,14): bytes 128..131
    const uchar256 ppart = PAT32(li ^ 1);
    const int64 even = SEL_EQ(L & 1, 0, one, zero);
    for (int b = is[1]; b < ie[1]; b++) {
        for (int g = is[0]; g < ie[0]; g++) {
            const int s0 = g * 64;
            const int bi = s0 / BS;
            const int blk = K1_LD_I(bt, bi, b, 0);
            const int owner = K1_LD_I(own, bi, b, 0);
            const int r0 = blk >= 0 ? blk * BS + (s0 - bi * BS) : 0;
            int64 pA = one, pB = one;
            if (tagcheck) {
                const int64 A = v_i32_ld_tnsr_b((int5){2 * r0, 0, 0, 0, 0}, tags);
                const int64 Bv = v_i32_ld_tnsr_b((int5){2 * r0 + 64, 0, 0, 0, 0}, tags);
                const int64 EA = SEL_EQ(even, 1, (int64)owner, (L >> 1) + s0);
                const int64 EB = SEL_EQ(even, 1, (int64)owner, (L >> 1) + (s0 + 32));
                pA = SEL_EQ(A, EA, one, zero);
                pB = SEL_EQ(Bv, EB, one, zero);
                pA = pA & v_i32_shuffle_b(pA, ppart, 0, zero, 1, 0);     // lanes 2i, 2i+1: row i of the 32
                pB = pB & v_i32_shuffle_b(pB, ppart, 0, zero, 1, 0);
            }
            if (blk < 0) { pA = zero; pB = zero; }
            int jb = 0;
            DG8(pA, 0) DG8(pA, 1) DG8(pA, 2) DG8(pA, 3)
            jb = 32;
            DG8(pB, 0) DG8(pB, 1) DG8(pB, 2) DG8(pB, 3)
        }
    }
}
