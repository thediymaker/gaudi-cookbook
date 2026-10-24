// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_deq (GLM-5.x prefill): MXFP4 (packed E2M1 + E8M0, the served layout) -> bf16, natural layout.
//   in  w u8 [E, N, K/2] (two E2M1 values per byte, low nibble first), s u8 [E, N, K/32] (one E8M0 per 32 values)
//   out bf16 [E, N, K], out[e, n, k] = W[e, n, k]
// The prefill path unpacks each local expert once with this kernel and feeds the stock bf16 MoE op on the matrix
// engine. The lookup (function 457) and the bf16 scale multiply are the ones the stock cast uses, so the unpack is
// exact and the bf16 MoE op on it gives the stock MXFP4 op's result.
// rowperm: kept in the ABI, always 0 (the glue rejects anything else).
// Index space: npts points (one per TPC); point ip owns rows [ip * N / npts, (ip + 1) * N / npts) of every expert.
#define FUNC_MXFP4_BF16 457

#define DQ_SCALES(bc, sb, zero8)                                                                        \
    {                                                                                                   \
        ushort128 u_ = as_ushort128(v_u8_unpack_b(sb, SW_UNPCK_8_TO_16 | SW_GROUP_0, zero8)) << 7;     \
        bc = v_u16_mov_dual_group_all_b(u_, 0xFFFFFFFF, 0, 0, 0, 0, MkWrA(0b11, 0b11, 0b11, 0b11), u_); \
    }
#define DQ_PICK(bc, PAT) as_bfloat128(v_u16_shuffle_b(bc, as_uchar256(PAT), 0, bc))

void main(tensor w, tensor s, tensor out, int rowperm, int npts)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int nch = get_dim_size(w, 0) / 256;
    const int N = get_dim_size(w, 1);
    const int E = get_dim_size(w, 2);
    ushort128 lid = read_lane_id_2b_b();
    const uchar256 zero8 = 0;

    // Output vector v of chunk c holds values 128v .. 128v+127 = packed bytes 64v .. 64v+63 = dual group v of the
    // 256-byte load (u16 lane j of the output reads one nibble of byte 64v + j/2):
    // 1. broadcast dual group v of the packed bytes to all four dual groups (mov_dual_group_all);
    // 2. byte shuffle inside each dual group: u8 lane b (dual group k = b / 64, local lb = b % 64) gets local byte
    //    16k + (lb/2)/2. The low byte of u16 lane j = 32k + lb/2 then holds byte 64v + j/2; the high byte is a copy
    //    the lookup ignores;
    // 3. per-lane shift: 4 on odd j (high nibble), 0 on even j; mask to 4 bits; lookup; multiply by the scale of
    //    block (128v + j) / 32 = 4v + j / 32.
    // (Merging two shuffles through the income with disabled pattern lanes does not work: the group bit 0x20 of the
    // enabled lanes is then ignored.)
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 lb = lid8 & 63;
    const uchar256 mm = lb >> 1;
    const uchar256 pat = (((lid8 >> 6) << 4) + (mm >> 1)) | 0x80;
    const uchar256 shv = (mm & 1) << 2;
    const ushort128 zero16 = 0;
    const ushort128 ps0 = ((lid >> 5) + 0) | 0x80, ps1 = ((lid >> 5) + 4) | 0x80;
    const ushort128 ps2 = ((lid >> 5) + 8) | 0x80, ps3 = ((lid >> 5) + 12) | 0x80;
#define DQN_VEC(V, PS)                                                                                       \
    {                                                                                                        \
        ushort128 x_ = v_u16_mov_dual_group_all_b(as_ushort128(b8), 0xFFFFFFFF, V, V, V, V,                  \
                                                  MkWrA(0b11, 0b11, 0b11, 0b11), zero16);                   \
        uchar256 y_ = v_u8_shuffle_b(as_uchar256(x_), pat, 0, zero8);                                       \
        y_ = (y_ >> shv) & 0xF;                                                                              \
        bfloat128 sc_ = DQ_PICK(bc, PS);                                                                     \
        bfloat128 w_ = as_bfloat128(v_i16_lookup_1c(as_ushort128(y_), FUNC_MXFP4_BF16, 0, (short128)0)) * sc_; \
        v_bf16_st_tnsr(co, out, w_);                                                                         \
        co[0] += 128;                                                                                        \
    }
    for (int ip = is[0]; ip < ie[0]; ip++)
    for (int n = ip * N / npts; n < (ip + 1) * N / npts; n++) {
        for (int e = 0; e < E; e++) {
            // software pipeline: chunk c+1's packed bytes and scales are in flight while chunk c is unpacked
            int5 cw = {0, n, e, 0, 0};
            uchar256 bn = v_u8_ld_tnsr_b(cw, w);
            uchar256 sn = v_u8_ld_tnsr_b(cw, s);
            for (int c = 0; c < nch; c++) {
                uchar256 b8 = bn, sb = sn;
                int5 cw2 = {(c + 1) * 256, n, e, 0, 0};
                int5 cs2 = {(c + 1) * 16, n, e, 0, 0};
                bn = v_u8_ld_tnsr_b(cw2, w);
                sn = v_u8_ld_tnsr_b(cs2, s);
                ushort128 bc;
                DQ_SCALES(bc, sb, zero8);
                int5 co = {c * 512, n, e, 0, 0};
                DQN_VEC(0, ps0) DQN_VEC(1, ps1) DQN_VEC(2, ps2) DQN_VEC(3, ps3)
            }
        }
    }
}
