// SPDX-License-Identifier: Apache-2.0
//
// Dequant + dot core of the MXFP4 MoE decode kernels for the Gaudi 2 TPC (TPC-C, built with tpc-clang
// -march=gaudi2). Included by mxk8_w13.h and mxk8_w2.h.
//
// Weights are read in the layout vLLM already keeps on the device, with no repack and no bf16 copy:
//   w u8 [E, N, K/2]   two E2M1 values per byte (low nibble = even k, high nibble = odd k)
//   s u8 [E, N, K/32]  one E8M0 scale per 32 values
// Per 512-k chunk of one weight row: one 256-byte load, nibble split, table lookup 457 (E2M1 -> bf16, the same
// table the stock cast uses), a per-lane bf16 scale built from 16 scale bytes, exact bf16 multiply, fp32 MAC.
// The activation row is pre-permuted on the host per 512-k chunk (lane i of sub-vector p holds k = 512c + 4i + p)
// so the four nibble planes line up with four activation vectors without lane shuffles.
//
// One pass = 4 weight rows of one expert against one activation row staged in vector local memory (VLM). Chunks are
// processed in pairs: one 256-byte scale load serves two chunks. A kernel walks one software-pipelined stream of
// passes per TPC: the last chunk pair of a pass issues the loads of the first chunk pair of the NEXT pass (its rows
// and its expert) instead of reading past the row.
//   mxk8_first():   issue the loads of the first pass of a stream.
//   mxk8_pass_*():  run one pass on the loads in flight and leave the next pass's first loads in flight.
// The prefetch registers (bA / bB / sP) belong to the caller and are passed by reference.
//
// Compile-time switches that are off in the release build and kept for experiments: MXK8_PF (TPC software
// prefetch; about 2x slower, the prefetch instructions stall the core), MXK8_HSUM_X2 (paired reductions, within
// +-2%), MXK8_NCH6 (fully unrolled passes for rows of at most 3072 values; 1.6-1.9x slower, see below) and the
// MXK8_DIAG_* timing variants (wrong results on purpose, for profiling only).
#pragma once
#define FUNC_MXFP4_BF16 457

#define MXK8_SCALES(bc, sb, zero8)                                                                      \
    {                                                                                                   \
        ushort128 u_ = as_ushort128(v_u8_unpack_b(sb, SW_UNPCK_8_TO_16 | SW_GROUP_0, zero8)) << 7;     \
        bc = v_u16_mov_dual_group_all_b(u_, 0xFFFFFFFF, 0, 0, 0, 0, MkWrA(0b11, 0b11, 0b11, 0b11), u_); \
    }
#define MXK8_PICK(bc, PAT) as_bfloat128(v_u16_shuffle_b(bc, as_uchar256(PAT), 0, bc))

#define MXK8_DOT(acc, b, sc, xv)                                                                        \
    {                                                                                                   \
        ushort128 lo_ = as_ushort128(v_u8_and_b(b, 0xF));                                               \
        ushort128 hi_ = as_ushort128(v_u8_shr_b(b, 4));                                                 \
        bfloat128 W0_ = as_bfloat128(v_i16_lookup_1c(lo_, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;       \
        bfloat128 W1_ = as_bfloat128(v_i16_lookup_1c(hi_, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;       \
        bfloat128 W2_ = as_bfloat128(v_i16_lookup_1c(lo_ >> 8, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;  \
        bfloat128 W3_ = as_bfloat128(v_i16_lookup_1c(hi_ >> 8, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;  \
        acc = v_bf16_mac_acc32_b(W0_, xv[0], acc);                                                      \
        acc = v_bf16_mac_acc32_b(W1_, xv[1], acc);                                                      \
        acc = v_bf16_mac_acc32_b(W2_, xv[2], acc);                                                      \
        acc = v_bf16_mac_acc32_b(W3_, xv[3], acc);                                                      \
    }

// activation row staged in vector local memory: chunk c vector p = v8_x[4c + p], up to 12 chunks (K <= 6144)
#ifndef MXK8_XCH
#define MXK8_XCH 12
#endif
__local__ bfloat128 v8_x[4 * MXK8_XCH];
// MXK8_XTAIL 1 (w13 only; the Kimi build, K = 7168): rows longer than the stage. Chunks c >= MXK8_XCH are read from
// the activation tensor itself (4 loads per chunk per pass), so the pass functions take the tensor as one more
// argument. 14 staged chunks do not fit: VLM is shared with the batch buffers and the register spills.
#ifndef MXK8_XTAIL
#define MXK8_XTAIL 0
#endif
#if MXK8_XTAIL
__local__ int s8_xrow;                     // activation row of the staged route
#define MXK8_XPARAM , tensor xp
#define MXK8_XARG , xp
#define MXK8_LDX(xv, c)                                                                                 \
    {                                                                                                   \
        if ((c) < MXK8_XCH) {                                                                           \
            xv[0] = v8_x[4 * (c)]; xv[1] = v8_x[4 * (c) + 1]; xv[2] = v8_x[4 * (c) + 2]; xv[3] = v8_x[4 * (c) + 3]; \
        } else {                                                                                        \
            int5 cx_ = {(c) * 512, s8_xrow, 0, 0, 0};                                                   \
            xv[0] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                           \
            xv[1] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                           \
            xv[2] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                           \
            xv[3] = v_bf16_ld_tnsr_b(cx_, xp);                                                          \
        }                                                                                               \
    }
#define MXK8_XNCH(nch) ((nch) < MXK8_XCH ? (nch) : MXK8_XCH)
#define MXK8_XROW(row) s8_xrow = (row);
#else
#define MXK8_XPARAM
#define MXK8_XARG
#define MXK8_LDX(xv, c) { xv[0] = v8_x[4 * (c)]; xv[1] = v8_x[4 * (c) + 1]; xv[2] = v8_x[4 * (c) + 2]; xv[3] = v8_x[4 * (c) + 3]; }
#define MXK8_XNCH(nch) (nch)
#define MXK8_XROW(row)
#endif
// copy activation row `row` (nch chunks, nch even) of xp into v8_x; chunks past the tensor read as zeros
#define MXK8_STAGE_X(xp, row, nch)                                                                      \
    {                                                                                                   \
        MXK8_XROW(row)                                                                                  \
        for (int c_ = 0; c_ < MXK8_XNCH(nch); c_++) {                                                   \
            int5 cx_ = {c_ * 512, (row), 0, 0, 0};                                                      \
            v8_x[4 * c_] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                    \
            v8_x[4 * c_ + 1] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                \
            v8_x[4 * c_ + 2] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                \
            v8_x[4 * c_ + 3] = v_bf16_ld_tnsr_b(cx_, xp);                                               \
        }                                                                                               \
    }

#ifdef MXK8_DIAG_NOHSUM          // timing only: no horizontal sum
#define MXK8_HSUM(a) ((a).v1 + (a).v2)
#else
#define MXK8_HSUM(a) v_f32_reduce_add((a).v1 + (a).v2)
#endif
#if defined(MXK8_HSUM_X2) && !defined(MXK8_DIAG_NOHSUM)
#define MXK8_HSUM4(acc, h0, h1, h2, h3)                                                                 \
    float128 q01_ = v_f32_reduce_x2_add(acc[0].v1 + acc[0].v2, acc[1].v1 + acc[1].v2);                  \
    float128 q23_ = v_f32_reduce_x2_add(acc[2].v1 + acc[2].v2, acc[3].v1 + acc[3].v2);                  \
    float64 h0 = q01_.v1, h1 = q01_.v2, h2 = q23_.v1, h3 = q23_.v2;
#else
#define MXK8_HSUM4(acc, h0, h1, h2, h3)                                                                 \
    float64 h0 = MXK8_HSUM(acc[0]), h1 = MXK8_HSUM(acc[1]), h2 = MXK8_HSUM(acc[2]), h3 = MXK8_HSUM(acc[3]);
#endif

inline void mxk8_first(int n0, int n1, int n2, int n3, int e, tensor w, tensor s,
                       uchar256 (&bA)[4], uchar256 (&bB)[4], uchar256 (&sP)[4])
{
    const int nn[4] = {n0, n1, n2, n3};
    _Pragma("unroll") for (int j = 0; j < 4; j++) {
        int5 cw = {0, nn[j], e, 0, 0};
        bA[j] = v_u8_ld_tnsr_b(cw, w);
        sP[j] = v_u8_ld_tnsr_b(cw, s);
        cw[0] = 256;
        bB[j] = v_u8_ld_tnsr_b(cw, w);
    }
}

// One pass: rows n0..n3 of expert e against the activation row in v8_x (nch chunks, nch even). Rows m0..m3 of
// expert f are the next pass (their first chunk pair is loaded in the last pair of this one).
// Result layouts of the callers:
//   mxk8_pass_dot: r.v1 = {lane 0: row 0, other lanes: row 2}, r.v2 = {row 1, row 3} with lsel = lane id 4 == 0,
//                  or {even lanes: row 0, odd lanes: row 2} / {even: row 1, odd: row 3} with lsel = lane id & 1
//                  (the packed w2 sums);
//   mxk8_pass_act: gated activation of rows [g0, g1, u0, u1]: r.v1 = act(g0, u0), r.v2 = act(g1, u1);
//   mxk8_pass_h4:  the four horizontal sums, for callers that batch their epilogue.
// MXK8_PF 1 (off): while pass i runs, chunk pair c also prefetches chunks c+2, c+3 of the next pass's 4 rows
// into the TPC cache, and the first pair prefetches the next pass's scale rows. MXK8_PF_STRIDE bytes per prefetch.
#ifndef MXK8_PF
#define MXK8_PF 0
#endif
#ifndef MXK8_PF_STRIDE
#define MXK8_PF_STRIDE 128
#endif
#ifndef MXK8_PF_SW
#define MXK8_PF_SW 0
#endif
#if MXK8_PF
#define MXK8_PREFETCH(c)                                                                                \
    {                                                                                                   \
        if ((c) + 2 < nch) {                                                                            \
            _Pragma("unroll") for (int j = 0; j < 4; j++) {                                             \
                _Pragma("unroll") for (int o = 0; o < 512; o += MXK8_PF_STRIDE) {                       \
                    int5 cp = {((c) + 2) * 256 + o, mm[j], f, 0, 0};                                    \
                    prefetch(gen_addr(cp, w), MXK8_PF_SW, 1, 0);                                        \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
        if ((c) == 0) {                                                                                 \
            _Pragma("unroll") for (int j = 0; j < 4; j++) {                                             \
                _Pragma("unroll") for (int o = 0; o < 16 * nch; o += MXK8_PF_STRIDE) {                  \
                    int5 cp = {o, mm[j], f, 0, 0};                                                      \
                    prefetch(gen_addr(cp, s), MXK8_PF_SW, 1, 0);                                        \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
    }
#else
#define MXK8_PREFETCH(c)
#endif

// Timing-only variants (wrong results): MXK8_DIAG_NOW (no weight loads), MXK8_DIAG_NOX (no x loads from VLM)
#ifdef MXK8_DIAG_NOW
#define MXK8_LDW(cw, w) ((uchar256)0x35)
#else
#define MXK8_LDW(cw, w) v_u8_ld_tnsr_b(cw, w)
#endif
#ifdef MXK8_DIAG_NOX
#undef MXK8_LDX
#define MXK8_LDX(xv, c) { xv[0] = (bfloat128)0.5f; xv[1] = (bfloat128)0.25f; xv[2] = (bfloat128)0.125f; xv[3] = (bfloat128)1.0f; }
#endif

// one chunk pair c of the pass; the loads it issues are chunk pair cn of rows RR[] of expert EE
#define MXK8_PAIR(c, cn, RR, EE)                                                                        \
    {                                                                                                   \
        MXK8_LDX(xv, c);                                                                                \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK8_SCALES(bc, sP[j], zero8);                                                              \
            bfloat128 sc = MXK8_PICK(bc, pat0);                                                         \
            uchar256 b = bA[j];                                                                         \
            int5 cw = {(cn) * 256, RR[j], EE, 0, 0};                                                    \
            bA[j] = MXK8_LDW(cw, w);                                                                    \
            MXK8_DOT(acc[j], b, sc, xv);                                                                \
        }                                                                                               \
        MXK8_PREFETCH(c)                                                                                \
        MXK8_LDX(xv, (c) + 1);                                                                          \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK8_SCALES(bc, sP[j], zero8);                                                              \
            bfloat128 sc = MXK8_PICK(bc, pat1);                                                         \
            uchar256 b = bB[j];                                                                         \
            int5 cw = {((cn) + 1) * 256, RR[j], EE, 0, 0};                                              \
            int5 cs = {(cn) * 16, RR[j], EE, 0, 0};                                                     \
            bB[j] = MXK8_LDW(cw, w);                                                                    \
            sP[j] = v_u8_ld_tnsr_b(cs, s);                                                              \
            MXK8_DOT(acc[j], b, sc, xv);                                                                \
        }                                                                                               \
    }

// the pass: pairs 0 .. nch-4 load the next pair of the same rows; the last pair (peeled) loads the first
// pair of the next pass
#define MXK8_PASS_BODY                                                                                  \
    const int nn[4] = {n0, n1, n2, n3};                                                                 \
    const int mm[4] = {m0, m1, m2, m3};                                                                 \
    const ushort128 pat1 = pat0 | 0x2020;                                                               \
    float128 acc[4];                                                                                    \
    _Pragma("unroll") for (int j = 0; j < 4; j++) { acc[j].v1 = 0; acc[j].v2 = 0; }                    \
    bfloat128 xv[4];                                                                                    \
    for (int c = 0; c < nch - 2; c += 2) MXK8_PAIR(c, c + 2, nn, e)                                     \
    MXK8_PAIR(nch - 2, 0, mm, f)                                                                        \
    MXK8_HSUM4(acc, h0, h1, h2, h3)

#ifndef MXK8_NCH6
#define MXK8_NCH6 0
#endif
#if MXK8_NCH6
// Short rows (K or I <= 3072: nch == 6, i.e. 3 chunk pairs per pass), the pass fully unrolled:
//  - the scales of a whole row (<= 96 bytes of the 256-byte vector) come from ONE load per row per pass (offset 0),
//    issued by the previous pass's last pair, instead of one per row per pair: 4 instead of 12 scale loads per
//    pass. Pair j reads its 32 scale bytes as dual group j of the unpacked vector (MXK8_SCALESJ): the same bytes
//    and therefore the same bits as the default path.
//  - sP[] (this pass's row scales) stays live over the 3 pairs; the last pair reloads it for the next pass.
// Callers must guarantee nch == 6. Measured on the gpt-oss shapes and kept OFF: 1.6-1.9x slower (keeping sP live
// over the unrolled pass costs more than the 8 scale loads it saves).
#define MXK8_SCALESJ(bc, sb, zero8, J)                                                                  \
    {                                                                                                   \
        ushort128 u_ = as_ushort128(v_u8_unpack_b(sb, SW_UNPCK_8_TO_16 | SW_GROUP_0, zero8)) << 7;     \
        bc = v_u16_mov_dual_group_all_b(u_, 0xFFFFFFFF, J, J, J, J, MkWrA(0b11, 0b11, 0b11, 0b11), u_); \
    }
// pair J (chunks 2J, 2J+1) of the pass; LAST: also load the next pass's first pair and row scales (rows RR, EE)
#define MXK8_PAIR6(J, LAST, RR, EE)                                                                     \
    {                                                                                                   \
        MXK8_LDX(xv, 2 * (J));                                                                          \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK8_SCALESJ(bc, sP[j], zero8, J);                                                          \
            bfloat128 sc = MXK8_PICK(bc, pat0);                                                         \
            uchar256 b = bA[j];                                                                         \
            int5 cw = {(LAST) ? 0 : (2 * (J) + 2) * 256, RR[j], EE, 0, 0};                              \
            bA[j] = MXK8_LDW(cw, w);                                                                    \
            MXK8_DOT(acc[j], b, sc, xv);                                                                \
        }                                                                                               \
        MXK8_LDX(xv, 2 * (J) + 1);                                                                      \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK8_SCALESJ(bc, sP[j], zero8, J);                                                          \
            bfloat128 sc = MXK8_PICK(bc, pat1);                                                         \
            uchar256 b = bB[j];                                                                         \
            int5 cw = {(LAST) ? 256 : (2 * (J) + 3) * 256, RR[j], EE, 0, 0};                            \
            bB[j] = MXK8_LDW(cw, w);                                                                    \
            if (LAST) { int5 cs = {0, RR[j], EE, 0, 0}; sP[j] = v_u8_ld_tnsr_b(cs, s); }                \
            MXK8_DOT(acc[j], b, sc, xv);                                                                \
        }                                                                                               \
    }
#undef MXK8_PASS_BODY
#define MXK8_PASS_BODY                                                                                  \
    const int nn[4] = {n0, n1, n2, n3};                                                                 \
    const int mm[4] = {m0, m1, m2, m3};                                                                 \
    const ushort128 pat1 = pat0 | 0x2020;                                                               \
    float128 acc[4];                                                                                    \
    _Pragma("unroll") for (int j = 0; j < 4; j++) { acc[j].v1 = 0; acc[j].v2 = 0; }                    \
    bfloat128 xv[4];                                                                                    \
    (void)nch;                                                                                          \
    MXK8_PAIR6(0, 0, nn, e)                                                                             \
    MXK8_PAIR6(1, 0, nn, e)                                                                             \
    MXK8_PAIR6(2, 1, mm, f)                                                                             \
    MXK8_HSUM4(acc, h0, h1, h2, h3)
#endif

inline float128 mxk8_pass_act(int n0, int n1, int n2, int n3, int e, int m0, int m1, int m2, int m3, int f,
                              int nch, tensor w, tensor s, ushort128 pat0, uchar256 zero8, float alpha,
                              float limit, float beta, float ascale,
                              uchar256 (&bA)[4], uchar256 (&bB)[4], uchar256 (&sP)[4] MXK8_XPARAM)
{
    MXK8_PASS_BODY
    float128 r;
#ifdef MXK8_DIAG_NOACT            // timing only: no activation
    r.v1 = h0 + h2; r.v2 = h1 + h3;
    return r;
#endif
    float64 ga = v_f32_min_b(h0, limit), gb = v_f32_min_b(h1, limit);
    float64 ua = v_f32_max_b(v_f32_min_b(h2, limit), -limit), ub = v_f32_max_b(v_f32_min_b(h3, limit), -limit);
    r.v1 = ga * v_sigmoid_f32(ga * alpha) * ((ua + beta) * ascale);
    r.v2 = gb * v_sigmoid_f32(gb * alpha) * ((ub + beta) * ascale);
    return r;
}

// the four horizontal sums (broadcast to all lanes), for callers that batch their epilogue
inline void mxk8_pass_h4(int n0, int n1, int n2, int n3, int e, int m0, int m1, int m2, int m3, int f,
                         int nch, tensor w, tensor s, ushort128 pat0, uchar256 zero8,
                         uchar256 (&bA)[4], uchar256 (&bB)[4], uchar256 (&sP)[4], float64 (&h)[4] MXK8_XPARAM)
{
    MXK8_PASS_BODY
    h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3;
}

// the gated activation, lane-wise: a = min(g, L) * sigmoid(alpha * min(g, L)) * (clamp(u, -L, L) + beta) * ascale
#define MXK8_ACT(g, u) (v_f32_min_b(g, limit) * v_sigmoid_f32(v_f32_min_b(g, limit) * alpha) *                  \
                        ((v_f32_max_b(v_f32_min_b(u, limit), -limit) + beta) * ascale))

// lsel: lane mask selecting row 0 / row 1 (lid4 == 0 for the unpacked layout, (lid4 & 1) == 0 for the packed one)
inline float128 mxk8_pass_dot(int n0, int n1, int n2, int n3, int e, int m0, int m1, int m2, int m3, int f,
                              int nch, tensor w, tensor s, ushort128 pat0, uchar256 zero8, uint64 lsel,
                              uchar256 (&bA)[4], uchar256 (&bB)[4], uchar256 (&sP)[4] MXK8_XPARAM)
{
    MXK8_PASS_BODY
    float128 r;
    r.v1 = v_f32_sel_eq_u32_b(lsel, 0u, h0, h2);
    r.v2 = v_f32_sel_eq_u32_b(lsel, 0u, h1, h3);
    return r;
}

// Route grouping prologue (scalar, SLM): the local routes ordered by local expert. s8_sorted[g] = route,
// s8_ge[g] = its local expert, s8_eq[q] = local expert of route q or -1 (expert on another EP rank). Returns the
// number of local routes.
#define MXK8_QMAX 128
#define MXK8_EMAX 128
__local__ int s8_eq[MXK8_QMAX];
__local__ int s8_cnt[MXK8_EMAX];
__local__ int s8_off[MXK8_EMAX];
__local__ int s8_sorted[MXK8_QMAX];
__local__ int s8_ge[MXK8_QMAX];

inline int mxk8_routes(tensor ids, int Q, int e_min, int e_local)
{
    for (int e = 0; e < e_local; e++) s8_cnt[e] = 0;
    for (int q = 0; q < Q; q++) {
        int5 ci = {q, 0, 0, 0, 0};
        int e = s_i32_ld_g((__global void*)gen_addr(ci, ids)) - e_min;
        if (e >= 0 && e < e_local) { s8_eq[q] = e; s8_cnt[e] = s8_cnt[e] + 1; }
        else s8_eq[q] = -1;
    }
    int off = 0;
    for (int e = 0; e < e_local; e++) { s8_off[e] = off; off += s8_cnt[e]; }
    for (int q = 0; q < Q; q++) {
        int e = s8_eq[q];
        if (e >= 0) { int o = s8_off[e]; s8_sorted[o] = q; s8_ge[o] = e; s8_off[e] = o + 1; }
    }
    return off;
}
