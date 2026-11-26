// SPDX-License-Identifier: Apache-2.0
//
// v9 grouped-route dequant + dot core (Gaudi2 TPC) for the MXFP4 MoE decode kernels mxk9_*.c.
//
// v8 (mxk8_core.h) runs one pass per (4 weight rows, route): every route re-reads and re-unpacks its expert's packed
// rows, so a step whose tokens share a local expert unpacks that expert once per token. v9 groups the routes by
// local expert (in-kernel counting sort, as v8) into UNITS of up to MXK9_GMAX routes of one expert, and runs one pass
// per (4 weight rows, unit): the 4 rows are unpacked once per chunk (nibble split, lookup 457, bf16 scale multiply)
// and multiplied into one fp32 accumulator per route of the unit.
//
// Numerics: each (row, route) accumulator sees exactly v8's MAC chain (same dequantised bf16 weights, same x values,
// same v_bf16_mac_acc32_b order over chunks and sub-vectors, same horizontal sum), so every per-route sum is
// bit-identical to v8. Only the walk order changes.
//
// A unit of one route runs v8's 4-row pass on its row staged in vector local memory (VLM), so a step without shared
// experts runs v8's walk. A grouped unit runs 2-row half passes (4 rows x 2 routes do not fit the 40 vector
// registers). Activation rows of a grouped unit: w2 (rows of I <= 3072) stages two in VLM; w13 (rows of K = 6144,
// 12 KB: two do not fit the 16 KB VLM) reads them from an SRAM copy (mxk9g_w13.h).
//
// The pass is stamped once per unit size G (mxk9_pass.h, preprocessor, as mxk6_core.h; tpc-clang mis-compiles
// loops on a constant int parameter of an inlined function) and dispatched at run time.
#pragma once
#define FUNC_MXFP4_BF16 457

// largest unit; 2 at the served shapes (3 routes' half-pass registers do not fit next to w13's 12 KB VLM x row)
#ifndef MXK9_GMAX
#define MXK9_GMAX 2
#endif

#define MXK9_SCALES(bc, sb, zero8)                                                                      \
    {                                                                                                   \
        ushort128 u_ = as_ushort128(v_u8_unpack_b(sb, SW_UNPCK_8_TO_16 | SW_GROUP_0, zero8)) << 7;     \
        bc = v_u16_mov_dual_group_all_b(u_, 0xFFFFFFFF, 0, 0, 0, 0, MkWrA(0b11, 0b11, 0b11, 0b11), u_); \
    }
#define MXK9_PICK(bc, PAT) as_bfloat128(v_u16_shuffle_b(bc, as_uchar256(PAT), 0, bc))

// the dequantised 4 sub-vectors of one 512-k chunk of one row (v8 MXK8_DOT, split so several routes share it)
#define MXK9_DEQ(W, b, sc)                                                                              \
    {                                                                                                   \
        ushort128 lo_ = as_ushort128(v_u8_and_b(b, 0xF));                                               \
        ushort128 hi_ = as_ushort128(v_u8_shr_b(b, 4));                                                 \
        W[0] = as_bfloat128(v_i16_lookup_1c(lo_, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;                \
        W[1] = as_bfloat128(v_i16_lookup_1c(hi_, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;                \
        W[2] = as_bfloat128(v_i16_lookup_1c(lo_ >> 8, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;           \
        W[3] = as_bfloat128(v_i16_lookup_1c(hi_ >> 8, FUNC_MXFP4_BF16, 0, (short128)0)) * sc;           \
    }
#define MXK9_MAC(acc, W, xv)                                                                            \
    {                                                                                                   \
        acc = v_bf16_mac_acc32_b(W[0], xv[0], acc);                                                     \
        acc = v_bf16_mac_acc32_b(W[1], xv[1], acc);                                                     \
        acc = v_bf16_mac_acc32_b(W[2], xv[2], acc);                                                     \
        acc = v_bf16_mac_acc32_b(W[3], xv[3], acc);                                                     \
    }

// activation rows staged in vector local memory: MXK9_XCH chunks of 512 (12 = one K 6144 row, or two rows of
// I <= 3072 at chunk offsets 0 and MXK9_XCH / 2)
#ifndef MXK9_XCH
#define MXK9_XCH 12
#endif
#define MXK9_NCHMAX 12          // K <= 6144
__local__ bfloat128 v9_x[4 * MXK9_XCH];
#define MXK9_LDX(xv, c) { xv[0] = v9_x[4 * (c)]; xv[1] = v9_x[4 * (c) + 1]; xv[2] = v9_x[4 * (c) + 2]; xv[3] = v9_x[4 * (c) + 3]; }
// copy activation row `row` (nch chunks) of xp into v9_x from chunk slot c0; chunks past the tensor read as zeros
#define MXK9_STAGE_X(xp, row, nch, c0)                                                                  \
    {                                                                                                   \
        for (int c_ = 0; c_ < (nch); c_++) {                                                            \
            int5 cx_ = {c_ * 512, (row), 0, 0, 0};                                                      \
            v9_x[4 * ((c0) + c_)] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                           \
            v9_x[4 * ((c0) + c_) + 1] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                       \
            v9_x[4 * ((c0) + c_) + 2] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                       \
            v9_x[4 * ((c0) + c_) + 3] = v_bf16_ld_tnsr_b(cx_, xp);                                      \
        }                                                                                               \
    }
#define MXK9_STAGE(xp, row, nch, c0) MXK9_STAGE_X(xp, row, ((nch) < MXK9_XCH ? (nch) : MXK9_XCH), c0)
// chunk c of activation row `row` straight from global memory
#define MXK9_LDXG(xv, xp, c, row)                                                                       \
    {                                                                                                   \
        int5 cx_ = {(c) * 512, (row), 0, 0, 0};                                                         \
        xv[0] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                               \
        xv[1] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                               \
        xv[2] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                               \
        xv[3] = v_bf16_ld_tnsr_b(cx_, xp);                                                              \
    }

#define MXK9_HSUM(a) v_f32_reduce_add((a).v1 + (a).v2)

inline void mxk9_first(int n0, int n1, int n2, int n3, int e, tensor w, tensor s,
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

// Route grouping prologue (v8 mxk8_routes with ids [T, topk] and up to MXK9_QMAX routes): local routes ordered by
// local expert (stable: ascending route index within an expert), s9_sorted[g] = route q = t * topk + j,
// s9_ge[g] = its local expert. Then units of up to MXK9_GMAX consecutive sorted routes of one expert:
// s9_ub[u] = first sorted index, s9_un[u] = route count. Returns the local route count; nunits = unit count.
// gmax < MXK9_GMAX caps the unit size at run time (1 = one route per unit = v8's walk).
#ifndef MXK9_QMAX
#define MXK9_QMAX 512
#endif
#define MXK9_EMAX 128
__local__ int s9_eq[MXK9_QMAX];
__local__ int s9_cnt[MXK9_EMAX];
__local__ int s9_off[MXK9_EMAX];
__local__ int s9_sorted[MXK9_QMAX];
__local__ int s9_ge[MXK9_QMAX];
__local__ int s9_ub[MXK9_QMAX];
__local__ int s9_un[MXK9_QMAX];
__local__ int s9_spin[1];     // sink of mxk9g_w13.h's wait loop

inline int mxk9_routes(tensor ids, int T, int topk, int e_min, int e_local, int gmax, int& nunits)
{
    const int Q = T * topk;
    if (gmax > MXK9_GMAX) gmax = MXK9_GMAX;   // the kernel has passes for up to MXK9_GMAX routes
    if (gmax < 1) gmax = 1;
    for (int e = 0; e < e_local; e++) s9_cnt[e] = 0;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < topk; j++) {
            int5 ci = {j, t, 0, 0, 0};
            const int q = t * topk + j;
            int e = s_i32_ld_g((__global void*)gen_addr(ci, ids)) - e_min;
            if (e >= 0 && e < e_local) { s9_eq[q] = e; s9_cnt[e] = s9_cnt[e] + 1; }
            else s9_eq[q] = -1;
        }
    int off = 0;
    for (int e = 0; e < e_local; e++) { s9_off[e] = off; off += s9_cnt[e]; }
    for (int q = 0; q < Q; q++) {
        int e = s9_eq[q];
        if (e >= 0) { int o = s9_off[e]; s9_sorted[o] = q; s9_ge[o] = e; s9_off[e] = o + 1; }
    }
    int nu = 0;
    for (int g = 0; g < off; g++) {
        if (g > 0 && s9_ge[g] == s9_ge[g - 1] && s9_un[nu - 1] < gmax) { s9_un[nu - 1] = s9_un[nu - 1] + 1; continue; }
        s9_ub[nu] = g; s9_un[nu] = 1; nu++;
    }
#ifdef MXK9_CTRL_GROUP_OFF
    // test control only (never in a release build): grouped units start MXK9_CTRL_GROUP_OFF sorted routes late
    for (int u = 0; u < nu; u++)
        if (s9_un[u] > 1 && s9_ub[u] + MXK9_CTRL_GROUP_OFF + s9_un[u] <= off) s9_ub[u] = s9_ub[u] + MXK9_CTRL_GROUP_OFF;
#endif
    nunits = nu;
    return off;
}
