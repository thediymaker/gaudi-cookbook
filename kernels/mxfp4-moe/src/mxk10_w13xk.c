// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w13xk (v10, Kimi-K2.7, K 7168): the v10 w13 levers (packed reduction tree carried into the next pass,
// MXG_W13C 2, and the one-branch flush, MXG_FB 2) on a w13 kernel for rows longer than the VLM stage. x is PERMUTED
// per 512-k chunk (mxk_w13x's contract); chunks 0-11 of the route's row are staged in VLM and the last chunk pair of
// every pass (chunks 12-13 at K 7168) reads its 8 x vectors from global memory. Same contract and bit-identical
// output as mxk_w13xt; K <= 7168 (the glue refuses longer rows). See mxk10_w13.h.
#define MXG_W13C 2
#define MXG_FB 2
#include "mxk10_core.h"

#define MXK8K_XVLM_CH 12
#undef MXK8_STAGE_X
// staging: chunks 0 .. min(nch, 12) - 1 of the permuted row
#define MXK8_STAGE_X(xp, row, nch)                                                                      \
    {                                                                                                   \
        const int nst_ = (nch) < MXK8K_XVLM_CH ? (nch) : MXK8K_XVLM_CH;                                 \
        for (int c_ = 0; c_ < nst_; c_++) {                                                             \
            int5 cx_ = {c_ * 512, (row), 0, 0, 0};                                                      \
            v8_x[4 * c_] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                    \
            v8_x[4 * c_ + 1] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                \
            v8_x[4 * c_ + 2] = v_bf16_ld_tnsr_b(cx_, xp); cx_[0] += 128;                                \
            v8_x[4 * c_ + 3] = v_bf16_ld_tnsr_b(cx_, xp);                                               \
        }                                                                                               \
    }

// x vectors of chunk c from global memory (row xrow of xp)
#define MXK8K_LDXG(xv, c)                                                                               \
    {                                                                                                   \
        int5 cg_ = {(c) * 512, xrow, 0, 0, 0};                                                          \
        xv[0] = v_bf16_ld_tnsr_b(cg_, xp); cg_[0] += 128;                                               \
        xv[1] = v_bf16_ld_tnsr_b(cg_, xp); cg_[0] += 128;                                               \
        xv[2] = v_bf16_ld_tnsr_b(cg_, xp); cg_[0] += 128;                                               \
        xv[3] = v_bf16_ld_tnsr_b(cg_, xp);                                                              \
    }

// MXK8_PAIR with the x vectors from global memory
#define MXK8K_PAIR_G(c, cn, RR, EE)                                                                     \
    {                                                                                                   \
        MXK8K_LDXG(xv, c);                                                                              \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK8_SCALES(bc, sP[j], zero8);                                                              \
            bfloat128 sc = MXK8_PICK(bc, pat0);                                                         \
            uchar256 b = bA[j];                                                                         \
            int5 cw = {(cn) * 256, RR[j], EE, 0, 0};                                                    \
            bA[j] = MXK8_LDW(cw, w);                                                                    \
            MXK8_DOT(acc[j], b, sc, xv);                                                                \
        }                                                                                               \
        MXK8K_LDXG(xv, (c) + 1);                                                                        \
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

// W13C pass: the last pair (chunks nch-2, nch-1 = 12, 13 at K 7168) from global memory
#define MXG_LAST_PAIR(c, cn, RR, EE)                                                                    \
    {                                                                                                   \
        const int xrow = s8_sorted[gi] / topk;                                                          \
        MXK8K_PAIR_G(c, cn, RR, EE)                                                                     \
    }

// the long-row 4-row pass (stands in for mxk8_pass_h4 of the core)
inline void mxk8k_pass_h4(tensor xp, int xrow, int n0, int n1, int n2, int n3, int e, int m0, int m1, int m2,
                          int m3, int f, int nch, tensor w, tensor s, ushort128 pat0, uchar256 zero8,
                          uchar256 (&bA)[4], uchar256 (&bB)[4], uchar256 (&sP)[4], float64 (&h)[4])
{
    const int nn[4] = {n0, n1, n2, n3};
    const int mm[4] = {m0, m1, m2, m3};
    const ushort128 pat1 = pat0 | 0x2020;
    float128 acc[4];
    _Pragma("unroll") for (int j = 0; j < 4; j++) { acc[j].v1 = 0; acc[j].v2 = 0; }
    bfloat128 xv[4];
    for (int c = 0; c < nch - 2; c += 2) MXK8_PAIR(c, c + 2, nn, e)
    MXK8K_PAIR_G(nch - 2, 0, mm, f)
    MXK8_HSUM4(acc, h0, h1, h2, h3)
    h[0] = h0; h[1] = h1; h[2] = h2; h[3] = h3;
}
#define mxk8_pass_h4(...) mxk8k_pass_h4(xp, s8_sorted[gi] / topk, __VA_ARGS__)

#ifndef MXK8_IL
#define MXK8_IL 0
#endif
#include "mxk10_w13.h"
