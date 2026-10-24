// SPDX-License-Identifier: Apache-2.0
//
// MoE w13 projection + gated activation for decode (Gaudi 2 TPC):
//   g = x . Wg[e, n], u = x . Wu[e, n] (fp32); g = min(g, limit); u = clamp(u, -limit, limit)
//   a[q, n] = g * sigmoid(alpha * g) * (u + beta) * ascale    -> bf16, in the permuted layout the w2 kernel reads
// The activation parameters cover the common gated MoE activations:
//   SiLU (silu(g) * u):                 alpha 1, limit +inf, beta 0, ascale 1
//   SwiGLU-OAI (gpt-oss):               alpha 1.702, limit 7, beta 1, ascale 1
// Row layout (MXK8_IL): 0 = w13 rows [gate (I); up (I)]; 1 = interleaved [g0, u0, g1, u1, ...] (gpt-oss). In the
// interleaved layout I % 512 may be 256: the padding rows of ap are written as exact zeros.
// Tensors: xp bf16 [T, Kp] permuted; ids i32 [Q]; w u8 [E, 2I, K/2]; s u8 [E, 2I, K/32]; out ap bf16 [Q, Ip].
// MXK8_BIAS 1 (gpt-oss, kernel mxk_w13xib): one more input b f32 [E, 4, Ip/2], the per-row bias in the kernel's own
// column order (built once on the host by mxfp4_tpc_moe_rt.bias13: column c of the permuted ap = intermediate index
// perm(c); plane 0 / 1 = gate bias of the even / odd columns, plane 2 / 3 = up bias of the even / odd columns, zeros
// past I), added in fp32 to the row sums before the activation: g = x . Wg[e, n] + bg[e, n], u = x . Wu[e, n] +
// bu[e, n]. Needs MXK8_ACT_BATCH: the bias of a whole run of up to 32 column pairs is 4 vector loads at the flush,
// instead of 4 scalar loads per pass (scalar global loads cost about a quarter of w13 at the gpt-oss shapes).
// K need not be a multiple of 512: chunks past the row read as zeros (weights, scales and the staged x row).
// I need only be a multiple of 8 in the interleaved layout (row pairs are then all-real or all-padding).
// Schedule: route-outer (the route's x row is staged in VLM once), this TPC's row pairs inner, all passes of the
// TPC one software-pipelined stream (mxk8_core.h): the last chunk pair of a pass loads the first pair of the next
// pass (next row pair of the same route, or the first row pair of the next route).
// Batched epilogue (MXK8_ACT_BATCH 1, default): the gated activation (two sigmoids on whole vectors) is the costly
// part of a pass tail, so the sums of a run of passes are collected lane by lane and the activation runs once per
// run. It is the same lane-wise formula on the same values, so the output bits do not depend on the batching.
#include "mxk8_core.h"
#ifndef MXK8_IL
#define MXK8_IL 0
#endif
#ifndef MXK8_SPLIT
#define MXK8_SPLIT 0
#endif
#define MXK8_PMAX 1536
__local__ int s8_pts[MXK8_PMAX];          // this TPC's real row pairs in walk order
#ifndef MXK8_ACT_BATCH
#define MXK8_ACT_BATCH 1
#endif
#if MXK8_ACT_BATCH
__local__ float64 v8_hb[2];
#endif

// point p (row pair) -> block blk, sub-position pp, i0; gate rows g0 = blk*512 + 4*i0 + pp, g1 = g0 + 4; output
// columns blk*512 + 128*pp + i0 + {0, 1} of the permuted ap
#define MXK8_PT(p, g0, col)                                                                             \
    {                                                                                                   \
        const int blk_ = (p) / 256;                                                                     \
        const int rem_ = (p) - blk_ * 256;                                                              \
        const int pp_ = rem_ / 64;                                                                      \
        const int i0_ = (rem_ - pp_ * 64) * 2;                                                          \
        g0 = blk_ * 512 + 4 * i0_ + pp_;                                                                \
        col = blk_ * 512 + 128 * pp_ + i0_;                                                             \
    }
#if defined(MXK8_DIAG_SAMEROWS)      // timing only: every pass reads the rows of the first pass (cache-resident)
#define MXK8_ROWS(g0, r0, r1, r2, r3) { r0 = 0; r1 = 4; r2 = I; r3 = I + 4; }
#elif defined(MXK8_DIAG_CONTIG)     // timing only: a pass reads 4 consecutive rows, consecutive passes follow on
#define MXK8_ROWS(g0, r0, r1, r2, r3) { r0 = 4 * ((g0) / 8 * 2 + ((g0) & 3)) % (2 * I); r1 = r0 + 1; r2 = r0 + 2; r3 = r0 + 3; }
#elif MXK8_IL
#define MXK8_ROWS(g0, r0, r1, r2, r3) { r0 = 2 * (g0); r1 = 2 * ((g0) + 4); r2 = r0 + 1; r3 = r1 + 1; }
#else
#define MXK8_ROWS(g0, r0, r1, r2, r3) { r0 = (g0); r1 = (g0) + 4; r2 = I + (g0); r3 = I + (g0) + 4; }
#endif

#ifndef MXK8_BIAS
#define MXK8_BIAS 0
#endif
#if MXK8_BIAS && !MXK8_ACT_BATCH
#error "MXK8_BIAS needs the batched epilogue (MXK8_ACT_BATCH 1)"
#endif
#if MXK8_BIAS
void main(tensor xp, tensor ids, tensor w, tensor s, tensor bias, tensor ap, int K, int topk, int e_min, int npts,
          float alpha, float limit, float beta, float ascale)
#else
void main(tensor xp, tensor ids, tensor w, tensor s, tensor ap, int K, int topk, int e_min, int npts,
          float alpha, float limit, float beta, float ascale)
#endif
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int nch = ((K + 511) / 512 + 1) & ~1;
    const int e_local = get_dim_size(w, 2);
    const int I = get_dim_size(w, 1) / 2;
    const int Ip = (I + 511) / 512 * 512;
    const int Q = get_dim_size(ids, 0);
    const int nr = mxk8_routes(ids, Q, e_min, e_local);

    ushort128 lid = read_lane_id_2b_b();
    ushort128 pat0 = (lid >> 3) | 0x80;
    const uchar256 zero8 = 0;

    const int P = Ip / 2;
    const int pb = is[0] * P / npts, pe = ie[0] * P / npts;
    if (nr == 0 || pb >= pe) return;

    // this TPC's row pairs: padding pairs (g0 >= I, interleaved I % 512 == 256 only) get exact zeros and no
    // loads; the real ones go to s8_pts in walk order. MXK8_SPLIT 0: the contiguous range [pb, pe);
    // 1: strided, pairs ip, ip + npts, ... (the same rows overall, another TPC <-> row assignment; no gain).
    int nreal = 0;
#if MXK8_SPLIT == 1
    for (int ip = is[0]; ip < ie[0]; ip++)
    for (int p = ip; p < P; p += npts) {
#else
    for (int p = pb; p < pe; p++) {
#endif
        int g0, col;
        MXK8_PT(p, g0, col)
        if (g0 < I) { s8_pts[nreal] = p; nreal++; continue; }
        for (int gi = 0; gi < nr; gi++) {
            int5 co = {col, s8_sorted[gi], 0, 0, 0};
            v_bf16_st_tnsr_partial(co, ap, (bfloat128)0.0f, 1, 0);
        }
    }
    if (nreal == 0) return;

    uchar256 bA[4], bB[4], sP[4];
    int gi = 0, u = 0;
    int e = s8_ge[0];
    int g0, col;
    MXK8_PT(s8_pts[0], g0, col)
    int r0, r1, r2, r3;
    MXK8_ROWS(g0, r0, r1, r2, r3)
    MXK8_STAGE_X(xp, s8_sorted[0] / topk, nch);
    mxk8_first(r0, r1, r2, r3, e, w, s, bA, bB, sP);
#if MXK8_ACT_BATCH
    // batched epilogue: the sums of up to 32 consecutive row pairs of one route (one 64-column run of the
    // permuted ap) are collected lane by lane: pair m -> v8_hb[0] lanes m / 32 + m = its g0 / g1 sums,
    // v8_hb[1] lanes m / 32 + m = its u0 / u1 sums. The activation then runs once per run on whole vectors.
    // Lanes 32..63 are moved down to 0..31 (dual groups 2, 3 -> 0, 1) so convert_all puts pair m's two outputs in
    // bf16 lanes 2m, 2m+1.
    uint64 lid4 = read_lane_id_4b_b();
    int m = 0, col0 = col;
#endif
    const int total = nr * nreal;
    for (int it = 0; it < total; it++) {
        // next pass: next row pair of this route, else the first one of the next route
        int u2 = u + 1, gi2 = gi;
        if (u2 >= nreal) { u2 = 0; gi2 = gi + 1; }
        const int more = gi2 < nr;
        if (!more) { u2 = u; gi2 = gi; }
        const int f = s8_ge[gi2];
        int ng0, ncol;
        MXK8_PT(s8_pts[u2], ng0, ncol)
        int m0, m1, m2, m3;
        MXK8_ROWS(ng0, m0, m1, m2, m3)
#if MXK8_ACT_BATCH
        float64 h[4];
        mxk8_pass_h4(r0, r1, r2, r3, e, m0, m1, m2, m3, f, nch, w, s, pat0, zero8, bA, bB, sP, h);
        {
            bool256 la = from_bool64(v_u32_cmp_eq_b(lid4, (unsigned)m, 0, to_bool64((bool256){0}), 1, 0));
            bool256 lb = from_bool64(v_u32_cmp_eq_b(lid4, (unsigned)(m + 32), 0, to_bool64((bool256){0}), 1, 0));
            float64 G = v8_hb[0], U = v8_hb[1];
            G = v_f32_mov_vb(h[0], 0, G, to_bool64(la), 0);
            G = v_f32_mov_vb(h[1], 0, G, to_bool64(lb), 0);
            U = v_f32_mov_vb(h[2], 0, U, to_bool64(la), 0);
            U = v_f32_mov_vb(h[3], 0, U, to_bool64(lb), 0);
            v8_hb[0] = G; v8_hb[1] = U;
        }
        m++;
        // flush when the run ends: last pass, new route, or the next pair is not the next column pair
        if (!more || gi2 != gi || ncol != col + 2 || m == 32) {
            float128 r;
#if MXK8_BIAS && !defined(MXK8_DIAG_NOBIAS)   // DIAG_NOBIAS: timing only
            {   // bias of the run's columns col0 .. col0 + 2m - 1 (expert e) in the G / U lane order: lanes j < 32 =
                // even column col0 + 2j (planes 0 / 2), lanes 32 + j = odd column col0 + 2j + 1 (planes 1 / 3, dual
                // groups 0, 1 moved up to 2, 3). Lanes past the run are never stored.
                int5 cb = {col0 / 2, 0, e, 0, 0};
                const float64 bge = v_f32_ld_tnsr_b(cb, bias); cb[1] = 1;
                const float64 bgo = v_f32_ld_tnsr_b(cb, bias); cb[1] = 2;
                const float64 bue = v_f32_ld_tnsr_b(cb, bias); cb[1] = 3;
                const float64 buo = v_f32_ld_tnsr_b(cb, bias);
                const float64 go = v_f32_mov_dual_group_all_b(bgo, 0xFFFFFFFF, 0, 1, 0, 1, MkWrA(0b11, 0b11, 0b11, 0b11), bgo);
                const float64 uo = v_f32_mov_dual_group_all_b(buo, 0xFFFFFFFF, 0, 1, 0, 1, MkWrA(0b11, 0b11, 0b11, 0b11), buo);
                const uint64 hi = lid4 >> 5;
                v8_hb[0] = v8_hb[0] + v_f32_sel_eq_u32_b(hi, 0u, bge, go);
                v8_hb[1] = v8_hb[1] + v_f32_sel_eq_u32_b(hi, 0u, bue, uo);
            }
#endif
            r.v1 = MXK8_ACT(v8_hb[0], v8_hb[1]);
            r.v2 = v_f32_mov_dual_group_all_b(r.v1, 0xFFFFFFFF, 2, 3, 2, 3, MkWrA(0b11, 0b11, 0b11, 0b11), r.v1);
            int5 co = {col0, s8_sorted[gi], 0, 0, 0};
            v_bf16_st_tnsr_partial(co, ap, v_convert_f32_to_bf16_all_b(r), 2 * m - 1, 0);
            m = 0; col0 = ncol;
        }
#else
        float128 r = mxk8_pass_act(r0, r1, r2, r3, e, m0, m1, m2, m3, f, nch, w, s, pat0, zero8, alpha, limit,
                                   beta, ascale, bA, bB, sP);
        int5 co = {col, s8_sorted[gi], 0, 0, 0};
        v_bf16_st_tnsr_partial(co, ap, v_convert_f32_to_bf16_all_b(r), 1, 0);
#endif
        if (gi2 != gi) MXK8_STAGE_X(xp, s8_sorted[gi2] / topk, nch);
        gi = gi2; u = u2; e = f; col = ncol;
        r0 = m0; r1 = m1; r2 = m2; r3 = m3;
    }
}
