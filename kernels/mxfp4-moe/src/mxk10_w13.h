// SPDX-License-Identifier: Apache-2.0
//
// v10 w13: the v8 w13 (mxk8_w13.h) on the v10 core (mxk10_core.h), plus MXG_FB / MXG_W13C (see the block
// below). Without them and without MXKX_NATX it compiles to the v8 kernel. Options: MXK8_HSUM_X2 and / or MXKX_NATX
// (natural-order x, see mxk10_core.h); with MXKX_NATX the input xp is x in NATURAL order [T, K] (no host permute).
// MoE w13 projection + gated activation for decode (Gaudi2 TPC), v8. Same contract and bit-identical output as
// v7 mxk7_w13.h with MXK7_XVLM (mxk_w13x / mxk_w13xi):
//   g = x . Wg[e, n], u = x . Wu[e, n] (fp32); g = min(g, limit); u = clamp(u, -limit, limit)
//   a[q, n] = g * sigmoid(alpha * g) * (u + beta) * ascale    -> bf16, permuted layout for the w2 kernels
// Layout (MXK8_IL): 0 = w13 rows [gate (I); up (I)]; 1 = interleaved [g0, u0, g1, u1, ...] (I % 512 may be 256:
// the padding rows of ap are written as exact zeros).
// Tensors: xp bf16 [T, Kp] permuted; ids i32 [Q]; w u8 [E, 2I, K/2]; s u8 [E, 2I, K/32]; out ap bf16 [Q, Ip].
// MXK8_BIAS 1 (gpt-oss, mxk_w13xib): one more input b f32 [E, 4, Ip/2], the per-row bias in the kernel's own order
// (host-built: column c of the permuted ap = intermediate index perm(c); plane 0 / 1 = gate bias of
// the even / odd columns, plane 2 / 3 = up bias of the even / odd columns, zeros past I), added in fp32 to the row
// sums before the activation: g = x . Wg[e, n] + bg[e, n], u = x . Wu[e, n] + bu[e, n]. Needs MXK8_ACT_BATCH: the
// bias of a whole run of up to 32 column pairs is 4 vector loads at the flush, instead of 4 scalar loads per pass
// (scalar global loads are a large share of w13 at the gpt-oss shapes, prefetched or not).
// K need not be a multiple of 512: chunks past the row read as zeros (weights, scales and the staged x row).
// I need only be a multiple of 8 in the interleaved layout (row pairs are then all-real or all-padding).
// Schedule: route-outer (the route's x row is staged in VLM once), this TPC's row pairs inner, all passes of the
// TPC one software-pipelined stream (mxk8_core.h): the last chunk pair of a pass loads the first pair of the next
// pass (next row pair of the same route, or the first row pair of the next route).
// MXG_LAST_PAIR: the last chunk pair of an MXG_W13C pass. Default MXK8_PAIR; a kernel whose rows are longer than
// the VLM stage (mxk10_w13xk.c, K 7168) sets it to a pair that reads its x vectors from global memory.
#ifndef MXG_LAST_PAIR
#define MXG_LAST_PAIR(c, cn, RR, EE) MXK8_PAIR(c, cn, RR, EE)
#endif
#include "mxk10_core.h"
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
// columns blk*512 + 128*pp + i0 + {0, 1} of the permuted ap (as v7)
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

// ---- v10 levers (default off = the v8 schedule) ----
//   MXG_FB 1    flush test of the batched epilogue as an | of compares (the compiler still emits 4 branches);
//          2    as integer arithmetic that is non-zero iff a flush is due: one branch, same decision.
//   MXG_W13C 1  packed reduction tree (mxk10_w2.h MXT_RED 1 on the 4 row sums, 24 ops instead of 52), placed into v8_hb
//               by 2 dual-group moves + 2 masked moves (lanes m and m + 32 at once). Same adds per lane, same bits.
//   MXG_W13C 2  as 1, and steps 3..6 + the placement of pass i run in pass i + 1's last-pair block (one carried
//               vector); a flush finishes the pass in place and drops the carry.
//   MXG_BND 1   timing only (wrong results): MXG_W13C without steps 3..6.
#ifndef MXG_FB
#define MXG_FB 0
#endif
#ifndef MXG_W13C
#define MXG_W13C 0
#endif
#if MXG_W13C
// MXG_W13C with MXK8_BIAS (mxk_w13xibn): the bias is added at the flush, after the run's last pass has been placed
// (MXG_W13C 2 finishes the carried pass in place before the flush), so every lane sees the v8 sequence: row sum (the
// same adds per lane as v_f32_reduce_add), then + bias, then the activation.
#if !MXK8_ACT_BATCH || MXK8_NCH6 || defined(MXK8_DIAG_NOSCALE)
#error "MXG_W13C: batched w13 only"
#endif
#define MXG_MOVDG(src, s0, s1, s2, s3, w0, w1, w2, w3, inc) \
    v_f32_mov_dual_group_all_b(src, 0xFFFFFFFF, s0, s1, s2, s3, MkWrA(w0, w1, w2, w3), inc)
// butterfly steps 1..2 of the 4 row sums packed in one vector: dg0 row 0, dg1 row 2, dg2 row 1, dg3 row 3
inline float64 mxg_pack(const float128 (&acc)[4])
{
    const float64 s0 = acc[0].v1 + acc[0].v2, s1 = acc[1].v1 + acc[1].v2;
    const float64 s2 = acc[2].v1 + acc[2].v2, s3 = acc[3].v1 + acc[3].v2;
    float64 und_;
    const float64 u02 = MXG_MOVDG(s2, 0, 1, 2, 3, 0, 3, 0, 3, s0);
    const float64 w02 = MXG_MOVDG(s2, 1, 0, 3, 2, 0, 3, 0, 3, MXG_MOVDG(s0, 1, 0, 3, 2, 3, 0, 3, 0, und_));
    const float64 u13 = MXG_MOVDG(s3, 0, 1, 2, 3, 0, 3, 0, 3, s1);
    const float64 w13 = MXG_MOVDG(s3, 1, 0, 3, 2, 0, 3, 0, 3, MXG_MOVDG(s1, 1, 0, 3, 2, 3, 0, 3, 0, und_));
    const float64 p02 = u02 + w02, p13 = u13 + w13;
    const float64 u = MXG_MOVDG(p13, 0, 1, 2, 3, 0, 0, 3, 3, p02);
    const float64 w = MXG_MOVDG(p13, 2, 3, 0, 1, 0, 0, 3, 3, MXG_MOVDG(p02, 2, 3, 0, 1, 3, 3, 0, 0, und_));
    return u + w;
}
// steps 3..6 (v_f32_reduce_add's mov_g + 3 LUT-287 shuffles); every lane of a dual group ends with its row's sum.
// The 3 shuffle tables are re-read per call from lane id + zs (zs = 0 at run time, but not provably so): hoisted
// tables take 3 VLM spill vectors, which mxk_w13x does not have (build fails). Costs 1 add + 2 lookups a call.
inline float64 mxg_finish(float64 p, uint64 lid4, unsigned zs)
{
#if MXG_BND != 1
    float64 und_;
    const uint64 lz = lid4 + zs;
    const uint64 t0 = v_u32_lookup_1c(lz, 287, SW_BV32, (uint64)0);
    const uint128 t12 = v_u32_lookup_2c(lz, 287, SW_BV32, (uint128){0, 0});
    p = p + v_f32_mov_group_b(p, 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |
                              SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(t0), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(t12.v1), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(t12.v2), 0, und_);
#endif
    return p;
}
#define MXG_FIN(p, mq) mxg_finish(p, lid4, (unsigned)(mq) >> 8)
// pair mq's sums into v8_hb: G lanes mq / mq + 32 = rows 0 / 1 (dg 0 / 2 of p), U = rows 2 / 3 (dg 1 / 3). The mask
// is ((lane ^ mq) & 31) == 0, not (lane & 31) == mq: a hoisted lane & 31 costs the last free VLM spill vector.
#define MXG_PLACE(p, mq)                                                                                \
    {                                                                                                   \
        float64 und_;                                                                                   \
        bool256 lm_ = from_bool64(v_u32_cmp_eq_b(v_u32_and_b(v_u32_xor_b(lid4, (unsigned)(mq)), 31u), 0u, 0, to_bool64((bool256){0}), 1, 0)); \
        const float64 gs_ = MXG_MOVDG(p, 0, 0, 2, 2, 3, 3, 3, 3, und_);                                 \
        const float64 us_ = MXG_MOVDG(p, 1, 1, 3, 3, 3, 3, 3, 3, und_);                                 \
        v8_hb[0] = v_f32_mov_vb(gs_, 0, v8_hb[0], to_bool64(lm_), 0);                                   \
        v8_hb[1] = v_f32_mov_vb(us_, 0, v8_hb[1], to_bool64(lm_), 0);                                   \
    }
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
    // loads; the real ones go to s8_pts in walk order. MXK8_SPLIT 0: the contiguous range [pb, pe) (v7);
    // 1: strided, pairs ip, ip + npts, ... (the same rows as range 0 overall, in another TPC <-> row assignment).
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
#ifdef MXKX_DBG   // debug only: TPC 0 dumps the staged row (v8_x[j] -> ap row j / 8, columns 128 (j % 8) ..)
    if (is[0] == 0) {
        for (int j = 0; j < 4 * nch; j++) {
            int5 cd = {128 * (j % 8), j / 8, 0, 0, 0};
            v_bf16_st_tnsr(cd, ap, v8_x[j]);
        }
    }
    return;
#endif
    mxk8_first(r0, r1, r2, r3, e, w, s, bA, bB, sP);
#if MXK8_ACT_BATCH
    // batched epilogue: the sums of up to 32 consecutive row pairs of one route (one 64-column run of the
    // permuted ap) are collected lane by lane: pair m -> v8_hb[0] lanes m / 32 + m = its g0 / g1 sums,
    // v8_hb[1] lanes m / 32 + m = its u0 / u1 sums. The activation then runs once per run on whole vectors: the
    // same lane-wise formula on the same values, so the same bits as v7's one-pair-at-a-time epilogue. Lanes
    // 32..63 are moved down to 0..31 (dual groups 2, 3 -> 0, 1) so convert_all puts pair m's two outputs in bf16
    // lanes 2m, 2m+1.
    uint64 lid4 = read_lane_id_4b_b();
    int m = 0, col0 = col;
#endif
#if MXG_W13C
    float64 cv = 0.0f;
    int mp = 99;       // lane of the carried pass; 99 = none (empty mask)
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
#if MXG_W13C
        {   // the pass (mxk8_pass_h4 without the 4 reduce_adds), then the packed tree
            const int nn[4] = {r0, r1, r2, r3};
            const int mm[4] = {m0, m1, m2, m3};
            const ushort128 pat1 = pat0 | 0x2020;
            float128 acc[4];
            _Pragma("unroll") for (int j = 0; j < 4; j++) { acc[j].v1 = 0; acc[j].v2 = 0; }
            bfloat128 xv[4];
            for (int c = 0; c < nch - 2; c += 2) MXK8_PAIR(c, c + 2, nn, e)
#if MXG_W13C == 2
            MXG_PLACE(MXG_FIN(cv, mp), mp)       // previous pass's tail, in the last-pair block
#endif
            MXG_LAST_PAIR(nch - 2, 0, mm, f)
            cv = mxg_pack(acc);
#if MXG_W13C == 1
            MXG_PLACE(MXG_FIN(cv, m), m)
#endif
        }
        const int mc = m;
#if MXG_W13C == 2
        mp = mc;
#endif
#else
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
#endif
        m++;
        // flush when the run ends: last pass, new route, or the next pair is not the next column pair
#if MXG_FB == 2
        const int fl_ = (gi2 - gi) | (ncol - col - 2) | (1 - more) | ((m >> 5) & 1);   // m <= 32
        if (fl_ != 0) {
#elif MXG_FB
        const int fl_ = (more == 0) | (gi2 != gi) | (ncol != col + 2) | (m == 32);
        if (fl_) {
#else
        if (!more || gi2 != gi || ncol != col + 2 || m == 32) {
#endif
#if MXG_W13C == 2
            MXG_PLACE(MXG_FIN(cv, mc), mc)       // finish this pass in place, drop the carry
            mp = 99;
#endif
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
