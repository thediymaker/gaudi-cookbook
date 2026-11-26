// SPDX-License-Identifier: Apache-2.0
//
// v10 w2 (mxk_w2xc: mxk_w2x's contract, combine mode) at nch = 2 (I_slice <= 1024). The v8 w2 (mxk8_w2.h,
// MXK8_ROUTE 0, no bias) with these -D options (default = v8 schedule and math), plus MXG_DG (accumulator by dual
// group, below MXT_STREAM) and MXG_PROBE (f32 shuffle format probe, development only):
//   MXT_N2 1   the stream loop runs a copy specialised for nch == 2 (pass = one chunk pair, no pair loop, so the
//              whole pass is one basic block); other nch take the v8 loop. Same math, same bits.
//   MXT_RED 1  packed reduction tree: v8 runs v_f32_reduce_add on each of the 4 row vectors (6 butterfly steps:
//              mov_dg xor 1 dual group, mov_dg xor 2, mov_g, 3 shuffles from LUT 287; 13 ops per row, 52 per pass).
//              The first two steps exchange whole dual groups, so rows can share a vector there: after step 1 rows
//              0 / 2 sit in one vector (dual groups 0,2 / 1,3), rows 1 / 3 in another; after step 2 all 4 rows sit
//              in one vector (dg0 row 0, dg1 row 2, dg2 row 1, dg3 row 3), and steps 3..6 run once on it. Every lane
//              adds the same two values in the same order as v8's butterfly, so the sums are bit-identical.
//              24 ops instead of 52, plus 4 dual-group broadcasts back to the v8 lane layout before the 2 sels.
//   MXT_RED 2  as 1, and steps 3..6 + the accumulate of pass i run in pass i + 1 (interleaved with its dot; one
//              carried vector), flushed at a token end. Needs MXT_N2.
//   MXT_RED 3  as 2 but the whole tree of pass i runs in pass i + 1 (4 carried vectors).
//   MXT_BK 1   token and token-end flag of each route precomputed once per call (SLM tables): no per-pass divide or
//              short-circuit branch. Bookkeeping only, same bits.
//   MXT_PK 1   steps 1..2 packed with dual-group selects (lane-bit masks) instead of merging dual-group moves.
//   MXK8_DIAG_NOHSUM (v8 option, timing only): no horizontal sums.
#include "mxk10_w2core.h"
#define RPP 4
#ifndef MXK8_TMAX
#define MXK8_TMAX 16
#endif
#if MXK8_ROUTE || MXK8_BIAS
#error "mxk10_w2.h: combine mode without bias only"
#endif
#define MXK8_W2_BLK 32
#ifndef MXK8_W2_NB
#define MXK8_W2_NB 4
#endif
#ifndef MXT_N2
#define MXT_N2 0
#endif
#ifndef MXT_RED
#define MXT_RED 0
#endif
#ifndef MXT_PK
#define MXT_PK 0
#endif
#ifndef MXT_BK
#define MXT_BK 0
#endif
#if MXT_RED >= 2 && !MXT_N2
#error "MXT_RED 2/3 need MXT_N2"
#endif

#if defined(MXG_DG) && MXG_DG
__local__ float64 v8_acc[4 * MXK8_W2_NB];  // MXG_DG: 4 vectors per block
#else
__local__ float64 v8_acc[2 * MXK8_W2_NB];
#endif
__local__ int s8_toff[MXK8_TMAX + 1];
__local__ int s8_tord[MXK8_QMAX];
#if MXT_BK
__local__ int s8_tk[MXK8_QMAX];              // MXT_BK: token of the k-th route in token order, and its last flag
__local__ int s8_tl[MXK8_QMAX];
#endif
__local__ uint64 vt_shuf[3];                 // LUT 287 shuffle tables of v_f32_reduce_add (steps 4..6)
__local__ uint64 vt_m[2];                    // MXT_PK: lane id & 16, lane id & 32

// one pass's dot products (v8 MXK8_PASS_BODY without the horizontal sums)
#define MXT_DOT(NCH)                                                                                    \
    {                                                                                                   \
        const ushort128 pat1 = pat0 | 0x2020;                                                           \
        _Pragma("unroll") for (int j = 0; j < 4; j++) { acc[j].v1 = 0; acc[j].v2 = 0; }                \
        bfloat128 xv[4];                                                                                \
        for (int c = 0; c < (NCH) - 2; c += 2) MXK8_PAIR(c, c + 2, nn, e)                               \
        MXK8_PAIR((NCH) - 2, 0, mm, f)                                                                  \
    }

#define MXT_MOVDG(src, s0, s1, s2, s3, w0, w1, w2, w3, inc) \
    v_f32_mov_dual_group_all_b(src, 0xFFFFFFFF, s0, s1, s2, s3, MkWrA(w0, w1, w2, w3), inc)

// steps 1..2 (+ the v1 + v2 add) of the 4 butterflies, packed: one vector, dg0 row 0, dg1 row 2, dg2 row 1, dg3 row 3
inline float64 mxt_pack(const float128 (&acc)[4])
{
#ifdef MXK8_DIAG_NOHSUM
    return acc[0].v1 + acc[1].v2 + acc[2].v1 + acc[3].v2;
#else
    const float64 s0 = acc[0].v1 + acc[0].v2, s1 = acc[1].v1 + acc[1].v2;
    const float64 s2 = acc[2].v1 + acc[2].v2, s3 = acc[3].v1 + acc[3].v2;
    float64 und_;                        // undefined income: whole-vector moves need no copy of a dst
#if MXT_PK
    // dual-group selects (lane bit 4 / bit 5 masks from VLM): u = own row, w = the partner dual group's value
    const uint64 m16 = vt_m[0], m32 = vt_m[1];
    const float64 u02 = v_f32_sel_eq_u32_b(m16, 0u, s0, s2), u13 = v_f32_sel_eq_u32_b(m16, 0u, s1, s3);
    const float64 w02 = MXT_MOVDG(v_f32_sel_eq_u32_b(m16, 0u, s2, s0), 1, 0, 3, 2, 3, 3, 3, 3, und_);
    const float64 w13 = MXT_MOVDG(v_f32_sel_eq_u32_b(m16, 0u, s3, s1), 1, 0, 3, 2, 3, 3, 3, 3, und_);
    const float64 p02 = u02 + w02, p13 = u13 + w13;
    const float64 u = v_f32_sel_eq_u32_b(m32, 0u, p02, p13);
    const float64 w = MXT_MOVDG(v_f32_sel_eq_u32_b(m32, 0u, p13, p02), 2, 3, 0, 1, 3, 3, 3, 3, und_);
#else
    // step 1 (lane i + lane i ^ 16): rows a / b in dual groups {0, 2} / {1, 3}
    const float64 u02 = MXT_MOVDG(s2, 0, 1, 2, 3, 0, 3, 0, 3, s0);
    const float64 w02 = MXT_MOVDG(s2, 1, 0, 3, 2, 0, 3, 0, 3, MXT_MOVDG(s0, 1, 0, 3, 2, 3, 0, 3, 0, und_));
    const float64 u13 = MXT_MOVDG(s3, 0, 1, 2, 3, 0, 3, 0, 3, s1);
    const float64 w13 = MXT_MOVDG(s3, 1, 0, 3, 2, 0, 3, 0, 3, MXT_MOVDG(s1, 1, 0, 3, 2, 3, 0, 3, 0, und_));
    const float64 p02 = u02 + w02, p13 = u13 + w13;
    // step 2 (lane i + lane i ^ 32): p02 in dual groups {0, 1}, p13 in {2, 3}
    const float64 u = MXT_MOVDG(p13, 0, 1, 2, 3, 0, 0, 3, 3, p02);
    const float64 w = MXT_MOVDG(p13, 2, 3, 0, 1, 0, 0, 3, 3, MXT_MOVDG(p02, 2, 3, 0, 1, 3, 3, 0, 0, und_));
#endif
    return u + w;
#endif
}

// steps 3..6 on the packed vector and the v8 lane layout: r.v1 = {even lanes: row 0, odd: row 2}, r.v2 = {row 1, row 3}
inline float128 mxt_finish(float64 p, uint64 lsel)
{
    float64 und_;
#ifndef MXK8_DIAG_NOHSUM
    p = p + v_f32_mov_group_b(p, 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |
                              SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vt_shuf[0]), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vt_shuf[1]), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vt_shuf[2]), 0, und_);
#endif
    float128 r;
    r.v1 = v_f32_sel_eq_u32_b(lsel, 0u, MXT_MOVDG(p, 0, 0, 0, 0, 3, 3, 3, 3, und_), MXT_MOVDG(p, 1, 1, 1, 1, 3, 3, 3, 3, und_));
    r.v2 = v_f32_sel_eq_u32_b(lsel, 0u, MXT_MOVDG(p, 2, 2, 2, 2, 3, 3, 3, 3, und_), MXT_MOVDG(p, 3, 3, 3, 3, 3, 3, 3, 3, und_));
    return r;
}

inline float128 mxt_red(const float128 (&acc)[4], uint64 lsel)
{
#if MXT_RED == 0
    MXK8_HSUM4(acc, h0, h1, h2, h3)
    float128 r;
    r.v1 = v_f32_sel_eq_u32_b(lsel, 0u, h0, h2);
    r.v2 = v_f32_sel_eq_u32_b(lsel, 0u, h1, h3);
    return r;
#else
    return mxt_finish(mxt_pack(acc), lsel);
#endif
}

// accumulate one pass's sums r (weight wq) into block bi, point j of the strip
#define MXT_ACC(r, wq, bi, j)                                                                           \
    {                                                                                                   \
        bool256 m_ = from_bool64(v_u32_cmp_eq_b(lpt, (unsigned)(j), 0, to_bool64((bool256){0}), 1, 0)); \
        v8_acc[2 * (bi)] = v_f32_mac_vb((r).v1, wq, v8_acc[2 * (bi)], 0, to_bool64(m_), 0);           \
        v8_acc[2 * (bi) + 1] = v_f32_mac_vb((r).v2, wq, v8_acc[2 * (bi) + 1], 0, to_bool64(m_), 0);   \
    }

// the per-pass bookkeeping of v8 (next pass, router weight, block / lane, token end)
#define MXT_NEXT                                                                                        \
            int p2 = p + 1, k2 = k;                                                                     \
            if (p2 >= se) { p2 = sb; k2 = k + 1; }                                                      \
            if (k2 >= nr) { p2 = p; k2 = k; }                                                           \
            const int gi2 = s8_tord[k2];                                                                \
            const int f = s8_ge[gi2];                                                                   \
            const int n0 = p * RPP, m0 = p2 * RPP;                                                      \
            const int nn[4] = {n0, n0 + 1, n0 + 2, n0 + 3};                                             \
            const int mm[4] = {m0, m0 + 1, m0 + 2, m0 + 3};                                             \
            const int q = s8_sorted[gi];                                                                \
            int5 cq = {q, 0, 0, 0, 0};                                                                  \
            const float wq = s_convert_bf16_to_f32(s_bf16_ld_g((__global void*)gen_addr(cq, tw)));     \
            const int pl = p - sb;                                                                      \
            const int bi = pl / MXK8_W2_BLK;                                                            \
            const int j = pl - bi * MXK8_W2_BLK;                                                        \

// token of the pass and token end, evaluated after the dot (as v8) so the pass stays one basic block
#if MXT_BK
#define MXT_TEND                                                                                        \
            const int t = s8_tk[k];                                                                     \
            const int tend = (p + 1 >= se) & s8_tl[k];
#else
#define MXT_TEND                                                                                        \
            const int t = q / topk;                                                                     \
            const int tend = p + 1 >= se && (k + 1 >= nr || s8_sorted[s8_tord[k + 1]] / topk != t);
#endif

#define MXT_STORE                                                                                       \
            for (int b = 0; b < nb; b++) {                                                              \
                float128 o; o.v1 = v8_acc[2 * b]; o.v2 = v8_acc[2 * b + 1];                             \
                const int q0 = sb + b * MXK8_W2_BLK;                                                    \
                const int qn = (se - q0) < MXK8_W2_BLK ? (se - q0) : MXK8_W2_BLK;                       \
                int5 co = {q0 * RPP, t, 0, 0, 0};                                                       \
                v_bf16_st_tnsr_partial(co, out, v_convert_f32_to_bf16_all_b(o), qn * RPP - 1, 0);       \
                v8_acc[2 * b] = zf; v8_acc[2 * b + 1] = zf;                                             \
            }

#define MXT_ADVANCE(NCH)                                                                                   \
            if (gi2 != gi) MXK8_STAGE_X(ap, s8_sorted[gi2], NCH);                                       \
            k = k2; p = p2; gi = gi2; e = f;

// the v8 stream (one strip): reduction and accumulate in the pass
#define MXT_STREAM(NCH)                                                                                 \
        for (int it = 0; it < total; it++) {                                                            \
            MXT_NEXT                                                                                    \
            float128 acc[4];                                                                            \
            MXT_DOT(NCH)                                                                                \
            const float128 r = mxt_red(acc, lsel);                                                      \
            MXT_ACC(r, wq, bi, j)                                                                       \
            MXT_TEND                                                                                    \
            if (tend) { MXT_STORE }                                                                     \
            MXT_ADVANCE(NCH)                                                                        \
        }

// carried stream: pass i's tail (finish + accumulate) runs at the top of pass i + 1; no carry after a token end
// (prev j = -1: the accumulate's lane mask is empty, so it is a no-op on block 0)
#if MXT_RED == 2
#define MXT_CARRY_T float64
#define MXT_CARRY_MK(acc) mxt_pack(acc)
#define MXT_CARRY_FIN(cv) mxt_finish(cv, lsel)
#elif MXT_RED == 3
struct mxt_c4 { float128 a[4]; };
#define MXT_CARRY_T mxt_c4
#define MXT_CARRY_MK(acc) mxt_c4{{acc[0], acc[1], acc[2], acc[3]}}
#define MXT_CARRY_FIN(cv) mxt_finish(mxt_pack(cv.a), lsel)
#endif
// ---- MXG_DG 1 (needs MXT_RED 2), carried stream only ----
// After steps 3..6 every lane of dual group d holds row R(d)'s sum (R = 0, 2, 1, 3, the packed tree's order). The
// accumulator of a 32-point block is 4 vectors (v8_acc[4 bi + j / 8], 8 points each) laid out so that one masked MAC
// of the finished vector does the whole accumulate: point j % 8 = jj sits in lane 2 jj of dual groups 0 / 2 (rows
// 0 / 1) and lane 2 jj + 1 of dual groups 1 / 3 (rows 2 / 3); mask = key == 2 jj, key = (lane ^ (lane >> 4 & 1)) & 15.
// That replaces 4 dual-group broadcasts + 2 sels + 2 MACs a pass. The token-end store rebuilds the v8 lane layout
// with dual-group moves and 2 parity sels per block (no shuffles: the
// f32 shuffle reads only inside a dual group). Same MAC per value, same bits. MXG_BND 2: timing only, the store without the
// rebuild (wrong results).
#ifndef MXG_DG
#define MXG_DG 0
#endif
#if MXG_DG && MXT_RED != 2
#error "MXG_DG needs MXT_RED 2"
#endif
#if MXG_DG
inline float64 mxg_fin(float64 p)
{
    float64 und_;
    p = p + v_f32_mov_group_b(p, 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |
                              SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vt_shuf[0]), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vt_shuf[1]), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vt_shuf[2]), 0, und_);
    return p;
}
// jv = j / 8 (vector in the block), jm = 2 (j % 8) (mask key; 99 = no lane)
#define MXG_ACC(p, wq, bi, jv, jm)                                                                      \
    {                                                                                                   \
        bool256 m_ = from_bool64(v_u32_cmp_eq_b(kdg, (unsigned)(jm), 0, to_bool64((bool256){0}), 1, 0)); \
        v8_acc[4 * (bi) + (jv)] = v_f32_mac_vb(p, wq, v8_acc[4 * (bi) + (jv)], 0, to_bool64(m_), 0);    \
    }
// dual group d of the result = dual group S of B[d] (B = the block's 4 vectors)
#define MXG_GATHER(B, S, r)                                                                             \
    {                                                                                                   \
        r = MXT_MOVDG(B[0], S, S, S, S, 3, 0, 0, 0, B[0]);                                              \
        r = MXT_MOVDG(B[1], S, S, S, S, 0, 3, 0, 0, r);                                                 \
        r = MXT_MOVDG(B[2], S, S, S, S, 0, 0, 3, 0, r);                                                 \
        r = MXT_MOVDG(B[3], S, S, S, S, 0, 0, 0, 3, r);                                                 \
    }
#if MXG_BND == 2
#define MXG_PERM(B, o) { (o).v1 = B[0] + B[1]; (o).v2 = B[2] + B[3]; }
#else
#define MXG_PERM(B, o)                                                                                  \
    {                                                                                                   \
        float64 x0_, x1_, x2_, x3_;                                                                     \
        MXG_GATHER(B, 0, x0_) MXG_GATHER(B, 1, x1_) MXG_GATHER(B, 2, x2_) MXG_GATHER(B, 3, x3_)         \
        (o).v1 = v_f32_sel_eq_u32_b(lsel, 0u, x0_, x1_);                                                \
        (o).v2 = v_f32_sel_eq_u32_b(lsel, 0u, x2_, x3_);                                                \
    }
#endif
#define MXG_STORE                                                                                       \
            for (int b = 0; b < nb; b++) {                                                              \
                float64 B_[4] = {v8_acc[4 * b], v8_acc[4 * b + 1], v8_acc[4 * b + 2], v8_acc[4 * b + 3]}; \
                float128 o; MXG_PERM(B_, o)                                                             \
                const int q0 = sb + b * MXK8_W2_BLK;                                                    \
                const int qn = (se - q0) < MXK8_W2_BLK ? (se - q0) : MXK8_W2_BLK;                       \
                int5 co = {q0 * RPP, t, 0, 0, 0};                                                       \
                v_bf16_st_tnsr_partial(co, out, v_convert_f32_to_bf16_all_b(o), qn * RPP - 1, 0);       \
                v8_acc[4 * b] = zf; v8_acc[4 * b + 1] = zf; v8_acc[4 * b + 2] = zf; v8_acc[4 * b + 3] = zf; \
            }
#define MXT_STREAM_C(NCH)                                                                               \
        {                                                                                               \
            float64 cv; float wqp = 0.0f; int bip = 0, jvp = 0, jmp = 99;                               \
            { float128 a0[4]; _Pragma("unroll") for (int jj = 0; jj < 4; jj++) { a0[jj].v1 = zf; a0[jj].v2 = zf; } \
              cv = mxt_pack(a0); }                                                                      \
            for (int it = 0; it < total; it++) {                                                        \
                MXT_NEXT                                                                                \
                { const float64 rp = mxg_fin(cv); MXG_ACC(rp, wqp, bip, jvp, jmp) }                     \
                float128 acc[4];                                                                        \
                MXT_DOT(NCH)                                                                            \
                cv = mxt_pack(acc); wqp = wq; bip = bi; jvp = j >> 3; jmp = 2 * (j & 7);                \
                MXT_TEND                                                                                \
                if (tend) {                                                                             \
                    { const float64 rc = mxg_fin(cv); MXG_ACC(rc, wq, bi, j >> 3, 2 * (j & 7)) }        \
                    jvp = 0; jmp = 99;                                                                  \
                    MXG_STORE                                                                           \
                }                                                                                       \
                MXT_ADVANCE(NCH)                                                                        \
            }                                                                                           \
        }
#endif

#if MXT_RED >= 2 && !MXG_DG
#define MXT_STREAM_C(NCH)                                                                               \
        {                                                                                               \
            MXT_CARRY_T cv; float wqp = 0.0f; int bip = 0, jp = -1;                                     \
            { float128 a0[4]; _Pragma("unroll") for (int jj = 0; jj < 4; jj++) { a0[jj].v1 = zf; a0[jj].v2 = zf; } \
              cv = MXT_CARRY_MK(a0); }                                                                  \
            for (int it = 0; it < total; it++) {                                                        \
                MXT_NEXT                                                                                \
                { const float128 rp = MXT_CARRY_FIN(cv); MXT_ACC(rp, wqp, bip, jp) }                    \
                float128 acc[4];                                                                        \
                MXT_DOT(NCH)                                                                            \
                cv = MXT_CARRY_MK(acc); wqp = wq; bip = bi; jp = j;                                     \
                MXT_TEND                                                                                \
                if (tend) {                                                                             \
                    { const float128 rc = MXT_CARRY_FIN(cv); MXT_ACC(rc, wq, bi, j) }                   \
                    jp = -1;                                                                            \
                    MXT_STORE                                                                           \
                }                                                                                       \
                MXT_ADVANCE(NCH)                                                                        \
            }                                                                                           \
        }
#endif

void main(tensor ap, tensor ids, tensor tw, tensor w, tensor s, tensor out, int I, int topk, int e_min, int npts)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int nch = ((I + 511) / 512 + 1) & ~1;
    const int e_local = get_dim_size(w, 2);
    const int Q = get_dim_size(ids, 0);
    const int nr = mxk8_routes(ids, Q, e_min, e_local);

    ushort128 lid = read_lane_id_2b_b();
    ushort128 pat0 = (lid >> 3) | 0x80;
    uint64 lid4 = read_lane_id_4b_b();
    const uchar256 zero8 = 0;
#if MXT_RED
    vt_shuf[0] = v_u32_lookup_1c(lid4, 287, SW_BV32, (uint64)0);
    { const uint128 t12 = v_u32_lookup_2c(lid4, 287, SW_BV32, (uint128){0, 0}); vt_shuf[1] = t12.v1; vt_shuf[2] = t12.v2; }
    vt_m[0] = lid4 & 16u; vt_m[1] = lid4 & 32u;
#endif
#ifdef MXG_PROBE   // probe only: f32 shuffle pattern format. TPC 0 writes vector q (lanes as bf16 of
                   // f32 lane values, v1 = even bf16 lanes) to out row 0, columns 128 q ..
    if (is[0] == 0) {
        const float64 lf = convert_uint64_to_float64(lid4, 0);
        float64 pv[20];
        const uint64 tt[3] = {vt_shuf[0], vt_shuf[1], vt_shuf[2]};
        for (int i = 0; i < 3; i++)
            for (int b = 0; b < 4; b++) pv[4 * i + b] = convert_uint64_to_float64((tt[i] >> (8 * b)) & 0xFFu, 0);
        const uint64 ix = ((lid4 & 1u) << 4) + ((lid4 & 31u) >> 1);
        float64 und_;
        pv[12] = v_f32_shuffle_b(lf, as_uchar256(tt[0]), 0, und_);
        pv[13] = v_f32_shuffle_b(lf, as_uchar256(ix | 0x80u), 0, und_);
        pv[14] = v_f32_shuffle_b(lf, as_uchar256(ix), 0, und_);
        const uchar256 l1 = read_lane_id_1b_b();
        const uchar256 i1 = (uchar256)(((l1 & (unsigned char)1) << 4) + ((l1 & (unsigned char)31) >> 1)) | (unsigned char)0x80;
        pv[15] = v_f32_shuffle_b(lf, i1, 0, und_);
        pv[16] = v_f32_shuffle_b(lf, as_uchar256(ix | 0xA0u), 0, und_);
        pv[17] = v_f32_shuffle_b(lf, as_uchar256((ix | 0x80u) * 0x01010101u), 0, und_);
        pv[18] = lf;
        pv[19] = convert_uint64_to_float64(ix, 0);
        for (int q = 0; q < 20; q++) {
            int5 co = {128 * q, 0, 0, 0, 0};
            float128 r; r.v1 = pv[q]; r.v2 = pv[q];
            v_bf16_st_tnsr(co, out, v_convert_f32_to_bf16_all_b(r));
        }
    }
    return;
#endif

    const int P = get_dim_size(w, 1) / RPP;
    const int pb = is[0] * P / npts, pe = ie[0] * P / npts;
    if (pb >= pe) return;
    uchar256 bA[4], bB[4], sP[4];

    const int T = Q / topk;
    for (int t = 0; t < T; t++) s8_toff[t] = 0;
    for (int gi = 0; gi < nr; gi++) { const int t = s8_sorted[gi] / topk; s8_toff[t] = s8_toff[t] + 1; }
    for (int t = 0; t < T; t++) {
        if (s8_toff[t] != 0) continue;
        for (int p = pb; p < pe; p += MXK8_W2_BLK) {
            const int qn = (pe - p) < MXK8_W2_BLK ? (pe - p) : MXK8_W2_BLK;
            int5 co = {p * RPP, t, 0, 0, 0};
            v_bf16_st_tnsr_partial(co, out, (bfloat128)0.0f, qn * RPP - 1, 0);
        }
    }
    if (nr == 0) return;
    {
        int off = 0;
        for (int t = 0; t < T; t++) { const int c = s8_toff[t]; s8_toff[t] = off; off += c; }
        for (int gi = 0; gi < nr; gi++) {
            const int t = s8_sorted[gi] / topk;
            const int o = s8_toff[t];
            s8_tord[o] = gi;
            s8_toff[t] = o + 1;
        }
#if MXT_BK
        for (int k = 0; k < nr; k++) s8_tk[k] = s8_sorted[s8_tord[k]] / topk;
        for (int k = 0; k < nr; k++) s8_tl[k] = (k + 1 >= nr || s8_tk[k + 1] != s8_tk[k]) ? 1 : 0;
#endif
    }
    const float64 zf = (float64)((float)npts * 0.0f);
    uint64 lsel = lid4 & 1u;
    uint64 lpt = lid4 >> 1;
#if MXG_DG
    const uint64 kdg = (lid4 ^ ((lid4 >> 4) & 1u)) & 15u;
#endif
    const int SP = MXK8_W2_NB * MXK8_W2_BLK;
    for (int sb = pb; sb < pe; sb += SP) {
        const int se = sb + SP < pe ? sb + SP : pe;
        const int nb = (se - sb + MXK8_W2_BLK - 1) / MXK8_W2_BLK;
#if MXG_DG
        for (int i = 0; i < 4 * nb; i++) v8_acc[i] = zf;
#else
        for (int i = 0; i < 2 * nb; i++) v8_acc[i] = zf;
#endif
        int k = 0, p = sb;
        int gi = s8_tord[0], e = s8_ge[gi];
        MXK8_STAGE_X(ap, s8_sorted[gi], nch);
        mxk8_first(p * RPP, p * RPP + 1, p * RPP + 2, p * RPP + 3, e, w, s, bA, bB, sP);
        const int total = nr * (se - sb);
#if MXT_N2
        if (nch == 2) {
#if MXT_RED >= 2
            MXT_STREAM_C(2)
#else
            MXT_STREAM(2)
#endif
        } else {
            MXT_STREAM(nch)
        }
#else
        MXT_STREAM(nch)
#endif
    }
}
