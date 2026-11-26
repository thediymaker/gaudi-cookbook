// SPDX-License-Identifier: Apache-2.0
//
// mxk_w2gp, v10: the grouped-route w2 kernel mxk_w2g (mxk9_w2.h) with a packed reduce + carried tail on the
// one-route passes (MXKP_*, below) and the same packed reduce + carried tail on the TWO-ROUTE half passes (G == 2 units = repeated experts) at nch == 2
// (MXKQ_MODE). A half pass has 4 row sums too: route 0 rows a / b (a0[0], a0[1]) and route 1 rows a / b (a1[0],
// a1[1]); packed as {a0[0], a1[0], a0[1], a1[1]}, mxkp_finish returns r.v1 = {even a0[0], odd a0[1]} (route 0's
// 2-lane store) and r.v2 = {a1[0], a1[1]} (route 1's), exactly mxk9_pass2.h's two MXK9_COLLECT2 stores.
// MXKQ_MODE 0: half passes as mxk_w2g; 3: carried, finish + store of half pass h written after half pass
// h + 1's dot; 2: carried, written before the dot.
#ifndef MXKQ_MODE
#define MXKQ_MODE 3
#endif
// One-route passes: mxk_w2g (mxk9_w2.h, MXK9_W2_COMBINE 0; that file is not edited) with the packed reduce + carried tail on the one-route pass at nch == 2 (MXKP_N2 1): the 4 row sums of
// pass p are packed after butterfly steps 1-2 and finished + stored at the top of pass p + 1, interleaved with its
// dot. Units of 2+ routes (half passes, mxk9_pass2.h) and nch != 2 run v9's code unchanged. Output = mxk_w2g's
// part, bit for bit (pair it with mxk_w2c). The pass loads / MACs are mxk9_pass.h's MXK9_PAIRG at c = cn = 0.
#ifndef MXKP_N2
#define MXKP_N2 1
#endif
// MXKP_MODE 2: carried (finish + store of pass p at the top of pass p + 1); 3: carried, finish + store of pass p
// written after pass p + 1's dot (source order only); 1: packed tree inside the pass (no carry)
#ifndef MXKP_MODE
#define MXKP_MODE 2
#endif
#include "mxk9_core.h"
#define RPP 4
#define MXK9_W2_BLK 32

// ---------------------------------------------------------------------------------------------- mxk_w2g
// per route g of the unit: the point's 4 sums as A lanes (even: row 0, odd: row 2) and B lanes (even: row 1, odd:
// row 3), two 2-lane stores into part at the block's A / B positions. Source-lane offsets differ per store (two
// back-to-back partial stores with the same size and offset lose the second: tpc-clang pitfall).
#define MXK9_QG(g) ((g) == 0 ? q0 : (g) == 1 ? q1 : (g) == 2 ? q2 : q3)
#define MXK9_CTX_PARAMS , tensor part, int ca0, int cb0, uint64 lsel, int q0, int q1, int q2, int q3
#define MXK9_CTX_ARGS , part, ca0, cb0, lsel, q0, q1, q2, q3
#define MXK9_COLLECT(g, h0, h1, h2, h3)                                                                 \
    {                                                                                                   \
        const float64 H0_ = h0, H1_ = h1, H2_ = h2, H3_ = h3;                                           \
        int5 c_ = {ca0, MXK9_QG(g), 0, 0, 0};                                                           \
        v_f32_st_tnsr_partial(c_, part, v_f32_sel_eq_u32_b(lsel, 0u, H0_, H2_), 1, 4 * (g));            \
        c_[0] = cb0;                                                                                    \
        v_f32_st_tnsr_partial(c_, part, v_f32_sel_eq_u32_b(lsel, 0u, H1_, H3_), 1, 4 * (g) + 2);        \
    }
// half passes: half 0 = rows 4p, 4p + 2 (the A position), half 1 = rows 4p + 1, 4p + 3 (the B position)
#define MXK9_CTX2_PARAMS , tensor part, int cpos, uint64 lsel, int q0, int q1, int q2, int q3
#define MXK9_CTX2_ARGS , part, cpos, lsel, q0, q1, q2, q3
#define MXK9_COLLECT2(g, ha, hb)                                                                        \
    {                                                                                                   \
        const float64 A_ = ha, B_ = hb;                                                                 \
        int5 c_ = {cpos, MXK9_QG(g), 0, 0, 0};                                                          \
        v_f32_st_tnsr_partial(c_, part, v_f32_sel_eq_u32_b(lsel, 0u, A_, B_), 1, 2 * (g));              \
    }
#define MXK9_NXV 2
#define MXK9_G 1
#include "mxk9_pass.h"
#if MXK9_GMAX >= 2
#define MXK9_G 2
#include "mxk9_pass2.h"
#endif
#if MXK9_GMAX >= 3
#define MXK9_G 3
#include "mxk9_pass2.h"
#endif
#if MXK9_GMAX >= 4
#define MXK9_G 4
#include "mxk9_pass2.h"
#endif


#if MXKP_N2
// ---------------------------------------------------------------------------------------------- packed tail
// The 4 row sums of a pass, v_f32_reduce_add(acc.v1 + acc.v2) each in v8 / v9, as the packed tree of mxk10_w2.h
// (MXT_RED 1/2, non-PK): steps 1-2 of the 4 butterflies merge the rows by dual
// group into one vector (dg0 row 0, dg1 row 2, dg2 row 1, dg3 row 3), steps 3-6 run once on it. Every lane adds the
// same two values in the same order as the butterfly, so the sums are bit-identical. mxkp_finish returns
// r.v1 = {even lanes row 0, odd row 2}, r.v2 = {row 1, row 3}: exactly the two vectors mxk_w2g's collect stores.
__local__ uint64 vp_shuf[3];                 // LUT 287 shuffle tables of v_f32_reduce_add (steps 4..6)
#define MXKP_MOVDG(src, s0, s1, s2, s3, w0, w1, w2, w3, inc) \
    v_f32_mov_dual_group_all_b(src, 0xFFFFFFFF, s0, s1, s2, s3, MkWrA(w0, w1, w2, w3), inc)
inline float64 mxkp_pack(float64 s0, float64 s1, float64 s2, float64 s3)
{
    float64 und_;
    const float64 u02 = MXKP_MOVDG(s2, 0, 1, 2, 3, 0, 3, 0, 3, s0);
    const float64 w02 = MXKP_MOVDG(s2, 1, 0, 3, 2, 0, 3, 0, 3, MXKP_MOVDG(s0, 1, 0, 3, 2, 3, 0, 3, 0, und_));
    const float64 u13 = MXKP_MOVDG(s3, 0, 1, 2, 3, 0, 3, 0, 3, s1);
    const float64 w13 = MXKP_MOVDG(s3, 1, 0, 3, 2, 0, 3, 0, 3, MXKP_MOVDG(s1, 1, 0, 3, 2, 3, 0, 3, 0, und_));
    const float64 p02 = u02 + w02, p13 = u13 + w13;
    const float64 u = MXKP_MOVDG(p13, 0, 1, 2, 3, 0, 0, 3, 3, p02);
    const float64 w = MXKP_MOVDG(p13, 2, 3, 0, 1, 0, 0, 3, 3, MXKP_MOVDG(p02, 2, 3, 0, 1, 3, 3, 0, 0, und_));
    return u + w;
}
inline float128 mxkp_finish(float64 p, uint64 lsel)
{
    float64 und_;
#if !MXKP_DIAG_NOFIN
    p = p + v_f32_mov_group_b(p, 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |
                              SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vp_shuf[0]), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vp_shuf[1]), 0, und_);
    p = p + v_f32_shuffle_b(p, as_uchar256(vp_shuf[2]), 0, und_);
#endif
    float128 r;
    r.v1 = v_f32_sel_eq_u32_b(lsel, 0u, MXKP_MOVDG(p, 0, 0, 0, 0, 3, 3, 3, 3, und_), MXKP_MOVDG(p, 1, 1, 1, 1, 3, 3, 3, 3, und_));
    r.v2 = v_f32_sel_eq_u32_b(lsel, 0u, MXKP_MOVDG(p, 2, 2, 2, 2, 3, 3, 3, 3, und_), MXKP_MOVDG(p, 3, 3, 3, 3, 3, 3, 3, 3, und_));
    return r;
}
// the two stores of mxk_w2g's MXK9_COLLECT(0, ...): A position lanes 0-1, B position lanes 2-3 (offsets differ)
#define MXKP_STORE(R, q, ca, cb)                                                                        \
    {                                                                                                   \
        const float128 R_ = R;                                                                          \
        int5 c_ = {ca, q, 0, 0, 0};                                                                     \
        v_f32_st_tnsr_partial(c_, part, R_.v1, 1, 0);                                                   \
        c_[0] = cb;                                                                                     \
        v_f32_st_tnsr_partial(c_, part, R_.v2, 1, 2);                                                   \
    }
// one-route pass at nch == 2 (one chunk pair: the pass is one basic block): the carried finish + store of the
// previous pass first (scheduled into this pass's dot), then the dot (mxk9_pass.h MXK9_PAIRG with c = 0, cn = 0 on
// the next pass's rows), returns this pass's packed sums
inline float64 mxkp_pass_n2(int n0, int n1, int n2, int n3, int e, int m0, int m1, int m2, int m3, int f, tensor w,
                            tensor s, ushort128 pat0, uchar256 zero8, uchar256 (&bA)[4], uchar256 (&bB)[4],
                            uchar256 (&sP)[4], tensor part, int q, int cap, int cbp, float64 cv, uint64 lsel)
{
#if MXKP_MODE == 2
    MXKP_STORE(mxkp_finish(cv, lsel), q, cap, cbp)
#else
    (void)cv;
#endif
    const int mm[4] = {m0, m1, m2, m3};
    (void)n0; (void)n1; (void)n2; (void)n3; (void)e;
    const ushort128 pat1 = pat0 | 0x2020;
    float128 a0[4];
    bfloat128 x0[4];
    _Pragma("unroll") for (int j = 0; j < 4; j++) { a0[j].v1 = 0; a0[j].v2 = 0; }
    MXK9_LDX(x0, 0)
    _Pragma("unroll") for (int j = 0; j < 4; j++) {
        ushort128 bc;
        MXK9_SCALES(bc, sP[j], zero8);
        bfloat128 sc = MXK9_PICK(bc, pat0);
        uchar256 b = bA[j];
        int5 cw = {0, mm[j], f, 0, 0};
        bA[j] = v_u8_ld_tnsr_b(cw, w);
        bfloat128 W[4];
        MXK9_DEQ(W, b, sc)
        MXK9_MAC(a0[j], W, x0)
    }
    MXK9_LDX(x0, 1)
    _Pragma("unroll") for (int j = 0; j < 4; j++) {
        ushort128 bc;
        MXK9_SCALES(bc, sP[j], zero8);
        bfloat128 sc = MXK9_PICK(bc, pat1);
        uchar256 b = bB[j];
        int5 cw = {256, mm[j], f, 0, 0};
        int5 cs = {0, mm[j], f, 0, 0};
        bB[j] = v_u8_ld_tnsr_b(cw, w);
        sP[j] = v_u8_ld_tnsr_b(cs, s);
        bfloat128 W[4];
        MXK9_DEQ(W, b, sc)
        MXK9_MAC(a0[j], W, x0)
    }
#if MXKP_MODE == 3
    MXKP_STORE(mxkp_finish(cv, lsel), q, cap, cbp)
#endif
    const float64 pk = mxkp_pack(a0[0].v1 + a0[0].v2, a0[1].v1 + a0[1].v2, a0[2].v1 + a0[2].v2, a0[3].v1 + a0[3].v2);
#if MXKP_MODE == 1
    MXKP_STORE(mxkp_finish(pk, lsel), q, cap, cbp)          // mode 1: the caller passes this pass's own positions
#endif
    return pk;
}
// the two stores of mxk9_pass2.h's MXK9_COLLECT2(0 / 1, ...) at block position cp: route 0 (row q0) from lanes 0-1,
// route 1 (row q1) from lanes 2-3
#define MXKQ_STORE(R, qa, qb, cp)                                                                       \
    {                                                                                                   \
        const float128 R_ = R;                                                                          \
        int5 c_ = {cp, qa, 0, 0, 0};                                                                    \
        v_f32_st_tnsr_partial(c_, part, R_.v1, 1, 0);                                                   \
        c_[1] = qb;                                                                                     \
        v_f32_st_tnsr_partial(c_, part, R_.v2, 1, 2);                                                   \
    }

#if MXKQ_MODE
// two-route half pass at nch == 2 (one chunk pair, one basic block): mxk9_pass2.h MXK9_PAIR2(0, 0, mm, f) at
// MXK9_G 2, MXK9_NXV 2 (route 0 from VLM slot 0, route 1 from slot MXK9_XCH / 2), then the carried finish + store
// of the previous half pass (rows q0 / q1 at cpos cp), returns this half pass's packed sums
inline float64 mxkq_half_n2(int ma0, int mb0, int f, tensor w, tensor s, ushort128 pat0, uchar256 zero8,
                            uchar256 (&bA)[2], uchar256 (&bB)[2], uchar256 (&sP)[2], tensor part, int q0, int q1,
                            int cp, float64 cv, uint64 lsel)
{
#if MXKQ_MODE == 2
    MXKQ_STORE(mxkp_finish(cv, lsel), q0, q1, cp)
#endif
    const int mm[2] = {ma0, mb0};
    const ushort128 pat1 = pat0 | 0x2020;
    float128 a0[2], a1[2];
    bfloat128 x0[4], x1[4];
    _Pragma("unroll") for (int j = 0; j < 2; j++) { a0[j].v1 = 0; a0[j].v2 = 0; a1[j].v1 = 0; a1[j].v2 = 0; }
    MXK9_LDX(x0, 0)
    MXK9_LDX(x1, MXK9_XCH / 2)
    _Pragma("unroll") for (int j = 0; j < 2; j++) {
        ushort128 bc;
        MXK9_SCALES(bc, sP[j], zero8);
        bfloat128 sc = MXK9_PICK(bc, pat0);
        uchar256 b = bA[j];
        int5 cw = {0, mm[j], f, 0, 0};
        bA[j] = v_u8_ld_tnsr_b(cw, w);
        bfloat128 W[4];
        MXK9_DEQ(W, b, sc)
        MXK9_MAC(a0[j], W, x0)
        MXK9_MAC(a1[j], W, x1)
    }
    MXK9_LDX(x0, 1)
    MXK9_LDX(x1, MXK9_XCH / 2 + 1)
    _Pragma("unroll") for (int j = 0; j < 2; j++) {
        ushort128 bc;
        MXK9_SCALES(bc, sP[j], zero8);
        bfloat128 sc = MXK9_PICK(bc, pat1);
        uchar256 b = bB[j];
        int5 cw = {256, mm[j], f, 0, 0};
        int5 cs = {0, mm[j], f, 0, 0};
        bB[j] = v_u8_ld_tnsr_b(cw, w);
        sP[j] = v_u8_ld_tnsr_b(cs, s);
        bfloat128 W[4];
        MXK9_DEQ(W, b, sc)
        MXK9_MAC(a0[j], W, x0)
        MXK9_MAC(a1[j], W, x1)
    }
#if MXKQ_MODE == 3
    MXKQ_STORE(mxkp_finish(cv, lsel), q0, q1, cp)
#endif
    return mxkp_pack(a0[0].v1 + a0[0].v2, a1[0].v1 + a1[0].v2, a0[1].v1 + a0[1].v2, a1[1].v1 + a1[1].v2);
}
#endif
#endif

void main(tensor ap, tensor ids, tensor w, tensor s, tensor part, int I, int e_min, int npts, int gmax)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int nch = ((I + 511) / 512 + 1) & ~1;
    const int e_local = get_dim_size(w, 2);
    const int topk = get_dim_size(ids, 0);
    const int T = get_dim_size(ids, 1);
    int nu = 0;
    const int nr = mxk9_routes(ids, T, topk, e_min, e_local, gmax, nu);

    ushort128 lid = read_lane_id_2b_b();
    ushort128 pat0 = (lid >> 3) | 0x80;
    const uchar256 zero8 = 0;
    const uint64 lsel = read_lane_id_4b_b() & 1u;
#if MXKP_N2
    {
        const uint64 lid4 = read_lane_id_4b_b();
        vp_shuf[0] = v_u32_lookup_1c(lid4, 287, SW_BV32, (uint64)0);
        const uint128 t12 = v_u32_lookup_2c(lid4, 287, SW_BV32, (uint128){0, 0});
        vp_shuf[1] = t12.v1; vp_shuf[2] = t12.v2;
    }
    const float64 zf = (float64)((float)npts * 0.0f);
#endif

    const int P = get_dim_size(w, 1) / RPP;
    const int pb = is[0] * P / npts, pe = ie[0] * P / npts;
    if (nr == 0 || pb >= pe) return;
    const int np = pe - pb;

    for (int u = 0; u < nu; u++) {
        const int ub = s9_ub[u];
        const int G = s9_un[u];
        const int e = s9_ge[ub];
        const int q0 = s9_sorted[ub];
        const int q1 = G > 1 ? s9_sorted[ub + 1] : 0;
        const int q2 = G > 2 ? s9_sorted[ub + 2] : 0;
        const int q3 = G > 3 ? s9_sorted[ub + 3] : 0;
        // activation rows of the unit's first two routes in VLM (slots 0 and MXK9_XCH / 2)
        MXK9_STAGE(ap, q0, nch, 0);
        if (G > 1) MXK9_STAGE(ap, q1, nch, MXK9_XCH / 2);
        if (G == 1) {
            uchar256 bA[4], bB[4], sP[4];
            mxk9_first(pb * RPP, pb * RPP + 1, pb * RPP + 2, pb * RPP + 3, e, w, s, bA, bB, sP);
#if MXKP_N2
            if (nch == 2) {
                // carried tail: pass p's packed sums (butterfly steps 1-2) finish + store at the top of pass p + 1.
                // The first pass stores a zero carry at its own positions (overwritten by its real sums one pass
                // later, same TPC, in order); the last pass's carry is flushed after the loop.
                float64 cv = mxkp_pack(zf, zf, zf, zf);
                int cap = pb * RPP, cbp = pb * RPP + 2 * ((pe - pb) < MXK9_W2_BLK ? (pe - pb) : MXK9_W2_BLK);
                for (int p = pb; p < pe; p++) {
                    const int p2 = p + 1 < pe ? p + 1 : p;
                    const int n0 = p * RPP, m0 = p2 * RPP;
                    const int bi = (p - pb) / MXK9_W2_BLK;
                    const int j = p - pb - bi * MXK9_W2_BLK;
                    const int b0 = pb + bi * MXK9_W2_BLK;
                    const int qn = (pe - b0) < MXK9_W2_BLK ? (pe - b0) : MXK9_W2_BLK;
                    const int ca0 = b0 * RPP + 2 * j, cb0 = b0 * RPP + 2 * qn + 2 * j;
#if MXKP_MODE == 1
                    cap = ca0; cbp = cb0;
#endif
                    cv = mxkp_pass_n2(n0, n0 + 1, n0 + 2, n0 + 3, e, m0, m0 + 1, m0 + 2, m0 + 3, e, w, s, pat0, zero8,
                                      bA, bB, sP, part, q0, cap, cbp, cv, lsel);
                    cap = ca0; cbp = cb0;
                }
#if MXKP_MODE >= 2
                MXKP_STORE(mxkp_finish(cv, lsel), q0, cap, cbp)
#endif
                continue;
            }
#endif
            for (int p = pb; p < pe; p++) {
                const int p2 = p + 1 < pe ? p + 1 : p;
                const int n0 = p * RPP, m0 = p2 * RPP;
                // block of point p: base row, points in it; this point's A / B positions
                const int bi = (p - pb) / MXK9_W2_BLK;
                const int j = p - pb - bi * MXK9_W2_BLK;
                const int b0 = pb + bi * MXK9_W2_BLK;
                const int qn = (pe - b0) < MXK9_W2_BLK ? (pe - b0) : MXK9_W2_BLK;
                const int ca0 = b0 * RPP + 2 * j, cb0 = b0 * RPP + 2 * qn + 2 * j;
                mxk9_pass_g1(n0, n0 + 1, n0 + 2, n0 + 3, e, m0, m0 + 1, m0 + 2, m0 + 3, e, nch, w, s, pat0, zero8,
                             bA, bB, sP MXK9_CTX_ARGS);
            }
            continue;
        }
#if MXK9_GMAX >= 2
        uchar256 bA[2], bB[2], sP[2];
        // half pass h: point pb + h / 2, rows 4p + (h & 1) and 4p + (h & 1) + 2
        {
            int5 cw = {0, pb * RPP, e, 0, 0};
            bA[0] = v_u8_ld_tnsr_b(cw, w); sP[0] = v_u8_ld_tnsr_b(cw, s); cw[0] = 256; bB[0] = v_u8_ld_tnsr_b(cw, w);
            cw[0] = 0; cw[1] = pb * RPP + 2;
            bA[1] = v_u8_ld_tnsr_b(cw, w); sP[1] = v_u8_ld_tnsr_b(cw, s); cw[0] = 256; bB[1] = v_u8_ld_tnsr_b(cw, w);
        }
        const int nh = 2 * np;
#if MXKP_N2 && MXKQ_MODE
        if (nch == 2 && G == 2) {
            // carried tail on the half passes: half pass h's packed sums finish + store in half pass h + 1. The
            // first half pass stores a zero carry at its own positions (overwritten one half pass later, same TPC,
            // in order); the last half pass's carry is flushed after the loop.
            float64 cv = mxkp_pack(zf, zf, zf, zf);
            int cp = pb * RPP;
            for (int h = 0; h < nh; h++) {
                const int h2 = h + 1 < nh ? h + 1 : h;
                const int p = pb + h / 2, hs = h & 1;
                const int ma = (pb + h2 / 2) * RPP + (h2 & 1);
                const int bi = (p - pb) / MXK9_W2_BLK;
                const int j = p - pb - bi * MXK9_W2_BLK;
                const int b0 = pb + bi * MXK9_W2_BLK;
                const int qn = (pe - b0) < MXK9_W2_BLK ? (pe - b0) : MXK9_W2_BLK;
                const int cpos = b0 * RPP + hs * 2 * qn + 2 * j;
                cv = mxkq_half_n2(ma, ma + 2, e, w, s, pat0, zero8, bA, bB, sP, part, q0, q1, cp, cv, lsel);
                cp = cpos;
            }
            MXKQ_STORE(mxkp_finish(cv, lsel), q0, q1, cp)
            continue;
        }
#endif
        for (int h = 0; h < nh; h++) {
            const int h2 = h + 1 < nh ? h + 1 : h;
            const int p = pb + h / 2, hs = h & 1;
            const int na = p * RPP + hs, ma = (pb + h2 / 2) * RPP + (h2 & 1);
            const int bi = (p - pb) / MXK9_W2_BLK;
            const int j = p - pb - bi * MXK9_W2_BLK;
            const int b0 = pb + bi * MXK9_W2_BLK;
            const int qn = (pe - b0) < MXK9_W2_BLK ? (pe - b0) : MXK9_W2_BLK;
            const int cpos = b0 * RPP + hs * 2 * qn + 2 * j;
#if MXK9_GMAX >= 2
            if (G == 2)
                mxk9_half_g2(na, na + 2, e, ma, ma + 2, e, nch, w, s, ap, q0, q1, q2, q3, pat0, zero8, bA, bB, sP MXK9_CTX2_ARGS);
#endif
#if MXK9_GMAX >= 3
            if (G == 3)
                mxk9_half_g3(na, na + 2, e, ma, ma + 2, e, nch, w, s, ap, q0, q1, q2, q3, pat0, zero8, bA, bB, sP MXK9_CTX2_ARGS);
#endif
#if MXK9_GMAX >= 4
            if (G == 4)
                mxk9_half_g4(na, na + 2, e, ma, ma + 2, e, nch, w, s, ap, q0, q1, q2, q3, pat0, zero8, bA, bB, sP MXK9_CTX2_ARGS);
#endif
        }
#endif
    }
}
