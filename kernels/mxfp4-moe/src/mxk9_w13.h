// SPDX-License-Identifier: Apache-2.0
//
// MoE w13 projection + gated activation for decode, grouped routes (Gaudi2 TPC), v9. Same math and, per route,
// bit-identical output to v8 mxk8_w13.h (mxk_w13x, concat [gate; up] rows):
//   g = x . Wg[e, n], u = x . Wu[e, n] (fp32); g = min(g, limit); u = clamp(u, -limit, limit)
//   a[q, n] = g * sigmoid(alpha * g) * (u + beta) * ascale    -> bf16, permuted layout for the w2 kernels
// Tensors: xp bf16 [T, Kp] permuted; ids i32 [T, topk]; w u8 [E, 2I, K/2]; s u8 [E, 2I, K/32]; out ap bf16 [Q, Ip],
// Q = T * topk <= MXK9_QMAX; aux xa bf16 [8 * npts, K] (SRAM scratch, below). Rows of ap that belong to non-local
// routes are not written (as v8). I % 512 == 0, K <= 6144 (one x row in VLM).
// Schedule: unit-outer (a unit = up to MXK9_GMAX routes of one local expert, mxk9_core.h), this TPC's row pairs
// inner, all passes of a unit one software-pipelined stream. Index space: one point per TPC (preferredSplitDim),
// each point owns a fixed slice of row pairs (v8), so the split does not depend on the routing.
//   one-route unit: v8's 4-row pass (mxk9_pass.h), x row staged in VLM: v8's walk and speed.
//   grouped unit (G >= 2): 2-row half passes (mxk9_pass2.h: gate pair, then up pair of each row pair), every route's
//   x row read from this TPC's SRAM copy of it.
// gmax (scalar) caps the unit size at run time: 1 runs v8's walk (one route per unit).
#include "mxk9_core.h"

// one-route units collect in v9_hb[0]; grouped units read no x from VLM, so the VLM x row is free during them and
// route g collects in its vector g (the next one-route unit stages its row again)
__local__ float64 v9_hb[1];
#define MXK9_HB2(g) (*(__local__ float64*)&v9_x[g])

// Batched epilogue (v8 MXK8_ACT_BATCH with runs of 16 row pairs): route g's sums of up to 16 consecutive row pairs
// m = 0..15 are collected lane by lane in ONE VLM vector: lane m = g0 sum, 16 + m = g1, 32 + m = u0,
// 48 + m = u1 (one vector per route instead of v8's two, so the collection of 2+ routes fits next to the 12 KB x row).
// At the flush the up half is moved down (dual groups 2, 3 -> 0, 1) and the activation runs once on the vector: the
// same lane-wise formula on the same fp32 values as v8, so the same bits. Masks: l0 = lane m, l1 = 16 + m,
// l2 = 32 + m, l3 = 48 + m.
#define MXK9_RUN 16
#define MXK9_CTX_PARAMS , bool256 l0, bool256 l1, bool256 l2, bool256 l3
#define MXK9_CTX_ARGS , l0, l1, l2, l3
#define MXK9_COLLECT(g, h0, h1, h2, h3)                                                                 \
    {                                                                                                   \
        float64 V_ = v9_hb[g];                                                                          \
        V_ = v_f32_mov_vb(h0, 0, V_, to_bool64(l0), 0);                                                 \
        V_ = v_f32_mov_vb(h1, 0, V_, to_bool64(l1), 0);                                                 \
        V_ = v_f32_mov_vb(h2, 0, V_, to_bool64(l2), 0);                                                 \
        V_ = v_f32_mov_vb(h3, 0, V_, to_bool64(l3), 0);                                                 \
        v9_hb[g] = V_;                                                                                  \
    }
// half passes: hs 0 = the two gate rows of the pair (lanes m, 16 + m), 1 = the two up rows (32 + m, 48 + m)
#define MXK9_CTX2_PARAMS , bool256 la, bool256 lb
#define MXK9_CTX2_ARGS , la, lb
#define MXK9_COLLECT2(g, ha, hb)                                                                        \
    {                                                                                                   \
        float64 V_ = MXK9_HB2(g);                                                                       \
        V_ = v_f32_mov_vb(ha, 0, V_, to_bool64(la), 0);                                                 \
        V_ = v_f32_mov_vb(hb, 0, V_, to_bool64(lb), 0);                                                 \
        MXK9_HB2(g) = V_;                                                                               \
    }
#define MXK9_LMASK(k) from_bool64(v_u32_cmp_eq_b(lid4, (unsigned)(k), 0, to_bool64((bool256){0}), 1, 0))
// flush the run of m row pairs collected in vector HB (sorted index gi) at column col0
#define MXK9_FLUSH(HB, gi)                                                                              \
    {                                                                                                   \
        const float64 V_ = HB;                                                                          \
        const float lim_ = s9_act[0], al_ = s9_act[1], be_ = s9_act[2], as_ = s9_act[3];               \
        const float64 U_ = v_f32_mov_dual_group_all_b(V_, 0xFFFFFFFF, 2, 3, 2, 3, MkWrA(0b11, 0b11, 0b11, 0b11), V_); \
        float128 r_;                                                                                    \
        r_.v1 = MXK9_ACT(V_, U_);                                                                       \
        r_.v2 = v_f32_mov_dual_group_all_b(r_.v1, 0xFFFFFFFF, 1, 1, 1, 1, MkWrA(0b11, 0b11, 0b11, 0b11), r_.v1); \
        int5 co_ = {col0, s9_sorted[gi], 0, 0, 0};                                                      \
        v_bf16_st_tnsr_partial(co_, ap, v_convert_f32_to_bf16_all_b(r_), 2 * m - 1, 0);                \
    }
#define MXK9_NXV 1
#define MXK9_X0AUX 1
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

// row pair p -> gate rows g0 = blk*512 + 4*i0 + pp and g0 + 4; output columns 2p, 2p + 1 of the permuted ap (v8
// MXK8_PT for the concat layout, where every pair is real)
#define MXK9_G0(p) ((p) / 256 * 512 + 8 * ((p) % 256 - (p) % 256 / 64 * 64) + (p) % 256 / 64)
#define MXK9_ROWS(p, r0, r1, r2, r3) { const int g0_ = MXK9_G0(p); r0 = g0_; r1 = g0_ + 4; r2 = I + g0_; r3 = I + g0_ + 4; }
// half pass h of the TPC's walk: pair pb + h / 2, half h & 1 (0: gate rows g0, g0 + 4; 1: up rows I + g0, I + g0 + 4)
#define MXK9_HROWS(h, ra, rb) { const int g0_ = MXK9_G0(pb + (h) / 2) + ((h) & 1) * I; ra = g0_; rb = g0_ + 4; }
// The activation's scalars are re-read from SLM (volatile) at every flush: otherwise the compiler hoists their vector
// broadcasts out of the unit walk and spills them to VLM, which then does not fit next to the 12 KB x row.
__local__ volatile float s9_act[4];
#define MXK9_ACT(g, u) (v_f32_min_b(g, lim_) * v_sigmoid_f32(v_f32_min_b(g, lim_) * al_) *                      \
                        ((v_f32_max_b(v_f32_min_b(u, lim_), -lim_) + be_) * as_))

// Grouped units read their x rows from an auxiliary scratch tensor xa [8 * npts, K] bf16 that the glue asks the
// graph compiler to place in SRAM, not from xp: re-reading from global memory an activation row the graph produced
// shortly before costs about as much as streaming weights (a grouped walk re-reads each row once per half pass),
// while a row in SRAM does not (docs/kernels/mxfp4-moe.md).
// Each TPC owns 2 sets of 4 rows (unit parity) and copies unit u + 1's rows into the other set while it runs unit u.
// A TPC's own stores to global memory reach its later loads only after a delay (measured on Gaudi2: between ~1k and
// ~3k cycles; aso and cache_invalidate do not close it), so a unit's rows are always written one whole unit earlier
// (a unit is np >= MXK9_NPSAFE passes, tens of thousands of cycles at the served shapes). The walk starts with a
// one-route unit when there is one (it needs no copy); otherwise, and when units are short, a grouped unit first
// waits MXK9_AUXSPIN iterations of a dependent scalar chain (~7.5 cycles each; 400 measured enough, 150 not).
#ifndef MXK9_AUXSPIN
#define MXK9_AUXSPIN 600
#endif
#define MXK9_NPSAFE 16
#define MXK9_AUXCOPY(v)                                                                                      \
    {                                                                                                       \
        const int ub_ = s9_ub[v], G_ = s9_un[v], sl_ = 8 * is[0] + 4 * ((v) & 1);                         \
        for (int g_ = 0; g_ < (G_ > 1 ? G_ : 0); g_++) {                                                \
            const int tr_ = s9_sorted[ub_ + g_] / topk;                                                     \
            for (int c_ = 0; c_ < nch; c_++) {                                                                \
                int5 cs_ = {c_ * 512, tr_, 0, 0, 0};                                                        \
                int5 cd_ = {c_ * 512, sl_ + g_, 0, 0, 0};                                                   \
                _Pragma("unroll") for (int p_ = 0; p_ < 4; p_++) {                                          \
                    v_bf16_st_tnsr(cd_, xa, v_bf16_ld_tnsr_b(cs_, xp));                                     \
                    cs_[0] += 128; cd_[0] += 128;                                                           \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
    }
void main(tensor xp, tensor ids, tensor w, tensor s, tensor ap, tensor xa, int K, int e_min, int npts, int gmax,
          float alpha, float limit, float beta, float ascale)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int nch = ((K + 511) / 512 + 1) & ~1;
    const int e_local = get_dim_size(w, 2);
    const int I = get_dim_size(w, 1) / 2;
    const int Ip = (I + 511) / 512 * 512;
    const int topk = get_dim_size(ids, 0);
    const int T = get_dim_size(ids, 1);
    int nu = 0;
    const int nr = mxk9_routes(ids, T, topk, e_min, e_local, gmax, nu);
    s9_act[0] = limit; s9_act[1] = alpha; s9_act[2] = beta; s9_act[3] = ascale;

    const int P = Ip / 2;
    const int pb = is[0] * P / npts, pe = ie[0] * P / npts;
    if (nr == 0 || pb >= pe) return;
    const int np = pe - pb;

    int xrow = -1;
    // run a one-route unit first when there is one, so that unit 0 needs no aux rows
    for (int v = 0; v < nu; v++)
        if (s9_un[v] == 1) {
            const int b_ = s9_ub[v];
            s9_ub[v] = s9_ub[0]; s9_un[v] = s9_un[0];
            s9_ub[0] = b_; s9_un[0] = 1;
            break;
        }
    MXK9_AUXCOPY(0)
    for (int u = 0; u < nu; u++) {
        const int ub = s9_ub[u];
        const int G = s9_un[u];
        const int e = s9_ge[ub];
        const int xr0 = s9_sorted[ub] / topk;
        if (G > 1) xrow = -1;   // v9_x holds the collection vectors of a grouped unit
        else if (xr0 != xrow) { MXK9_STAGE(xp, xr0, nch, 0); xrow = xr0; }
        if (u + 1 < nu) MXK9_AUXCOPY(u + 1)
        if (G > 1 && (u == 0 || np < MXK9_NPSAFE)) {
            int z = nu;
            for (int q = 0; q < MXK9_AUXSPIN; q++) z = z * 3 + 1;
            s9_spin[0] = z;
        }
        const int sl = 8 * is[0] + 4 * (u & 1);   // this unit's aux rows: route g at row sl + g
        // lane constants, made per unit (short live ranges: they are not spilled across the walk)
        const ushort128 pat0 = (read_lane_id_2b_b() >> 3) | 0x80;
        const uchar256 zero8 = 0;
        const uint64 lid4 = read_lane_id_4b_b();
        int m = 0, col0 = 2 * pb;
        if (G == 1) {
            // one route: v8's 4-row passes, one software-pipelined stream over this TPC's row pairs
            uchar256 bA[4], bB[4], sP[4];
            int r0, r1, r2, r3;
            MXK9_ROWS(pb, r0, r1, r2, r3)
            mxk9_first(r0, r1, r2, r3, e, w, s, bA, bB, sP);
            for (int p = pb; p < pe; p++) {
                const int p2 = p + 1 < pe ? p + 1 : p;      // the last pass reloads its own rows (harmless)
                int m0, m1, m2, m3;
                MXK9_ROWS(p2, m0, m1, m2, m3)
                bool256 l0 = MXK9_LMASK(m), l1 = MXK9_LMASK(m + 16), l2 = MXK9_LMASK(m + 32), l3 = MXK9_LMASK(m + 48);
                mxk9_pass_g1(r0, r1, r2, r3, e, m0, m1, m2, m3, e, nch, w, s, pat0, zero8, bA, bB, sP
                             MXK9_CTX_ARGS);
                m++;
                r0 = m0; r1 = m1; r2 = m2; r3 = m3;
                if (m == MXK9_RUN || p + 1 == pe) {
                    MXK9_FLUSH(v9_hb[0], ub)
                    m = 0; col0 = 2 * (p + 1);
                }
            }
            continue;
        }
#if MXK9_GMAX >= 2
        // a unit of G >= 2 routes: 2-row half passes (gate pair, then up pair of each row pair), one stream
        uchar256 bA[2], bB[2], sP[2];
        int ra, rb;
        MXK9_HROWS(0, ra, rb)
        {
            int5 cw = {0, ra, e, 0, 0};
            bA[0] = v_u8_ld_tnsr_b(cw, w); sP[0] = v_u8_ld_tnsr_b(cw, s); cw[0] = 256; bB[0] = v_u8_ld_tnsr_b(cw, w);
            cw[0] = 0; cw[1] = rb;
            bA[1] = v_u8_ld_tnsr_b(cw, w); sP[1] = v_u8_ld_tnsr_b(cw, s); cw[0] = 256; bB[1] = v_u8_ld_tnsr_b(cw, w);
        }
        const int nh = 2 * np;
        for (int h = 0; h < nh; h++) {
            const int h2 = h + 1 < nh ? h + 1 : h;
            int ma, mb;
            MXK9_HROWS(h2, ma, mb)
            const int hs = h & 1;
            bool256 la = MXK9_LMASK(m + 32 * hs), lb = MXK9_LMASK(m + 16 + 32 * hs);
#if MXK9_GMAX >= 2
            if (G == 2)
                mxk9_half_g2(ra, rb, e, ma, mb, e, nch, w, s, xa, sl, sl + 1, sl + 2, sl + 3, pat0, zero8, bA, bB, sP
                             MXK9_CTX2_ARGS);
#endif
#if MXK9_GMAX >= 3
            if (G == 3)
                mxk9_half_g3(ra, rb, e, ma, mb, e, nch, w, s, xa, sl, sl + 1, sl + 2, sl + 3, pat0, zero8, bA, bB, sP
                             MXK9_CTX2_ARGS);
#endif
#if MXK9_GMAX >= 4
            if (G == 4)
                mxk9_half_g4(ra, rb, e, ma, mb, e, nch, w, s, xa, sl, sl + 1, sl + 2, sl + 3, pat0, zero8, bA, bB, sP
                             MXK9_CTX2_ARGS);
#endif
            ra = ma; rb = mb;
            if (hs) {
                m++;
                const int p = pb + h / 2;
                if (m == MXK9_RUN || p + 1 == pe) {
                    for (int g = 0; g < G; g++) MXK9_FLUSH(MXK9_HB2(g), ub + g)
                    m = 0; col0 = 2 * (p + 1);
                }
            }
        }
#endif
    }
}
