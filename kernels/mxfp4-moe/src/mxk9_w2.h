// SPDX-License-Identifier: Apache-2.0
//
// MoE w2 projection for decode, grouped routes (Gaudi2 TPC), v9, in two kernels:
//   mxk_w2g  part[q, n] = a[q] . W2[e_q, n]  (fp32 row sums of every local route, v8's per-route sums bit for bit)
//   mxk_w2c  out[t, n] = sum over local routes q of token t, ascending local expert, of tw[q] * part[q, n]
//            (fp32, v8 mxk_w2x's per-token MAC chain: acc = mac(part, tw, acc) from 0), -> bf16 [T, K]
// The two together give mxk_w2x's output bit for bit. The split is what lets w2g walk routes expert-outer (one
// unpack per unit of routes of one expert, mxk9_core.h) while keeping v8's per-token summation order: the per-token
// fp32 sums of a whole step do not fit vector local memory (T x 1 KB per TPC), and the combine needs the partial
// sums of routes that other units produced. A second kernel reads them after the first has finished (a graph edge),
// so no same-kernel read-after-write of global memory is relied on.
//
// part layout (fp32 [Q, K], private to the pair): for each TPC's point range [pb, pe) (4 rows per point, the v8 w2
// split), blocks of up to 32 points from pb: block base row R = 4 (pb + 32 b), qn points in the block. Even rows of
// the block (4j, 4j+2 of point j) at R + 2j, R + 2j + 1; odd rows (4j+1, 4j+3) at R + 2 qn + 2j, +1. That is v8's
// packed accumulator (A / B vectors) laid out in the block's own rows, so convert_all(A, B) gives rows in order.
// Rows of non-local routes are never written or read.
// Inputs: w2g: ap bf16 [Q, Ip], ids i32 [T, topk], w u8 [E, K, I/2], s u8 [E, K, I/32] -> part f32 [Q, K];
//         w2c: part, ids, tw bf16 [T, topk] -> out bf16 [T, K]. I % 256 == 0, I <= 3072 (two activation rows in VLM).
#include "mxk9_core.h"
#define RPP 4
#define MXK9_W2_BLK 32

#if MXK9_W2_COMBINE
// ---------------------------------------------------------------------------------------------- mxk_w2c
__local__ int s9_tq[16];
void main(tensor part, tensor ids, tensor tw, tensor out, int e_min, int e_local, int npts)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int topk = get_dim_size(ids, 0);
    const int T = get_dim_size(ids, 1);
    const int P = get_dim_size(out, 0) / RPP;
    const int pb = is[0] * P / npts, pe = ie[0] * P / npts;
    if (pb >= pe) return;
    const float64 zf = (float64)((float)npts * 0.0f);
    for (int t = 0; t < T; t++) {
        // this token's local routes in ascending local expert (insertion sort of <= topk entries)
        int n = 0;
        for (int j = 0; j < topk; j++) {
            int5 ci = {j, t, 0, 0, 0};
            const int e = s_i32_ld_g((__global void*)gen_addr(ci, ids)) - e_min;
            if (e < 0 || e >= e_local) continue;
            int k = n;
            while (k > 0 && s9_eq[k - 1] > e) { s9_eq[k] = s9_eq[k - 1]; s9_tq[k] = s9_tq[k - 1]; k--; }
            s9_eq[k] = e; s9_tq[k] = j;
            n++;
        }
        for (int b0 = pb; b0 < pe; b0 += MXK9_W2_BLK) {
            const int qn = (pe - b0) < MXK9_W2_BLK ? (pe - b0) : MXK9_W2_BLK;
            float128 o; o.v1 = zf; o.v2 = zf;
            for (int k = 0; k < n; k++) {
                const int j = s9_tq[k];
                int5 cq = {j, t, 0, 0, 0};
                const float wq = s_convert_bf16_to_f32(s_bf16_ld_g((__global void*)gen_addr(cq, tw)));
                int5 ca = {b0 * RPP, t * topk + j, 0, 0, 0};
                const float64 A = v_f32_ld_tnsr_b(ca, part);
                ca[0] += 2 * qn;
                const float64 B = v_f32_ld_tnsr_b(ca, part);
                o.v1 = v_f32_mac_b(A, wq, o.v1, 0, 1, 0);
                o.v2 = v_f32_mac_b(B, wq, o.v2, 0, 1, 0);
            }
            int5 co = {b0 * RPP, t, 0, 0, 0};
            v_bf16_st_tnsr_partial(co, out, v_convert_f32_to_bf16_all_b(o), qn * RPP - 1, 0);
        }
    }
}
#else
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
#endif
