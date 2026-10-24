// SPDX-License-Identifier: Apache-2.0
//
// MoE w2 projection for decode (Gaudi 2 TPC).
// MXK8_ROUTE 0 (combine): out[t, n] = sum over local routes q of token t of tw[q] * (a[q] . W2[e_q, n]),
//   bf16 [T, K], T = Q / topk <= MXK8_TMAX; tokens with no local route = 0. Inputs ap bf16 [Q, Ip], ids i32 [Q],
//   tw bf16 [Q], w u8 [E, K, I/2], s u8 [E, K, I/32].
// MXK8_ROUTE 1 (per route): out[q, n] = a[q] . W2[e_q, n], bf16 [Q, K]; rows of routes whose expert is not local are
//   zeros (for models that apply a per-route operation before the token sum). Inputs ap, ids, w, s.
// I need not be a multiple of 512 (gpt-oss I = 2880): chunks past the row read as zeros, so any I with a
// zero-padded activation row works.
// MXK8_BIAS 1 (gpt-oss, kernel mxk_w2xb, combine mode only): one more input b f32 [E, 2, K/2] (per-expert output
// bias, built once on the host by mxfp4_tpc_moe_rt.bias2: plane 0 = even rows, plane 1 = odd rows, i.e. the A / B
// lane order of the packed sums), added once per (route, strip) as two vector MACs per 128-row block:
//   out[t, n] = sum_q tw[q] * (a[q] . W2[e_q, n] + b[e_q, n]).
// MXK8_TMAX (default 16) bounds the tokens per call in combine mode (gpt-oss build: 32).
//
// Schedule: the activation row is staged in VLM and the kernel runs route-outer: tokens one at a time, each token's
// local routes in ascending local expert order, this TPC's rows inner. The fp32 sums of all of this TPC's rows for
// the current token stay in VLM, packed: per block of 32 points (128 rows) two float64 vectors, A lane 2j / 2j+1 =
// rows 4j / 4j+2 and B lane 2j / 2j+1 = rows 4j+1 / 4j+3, updated with a lane-masked fp32 MAC; convert_all(A, B)
// then puts rows 0..127 in bf16 lanes 0..127 for one store. All passes of the TPC form one software-pipelined stream.
#include "mxk8_core.h"
#define RPP 4
#ifndef MXK8_TMAX
#define MXK8_TMAX 16
#endif
#ifndef MXK8_BIAS
#define MXK8_BIAS 0
#endif
#if MXK8_XTAIL
#error "MXK8_XTAIL is for the w13 kernels only"
#endif
#if MXK8_BIAS && MXK8_ROUTE
#error "MXK8_BIAS is combine mode (mxk_w2xb) only"
#endif
#define MXK8_W2_BLK 32                       // points per accumulator block (128 rows)
#ifndef MXK8_W2_NB
#define MXK8_W2_NB 4                         // blocks held at once (512 rows; the point range is walked in strips)
#endif

#if MXK8_ROUTE
void main(tensor ap, tensor ids, tensor w, tensor s, tensor out, int I, int topk, int e_min, int npts)
#else
__local__ float64 v8_acc[2 * MXK8_W2_NB];
__local__ int s8_toff[MXK8_TMAX + 1];
__local__ int s8_tord[MXK8_QMAX];
#if MXK8_BIAS
void main(tensor ap, tensor ids, tensor tw, tensor w, tensor s, tensor bias, tensor out, int I, int topk, int e_min,
          int npts)
#else
void main(tensor ap, tensor ids, tensor tw, tensor w, tensor s, tensor out, int I, int topk, int e_min, int npts)
#endif
#endif
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

    const int P = get_dim_size(w, 1) / RPP;
    const int pb = is[0] * P / npts, pe = ie[0] * P / npts;
    if (pb >= pe) return;
    uchar256 bA[4], bB[4], sP[4];

#if MXK8_ROUTE
    // zero rows for routes that are not local
    for (int q = 0; q < Q; q++) {
        if (s8_eq[q] >= 0) continue;
        for (int p = pb; p < pe; p += MXK8_W2_BLK) {          // 128 rows per store
            const int qn = (pe - p) < MXK8_W2_BLK ? (pe - p) : MXK8_W2_BLK;
            int5 co = {p * RPP, q, 0, 0, 0};
            v_bf16_st_tnsr_partial(co, out, (bfloat128)0.0f, qn * RPP - 1, 0);
        }
    }
    if (nr == 0) return;
    const int np = pe - pb;
    int gi = 0, p = pb, e = s8_ge[0];
    MXK8_STAGE_X(ap, s8_sorted[0], nch);
    mxk8_first(p * RPP, p * RPP + 1, p * RPP + 2, p * RPP + 3, e, w, s, bA, bB, sP);
    const int total = nr * np;
    for (int it = 0; it < total; it++) {
        int p2 = p + 1, gi2 = gi;
        if (p2 >= pe) { p2 = pb; gi2 = gi + 1; }
        if (gi2 >= nr) { p2 = p; gi2 = gi; }
        const int f = s8_ge[gi2];
        const int n0 = p * RPP, m0 = p2 * RPP;
        float128 o = mxk8_pass_dot(n0, n0 + 1, n0 + 2, n0 + 3, e, m0, m0 + 1, m0 + 2, m0 + 3, f, nch, w, s, pat0,
                                   zero8, lid4, bA, bB, sP);
        int5 co = {n0, s8_sorted[gi], 0, 0, 0};
        v_bf16_st_tnsr_partial(co, out, v_convert_f32_to_bf16_all_b(o), RPP - 1, 0);
        if (gi2 != gi) MXK8_STAGE_X(ap, s8_sorted[gi2], nch);
        gi = gi2; p = p2; e = f;
    }
#else
    const int T = Q / topk;
    // tokens with no local route: zero rows
    for (int t = 0; t < T; t++) s8_toff[t] = 0;
    for (int gi = 0; gi < nr; gi++) { const int t = s8_sorted[gi] / topk; s8_toff[t] = s8_toff[t] + 1; }
    for (int t = 0; t < T; t++) {
        if (s8_toff[t] != 0) continue;
        for (int p = pb; p < pe; p += MXK8_W2_BLK) {          // 128 rows per store
            const int qn = (pe - p) < MXK8_W2_BLK ? (pe - p) : MXK8_W2_BLK;
            int5 co = {p * RPP, t, 0, 0, 0};
            v_bf16_st_tnsr_partial(co, out, (bfloat128)0.0f, qn * RPP - 1, 0);
        }
    }
    if (nr == 0) return;
    // per-token route lists, each in ascending local expert order (the expert-sorted order);
    // s8_toff[t] = count, then cursor (no shifting copy loop: the backend cannot lower a VLM/SLM memmove)
    {
        int off = 0;
        for (int t = 0; t < T; t++) { const int c = s8_toff[t]; s8_toff[t] = off; off += c; }
        for (int gi = 0; gi < nr; gi++) {
            const int t = s8_sorted[gi] / topk;
            const int o = s8_toff[t];
            s8_tord[o] = gi;
            s8_toff[t] = o + 1;
        }
    }
    // zeros from a runtime value: a loop storing a constant 0 to VLM becomes a memset the backend cannot lower
    const float64 zf = (float64)((float)npts * 0.0f);
    uint64 lsel = lid4 & 1u;
    uint64 lpt = lid4 >> 1;
    // strips of up to NB * 32 points; per strip one pipelined stream over (token, route, point)
    const int SP = MXK8_W2_NB * MXK8_W2_BLK;
    for (int sb = pb; sb < pe; sb += SP) {
        const int se = sb + SP < pe ? sb + SP : pe;
        const int nb = (se - sb + MXK8_W2_BLK - 1) / MXK8_W2_BLK;
        for (int i = 0; i < 2 * nb; i++) v8_acc[i] = zf;
        int k = 0, p = sb;                       // pass = (k-th local route in token order, point p)
        int gi = s8_tord[0], e = s8_ge[gi];
        MXK8_STAGE_X(ap, s8_sorted[gi], nch);
        mxk8_first(p * RPP, p * RPP + 1, p * RPP + 2, p * RPP + 3, e, w, s, bA, bB, sP);
        const int total = nr * (se - sb);
        for (int it = 0; it < total; it++) {
            int p2 = p + 1, k2 = k;
            if (p2 >= se) { p2 = sb; k2 = k + 1; }
            if (k2 >= nr) { p2 = p; k2 = k; }
            const int gi2 = s8_tord[k2];
            const int f = s8_ge[gi2];
            const int n0 = p * RPP, m0 = p2 * RPP;
            float128 r = mxk8_pass_dot(n0, n0 + 1, n0 + 2, n0 + 3, e, m0, m0 + 1, m0 + 2, m0 + 3, f, nch, w, s,
                                       pat0, zero8, lsel, bA, bB, sP);
            const int q = s8_sorted[gi];
            int5 cq = {q, 0, 0, 0, 0};
            const float wq = s_convert_bf16_to_f32(s_bf16_ld_g((__global void*)gen_addr(cq, tw)));
#if MXK8_BIAS && !defined(MXK8_DIAG_NOBIAS)   // DIAG_NOBIAS: timing only
            if (p == sb) {
                // first pass of this route in the strip: tw * bias of expert e for every row of the strip. Block b
                // (rows 4 * (sb + 32 b) ..): A lanes = its even rows (plane 0), B lanes = its odd rows (plane 1).
                for (int b = 0; b < nb; b++) {
                    int5 cb = {(sb + b * MXK8_W2_BLK) * 2, 0, e, 0, 0};
                    const float64 be = v_f32_ld_tnsr_b(cb, bias); cb[1] = 1;
                    const float64 bo = v_f32_ld_tnsr_b(cb, bias);
                    v8_acc[2 * b] = v_f32_mac_b(be, wq, v8_acc[2 * b], 0, 1, 0);
                    v8_acc[2 * b + 1] = v_f32_mac_b(bo, wq, v8_acc[2 * b + 1], 0, 1, 0);
                }
            }
#endif
            const int pl = p - sb;
            const int bi = pl / MXK8_W2_BLK;
            const int j = pl - bi * MXK8_W2_BLK;
            bool256 m = from_bool64(v_u32_cmp_eq_b(lpt, (unsigned)j, 0, to_bool64((bool256){0}), 1, 0));
            v8_acc[2 * bi] = v_f32_mac_vb(r.v1, wq, v8_acc[2 * bi], 0, to_bool64(m), 0);
            v8_acc[2 * bi + 1] = v_f32_mac_vb(r.v2, wq, v8_acc[2 * bi + 1], 0, to_bool64(m), 0);
            const int t = q / topk;
            const int tend = p + 1 >= se && (k + 1 >= nr || s8_sorted[s8_tord[k + 1]] / topk != t);
            if (tend) {
                // last pass of token t in this strip: store its rows, clear the sums
                for (int b = 0; b < nb; b++) {
                    float128 o; o.v1 = v8_acc[2 * b]; o.v2 = v8_acc[2 * b + 1];
                    const int q0 = sb + b * MXK8_W2_BLK;
                    const int qn = (se - q0) < MXK8_W2_BLK ? (se - q0) : MXK8_W2_BLK;
                    int5 co = {q0 * RPP, t, 0, 0, 0};
                    v_bf16_st_tnsr_partial(co, out, v_convert_f32_to_bf16_all_b(o), qn * RPP - 1, 0);
                    v8_acc[2 * b] = zf; v8_acc[2 * b + 1] = zf;
                }
            }
            if (gi2 != gi) MXK8_STAGE_X(ap, s8_sorted[gi2], nch);
            k = k2; p = p2; gi = gi2; e = f;
        }
    }
#endif
}
