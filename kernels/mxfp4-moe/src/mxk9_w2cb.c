// SPDX-License-Identifier: Apache-2.0
//
// Kernel mxk_w2cb (v10, gpt-oss): mxk_w2c (mxk9_w2.h, MXK9_W2_COMBINE 1) plus the per-expert output bias of the
// gpt-oss mxk_w2xb (mxk8_w2.h, MXK8_BIAS 1):
//   out[t, n] = sum over local routes q of token t, ascending local expert, of tw[q] * (part[q, n] + b[e_q, n])
// in mxk_w2xb's fp32 order per lane: acc = 0; per route: acc = mac(b, tw, acc), then acc = mac(part, tw, acc). With
// part = mxk_w2g / mxk_w2gn (mxk_w2xb's per-route row sums bit for bit), the output is mxk_w2xb's bit for bit.
// Inputs: part f32 [Q, K] (mxk9_w2.h block layout), ids i32 [T, topk], tw bf16 [T, topk], b f32 [E, 2, K/2] (the
// mxk_w2xb bias layout: plane 0 = even rows, plane 1 = odd rows of the expert's K outputs) -> out bf16 [T, K].
// Block b0 (32 points from this TPC's pb) = rows 4 b0 ..: its even rows are b[e, 0, 2 b0 ..], its odd rows
// b[e, 1, 2 b0 ..] (the A / B lane order of the part block, as mxk_w2xb's v8_acc[2b] / [2b + 1]). Tokens with no
// local route are zeros (no bias), as mxk_w2xb.
#include "mxk9_core.h"
#define RPP 4
#define MXK9_W2_BLK 32
__local__ int s9_tq[16];
void main(tensor part, tensor ids, tensor tw, tensor bias, tensor out, int e_min, int e_local, int npts)
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
                int5 cb = {b0 * 2, 0, s9_eq[k], 0, 0};
                const float64 be = v_f32_ld_tnsr_b(cb, bias); cb[1] = 1;
                const float64 bo = v_f32_ld_tnsr_b(cb, bias);
                int5 ca = {b0 * RPP, t * topk + j, 0, 0, 0};
                const float64 A = v_f32_ld_tnsr_b(ca, part);
                ca[0] += 2 * qn;
                const float64 B = v_f32_ld_tnsr_b(ca, part);
                o.v1 = v_f32_mac_b(be, wq, o.v1, 0, 1, 0);
                o.v2 = v_f32_mac_b(bo, wq, o.v2, 0, 1, 0);
                o.v1 = v_f32_mac_b(A, wq, o.v1, 0, 1, 0);
                o.v2 = v_f32_mac_b(B, wq, o.v2, 0, 1, 0);
            }
            int5 co = {b0 * RPP, t, 0, 0, 0};
            v_bf16_st_tnsr_partial(co, out, v_convert_f32_to_bf16_all_b(o), qn * RPP - 1, 0);
        }
    }
}
