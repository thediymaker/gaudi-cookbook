// SPDX-License-Identifier: Apache-2.0
//
// dsv41_sinkhorn (Gaudi 2 TPC): every Sinkhorn-Knopp iteration of a batch of DeepSeek-V4.1 mHC comb matrices in ONE
// kernel (DSV41_HC_SINKHORN=kernel, dsv41_hpu/mhc.py). The torch form is ~40 dependent tiny TPC nodes per call, 80
// calls per decode step: ~3,400 graph nodes, each mostly launch gap.
//
// in  c  f32, torch shape [16, N]: TPC dim0 = N matrices (lanes), dim1 = element i*4 + j of the 4x4 matrix
//        (softmax + hc_eps already applied, as mhc.hc_mixes does before its loop)
// out o  f32 [16, N]
// params iters, eps:  c /= colsum(c) + eps;  then iters - 1 times { c /= rowsum(c) + eps;  c /= colsum(c) + eps }
// Sums run in mhc._row_sums / _col_sums order (((x0 + x1) + x2) + x3), each element divided by its sum, all fp32.
// Index space dim 0: 64-matrix chunks.
#define C(i, j) m[(i) * 4 + (j)]

void main(tensor cin, tensor cout, int iters, float eps)
{
    const int5 s = get_index_space_offset();
    const int5 e = get_index_space_size() + s;
    int5 c = {0, 0, 0, 0, 0};
    for (int d = s[0]; d < e[0]; d++) {
        c[0] = d * 64;
        float64 m[16];
#pragma unroll
        for (int k = 0; k < 16; k++) {
            c[1] = k;
            m[k] = v_f32_ld_tnsr_b(c, cin);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float64 t = C(0, j) + C(1, j);
            t = t + C(2, j);
            t = t + C(3, j);
            t = t + eps;
            C(0, j) = C(0, j) / t; C(1, j) = C(1, j) / t; C(2, j) = C(2, j) / t; C(3, j) = C(3, j) / t;
        }
        for (int it = 1; it < iters; it++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float64 t = C(i, 0) + C(i, 1);
                t = t + C(i, 2);
                t = t + C(i, 3);
                t = t + eps;
                C(i, 0) = C(i, 0) / t; C(i, 1) = C(i, 1) / t; C(i, 2) = C(i, 2) / t; C(i, 3) = C(i, 3) / t;
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float64 t = C(0, j) + C(1, j);
                t = t + C(2, j);
                t = t + C(3, j);
                t = t + eps;
                C(0, j) = C(0, j) / t; C(1, j) = C(1, j) / t; C(2, j) = C(2, j) / t; C(3, j) = C(3, j) / t;
            }
        }
#pragma unroll
        for (int k = 0; k < 16; k++) {
            c[1] = k;
            v_f32_st_tnsr(c, cout, m[k]);
        }
    }
}
