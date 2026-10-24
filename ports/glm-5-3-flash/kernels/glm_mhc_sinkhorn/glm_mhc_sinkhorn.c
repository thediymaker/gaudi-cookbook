// SPDX-License-Identifier: Apache-2.0
// glm_mhc_sinkhorn (Gaudi 2 TPC): the 20-iteration mHC Sinkhorn of GLM-5.3-Flash in one kernel. vLLM's torch loop is
// about 78 small dependent device ops per call, 90 calls per decode step.
//
// Bit-identical to that loop as Gaudi 2 runs it, which takes the device's own order and divide:
//   - column sums (torch .sum(dim=-2)): sequential, ((c0 + c1) + c2) + c3;
//   - row sums (torch .sum(dim=-1)): pairwise, (c0 + c1) + (c2 + c3), the order the device reduce_sum uses over the
//     innermost dim;
//   - division: plain `/`, which compiles to the same x * reciprocal(y) as the device's fused divide.
//
// in  c  f32, torch shape [16, N]: TPC dim0 = N matrices (lanes), dim1 = element i*4 + j of the 4x4 matrix
//        (softmax + eps already applied, as vLLM does before its loop)
// out o  f32 [16, N]
// params iters, eps:  c /= colsum(c) + eps;  then iters - 1 times { c /= rowsum(c) + eps;  c /= colsum(c) + eps }
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
                t = t + (C(i, 2) + C(i, 3));        // pairwise, as the device reduce_sum over the last dim
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
