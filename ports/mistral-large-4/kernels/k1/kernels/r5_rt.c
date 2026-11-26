// r5_rt: the decode MoE router after the native softmax (moe.route + _forward_decode, placement B):
//   top-4 of p [T][128] (descending, ties to the lower index as the HPU topk), w / (sum w + 1e-20) with the native
//   division (x * reciprocal), virtual ids [i0, i0+128, i1, i1+128, ...] int32 and the routing weights repeated per
//   half, rounded to bf16 (the mxk call's cast).
//   z i32 [1] (unused: input 0 sets the outputs' dtype under graph capture), p f32 [T][128] -> vid i32 [T][8],
//   vwb i32 [T][4] (8 bf16 bits); index {T}; scalar mode
//   (bits 0-1 sum order: 0 ((w0+w1)+w2)+w3, 1 (w0+w1)+(w2+w3), 2 (w0+w2)+(w1+w3); bit 2: the division's reciprocal
//   as reciprocal_cephes_f32 (the native broadcast division) instead of reciprocal_f32)
#include "r5_common.h"
void main(tensor z, tensor p, tensor vid, tensor vwb, int mode)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 L = LANE();
    const int64 L2 = L + 64;
    const int64 ninf = (int64)0xFF800000;
    for (int t = is[0]; t < ie[0]; t++) {
        int64 v0 = as_int64(v_f32_ld_tnsr_b((int5){0, t, 0, 0, 0}, p));
        int64 v1 = as_int64(v_f32_ld_tnsr_b((int5){64, t, 0, 0, 0}, p));
        float64 w[4];
        int64 id[4];
        for (int k = 0; k < 4; k++) {
            const float64 m = v_f32_reduce_max(v_f32_max_b(as_float64(v0), as_float64(v1)));
            const int64 mb = as_int64(m);
            const int64 c0 = SEL_EQ(v0, mb, L, (int64)1000), c1 = SEL_EQ(v1, mb, L2, (int64)1000);
            const int64 im = v_i32_reduce_min(v_i32_min_b(c0, c1));
            v0 = SEL_EQ(L, im, ninf, v0);
            v1 = SEL_EQ(L2, im, ninf, v1);
            w[k] = m; id[k] = im;
        }
        float64 den;
        if ((mode & 3) == 1) den = (w[0] + w[1]) + (w[2] + w[3]);
        else if ((mode & 3) == 2) den = (w[0] + w[2]) + (w[1] + w[3]);
        else den = ((w[0] + w[1]) + w[2]) + w[3];
        den = den + 1e-20f;
        const float64 rc = (mode & 4) ? reciprocal_cephes_f32(den) : v_reciprocal_f32(den);
        int64 ov = 0, ow = 0;
        for (int k = 0; k < 4; k++) {
            const int64 hb = (RNE16(as_int64(w[k] * rc)) >> 16) & 0xFFFF;
            ow = SEL_EQ(L, (int64)k, hb | (hb << 16), ow);
            ov = SEL_EQ(L, (int64)(2 * k), id[k], ov);
            ov = SEL_EQ(L, (int64)(2 * k + 1), id[k] + 128, ov);
        }
        v_i32_st_tnsr((int5){0, t, 0, 0, 0}, vid, ov);
        v_i32_st_tnsr((int5){0, t, 0, 0, 0}, vwb, ow);
    }
}
