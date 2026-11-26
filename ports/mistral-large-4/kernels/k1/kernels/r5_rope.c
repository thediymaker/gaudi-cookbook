// r5_rope: rope.rope_head (off 0) / rope_tail (off 64) on 128-wide heads, bf16 in and out.
//   x bf16 [N][H][128], cos f32 [N][32], sin f32 [N][32] -> y bf16 [N][H][128]; index {H, N}; scalar off (0 | 64)
// Pair i of the rotated 64 dims = elements off + 2i, off + 2i + 1 = lane off/2 + i of the convert_all halves.
// y0 = x0 c - x1 s, y1 = x1 c + x0 s: separate IEEE fp32 multiplies and adds (no mac), one RNE to bf16 (rope.rotate).
// r6 (off bit 8, ML4_R6_ROPEFMA): the form the graph compiler uses for small rotations:
// y0 = fma(-x1, s, fl(x0 c)), y1 = fma(x1, c, fl(x0 s)).
#include "r5_common.h"
void main(tensor x, tensor cs, tensor sn, tensor y, int off)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 L = LANE();
    const int fm = (off >> 8) & 1;
    off = off & 0xFF;
    const int l0 = off >> 1;
    for (int n = is[1]; n < ie[1]; n++) {
        const float64 c = v_f32_ld_tnsr_b((int5){-l0, n, 0, 0, 0}, cs);
        const float64 s = v_f32_ld_tnsr_b((int5){-l0, n, 0, 0, 0}, sn);
        for (int h = is[0]; h < ie[0]; h++) {
            const float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0, h, n, 0, 0}, x));
            const float64 a = xf.v1 * c, b = xf.v2 * s, d = xf.v2 * c, e = xf.v1 * s;
            float64 r0 = a - b, r1 = d + e;
            if (fm) {
                r0 = v_f32_mac_b(xf.v2, s, a, SW_NEG, 1, 0);
                r1 = v_f32_mac_b(xf.v2, c, e, 0, 1, 0);
            }
            const int64 in = SEL_LT(L, (int64)l0, (int64)0, SEL_LT(L, (int64)(l0 + 32), (int64)1, (int64)0));
            const int64 o0 = SEL_EQ(in, (int64)1, RNE16(as_int64(r0)), as_int64(xf.v1));
            const int64 o1 = SEL_EQ(in, (int64)1, RNE16(as_int64(r1)), as_int64(xf.v2));
            v_bf16_st_tnsr((int5){0, h, n, 0, 0}, y, r5_pack(o0, o1));
        }
    }
}
