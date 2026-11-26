// r5_rms: the tail of numerics.rms_norm after the native rsqrt: y = bf16( round_bf16(f32(x) * r) * f32(w) ).
//   x bf16 [N][Dx] (Dx >= D: reads the first D columns), r f32 [N][1], w bf16 [D] -> y bf16 [N][D]; index {D/128, N}
// Every op is exact or one IEEE fp32 multiply (the native mult), roundings on the bits as round_bf16 / the RNE cast.
#include "r5_common.h"
void main(tensor x, tensor r, tensor w, tensor y)
{
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    for (int n = is[1]; n < ie[1]; n++) {
        const float rr = R5_LD_F(r, 0, n);
        const float64 rv = rr;
        for (int c = is[0]; c < ie[0]; c++) {
            const float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){128 * c, n, 0, 0, 0}, x));
            const float128 wf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){128 * c, 0, 0, 0, 0}, w));
            const int64 t1 = RNE16(as_int64(xf.v1 * rv)), t2 = RNE16(as_int64(xf.v2 * rv));
            const int64 y1 = RNE16(as_int64(as_float64(t1) * wf.v1)), y2 = RNE16(as_int64(as_float64(t2) * wf.v2));
            v_bf16_st_tnsr((int5){128 * c, n, 0, 0, 0}, y, r5_pack(y1, y2));
        }
    }
}
