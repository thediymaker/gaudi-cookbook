// r6_rmsd: numerics.rms_norm's tail after the native rsqrt with both values the indexer layer's consumers see:
//   y = bf16(p) (the attention / indexer linears) and p = round_bf16(f32(x) * r) * f32(w) in fp32 (what the graph
//   compiler hands the indexer's layer_norm: it drops the y.float() round trip).
//   r f32 [N][1] (input 0: the outputs are fp32-typed under capture), x bf16 [N][Dx] (reads D columns), w bf16 [D]
//   -> y f32 [N][D/2] (bf16 pairs, natural order: view as bf16 [N][D]), p f32 [N][D]; index {D/128, N}
#include "r6_common.h"
void main(tensor r, tensor x, tensor w, tensor y, tensor po)
{
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    for (int n = is[1]; n < ie[1]; n++) {
        const float64 rv = R5_LD_F(r, 0, n);
        for (int c = is[0]; c < ie[0]; c++) {
            int64 o[2];
            for (int h = 0; h < 2; h++) {
                const int e = 128 * c + 64 * h;
                const float64 xf = as_float64(v_i32_ld_tnsr_b((int5){e, n, 0, 0, 0}, x, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
                const float64 wf = as_float64(v_i32_ld_tnsr_b((int5){e, 0, 0, 0, 0}, w, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
                const float64 p = as_float64(RNE16(as_int64(xf * rv))) * wf;
                v_f32_st_tnsr((int5){e, n, 0, 0, 0}, po, p);
                o[h] = RNE16(as_int64(p));
            }
            v_i32_st_tnsr((int5){64 * c, n, 0, 0, 0}, y, ((o[0] >> 16) & 0xFFFF) | o[1], SW_PACK | SW_PCK_32_TO_16, 1, 0);
        }
    }
}
