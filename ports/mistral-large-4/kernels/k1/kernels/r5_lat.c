// r5_lat: the owner layer's latent cache row in one kernel (ML4_KV_CODEC i8-t128-f32):
//   c = rms_norm tail of kv[:512] (as r5_rms: round_bf16(f32(kv) * r) * f32(w), RNE to bf16),
//   kpe = rope.rotate(kv[512:576]) (RoPE layers; zeros on NoPE layers), then paged.fp8_latent_pack:
//   per tile of 128: amax = max|c|, scale = amax > 0 ? amax / 127 : 1, code = clamp(round(c / scale), +-127) as int8,
//   both divisions as the native broadcast division evaluates them (x * reciprocal_cephes_f32(y), r5 probe u3d); row = 512 codes | 4 fp32 scales | 64 bf16 kpe.
//   zi i8 [1] (unused: output dtype under capture), kv bf16 [N][576|512], r f32 [N][1], w bf16 [512],
//   cos f32 [N][32], sin f32 [N][32] -> row i8 [N][656]; index {N}; scalar rope (0 | 1; bit 8: the k_pe rotation in
//   the fma form of r5_rope, ML4_R6_ROPEFMA)
#include "r5_common.h"
void main(tensor zi, tensor kv, tensor r, tensor w, tensor cs, tensor sn, tensor row, int rope)
{
    const int fm = (rope >> 8) & 1;
    rope = rope & 0xFF;
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int64 L = LANE();
    const float64 rc127 = reciprocal_cephes_f32((float64)127.0f);
    for (int n = is[0]; n < ie[0]; n++) {
        const float64 rv = R5_LD_F(r, 0, n);
        float64 sc[4];
        for (int h = 0; h < 2; h++) {             // codes 256 h .. 256 h + 255 = tiles 2h, 2h + 1
            int64 word = 0;
            float64 cv[4];
            for (int j = 0; j < 4; j++) {
                const int e0 = 256 * h + 64 * j;
                const float64 x = as_float64(v_i32_ld_tnsr_b((int5){e0, n, 0, 0, 0}, kv, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
                const float64 wf = as_float64(v_i32_ld_tnsr_b((int5){e0, 0, 0, 0, 0}, w, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
                const float64 t = as_float64(RNE16(as_int64(x * rv)));
                cv[j] = as_float64(RNE16(as_int64(t * wf)));
            }
            for (int tt = 0; tt < 2; tt++) {
                const float64 a = v_f32_reduce_max(v_f32_max_b(v_f32_abs_b(cv[2 * tt], 0, 0, 1, 0), v_f32_abs_b(cv[2 * tt + 1], 0, 0, 1, 0)));
                const float64 s = as_float64(v_i32_sel_grt_i32_b(as_int64(a), (int64)0, as_int64(a * rc127), as_int64((float64)1.0f), 0, 0, 1, 0));
                const float64 inv = reciprocal_cephes_f32(s);
                sc[2 * h + tt] = s;
                for (int u = 0; u < 2; u++) {
                    const int j = 2 * tt + u;
                    float64 q = v_f32_nearbyint_b(cv[j] * inv, SW_RHNE, 0, 1, 0);
                    q = v_f32_min_b(v_f32_max_b(q, -127.0f), 127.0f);
                    const int64 qi = v_convert_f32_to_i32_b(q, SW_RHNE, 0, 1, 0);
                    word = word | ((qi & 0xFF) << (8 * j));
                }
            }
            v_i32_st_tnsr((int5){256 * h, n, 0, 0, 0}, row, word, SW_PACK | SW_PCK_32_TO_8, 1, 0);
        }
        // bytes 512..527: the 4 scales; 528..655: kpe as bf16 pairs (lane 4 + i = elements 2i, 2i + 1)
        int64 kw = 0;
        if (rope) {
            const float128 xf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){504, n, 0, 0, 0}, kv));
            const float64 c = v_f32_ld_tnsr_b((int5){-4, n, 0, 0, 0}, cs);
            const float64 s = v_f32_ld_tnsr_b((int5){-4, n, 0, 0, 0}, sn);
            float64 r0 = xf.v1 * c - xf.v2 * s, r1 = xf.v2 * c + xf.v1 * s;
            if (fm) {
                r0 = v_f32_mac_b(xf.v2, s, xf.v1 * c, SW_NEG, 1, 0);
                r1 = v_f32_mac_b(xf.v2, c, xf.v1 * s, 0, 1, 0);
            }
            kw = ((RNE16(as_int64(r0)) >> 16) & 0xFFFF) | RNE16(as_int64(r1));
        }
        int64 tail = SEL_LT(L, (int64)36, kw, (int64)0);
        tail = SEL_EQ(L, (int64)0, as_int64(sc[0]), tail);
        tail = SEL_EQ(L, (int64)1, as_int64(sc[1]), tail);
        tail = SEL_EQ(L, (int64)2, as_int64(sc[2]), tail);
        tail = SEL_EQ(L, (int64)3, as_int64(sc[3]), tail);
        v_i32_st_tnsr((int5){512, n, 0, 0, 0}, row, tail);
    }
}
