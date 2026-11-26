// r5_ssl: numerics.ssmax_lambda: lam[n, h] = round_bf16(round_bf16(lp[n]) * f32(s[h])) + f32(b[h]), then to bf16 (r5_ssl)
// or kept fp32 (r5_sslf, SSL_F32: the value the native graph hands a fused .float() consumer).
//   r5_ssl: s bf16 [H], b bf16 [H], lp f32 [N][1] -> lam bf16 [N][H];  r5_sslf: lp f32 [N][1], s, b -> lam f32 [N][H]
//   index {N}; H <= 64
#include "r5_common.h"
#ifdef SSL_F32
void main(tensor lp, tensor s, tensor b, tensor lam)
#else
void main(tensor s, tensor b, tensor lp, tensor lam)
#endif
{
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const float64 sf = as_float64(v_i32_ld_tnsr_b((int5){0, 0, 0, 0, 0}, s, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
    const float64 bf = as_float64(v_i32_ld_tnsr_b((int5){0, 0, 0, 0, 0}, b, SW_UNPACK | SW_UNPCK_16_TO_32, 0, 1, 0) << 16);
    for (int n = is[0]; n < ie[0]; n++) {
        const float l = R5_LD_F(lp, 0, n);
        const float64 lr = as_float64(RNE16(as_int64((float64)l)));
        const float64 prod = as_float64(RNE16(as_int64(lr * sf)));
        const float64 v = prod + bf;
#ifdef SSL_F32
        v_f32_st_tnsr((int5){0, n, 0, 0, 0}, lam, v);
#else
        v_i32_st_tnsr((int5){0, n, 0, 0, 0}, lam, (RNE16(as_int64(v)) >> 16) & 0xFFFF, SW_PACK | SW_PCK_32_TO_16, 1, 0);
#endif
    }
}
