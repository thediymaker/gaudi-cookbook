// r5 shared helpers (Gaudi2 TPC-C)
#pragma once
#define LANE() ((int64)read_lane_id_4b_b())
// fp32 bits -> the bits of the nearest bf16 (RNE, finite values), kept as fp32 bits (low 16 zero): numerics.round_bf16
#define RNE16(b) (((b) + 0x7FFF + (((b) >> 16) & 1)) & (int64)0xFFFF0000)
#define R5_LD_F(t, c0, c1) s_f32_ld_g((__global void*)gen_addr((int5){c0, c1, 0, 0, 0}, t), 0, 0, 1, 0)
#define R5_LD_I(t, c0, c1) s_i32_ld_g((__global void*)gen_addr((int5){c0, c1, 0, 0, 0}, t), 0, 0, 1, 0)
#define SEL_EQ(a, b, x, y) v_i32_sel_eq_i32_b(a, b, x, y, 0, 0, 1, 0)
#define SEL_LT(a, b, x, y) v_i32_sel_less_i32_b(a, b, x, y, 0, 0, 1, 0)
// float128 (v1 = even elements, v2 = odd: convert_all order) of exactly-bf16 fp32 values -> bfloat128, natural order
inline bfloat128 r5_pack(int64 e, int64 o)
{
    float128 g; g.v1 = as_float64(e); g.v2 = as_float64(o);
    return v_convert_f32_to_bf16_all_b(g);
}
