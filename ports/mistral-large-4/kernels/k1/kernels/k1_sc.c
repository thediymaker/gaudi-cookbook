// k1_sc: score + order key + radix pass 1 for decode rows.
//   d f32 [B][S*32][Lk] (the MME dot, rows t*32+h), w f32 [B][S][32] (weights incl. both scales), pos i32 [B][S],
//   score = sum_c (sum_{h in chunk c} relu(d_h) * w_h) in the plugin's order, rounded to bf16;
//   key u = order-preserving u16 of the bf16 score (-0 as +0), unseen (s > pos, or a NaN dot = tag-invalid key) =
//   0x80 (-inf's key under indexer._ukey: the lowest finite bf16) with bit 16 set (the unseen flag k1_cm writes
//   as -1). out keys i32 [B][S][Lk], cnt i32 [B][S][NCH][64] (pass 1: lane j =
//   #(u >> 12 >= j)). index space {NCH, S, B}; scalars NCH. kref (k1_kg's int32 output, not read) is input 0 so the outputs
//   are typed int32 under graph capture (outputs take input 0's dtype); pos is int32
#include "k1_common.h"
#ifndef K1_PART
#define K1_PART 0
#endif
void main(tensor kref, tensor d, tensor pos, tensor w, tensor keys, tensor cnt, int NCH)
{
    // scalar loads below read tensors written by earlier kernels; the scalar data cache can hold lines of an earlier
    // launch on this core at the same addresses (seen: one stale k1_cm offset/mask in a 252-case run). Drop them.
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int Lk = get_dim_size(d, 0);
    const int nv = Lk / 64;
    const int per = (nv + NCH - 1) / NCH;
    const int64 L = LANE();
    const int64 one = 1, zero = 0;
    const float64 fz = 0.0f;
    for (int b = is[2]; b < ie[2]; b++)
    for (int t = is[1]; t < ie[1]; t++)
    for (int c = is[0]; c < ie[0]; c++) {
        float wh[32];
        for (int h = 0; h < 32; h++) wh[h] = K1_LD_F(w, h, t, b);
        const int p = K1_LD_I(pos, t, b, 0);
        const int v0 = c * per;
        const int v1 = v0 + per < nv ? v0 + per : nv;
        K1_CNT_DECL
        for (int v = v0; v < v1; v++) {
            const int s0 = v * 64;
            float64 sc = 0.0f;
            int64 nanm = 0;
            for (int cc = 0; cc < 8; cc++) {
                float64 r[4];
                for (int i = 0; i < 4; i++) {
                    const int h = cc * 4 + i;
                    const float64 dv = v_f32_ld_tnsr_b((int5){s0, t * 32 + h, b, 0, 0}, d);
                    if (h == 0) nanm = SEL_GT(as_int64(dv) & 0x7FFFFFFF, 0x7F800000, one, zero);
                    r[i] = v_f32_max_b(dv, fz) * wh[h];
                }
#if K1_PART == 0
                const float64 part = ((r[0] + r[1]) + r[2]) + r[3];
#else
                const float64 part = (r[0] + r[1]) + (r[2] + r[3]);
#endif
                sc = sc + part;
            }
            // RNE to bf16 bits, -0 -> +0, order key
            const int64 bits = as_int64(sc);
            int64 r16 = ((bits + (0x7FFF + ((bits >> 16) & 1))) >> 16) & 0xFFFF;
            r16 = SEL_EQ(r16, 0x8000, zero, r16);
            int64 u = SEL_GE(r16, 0x8000, 0xFFFF - r16, r16 + 0x8000);
            const int64 sv = L + s0;
            u = SEL_LE(sv, p, u, (int64)0x10080);
            u = SEL_EQ(nanm, 1, (int64)0x10080, u);
            v_i32_st_tnsr((int5){s0, t, b, 0, 0}, keys, u);
            u = u & 0xFFFF;
            K1_CNT_ADD(u >> 12)
        }
        int64 out;
        K1_CNT_OUT(out)
        v_i32_st_tnsr((int5){0, c, t, b, 0}, cnt, out);
    }
}
