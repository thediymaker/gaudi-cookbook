// k1_cm1: compaction phase 1 (r3 split of k1_cm). Per chunk c of row (t, b), per 64-key vector of the chunk: the take
// lane mask (key > v, or key == v within the chunk's tie quota, lower index first) and the see mask (not unseen) as
// 2 x 32-bit words each, packed 16 vectors per 64-lane store into mk [B][S][NCH*MW] at c*MW. The scalar read-back is
// in k1_cm2 (a kernel boundary orders these vector stores before its scalar loads; the r2 same-kernel sentinel
// protocol gave a rare wrong element). Rows at pos <= K-1 write nothing (k1_cm2 writes dense_topk).
// keys (int32) is input 0 so mk is typed int32 under graph capture. index space {NCH, S, B}; scalars NCH, K, MW.
#include "k1_common.h"
void main(tensor keys, tensor pos, tensor fin, tensor mk, int NCH, int K, int MW)
{
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);     // scalar loads of earlier kernels' outputs: no stale D$ lines
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int Lk = get_dim_size(keys, 0);
    const int nv = Lk / 64;
    const int per = (nv + NCH - 1) / NCH;
    const int64 L = LANE();
    const int64 li = L & 15;
    const int64 one = 1, zero = 0;
    const int64 sh = L & 31;
    for (int b = is[2]; b < ie[2]; b++)
    for (int t = is[1]; t < ie[1]; t++)
    for (int c = is[0]; c < ie[0]; c++) {
        const int p = K1_LD_I(pos, t, b, 0);
        if (p <= K - 1) continue;
        const int q = K1_LD_I(fin, 64 + c, t, b);
        const int v = K1_LD_I(fin, 128, t, b);
        const int v0 = c * per;
        const int v1 = v0 + per < nv ? v0 + per : nv;
        const int mb = c * MW;
        int64 cum = 0, pk = 0;
        for (int vv = v0; vv < v1; vv++) {
            const int s0 = vv * 64;
            const int64 kk = v_i32_ld_tnsr_b((int5){s0, t, b, 0, 0}, keys);
            const int64 u = kk & 0xFFFF;
            const int64 e01 = SEL_EQ(u, v, one, zero);
            const int64 pre = k1_prefix64(e01, li);
            int64 take = SEL_GT(u, v, one, zero);
            take = take | SEL_LE(cum + pre, q, e01, zero);
            cum += v_i32_reduce_add(e01);
            const int64 see = SEL_EQ(kk >> 16, 0, one, zero);    // (k1_sc folded s <= pos into the flag)
            const int64 tb = take << sh, sb = see << sh;
            const int64 tlo = v_i32_reduce_add(SEL_LE(L, 31, tb, zero));
            const int64 thi = v_i32_reduce_add(SEL_GE(L, 32, tb, zero));
            const int64 slo = v_i32_reduce_add(SEL_LE(L, 31, sb, zero));
            const int64 shi = v_i32_reduce_add(SEL_GE(L, 32, sb, zero));
            const int i4 = ((vv - v0) & 15) * 4;
            pk = SEL_EQ(L, i4, tlo, pk);
            pk = SEL_EQ(L, i4 + 1, thi, pk);
            pk = SEL_EQ(L, i4 + 2, slo, pk);
            pk = SEL_EQ(L, i4 + 3, shi, pk);
            if (((vv - v0) & 15) == 15 || vv == v1 - 1)
                v_i32_st_tnsr((int5){mb + ((vv - v0) >> 4) * 64, t, b, 0, 0}, mk, pk);
        }
    }
}
