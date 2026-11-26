// k1_cm2: compaction phase 2 (r3 split of k1_cm). Per chunk c of row (t, b): read k1_cm1's mask words with scalar
// loads and write each taken key's index (or -1 if unseen) with a scalar store from the chunk's output offset, in
// ascending index order. Rows at pos <= K-1: dense_topk (j <= pos ? j : -1), written by chunk 0 with vector stores.
// out int32 [B][S][K] (the caller casts to int64 natively); keys (int32, not read) is input 0 so out is typed int32
// under graph capture. index space {NCH, S, B}; scalars NCH, K, MW. (Gaudi2's squeeze store cannot reset its
// counter: not used.)
#include "k1_common.h"
#define POPC(x) s_u32_popcnt((unsigned)(x), SW_COUNT_ONES, 0, 1, 0)
void main(tensor keys, tensor pos, tensor fin, tensor mk, tensor out, int NCH, int K, int MW)
{
    cache_invalidate(SW_SB | SW_D | SW_LU, 1, 0);     // scalar loads of earlier kernels' outputs: no stale D$ lines
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int Lk = get_dim_size(keys, 0);
    const int nv = Lk / 64;
    const int per = (nv + NCH - 1) / NCH;
    const int64 L = LANE();
    const int64 m1 = -1;
    for (int b = is[2]; b < ie[2]; b++)
    for (int t = is[1]; t < ie[1]; t++)
    for (int c = is[0]; c < ie[0]; c++) {
        const int p = K1_LD_I(pos, t, b, 0);
        if (p <= K - 1) {
            if (c == 0)
                for (int j = 0; j < K; j += 64) {
                    const int64 jj = L + j;
                    v_i32_st_tnsr((int5){j, t, b, 0, 0}, out, SEL_LE(jj, p, jj, m1));
                }
            continue;
        }
        const int off = K1_LD_I(fin, c, t, b);
        const int v0 = c * per;
        const int v1 = v0 + per < nv ? v0 + per : nv;
        const int mb = c * MW;
        int o = off;
        for (int vv = v0; vv < v1; vv++) {
            const int i4 = (vv - v0) * 4;
            const int wb = (i4 >> 6) * 64 + (i4 & 63);
            for (int h = 0; h < 2; h++) {
                unsigned m = (unsigned)K1_LD_I(mk, mb + wb + h, t, b);
                const unsigned sm = (unsigned)K1_LD_I(mk, mb + wb + 2 + h, t, b);
                const int sbase = vv * 64 + h * 32;
                while (m) {
                    const unsigned lb = m & (0u - m);
                    const int bit = POPC(lb - 1);
                    const int val = (sm & lb) ? sbase + bit : -1;
                    s_i32_st_g((__global void*)gen_addr((int5){o, t, b, 0, 0}, out), val, 0, 1, 0);
                    o++;
                    m ^= lb;
                }
            }
        }
    }
}
