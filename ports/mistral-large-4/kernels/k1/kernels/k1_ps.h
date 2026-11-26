// k1_ps<NPREV>: radix pass NPREV+1 (4 bits per pass over the 16-bit key). Inputs keys + the previous passes'
// count vectors; out cnt [B][S][NCH][64], lane j = #(((u - base) >> shift) >= j). index space {NCH, S, B}.
#include "k1_common.h"
#define PSHIFT (12 - 4 * NPREV)
void main(tensor keys, tensor c1
#if NPREV >= 2
          , tensor c2
#endif
#if NPREV >= 3
          , tensor c3
#endif
          , tensor cnt, int NCH, int K)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    const int Lk = get_dim_size(keys, 0);
    const int nv = Lk / 64;
    const int per = (nv + NCH - 1) / NCH;
    const int64 L = LANE();
    const int64 one = 1, zero = 0;
    for (int b = is[2]; b < ie[2]; b++)
    for (int t = is[1]; t < ie[1]; t++) {
        int64 base = k1_digit(k1_tot(c1, NCH, t, b), K) << 12;
#if NPREV >= 2
        base += k1_digit(k1_tot(c2, NCH, t, b), K) << 8;
#endif
#if NPREV >= 3
        base += k1_digit(k1_tot(c3, NCH, t, b), K) << 4;
#endif
        for (int c = is[0]; c < ie[0]; c++) {
            const int v0 = c * per;
            const int v1 = v0 + per < nv ? v0 + per : nv;
            K1_CNT_DECL
            for (int v = v0; v < v1; v++) {
                const int64 u = v_i32_ld_tnsr_b((int5){v * 64, t, b, 0, 0}, keys) & 0xFFFF;
                K1_CNT_ADD((u - base) >> PSHIFT)
            }
            int64 out;
            K1_CNT_OUT(out)
            v_i32_st_tnsr((int5){0, c, t, b, 0}, cnt, out);
        }
    }
}
