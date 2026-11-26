// SPDX-License-Identifier: Apache-2.0
//
// The v9 pass for a unit of one route (included once by each kernel; see mxk9_core.h): v8's 4-row pass
// (mxk8_core.h MXK8_PASS_BODY). Rows n0..n3 of expert e against the route's activation row, staged in VLM (v9_x).
// Rows m0..m3 of expert f are the next pass: its first chunk pair is loaded by the last pair of this one (the v8
// stream). The four horizontal sums go to the kernel's MXK9_COLLECT(0, h0, h1, h2, h3), which reads the context the
// kernel passes through MXK9_CTX_PARAMS.
// MAC chain: acc[j] over chunk pairs, sub-vectors 0..3, then v_f32_reduce_add(acc.v1 + acc.v2), as v8.
#if !defined(MXK9_G) || MXK9_G != 1
#error "mxk9g_pass.h is the one-route pass: define MXK9_G 1"
#endif

// one chunk pair c of the pass; the loads it issues are chunk pair cn of rows RR[] of expert EE (v8 MXK8_PAIR)
#define MXK9_PAIRG(c, cn, RR, EE)                                                                       \
    {                                                                                                   \
        MXK9_LDX(x0, c)                                                                                 \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK9_SCALES(bc, sP[j], zero8);                                                              \
            bfloat128 sc = MXK9_PICK(bc, pat0);                                                         \
            uchar256 b = bA[j];                                                                         \
            int5 cw = {(cn) * 256, RR[j], EE, 0, 0};                                                    \
            bA[j] = v_u8_ld_tnsr_b(cw, w);                                                              \
            bfloat128 W[4];                                                                             \
            MXK9_DEQ(W, b, sc)                                                                          \
            MXK9_MAC(a0[j], W, x0)                                                                      \
        }                                                                                               \
        MXK9_LDX(x0, (c) + 1)                                                                           \
        _Pragma("unroll") for (int j = 0; j < 4; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK9_SCALES(bc, sP[j], zero8);                                                              \
            bfloat128 sc = MXK9_PICK(bc, pat1);                                                         \
            uchar256 b = bB[j];                                                                         \
            int5 cw = {((cn) + 1) * 256, RR[j], EE, 0, 0};                                              \
            int5 cs = {(cn) * 16, RR[j], EE, 0, 0};                                                     \
            bB[j] = v_u8_ld_tnsr_b(cw, w);                                                              \
            sP[j] = v_u8_ld_tnsr_b(cs, s);                                                              \
            bfloat128 W[4];                                                                             \
            MXK9_DEQ(W, b, sc)                                                                          \
            MXK9_MAC(a0[j], W, x0)                                                                      \
        }                                                                                               \
    }

inline void mxk9_pass_g1(int n0, int n1, int n2, int n3, int e, int m0, int m1, int m2, int m3, int f, int nch,
                         tensor w, tensor s, ushort128 pat0, uchar256 zero8, uchar256 (&bA)[4], uchar256 (&bB)[4],
                         uchar256 (&sP)[4] MXK9_CTX_PARAMS)
{
    const int nn[4] = {n0, n1, n2, n3};
    const int mm[4] = {m0, m1, m2, m3};
    const ushort128 pat1 = pat0 | 0x2020;
    float128 a0[4];
    bfloat128 x0[4];
    _Pragma("unroll") for (int j = 0; j < 4; j++) { a0[j].v1 = 0; a0[j].v2 = 0; }
    for (int c = 0; c < nch - 2; c += 2) MXK9_PAIRG(c, c + 2, nn, e)
    MXK9_PAIRG(nch - 2, 0, mm, f)
    MXK9_COLLECT(0, MXK9_HSUM(a0[0]), MXK9_HSUM(a0[1]), MXK9_HSUM(a0[2]), MXK9_HSUM(a0[3]))
}

#undef MXK9_PAIRG
#undef MXK9_G
