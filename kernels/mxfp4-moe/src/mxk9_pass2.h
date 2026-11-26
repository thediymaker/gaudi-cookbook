// SPDX-License-Identifier: Apache-2.0
//
// One v9 half pass for a unit of MXK9_G routes of one expert (included once per G by the kernel; see
// mxk9_core.h): 2 weight rows a, b of expert e against every route of the unit, i.e. half of mxk9_pass.h's 4 rows,
// so the per-route accumulators (2 x G float128) and the stream registers (bA / bB / sP for 2 rows) fit the 40
// vector registers without spilling (4 rows x 2 routes spilled ~80 VLM accesses per chunk pair).
// Activation rows: routes g < MXK9_NXV from VLM slot g (chunk offset g * MXK9_XCH / MXK9_NXV), the others from
// rows r1, r2, r3 of xq (the kernel's SRAM copy of those rows, or xp). MXK9_X0AUX 1: route 0 from row rx0 of xq too.
// Rows na, nb of expert f are the next half pass (its first chunk pair is loaded by the last pair of this one).
// Per route the MAC chain is v8's: acc over chunk pairs, sub-vectors 0..3, then v_f32_reduce_add(acc.v1 + acc.v2).
// Sums go to the kernel's MXK9_COLLECT2(g, ha, hb).
#ifndef MXK9_G
#error "define MXK9_G before including mxk9g_pass2.h"
#endif
#ifndef MXK9_NXV
#define MXK9_NXV 1
#endif
#ifndef MXK9_X0AUX
#define MXK9_X0AUX 0
#endif
#define MXK9_CAT_(a, b) a##b
#define MXK9_CAT(a, b) MXK9_CAT_(a, b)
#define MXK9_XSLOT(g) ((g) * (MXK9_XCH / MXK9_NXV))

// Every chunk is loaded at the half that uses it: VLM routes from their slot, global routes from row r_g of xq.
#define MXK9_XV(g, xg, c) MXK9_LDX(xg, (c) + MXK9_XSLOT(g))
#if MXK9_NXV >= 2
#define MXK9_L1(c) MXK9_XV(1, x1, c)
#else
#define MXK9_L1(c) MXK9_LDXG(x1, xq, c, r1)
#endif
#define MXK9_L2(c) MXK9_LDXG(x2, xq, c, r2)
#define MXK9_L3(c) MXK9_LDXG(x3, xq, c, r3)
#if MXK9_X0AUX              // route 0 from row rx0 of xq too (no VLM activation reads in the half passes)
#define MXK9_X0LD(c) MXK9_LDXG(x0, xq, c, rx0)
#else
#define MXK9_X0LD(c) MXK9_LDX(x0, c)
#endif
#if MXK9_G == 1
#define MXK9_XLD(c) MXK9_X0LD(c)
#define MXK9_MACS(j, W) MXK9_MAC(a0[j], W, x0)
#elif MXK9_G == 2
#define MXK9_XLD(c) { MXK9_X0LD(c); MXK9_L1(c) }
#define MXK9_MACS(j, W) { MXK9_MAC(a0[j], W, x0); MXK9_MAC(a1[j], W, x1); }
#elif MXK9_G == 3
#define MXK9_XLD(c) { MXK9_X0LD(c); MXK9_L1(c) MXK9_L2(c) }
#define MXK9_MACS(j, W) { MXK9_MAC(a0[j], W, x0); MXK9_MAC(a1[j], W, x1); MXK9_MAC(a2[j], W, x2); }
#elif MXK9_G == 4
#define MXK9_XLD(c) { MXK9_X0LD(c); MXK9_L1(c) MXK9_L2(c) MXK9_L3(c) }
#define MXK9_MACS(j, W) { MXK9_MAC(a0[j], W, x0); MXK9_MAC(a1[j], W, x1); MXK9_MAC(a2[j], W, x2); MXK9_MAC(a3[j], W, x3); }
#else
#error "mxk9g_pass2.h: MXK9_G must be 1..4"
#endif

// one chunk pair c of the half pass; the weight loads it issues are chunk pair cn of rows RR[] of expert EE
#define MXK9_PAIR2(c, cn, RR, EE)                                                                   \
    {                                                                                                   \
        MXK9_XLD(c)                                                                                     \
        _Pragma("unroll") for (int j = 0; j < 2; j++) {                                                 \
            ushort128 bc;                                                                               \
            MXK9_SCALES(bc, sP[j], zero8);                                                              \
            bfloat128 sc = MXK9_PICK(bc, pat0);                                                         \
            uchar256 b = bA[j];                                                                         \
            int5 cw = {(cn) * 256, RR[j], EE, 0, 0};                                                    \
            bA[j] = v_u8_ld_tnsr_b(cw, w);                                                              \
            bfloat128 W[4];                                                                             \
            MXK9_DEQ(W, b, sc)                                                                          \
            MXK9_MACS(j, W)                                                                             \
        }                                                                                               \
        MXK9_XLD((c) + 1)                                                                               \
        _Pragma("unroll") for (int j = 0; j < 2; j++) {                                                 \
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
            MXK9_MACS(j, W)                                                                             \
        }                                                                                               \
    }

#define MXK9_ZERO2(a) _Pragma("unroll") for (int j = 0; j < 2; j++) { a[j].v1 = 0; a[j].v2 = 0; }

inline void MXK9_CAT(mxk9_half_g, MXK9_G)(int na0, int nb0, int e, int ma0, int mb0, int f, int nch, tensor w,
                                          tensor s, tensor xq, int rx0, int r1, int r2, int r3, ushort128 pat0, uchar256 zero8,
                                          uchar256 (&bA)[2], uchar256 (&bB)[2], uchar256 (&sP)[2]
                                          MXK9_CTX2_PARAMS)
{
    const int nn[2] = {na0, nb0};
    const int mm[2] = {ma0, mb0};
    const ushort128 pat1 = pat0 | 0x2020;
    (void)rx0; (void)r1; (void)r2; (void)r3;
    float128 a0[2];
    bfloat128 x0[4];
    MXK9_ZERO2(a0)
#if MXK9_G >= 2
    float128 a1[2];
    bfloat128 x1[4];
    MXK9_ZERO2(a1)
#endif
#if MXK9_G >= 3
    float128 a2[2];
    bfloat128 x2[4];
    MXK9_ZERO2(a2)
#endif
#if MXK9_G >= 4
    float128 a3[2];
    bfloat128 x3[4];
    MXK9_ZERO2(a3)
#endif
    for (int c = 0; c < nch - 2; c += 2) MXK9_PAIR2(c, c + 2, nn, e)
    MXK9_PAIR2(nch - 2, 0, mm, f)
    MXK9_COLLECT2(0, MXK9_HSUM(a0[0]), MXK9_HSUM(a0[1]))
#if MXK9_G >= 2
    MXK9_COLLECT2(1, MXK9_HSUM(a1[0]), MXK9_HSUM(a1[1]))
#endif
#if MXK9_G >= 3
    MXK9_COLLECT2(2, MXK9_HSUM(a2[0]), MXK9_HSUM(a2[1]))
#endif
#if MXK9_G >= 4
    MXK9_COLLECT2(3, MXK9_HSUM(a3[0]), MXK9_HSUM(a3[1]))
#endif
}

#undef MXK9_XLD
#undef MXK9_X0LD
#undef MXK9_MACS
#undef MXK9_L1
#undef MXK9_L2
#undef MXK9_L3
#undef MXK9_XV
#undef MXK9_XSLOT
#undef MXK9_PAIR2
#undef MXK9_ZERO2
#undef MXK9_G
