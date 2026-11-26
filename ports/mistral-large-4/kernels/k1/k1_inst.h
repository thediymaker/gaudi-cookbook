// k1_kg: data u8, tags i32, bt i32, own i32 -> K f32 [B][Lk][128], kv i32 [B][Lk]; {Lk/64, B}; BS, tagcheck
static GlueCodeReturn inst_kg(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 4, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_BF16, DATA_I32, DATA_I32, DATA_I32)
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = osz(in, 0, 1) / 64; out->indexSpaceGeometry[1] = osz(in, 0, 2);
    allRequired(in, out); userScalars(in, out, 2);
    return copyElf(out, ELF(k1_kg));
}
// k1_sc: d f32, w f32, pos i32, kv i32 -> keys i32 [B][S][Lk], cnt i32 [B][S][NCH][64]; {NCH, S, B}; NCH
static GlueCodeReturn inst_sc(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 4, 2); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_F32, DATA_I32, DATA_F32, DATA_I32, DATA_I32)
    out->indexSpaceRank = 3; out->indexSpaceGeometry[0] = osz(in, 1, 1);
    out->indexSpaceGeometry[1] = osz(in, 1, 2); out->indexSpaceGeometry[2] = osz(in, 1, 3);
    allRequired(in, out); userScalars(in, out, 1);
    return copyElf(out, ELF(k1_sc));
}
#define INST_PS(N, NI, ELFN)                                                                              \
static GlueCodeReturn inst_p##N(HabanaKernelParams* in, HabanaKernelInstantiation* out) {                \
    GlueCodeReturn r = counts(in, NI, 1); if (r != GLUE_SUCCESS) return r;                               \
    TYPES(DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32)                                               \
    out->indexSpaceRank = 3; out->indexSpaceGeometry[0] = osz(in, 0, 1);                                  \
    out->indexSpaceGeometry[1] = osz(in, 0, 2); out->indexSpaceGeometry[2] = osz(in, 0, 3);               \
    allRequired(in, out); userScalars(in, out, 2);                                                       \
    return copyElf(out, ELF(ELFN));                                                                      \
}
INST_PS(2, 2, k1_p2)
INST_PS(3, 3, k1_p3)
INST_PS(4, 4, k1_p4)
// k1_fin: c1..c4 -> fin i32 [B][S][4][64]; {S, B}; NCH, K
static GlueCodeReturn inst_fin(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 4, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32)
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = osz(in, 0, 1); out->indexSpaceGeometry[1] = osz(in, 0, 2);
    allRequired(in, out); userScalars(in, out, 2);
    return copyElf(out, ELF(k1_fin));
}
// k1_qt: cs f32, qraw bf16, wraw bf16, lp f32, ssm bf16 -> qf f32 [B][S][4096], wf f32 [B][S][32]; {32, S, B}; use_rope
// k1_cm1: keys, pos, fin -> mk i32 [B][S][NCH*MW]; {NCH, S, B}; NCH, K, MW
static GlueCodeReturn inst_cm1(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 3, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_I32, DATA_I32, DATA_I32)
    const int32_t* u = (const int32_t*)in->nodeParams.nodeParams;
    out->indexSpaceRank = 3; out->indexSpaceGeometry[0] = u ? u[0] : 1;
    out->indexSpaceGeometry[1] = osz(in, 0, 1); out->indexSpaceGeometry[2] = osz(in, 0, 2);
    allRequired(in, out); userScalars(in, out, 3);
    return copyElf(out, ELF(k1_cm1));
}
// k1_cm2: keys, pos, fin, mk -> out i32 [B][S][K]; {NCH, S, B}; NCH, K, MW
static GlueCodeReturn inst_cm2(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 4, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32)
    const int32_t* u = (const int32_t*)in->nodeParams.nodeParams;
    out->indexSpaceRank = 3; out->indexSpaceGeometry[0] = u ? u[0] : 1;
    out->indexSpaceGeometry[1] = osz(in, 0, 1); out->indexSpaceGeometry[2] = osz(in, 0, 2);
    allRequired(in, out); userScalars(in, out, 3);
    return copyElf(out, ELF(k1_cm2));
}
// k2_g2 (ML4_ATTN_KERNEL): tags, data bf16, idx, prow, pown -> rows i32 [R][K][576], valid i32 [R][K];
// {K/64, R}; tagcheck
static GlueCodeReturn inst_k2g2(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 5, 2); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_BF16, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32)
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = osz(in, 0, 1) / 64; out->indexSpaceGeometry[1] = osz(in, 0, 2);
    allRequired(in, out); userScalars(in, out, 1);
    return copyElf(out, ELF(k2_g2));
}
#include "r5_inst.h"
#include "r6_inst.h"
struct KEntry { const char* name; InstFn fn; };
static KEntry g_k[] = {{"k1_kg", inst_kg}, {"k1_sc", inst_sc}, {"k1_p2", inst_p2}, {"k1_p3", inst_p3},
                       {"k1_p4", inst_p4}, {"k1_fin", inst_fin}, {"k1_cm1", inst_cm1}, {"k1_cm2", inst_cm2},
                       {"k2_g2", inst_k2g2} R5_ENTRIES R6_ENTRIES};
static const unsigned g_nk = sizeof(g_k) / sizeof(g_k[0]);
