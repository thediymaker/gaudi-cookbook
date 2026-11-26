// Node-cut kernel glue, set 1 (r5_*): instantiation functions + KEntry list fragment R5_ENTRIES.
// r5_rms: x bf16 [N][Dx], r f32 [N][1], w bf16 [D] -> y bf16 [N][D]; {D/128, N}
static GlueCodeReturn inst_r5rms(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 3, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_BF16, DATA_F32, DATA_BF16, DATA_BF16)
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = osz(in, 0, 0) / 128; out->indexSpaceGeometry[1] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 0);
    return copyElf(out, ELF(r5_rms));
}
// r5_rope: x bf16 [N][H][128], cos f32 [N][32], sin f32 [N][32] -> y bf16 [N][H][128]; {H, N}; off
static GlueCodeReturn inst_r5rope(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 3, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_BF16, DATA_F32, DATA_F32, DATA_BF16)
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = osz(in, 0, 1); out->indexSpaceGeometry[1] = osz(in, 0, 2);
    allRequired(in, out); userScalars(in, out, 1);
    return copyElf(out, ELF(r5_rope));
}
// r5_rt: z i32 [1], p f32 [T][128] -> vid i32 [T][8], vwb i32 (bf16 pairs) [T][4]; {T}; mode
static GlueCodeReturn inst_r5rt(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 2, 2); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_F32, DATA_I32, DATA_I32)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 1);
    return copyElf(out, ELF(r5_rt));
}
// r5_ff8: x bf16 [M][128] -> y bf16 [M][128]; r5_ff8f: zb bf16 [1], x f32 [M][128] -> y bf16; {M}
static GlueCodeReturn inst_r5ff8(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 1, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_BF16, DATA_BF16)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 0);
    return copyElf(out, ELF(r5_ff8));
}
static GlueCodeReturn inst_r5ff8f(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 2, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_BF16, DATA_F32, DATA_BF16)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 0);
    return copyElf(out, ELF(r5_ff8f));
}
// r5_ssl: s bf16 [H], b bf16 [H], lp f32 [N][1] -> lam bf16 [N][H]; r5_sslf: lp, s, b -> lam f32 [N][H]; {N}
static GlueCodeReturn inst_r5ssl(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 3, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_BF16, DATA_BF16, DATA_F32, DATA_BF16)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 0);
    return copyElf(out, ELF(r5_ssl));
}
static GlueCodeReturn inst_r5sslf(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 3, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_F32, DATA_BF16, DATA_BF16, DATA_F32)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 0);
    return copyElf(out, ELF(r5_sslf));
}
// r5_lat: zi i8 [1], kv bf16 [N][576|512], r f32 [N][1], w bf16 [512], cos f32 [N][32], sin f32 [N][32]
//   -> row i8 [N][656]; {N}; rope
static GlueCodeReturn inst_r5lat(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 6, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I8, DATA_BF16, DATA_F32, DATA_BF16, DATA_F32, DATA_F32, DATA_I8)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 1);
    return copyElf(out, ELF(r5_lat));
}
// k2_g3: tags i32, data bf16, tk i32 [R][K], bt i32 [B][NB], ep i32 [B][NB] -> rows i32 [R][K][576], valid i32 [R][K];
//   {K/64, R}; tagcheck, S
static GlueCodeReturn inst_k2g3(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 5, 2); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I32, DATA_BF16, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32)
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = osz(in, 0, 1) / 64; out->indexSpaceGeometry[1] = osz(in, 0, 2);
    allRequired(in, out); userScalars(in, out, 2);
    return copyElf(out, ELF(k2_g3));
}
#define R5_ENTRIES , {"r5_rms", inst_r5rms}, {"r5_rope", inst_r5rope}, {"r5_rt", inst_r5rt}, {"r5_ff8", inst_r5ff8}, \
    {"r5_ff8f", inst_r5ff8f}, {"r5_ssl", inst_r5ssl}, {"r5_sslf", inst_r5sslf}, {"r5_lat", inst_r5lat}, {"k2_g3", inst_k2g3}
