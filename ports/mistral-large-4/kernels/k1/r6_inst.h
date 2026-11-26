// Node-cut kernel glue, set 2 (r6_*): instantiation functions + KEntry list fragment R6_ENTRIES.
// r6_ffk: zi i8 [1], x bf16 [M][128] -> row i8 [M][132]; r6_ffkf: zi i8 [1], x f32 [M][128] -> row i8 [M][132]; {M}; mode
static GlueCodeReturn inst_r6ffk(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 2, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I8, DATA_BF16, DATA_I8)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 1);
    return copyElf(out, ELF(r6_ffk));
}
static GlueCodeReturn inst_r6ffkf(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 2, 1); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_I8, DATA_F32, DATA_I8)
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = osz(in, 0, 1);
    allRequired(in, out); userScalars(in, out, 1);
    return copyElf(out, ELF(r6_ffkf));
}
// r6_rmsd: r f32 [N][1], x bf16 [N][Dx], w bf16 [D] -> y f32 [N][D/2] (bf16 pairs), p f32 [N][D]; {D/128, N}
static GlueCodeReturn inst_r6rmsd(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    GlueCodeReturn r = counts(in, 3, 2); if (r != GLUE_SUCCESS) return r;
    TYPES(DATA_F32, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32)
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = osz(in, 1, 0) / 128; out->indexSpaceGeometry[1] = osz(in, 1, 1);
    allRequired(in, out); userScalars(in, out, 0);
    return copyElf(out, ELF(r6_rmsd));
}
#define R6_ENTRIES , {"r6_ffk", inst_r6ffk}, {"r6_ffkf", inst_r6ffkf}, {"r6_rmsd", inst_r6rmsd}
