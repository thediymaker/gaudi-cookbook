// SPDX-License-Identifier: Apache-2.0
// Graph-compiler glue for glm_mhc_sinkhorn, included into libmxk_tpc.so by kernels/mxfp4-moe/build.sh --extra-kernel.
// in f32 [16, N] (dim0 = N matrices), out f32 [16, N]; params {int iters; float eps;}; one index point per 64 matrices.
struct GlmMhcSinkParams { int32_t iters; float eps; };

static GlueCodeReturn inst_glm_mhc_sinkhorn(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[1] = {DATA_F32};
    static const TensorDataType to[1] = {DATA_F32};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* xs = in->inputTensors[0].geometry.maxSizes;
    const uint64_t* os = in->outputTensors[0].geometry.maxSizes;
    if (xs[1] != 16 || os[0] != xs[0] || os[1] != 16 || xs[0] == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    for (unsigned d = 2; d < MAX_TENSOR_DIM; d++)
        if (xs[d] > 1 || os[d] > 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    GlmMhcSinkParams prm{20, 1e-6f};
    if (in->nodeParams.nodeParams && in->nodeParams.nodeParamsSize >= sizeof(GlmMhcSinkParams))
        prm = *(const GlmMhcSinkParams*)in->nodeParams.nodeParams;
    if (prm.iters < 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = (xs[0] + 63) / 64;
    allRequired(in, out);
    out->kernel.scalarParams[0] = (uint32_t)prm.iters;
    memcpy(&out->kernel.scalarParams[1], &prm.eps, 4);
    out->kernel.paramsNr = 2;
    return copyElf(out, ELF(glm_mhc_sinkhorn));
}
