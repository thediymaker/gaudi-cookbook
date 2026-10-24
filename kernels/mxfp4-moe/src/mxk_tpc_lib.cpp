// SPDX-License-Identifier: Apache-2.0
//
// libmxk_tpc.so: a TPC kernel library for the Gaudi 2 graph compiler that serves two MXFP4 MoE decode kernels
// (mxk_w13xib, mxk_w2xb; sources mxk8_w13xib.c, mxk8_w2xb.c) AND every stock GUID, by forwarding to the stock
// kernel library (default /usr/lib/habanalabs/libtpc_kernels.so, override with MXK_TPC_FWD_LIB). One file therefore
// replaces the stock library in GC_KERNEL_PATH. That matters because with PT_HPU_RECIPE_CACHE_CONFIG set, the bridge
// accepts GC_KERNEL_PATH only as a single regular file; a "custom.so:stock.so" list fails device init.
//
// Interface: tpc_kernel_lib_interface.h from the installed Gaudi software (/usr/include/habanalabs); it is not part
// of this repository. Kernel contracts:
//   mxk_w13xib: in xp bf16 [T, Kp] (x zero-padded to Kp = K rounded up to 512, then permuted per 512-k chunk),
//               ids i32 [Q], w u8 [E, 2I, K/2] (gate/up rows interleaved), s u8 [E, 2I, K/32],
//               b f32 [E, 4, Ip/2] (bias in the kernel's column order, built by the host: mxfp4_tpc_moe_rt.bias13);
//               out ap bf16 [Q, Ip]; params {int topk; int e_min; float alpha; float limit; float beta; float ascale;}
//   mxk_w2xb  : in ap bf16 [Q, Ip], ids i32 [Q], tw bf16 [Q], w u8 [E, K, I/2], s u8 [E, K, I/32],
//               b f32 [E, 2, K/2] (bias, even rows then odd rows: mxfp4_tpc_moe_rt.bias2);
//               out bf16 [Q/topk, K] (sum over the token's local routes of tw * (a . W2 + b)); params {topk; e_min}
// The two GUIDs have different input counts (5 and 6). Lazy-mode legacy custom ops are cached by input signature,
// so two custom ops with identical input shapes in one process would share a recipe; different input counts rule
// that out for this pair.
#include <dlfcn.h>
#include <limits.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "tpc_kernel_lib_interface.h"

using namespace tpc_lib_api;

#ifndef MXK_VERSION
#define MXK_VERSION "dev"
#endif
#define EXPORT extern "C" __attribute__((visibility("default")))

EXPORT const char mxk_tpc_version[] = "mxk_tpc " MXK_VERSION;

// ------------------------------------------------------------------------------------------------ kernels
// The TPC ELFs are linked in as binary blobs (objcopy -I binary, see build.sh).
#define DECL_ELF(n) extern unsigned char _binary_##n##_o_start; extern unsigned char _binary_##n##_o_end;
#define ELF(n) &_binary_##n##_o_start, &_binary_##n##_o_end
DECL_ELF(mxk_w13xib)
DECL_ELF(mxk_w2xb)

static GlueCodeReturn copyElf(HabanaKernelInstantiation* out, unsigned char* b, unsigned char* e) {
    unsigned sz = (unsigned)(e - b);
    unsigned given = out->kernel.elfSize;
    out->kernel.elfSize = sz;
    if (given < sz) return GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(out->kernel.kernelElf, b, sz);
    return GLUE_SUCCESS;
}

static void allRequired(const HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    for (unsigned t = 0; t < in->inputTensorNr; t++) {
        out->inputTensorAccessPattern[t].allRequired = 1;
        for (unsigned d = 0; d < MAX_TENSOR_DIM; d++) {
            out->inputTensorAccessPattern[t].mapping[d].indexSpaceDim = 0;
            out->inputTensorAccessPattern[t].mapping[d].a = 0;
            out->inputTensorAccessPattern[t].mapping[d].start_b = 0;
            out->inputTensorAccessPattern[t].mapping[d].end_b =
                (float)(in->inputTensors[t].geometry.maxSizes[d] ? in->inputTensors[t].geometry.maxSizes[d] - 1 : 0);
        }
    }
    for (unsigned t = 0; t < in->outputTensorNr; t++) {
        out->outputTensorAccessPattern[t].allRequired = 1;
        for (unsigned d = 0; d < MAX_TENSOR_DIM; d++) {
            out->outputTensorAccessPattern[t].mapping[d].indexSpaceDim = 0;
            out->outputTensorAccessPattern[t].mapping[d].a = 0;
            out->outputTensorAccessPattern[t].mapping[d].start_b = 0;
            out->outputTensorAccessPattern[t].mapping[d].end_b =
                (float)(in->outputTensors[t].geometry.maxSizes[d] ? in->outputTensors[t].geometry.maxSizes[d] - 1 : 0);
        }
    }
}

// Strict data types: returning GLUE_INCOMPATIBLE_DATA_TYPE (expected types written back) stops the graph
// compiler from re-typing these tensors (it otherwise propagates a downstream cast into the kernel output).
static bool checkTypes(HabanaKernelParams* in, const TensorDataType* tin, const TensorDataType* tout) {
    bool ok = true;
    for (unsigned t = 0; t < in->inputTensorNr; t++)
        if (in->inputTensors[t].geometry.dataType != tin[t]) { in->inputTensors[t].geometry.dataType = tin[t]; ok = false; }
    for (unsigned t = 0; t < in->outputTensorNr; t++)
        if (in->outputTensors[t].geometry.dataType != tout[t]) { in->outputTensors[t].geometry.dataType = tout[t]; ok = false; }
    return ok;
}

struct MoeV7Params { int32_t topk; int32_t e_min; float alpha; float limit; float beta; float ascale; };
// one index point per TPC (24 on Gaudi 2), never more points than output units
static uint32_t v7_npts(const HabanaKernelParams* in, uint64_t P) {
    uint32_t npts = in->maxAvailableTpc ? in->maxAvailableTpc : 24;
    if (npts > P) npts = (uint32_t)P;
    return npts;
}
// Size limits of this build: K <= 3072 and I <= 3072 (x / activation row staged in 6 chunks of VLM),
// K % 32 == 0 (whole scale groups), I % 32 == 0 (whole scale groups along w2's K; also makes the interleaved row
// pairs all-real or all-padding), Q <= 128 routes, T = Q / topk <= 32 tokens (w2 per-token lists), E <= 128.
static GlueCodeReturn inst_w13xib(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[5] = {DATA_BF16, DATA_I32, DATA_U8, DATA_U8, DATA_F32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[2].geometry.maxSizes;
    const uint64_t* ss = in->inputTensors[3].geometry.maxSizes;
    const uint64_t* bs = in->inputTensors[4].geometry.maxSizes;
    uint64_t K = ws[0] * 2, I = ws[1] / 2, E = ws[2];
    uint64_t Q = in->inputTensors[1].geometry.maxSizes[0];
    MoeV7Params prm{1, 0, 1.f, 3.0e38f, 0.f, 1.f};
    if (in->nodeParams.nodeParams && in->nodeParams.nodeParamsSize >= sizeof(MoeV7Params))
        prm = *(const MoeV7Params*)in->nodeParams.nodeParams;
    if (Q > 128 || prm.topk < 1 || Q % prm.topk || Q / prm.topk > 32 || E > 128 || E == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (K % 32 || K > 3072 || I % 32 || I == 0 || I > 3072) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    uint64_t Kp = (K + 511) / 512 * 512, Ip = (I + 511) / 512 * 512;
    if (ss[0] != K / 32 || ss[1] != 2 * I || ss[2] != E || bs[0] != Ip / 2 || bs[1] != 4 || bs[2] != E)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (in->outputTensors[0].geometry.maxSizes[0] != Ip || in->outputTensors[0].geometry.maxSizes[1] != Q ||
        in->inputTensors[0].geometry.maxSizes[0] != Kp)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    uint32_t npts = v7_npts(in, Ip / 2);
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = npts;
    out->preferredSplitDim = 1;           // one index point per TPC
    allRequired(in, out);
    out->kernel.scalarParams[0] = (uint32_t)K;
    out->kernel.scalarParams[1] = (uint32_t)prm.topk;
    out->kernel.scalarParams[2] = (uint32_t)prm.e_min;
    out->kernel.scalarParams[3] = npts;
    memcpy(&out->kernel.scalarParams[4], &prm.alpha, 4);
    memcpy(&out->kernel.scalarParams[5], &prm.limit, 4);
    memcpy(&out->kernel.scalarParams[6], &prm.beta, 4);
    memcpy(&out->kernel.scalarParams[7], &prm.ascale, 4);
    out->kernel.paramsNr = 8;
    return copyElf(out, ELF(mxk_w13xib));
}
static GlueCodeReturn inst_w2xb(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    if (in->inputTensorNr != 6) { in->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    static const TensorDataType ti[6] = {DATA_BF16, DATA_I32, DATA_BF16, DATA_U8, DATA_U8, DATA_F32};
    static const TensorDataType to[1] = {DATA_BF16};
    if (!checkTypes(in, ti, to)) return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t* ws = in->inputTensors[3].geometry.maxSizes;
    const uint64_t* ss = in->inputTensors[4].geometry.maxSizes;
    const uint64_t* bs = in->inputTensors[5].geometry.maxSizes;
    uint64_t I = ws[0] * 2, K = ws[1], E = ws[2];
    uint64_t Q = in->inputTensors[1].geometry.maxSizes[0];
    MoeV7Params prm{1, 0, 1.f, 3.0e38f, 0.f, 1.f};
    if (in->nodeParams.nodeParams && in->nodeParams.nodeParamsSize >= 2 * sizeof(int32_t))
        memcpy(&prm, in->nodeParams.nodeParams, 2 * sizeof(int32_t));
    if (Q > 128 || prm.topk < 1 || Q % prm.topk || Q / prm.topk > 32 || E > 128 || E == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (K % 4 || I % 32 || I == 0 || I > 3072) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (ss[0] != I / 32 || ss[1] != K || ss[2] != E || bs[0] != K / 2 || bs[1] != 2 || bs[2] != E)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (in->inputTensors[0].geometry.maxSizes[0] != (I + 511) / 512 * 512 || in->inputTensors[2].geometry.maxSizes[0] != Q ||
        in->outputTensors[0].geometry.maxSizes[0] != K || in->outputTensors[0].geometry.maxSizes[1] != Q / prm.topk)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    uint32_t npts = v7_npts(in, K / 4);
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = npts;
    out->preferredSplitDim = 1;
    allRequired(in, out);
    out->kernel.scalarParams[0] = (uint32_t)I;
    out->kernel.scalarParams[1] = (uint32_t)prm.topk;
    out->kernel.scalarParams[2] = (uint32_t)prm.e_min;
    out->kernel.scalarParams[3] = npts;
    out->kernel.paramsNr = 4;
    return copyElf(out, ELF(mxk_w2xb));
}

typedef GlueCodeReturn (*InstFn)(HabanaKernelParams*, HabanaKernelInstantiation*);
struct KEntry { const char* name; InstFn fn; };
static const KEntry g_kernels[] = {{"mxk_w13xib", inst_w13xib}, {"mxk_w2xb", inst_w2xb}};
static const unsigned g_nk = sizeof(g_kernels) / sizeof(g_kernels[0]);

static const KEntry* ours(const char* name) {
    if (!name) return nullptr;
    for (unsigned i = 0; i < g_nk; i++)
        if (strncmp(name, g_kernels[i].name, MAX_NODE_NAME) == 0) return &g_kernels[i];
    return nullptr;
}

// ------------------------------------------------------------------------------------------------ forwarding
// The graph compiler (Gaudi software 1.24.1) reads six entry points per kernel library: GetLibVersion,
// InstantiateTpcKernel, GetKernelGuids, GetSupportedDataLayouts, GetSuggestedManipulation, GetShapeInference.
// They answer for this library's GUIDs and forward the rest. The remaining stock C entry points are exported as bare jumps
// into the stock library.
typedef GlueCodeReturn (*pfnSuggest)(const HabanaKernelParams*, void*);
typedef void (*pfnResetOpType)(const HabanaKernelParams*, void*);
static void* g_stock;
static pfnGetKernelGuids s_guids;
static pfnInstantiateTpcKernel s_inst;
static pfnGetSupportedDataLayout s_layouts;
static pfnGetShapeInference s_sif;
static pfnGetLibVersion s_libver;
static pfnSuggest s_suggest;

// Stock entry points this library does not interpret: exported as a bare jump through a pointer that the
// constructor fills from the stock library, so any signature passes through untouched.
#define MXK_PASS_LIST(X) X(GetClangGitVersion) X(GetKernelISAs) X(GetKernelNames)                       \
    X(GetKernelSupportingDynamicShapeNames) X(GetShapeInferenceFunction) X(GetShapeInferenceLibVersion) \
    X(GetShapeInferenceUniqueId) X(GetSuggestedTensorManipulation) X(GetSuggestedTensorManipulationV2) \
    X(GetSupportedLayouts) X(GetSupportedLayoutsV2) X(GetVersion) X(HabanaKernel) X(HabanaKernelV2)

extern "C" __attribute__((visibility("hidden"))) uint64_t mxk_pass_missing() { return GLUE_FAILED; }
#define MXK_PTR(n) extern "C" __attribute__((visibility("hidden"))) void* mxk_fp_##n;                  \
    void* mxk_fp_##n = (void*)&mxk_pass_missing;
MXK_PASS_LIST(MXK_PTR)
#define MXK_TRAMP(n) ".globl " #n "\n.type " #n ",@function\n" #n ":\n\tjmp *mxk_fp_" #n "(%rip)\n.size " #n ",.-" #n "\n"
asm(".text\n" MXK_PASS_LIST(MXK_TRAMP));

__attribute__((constructor)) static void mxk_load_stock() {
    const char* p = getenv("MXK_TPC_FWD_LIB");
    if (!p || !*p) p = "/usr/lib/habanalabs/libtpc_kernels.so";
    Dl_info self{};
    char a[PATH_MAX], b[PATH_MAX];
    if (dladdr((void*)&mxk_load_stock, &self) && self.dli_fname && realpath(p, a) && realpath(self.dli_fname, b) &&
        strcmp(a, b) == 0) {
        fprintf(stderr, "[mxk_tpc] MXK_TPC_FWD_LIB points at this library; using the stock default\n");
        p = "/usr/lib/habanalabs/libtpc_kernels.so";
    }
    g_stock = dlopen(p, RTLD_NOW | RTLD_LOCAL);
    if (!g_stock) {
        fprintf(stderr, "[mxk_tpc] cannot load the stock kernel library %s: %s\n", p, dlerror());
        return;
    }
    s_guids = (pfnGetKernelGuids)dlsym(g_stock, "GetKernelGuids");
    s_inst = (pfnInstantiateTpcKernel)dlsym(g_stock, "InstantiateTpcKernel");
    s_layouts = (pfnGetSupportedDataLayout)dlsym(g_stock, "GetSupportedDataLayouts");
    s_sif = (pfnGetShapeInference)dlsym(g_stock, "GetShapeInference");
    s_libver = (pfnGetLibVersion)dlsym(g_stock, "GetLibVersion");
    s_suggest = (pfnSuggest)dlsym(g_stock, "GetSuggestedManipulation");
#define MXK_RESOLVE(n) { void* f = dlsym(g_stock, #n); if (f) mxk_fp_##n = f; }
    MXK_PASS_LIST(MXK_RESOLVE)
}

EXPORT GlueCodeReturn GetKernelGuids(DeviceId deviceId, uint32_t* kernelCount, GuidInfo* guids) {
    const uint32_t nOurs = deviceId == DEVICE_ID_GAUDI2 ? g_nk : 0;
    if (!kernelCount) return s_guids ? s_guids(deviceId, kernelCount, guids) : GLUE_FAILED;
    uint32_t nStock = 0;
    if (s_guids) {
        GlueCodeReturn r = s_guids(deviceId, &nStock, nullptr);
        if (r != GLUE_SUCCESS) return r;
    }
    const uint32_t want = *kernelCount;
    if (want == 0 || !guids) { *kernelCount = nStock + nOurs; return GLUE_SUCCESS; }
    uint32_t got = want < nStock ? want : nStock;
    if (s_guids && got) {
        GlueCodeReturn r = s_guids(deviceId, &got, guids);
        if (r != GLUE_SUCCESS) return r;
    }
    uint32_t k = got;
    for (uint32_t i = 0; i < nOurs && k < want; i++, k++) {
        memset(&guids[k], 0, sizeof(GuidInfo));
        strncpy(guids[k].name, g_kernels[i].name, MAX_NODE_NAME - 1);
    }
    *kernelCount = k;
    return GLUE_SUCCESS;
}

EXPORT GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (const KEntry* k = ours(params->guid.name)) {
        if (params->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
        // The graph compiler hands the params in writable memory; the glue writes back expected counts/types.
        return k->fn(const_cast<HabanaKernelParams*>(params), instance);
    }
    return s_inst ? s_inst(params, instance) : GLUE_NODE_NOT_FOUND;
}

// Without GetShapeInference a custom library is skipped silently. This library's GUIDs have static shapes only.
EXPORT GlueCodeReturn GetShapeInference(DeviceId deviceId, const ShapeInferenceParams* in, ShapeInferenceOutput* out) {
    const char* name = in ? (in->pGuid ? in->pGuid->name : in->guid.name) : nullptr;
    if (ours(name)) return GLUE_SUCCESS;
    return s_sif ? s_sif(deviceId, in, out) : GLUE_FAILED;
}

// This library's GUIDs: a standalone library has no GetSupportedDataLayouts and the graph compiler then marks every
// input/output layout don't-care. An error return fails the setSupportedLayouts pass instead, and the stock
// library answers GLUE_NODE_NOT_FOUND for a GUID it does not own, so answer here: one layout set, every
// tensor "" (no layout = don't-care).
static void dontCare(TensorDataLayout* l, uint32_t n) {
    if (l)
        for (uint32_t i = 0; i < n; i++) memset(l[i].layout, 0, sizeof(l[i].layout));
}
EXPORT GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* params, NodeDataLayouts* layouts,
                                              uint32_t* layoutCount) {
    if (params && ours(params->guid.name)) {
        if (!layoutCount) return GLUE_FAILED;
        if (layouts && *layoutCount >= 1) {
            dontCare(layouts[0].inputs, layouts[0].inputTensorNr);
            dontCare(layouts[0].outputs, layouts[0].outputTensorNr);
            dontCare(layouts[0].shapeTensors, layouts[0].shapeTensorNr);
        }
        *layoutCount = 1;
        return GLUE_SUCCESS;
    }
    return s_layouts ? s_layouts(params, layouts, layoutCount) : GLUE_FAILED;
}

// This library's GUIDs: do what the graph compiler does for a library without GetSuggestedManipulation (call the graph
// compiler's own resetOpType on the suggestion, return success).
EXPORT GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams* params, void* suggestion) {
    if (params && ours(params->guid.name)) {
        static pfnResetOpType reset = (pfnResetOpType)dlsym(
            RTLD_DEFAULT, "_Z11resetOpTypePKN11tpc_lib_api19_HabanaKernelParamsEPNS_29_TensorManipulationSuggestionE");
        if (reset && suggestion) reset(params, suggestion);
        return GLUE_SUCCESS;
    }
    return s_suggest ? s_suggest(params, suggestion) : GLUE_FAILED;
}

EXPORT uint64_t GetLibVersion() { return s_libver ? s_libver() : 0; }
