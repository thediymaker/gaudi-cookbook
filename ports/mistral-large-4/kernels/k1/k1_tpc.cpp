// libk1_tpc.so (the ML4 decode-step kernels): the eight k1 kernels in ONE kernel library that forwards every
// other GUID to the next library (default /usr/lib/habanalabs/libtpc_kernels.so, override with K1_TPC_FWD_LIB).
// With the mxk MoE library loaded (mistral_large4_hpu.mxk.setup_env) the chain is
//   GC_KERNEL_PATH=libmxk_tpc.so -> MXK_TPC_FWD_LIB=libk1_tpc.so -> K1_TPC_FWD_LIB=libtpc_kernels.so
// so GC_KERNEL_PATH stays one regular file (the recipe cache requirement). The forwarding section is
// kernels/mxfp4-moe/src/mxk_tpc_lib.cpp's with the mxk names changed to k1.
#include <dlfcn.h>
#include <limits.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "tpc_kernel_lib_interface.h"
using namespace tpc_lib_api;
#ifndef K1_VERSION
#define K1_VERSION "dev"
#endif
#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT const char k1_tpc_version[] = "k1_tpc " K1_VERSION;
#define DECL_ELF(n) extern unsigned char _binary_##n##_o_start; extern unsigned char _binary_##n##_o_end;
#define ELF(n) &_binary_##n##_o_start, &_binary_##n##_o_end
DECL_ELF(k1_kg) DECL_ELF(k1_sc) DECL_ELF(k1_p2) DECL_ELF(k1_p3) DECL_ELF(k1_p4) DECL_ELF(k1_fin) DECL_ELF(k1_cm1) DECL_ELF(k1_cm2) DECL_ELF(k2_g2)
#include "r5_elfs.h"
#include "r6_elfs.h"
typedef GlueCodeReturn (*InstFn)(HabanaKernelParams*, HabanaKernelInstantiation*);
static GlueCodeReturn copyElf(HabanaKernelInstantiation* out, unsigned char* b, unsigned char* e) {
    unsigned sz = (unsigned)(e - b), given = out->kernel.elfSize;
    out->kernel.elfSize = sz;
    if (given < sz) return GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(out->kernel.kernelElf, b, sz);
    return GLUE_SUCCESS;
}
static void allRequired(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    for (unsigned io = 0; io < 2; io++) {
        unsigned n = io ? in->outputTensorNr : in->inputTensorNr;
        for (unsigned t = 0; t < n; t++) {
            TensorAccessPattern* p = io ? &out->outputTensorAccessPattern[t] : &out->inputTensorAccessPattern[t];
            const Tensor& T = io ? in->outputTensors[t] : in->inputTensors[t];
            p->allRequired = 1;
            for (unsigned d = 0; d < MAX_TENSOR_DIM; d++) {
                p->mapping[d].indexSpaceDim = 0; p->mapping[d].a = 0; p->mapping[d].start_b = 0;
                p->mapping[d].end_b = (float)(T.geometry.maxSizes[d] ? T.geometry.maxSizes[d] - 1 : 0);
            }
        }
    }
}
static int envi(const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; }
static GlueCodeReturn counts(HabanaKernelParams* in, unsigned ni, unsigned no) {
    if (in->inputTensorNr != ni) { in->inputTensorNr = ni; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != no) { in->outputTensorNr = no; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    return GLUE_SUCCESS;
}
static uint64_t isz(HabanaKernelParams* in, int t, int d) { return in->inputTensors[t].geometry.maxSizes[d]; }
static uint64_t osz(HabanaKernelParams* in, int t, int d) { return in->outputTensors[t].geometry.maxSizes[d]; }
// params from the op's user params blob: up to 8 int32 scalars (the op passes them)
static void userScalars(HabanaKernelParams* in, HabanaKernelInstantiation* out, int n) {
    const int32_t* u = (const int32_t*)in->nodeParams.nodeParams;
    for (int i = 0; i < n; i++) out->kernel.scalarParams[i] = u ? (uint32_t)u[i] : 0;
    out->kernel.paramsNr = n;
}
static bool checkTypes(HabanaKernelParams* in, const TensorDataType* tin, const TensorDataType* tout) {
    bool ok = true;
    if (getenv("K1_GLUE_DEBUG")) {
        for (unsigned t = 0; t < in->inputTensorNr; t++) fprintf(stderr, " in%u=%d(want %d)", t, (int)in->inputTensors[t].geometry.dataType, (int)tin[t]);
        for (unsigned t = 0; t < in->outputTensorNr; t++) fprintf(stderr, " out%u=%d(want %d)", t, (int)in->outputTensors[t].geometry.dataType, (int)tout[t]);
        fprintf(stderr, "\n");
    }
    for (unsigned t = 0; t < in->inputTensorNr; t++)
        if (in->inputTensors[t].geometry.dataType != tin[t]) { in->inputTensors[t].geometry.dataType = tin[t]; ok = false; }
    for (unsigned t = 0; t < in->outputTensorNr; t++)
        if (in->outputTensors[t].geometry.dataType != tout[t]) { in->outputTensors[t].geometry.dataType = tout[t]; ok = false; }
    return ok;
}
#define TYPES(...) { static const TensorDataType ty_[] = {__VA_ARGS__}; \
    if (!checkTypes(in, ty_, ty_ + in->inputTensorNr)) return GLUE_INCOMPATIBLE_DATA_TYPE; }
#include "k1_inst.h"
static const KEntry* ours(const char* name) {
    if (!name) return nullptr;
    for (unsigned i = 0; i < g_nk; i++)
        if (strncmp(name, g_k[i].name, MAX_NODE_NAME) == 0) return &g_k[i];
    return nullptr;
}

// ------------------------------------------------------------------------------------------------ forwarding
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
#define K1_PASS_LIST(X) X(GetClangGitVersion) X(GetKernelISAs) X(GetKernelNames)                       \
    X(GetKernelSupportingDynamicShapeNames) X(GetShapeInferenceFunction) X(GetShapeInferenceLibVersion) \
    X(GetShapeInferenceUniqueId) X(GetSuggestedTensorManipulation) X(GetSuggestedTensorManipulationV2) \
    X(GetSupportedLayouts) X(GetSupportedLayoutsV2) X(GetVersion) X(HabanaKernel) X(HabanaKernelV2)

extern "C" __attribute__((visibility("hidden"))) uint64_t k1_pass_missing() { return GLUE_FAILED; }
#define K1_PTR(n) extern "C" __attribute__((visibility("hidden"))) void* k1_fp_##n;                  \
    void* k1_fp_##n = (void*)&k1_pass_missing;
K1_PASS_LIST(K1_PTR)
#define K1_TRAMP(n) ".globl " #n "\n.type " #n ",@function\n" #n ":\n\tjmp *k1_fp_" #n "(%rip)\n.size " #n ",.-" #n "\n"
asm(".text\n" K1_PASS_LIST(K1_TRAMP));

__attribute__((constructor)) static void k1_load_stock() {
    const char* p = getenv("K1_TPC_FWD_LIB");
    if (!p || !*p) p = "/usr/lib/habanalabs/libtpc_kernels.so";
    Dl_info self{};
    char a[PATH_MAX], b[PATH_MAX];
    if (dladdr((void*)&k1_load_stock, &self) && self.dli_fname && realpath(p, a) && realpath(self.dli_fname, b) &&
        strcmp(a, b) == 0) {
        fprintf(stderr, "[k1_tpc] K1_TPC_FWD_LIB points at this library; using the stock default\n");
        p = "/usr/lib/habanalabs/libtpc_kernels.so";
    }
    g_stock = dlopen(p, RTLD_NOW | RTLD_LOCAL);
    if (!g_stock) {
        fprintf(stderr, "[k1_tpc] cannot load the stock kernel library %s: %s\n", p, dlerror());
        return;
    }
    s_guids = (pfnGetKernelGuids)dlsym(g_stock, "GetKernelGuids");
    s_inst = (pfnInstantiateTpcKernel)dlsym(g_stock, "InstantiateTpcKernel");
    s_layouts = (pfnGetSupportedDataLayout)dlsym(g_stock, "GetSupportedDataLayouts");
    s_sif = (pfnGetShapeInference)dlsym(g_stock, "GetShapeInference");
    s_libver = (pfnGetLibVersion)dlsym(g_stock, "GetLibVersion");
    s_suggest = (pfnSuggest)dlsym(g_stock, "GetSuggestedManipulation");
#define K1_RESOLVE(n) { void* f = dlsym(g_stock, #n); if (f) k1_fp_##n = f; }
    K1_PASS_LIST(K1_RESOLVE)
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
        strncpy(guids[k].name, g_k[i].name, MAX_NODE_NAME - 1);
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

EXPORT GlueCodeReturn GetShapeInference(DeviceId deviceId, const ShapeInferenceParams* in, ShapeInferenceOutput* out) {
    const char* name = in ? (in->pGuid ? in->pGuid->name : in->guid.name) : nullptr;
    if (ours(name)) return GLUE_SUCCESS;  // static shapes only (as the standalone library)
    return s_sif ? s_sif(deviceId, in, out) : GLUE_FAILED;
}

// Our GUIDs: a standalone library has no GetSupportedDataLayouts and the graph compiler then marks every
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

// Our GUIDs: what the graph compiler does for a library without GetSuggestedManipulation
// (KernelDB::GetSuggestedTensorManipulation: resetOpType(params, suggestion), return success).
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
