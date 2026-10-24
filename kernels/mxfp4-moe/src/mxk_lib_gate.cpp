// SPDX-License-Identifier: Apache-2.0
//
// Build gate for libmxk_tpc.so (no Gaudi device needed): dlopen the library the way the graph compiler does and
// check that GetKernelGuids(Gaudi 2) lists every stock GUID unchanged and in order, followed by exactly the GUIDs
// named on the command line, that the six entry points the graph compiler reads are exported, and that
// GetLibVersion matches the stock library. Exit 0 = pass.
//   mxk_lib_gate <libmxk_tpc.so> <stock libtpc_kernels.so> <guid>...
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "tpc_kernel_lib_interface.h"

using namespace tpc_lib_api;

static std::vector<std::string> guids(void* h, const char* what) {
    auto f = (pfnGetKernelGuids)dlsym(h, "GetKernelGuids");
    if (!f) { fprintf(stderr, "%s: no GetKernelGuids\n", what); return {}; }
    uint32_t n = 0;
    if (f(DEVICE_ID_GAUDI2, &n, nullptr) != GLUE_SUCCESS || n == 0) { fprintf(stderr, "%s: count failed\n", what); return {}; }
    std::vector<GuidInfo> g(n);
    memset(g.data(), 0, sizeof(GuidInfo) * n);
    if (f(DEVICE_ID_GAUDI2, &n, g.data()) != GLUE_SUCCESS) { fprintf(stderr, "%s: list failed\n", what); return {}; }
    std::vector<std::string> out;
    for (uint32_t i = 0; i < n; i++) out.emplace_back(g[i].name);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <libmxk_tpc.so> <stock lib> <guid>...\n", argv[0]); return 2; }
    void* stock = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    void* ours = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!stock || !ours) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    const char* ver = (const char*)dlsym(ours, "mxk_tpc_version");
    auto gs = guids(stock, "stock"), go = guids(ours, "ours");
    if (gs.empty() || go.empty()) return 1;
    std::vector<std::string> want(argv + 3, argv + argc);
    int bad = 0;
    if (go.size() != gs.size() + want.size()) {
        fprintf(stderr, "guid count %zu != stock %zu + %zu\n", go.size(), gs.size(), want.size());
        bad++;
    }
    for (size_t i = 0; i < gs.size() && i < go.size(); i++)
        if (gs[i] != go[i]) { fprintf(stderr, "stock guid %zu differs: %s vs %s\n", i, gs[i].c_str(), go[i].c_str()); bad++; break; }
    std::string list;
    for (size_t i = 0; i < want.size(); i++) {
        size_t j = gs.size() + i;
        if (j >= go.size() || go[j] != want[i]) { fprintf(stderr, "missing or out of order: %s\n", want[i].c_str()); bad++; }
        list += (i ? ", " : "") + want[i];
    }
    for (const char* n : {"GetLibVersion", "InstantiateTpcKernel", "GetSupportedDataLayouts", "GetShapeInference",
                          "GetSuggestedManipulation", "HabanaKernel"})
        if (!dlsym(ours, n)) { fprintf(stderr, "missing entry point %s\n", n); bad++; }
    auto lv = (pfnGetLibVersion)dlsym(ours, "GetLibVersion");
    auto lvs = (pfnGetLibVersion)dlsym(stock, "GetLibVersion");
    if (lv && lvs && lv() != lvs()) { fprintf(stderr, "GetLibVersion differs from stock\n"); bad++; }
    printf("%s: %zu GUIDs (stock %zu + %s), %s\n", ver ? ver : "?", go.size(), gs.size(), list.c_str(),
           bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
