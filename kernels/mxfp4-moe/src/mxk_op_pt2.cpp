// SPDX-License-Identifier: Apache-2.0
//
// libmxk_op_pt2.so: the PyTorch ops of one build variant for torch.compile mode (PT_HPU_LAZY_MODE=0), built by
// `build.sh --pt2`. Same op names and schemas as libmxk_op_lazy.so (see mxk_op_lazy.cpp for the ops and their
// layouts), on the PT2 custom-op API of the installed habana-torch-plugin (hpu_custom_op_pt2.h, not part of this
// repository), with Meta implementations so torch.compile can trace them. build.sh compiles this file with
// -DMXK_HAS_<guid> for every kernel of the variant, so the library registers exactly the ops that the variant's
// libmxk_tpc.so can run. A process loads one op library: this one in compile mode, libmxk_op_lazy.so in lazy mode
// (the PT2 API fails in lazy mode when the op's input comes from a native op in the same lazy graph).
//
// The PT2 API registers an op's kernel with the bridge on the op's first eager call, and a compiled graph looks it
// up there without making that call: call each op once eagerly per process before the first compiled graph that
// uses it ("custom_op::mxk_w13x isn't registered in KernelRegistry" otherwise).
//
// The gpt-oss ops (mxk_w13xib, mxk_w2xb) are not built here; gpt-oss serves in lazy mode.
#include "hpu_custom_op_pt2.h"
#include <torch/extension.h>
#include <mutex>

namespace {
struct MoeParams { int32_t topk; int32_t e_min; float alpha; float limit; float beta; float ascale; };

int64_t r512(int64_t n) { return (n + 511) / 512 * 512; }

at::Tensor run(const char* schema, std::vector<c10::IValue> in) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return op.execute(in)[0];
}

#if defined(MXK_HAS_mxk_w13x) || defined(MXK_HAS_mxk_w13xi) || defined(MXK_HAS_mxk_w13xt)
// ap [Q, Ip]: Q from ids (input 1), Ip from w (input 2, [E, 2I, K/2]); params from the scalars after 4 tensors
void reg_w13(const char* schema, const char* guid) {
    auto meta = [](const at::Stack& st) -> habana::PartialOutputMetaDataVector {
        return {habana::PartialOutputMetaData{c10::ScalarType::BFloat16,
                {st[1].toTensor().size(0), r512(st[2].toTensor().size(1) / 2)}}};
    };
    auto pf = [](const at::Stack& st, size_t& size) {
        HPU_PARAMS_STUB(MoeParams);
        *params = MoeParams{(int32_t)st[4].toInt(), (int32_t)st[5].toInt(), (float)st[6].toDouble(),
                            (float)st[7].toDouble(), (float)st[8].toDouble(), (float)st[9].toDouble()};
        return params;
    };
    habana::custom_op::registerUserCustomOp(schema, guid, meta, pf);
}

at::Tensor w13x(const at::Tensor& xp, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, int64_t topk,
                int64_t e_min, double alpha, double limit, double beta, double ascale, bool interleaved) {
    std::vector<c10::IValue> in{xp, ids, w, s, topk, e_min, alpha, limit, beta, ascale, interleaved};
    if (interleaved) {
#ifdef MXK_HAS_mxk_w13xi
        static std::once_flag f;
        std::call_once(f, []() { reg_w13("custom_op::mxk_w13xi", "mxk_w13xi"); });
        return run("custom_op::mxk_w13xi", in);
#else
        TORCH_CHECK(false, "mxk_w13x: this build has no interleaved w13 kernel (mxk_w13xi)");
#endif
    }
#if defined(MXK_HAS_mxk_w13x)
    static std::once_flag f;
    std::call_once(f, []() { reg_w13("custom_op::mxk_w13x", "mxk_w13x"); });
    return run("custom_op::mxk_w13x", in);
#elif defined(MXK_HAS_mxk_w13xt)
    static std::once_flag f;
    std::call_once(f, []() { reg_w13("custom_op::mxk_w13xt", "mxk_w13xt"); });
    return run("custom_op::mxk_w13xt", in);
#else
    TORCH_CHECK(false, "mxk_w13x: this build has only the interleaved w13 kernel (interleaved=True)");
#endif
}

at::Tensor w13x_meta(const at::Tensor& xp, const at::Tensor& ids, const at::Tensor& w, const at::Tensor&, int64_t,
                     int64_t, double, double, double, double, bool) {
    return at::empty({ids.size(0), r512(w.size(1) / 2)}, xp.options());
}
#endif

#if defined(MXK_HAS_mxk_w2x) || defined(MXK_HAS_mxk_w2xr)
// w2 params: topk, e_min after T tensors; the activation fields are unused
template <int T, typename Meta>
void reg_w2(const char* schema, const char* guid, Meta meta) {
    auto pf = [](const at::Stack& st, size_t& size) {
        HPU_PARAMS_STUB(MoeParams);
        *params = MoeParams{(int32_t)st[T].toInt(), (int32_t)st[T + 1].toInt(), 1.f, 3.0e38f, 0.f, 1.f};
        return params;
    };
    habana::custom_op::registerUserCustomOp(schema, guid, meta, pf);
}
#endif

#ifdef MXK_HAS_mxk_w2x
at::Tensor w2x(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& tw, const at::Tensor& w,
               const at::Tensor& s, int64_t topk, int64_t e_min) {
    static std::once_flag f;
    std::call_once(f, []() {
        reg_w2<5>("custom_op::mxk_w2x", "mxk_w2x", [](const at::Stack& st) -> habana::PartialOutputMetaDataVector {
            return {habana::PartialOutputMetaData{c10::ScalarType::BFloat16,
                    {st[1].toTensor().size(0) / st[5].toInt(), st[3].toTensor().size(1)}}};
        });
    });
    return run("custom_op::mxk_w2x", {ap, ids, tw, w, s, topk, e_min});
}

at::Tensor w2x_meta(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor&, const at::Tensor& w,
                    const at::Tensor&, int64_t topk, int64_t) {
    return at::empty({ids.size(0) / topk, w.size(1)}, ap.options());
}
#endif

#ifdef MXK_HAS_mxk_w2xr
at::Tensor w2xr(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, int64_t topk,
                int64_t e_min) {
    static std::once_flag f;
    std::call_once(f, []() {
        reg_w2<4>("custom_op::mxk_w2xr", "mxk_w2xr", [](const at::Stack& st) -> habana::PartialOutputMetaDataVector {
            return {habana::PartialOutputMetaData{c10::ScalarType::BFloat16,
                    {st[1].toTensor().size(0), st[2].toTensor().size(1)}}};
        });
    });
    return run("custom_op::mxk_w2xr", {ap, ids, w, s, topk, e_min});
}

at::Tensor w2xr_meta(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& w, const at::Tensor&, int64_t,
                     int64_t) {
    return at::empty({ids.size(0), w.size(1)}, ap.options());
}
#endif

#ifdef MXK_HAS_mxk_deq
at::Tensor deq(const at::Tensor& w, const at::Tensor& s, int64_t rowperm) {
    TORCH_CHECK(rowperm == 0, "mxk_deq: only rowperm=0 (natural layout) is built");
    static std::once_flag f;
    std::call_once(f, []() {
        auto meta = [](const at::Stack& st) -> habana::PartialOutputMetaDataVector {
            auto sz = st[0].toTensor().sizes().vec();
            sz.back() *= 2;
            return {habana::PartialOutputMetaData{c10::ScalarType::BFloat16, sz}};
        };
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(int32_t);
            *params = (int32_t)st[2].toInt();
            return params;
        };
        habana::custom_op::registerUserCustomOp("custom_op::mxk_deq", "mxk_deq", meta, pf);
    });
    return run("custom_op::mxk_deq", {w, s, rowperm});
}

at::Tensor deq_meta(const at::Tensor& w, const at::Tensor&, int64_t) {
    auto sz = w.sizes().vec();
    sz.back() *= 2;
    return at::empty(sz, w.options().dtype(at::kBFloat16));
}
#endif
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
#if defined(MXK_HAS_mxk_w13x) || defined(MXK_HAS_mxk_w13xi) || defined(MXK_HAS_mxk_w13xt)
    m.def("mxk_w13x(Tensor xp, Tensor ids, Tensor w, Tensor s, int topk, int e_min, float alpha, float limit, "
          "float beta, float ascale, bool interleaved) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w13xi
    // schema of the interleaved GUID's descriptor (no impl: called through mxk_w13x(..., interleaved=True))
    m.def("mxk_w13xi(Tensor xp, Tensor ids, Tensor w, Tensor s, int topk, int e_min, float alpha, float limit, "
          "float beta, float ascale, bool interleaved) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w13xt
    // schema of the long-row GUID's descriptor (no impl: called through mxk_w13x(..., interleaved=False))
    m.def("mxk_w13xt(Tensor xp, Tensor ids, Tensor w, Tensor s, int topk, int e_min, float alpha, float limit, "
          "float beta, float ascale, bool interleaved) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2x
    m.def("mxk_w2x(Tensor ap, Tensor ids, Tensor tw, Tensor w, Tensor s, int topk, int e_min) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2xr
    m.def("mxk_w2xr(Tensor ap, Tensor ids, Tensor w, Tensor s, int topk, int e_min) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_deq
    m.def("mxk_deq(Tensor w, Tensor s, int rowperm) -> Tensor");
#endif
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
#if defined(MXK_HAS_mxk_w13x) || defined(MXK_HAS_mxk_w13xi) || defined(MXK_HAS_mxk_w13xt)
    m.impl("mxk_w13x", w13x);
#endif
#ifdef MXK_HAS_mxk_w2x
    m.impl("mxk_w2x", w2x);
#endif
#ifdef MXK_HAS_mxk_w2xr
    m.impl("mxk_w2xr", w2xr);
#endif
#ifdef MXK_HAS_mxk_deq
    m.impl("mxk_deq", deq);
#endif
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
#if defined(MXK_HAS_mxk_w13x) || defined(MXK_HAS_mxk_w13xi) || defined(MXK_HAS_mxk_w13xt)
    m.impl("mxk_w13x", w13x_meta);
#endif
#ifdef MXK_HAS_mxk_w2x
    m.impl("mxk_w2x", w2x_meta);
#endif
#ifdef MXK_HAS_mxk_w2xr
    m.impl("mxk_w2xr", w2xr_meta);
#endif
#ifdef MXK_HAS_mxk_deq
    m.impl("mxk_deq", deq_meta);
#endif
}
