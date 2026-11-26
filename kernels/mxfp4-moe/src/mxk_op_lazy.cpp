// SPDX-License-Identifier: Apache-2.0
//
// libmxk_op_lazy.so: the PyTorch ops of one build variant for Gaudi lazy mode (PT_HPU_LAZY_MODE=1). build.sh
// compiles this file with -DMXK_HAS_<guid> for every kernel of the variant, so the library registers exactly the
// ops that the variant's libmxk_tpc.so can run. Uses the legacy hpu_custom_op.h API from the installed
// habana-torch-plugin (not part of this repository): the PT2 custom-op API fails in lazy mode when the op's input
// is produced by a native op in the same lazy graph. The kernels themselves live in libmxk_tpc.so, which must be
// the graph compiler's GC_KERNEL_PATH. Build against the torch and habana-torch-plugin of the serving image.
//
// Ops (torch.ops.custom_op.*; Ip = I rounded up to 512, Q = T * topk; x is permuted per 512-k chunk, see README):
//   mxk_w13x  (xp [T, K] bf16, ids [Q] i32, w [E, 2I, K/2] u8, s [E, 2I, K/32] u8, topk, e_min, alpha, limit,
//              beta, ascale, interleaved) -> ap bf16 [Q, Ip]
//              interleaved=False runs GUID mxk_w13x ([gate; up] rows; mxk_w13xt in the kimi build, K > 6144),
//              True runs GUID mxk_w13xi (interleaved rows)
//   mxk_w2x   (ap, ids, tw [Q] bf16, w [E, K, I/2] u8, s [E, K, I/32] u8, topk, e_min) -> bf16 [Q/topk, K]
//   mxk_w2xr  (ap, ids, w, s, topk, e_min) -> bf16 [Q, K], one row per route, zeros for routes that are not local
//   mxk_w13xib(xp [T, Kp] bf16 zero-padded, ids, w (interleaved), s, b [E, 4, Ip/2] f32, topk, e_min, alpha, limit,
//              beta, ascale) -> ap bf16 [Q, Ip]                                        (gpt-oss, bias13 layout)
//   mxk_w2xb  (ap, ids, tw, w, s, b [E, 2, K/2] f32, topk, e_min) -> bf16 [Q/topk, K]  (gpt-oss, bias2 layout)
//   mxk_deq   (w [E, N, K/2] or [N, K/2] u8, s [.., N, K/32] u8, rowperm=0) -> bf16 [.., N, K]
// v10 ops (the glm, kimi, dsv41 and gptoss builds):
//   mxk_w13xn (x [T, K] bf16 in NATURAL order, then as mxk_w13x; interleaved=False) -> ap bf16 [Q, Ip]   (glm)
//   mxk_w13xk (as mxk_w13x, permuted x, K up to 7168; interleaved=False)            -> ap bf16 [Q, Ip]   (kimi)
//   mxk_w13xin (x in NATURAL order, then as mxk_w13x; interleaved=True)             -> ap bf16 [Q, Ip]   (dsv41)
//   mxk_w13xibn (x [T, K] in NATURAL order, unpadded, then as mxk_w13xib)           -> ap bf16 [Q, Ip]   (gptoss)
//   mxk_w2gn  (as mxk_w2gp, nch 4 / 6)                                              -> part f32 [Q, K]
//   mxk_w2cb  (part, ids [T, topk], tw [T, topk], b [E, 2, K/2] f32, e_min, e_local) -> bf16 [T, K]       (gptoss)
//   mxk_w2xc  (as mxk_w2x)                                                           -> bf16 [Q/topk, K]
//   mxk_w2gp  (ap, ids [T, topk] i32, w, s, e_min, gmax)                             -> part f32 [Q, K]
//   mxk_w2c   (part, ids [T, topk] i32, tw [T, topk] bf16, e_min, e_local)           -> bf16 [T, K]
// grouped-route ops of the ml4 build (with mxk_w2c above):
//   mxk_w13g  (xp [T, K] bf16 permuted, ids [T, topk] i32, w, s, e_min, gmax, alpha, limit, beta, ascale)
//                                                                                    -> ap bf16 [Q, Ip]
//   mxk_w2g   (ap, ids [T, topk] i32, w, s, e_min, gmax)                             -> part f32 [Q, K]
//   mxk_w13g gives mxk_w13x's output bit for bit, mxk_w2g + mxk_w2c mxk_w2x's. ids is 2-D in both, so neither shares
//   a lazy input signature with a v8 op.
//   mxk_w2gp / mxk_w2gn + mxk_w2c give mxk_w2x's output bit for bit, mxk_w2gn + mxk_w2cb mxk_w2xb's; each v10 w13 op
//   gives its v8 twin's (mxk_w13x / mxk_w13xi / mxk_w13xib), mxk_w2xc mxk_w2x's. The v10 w13 ops can have their
//   v8 twin's input signature (mxk_w2xc has mxk_w2x's): a process calls the v10 ops or the v8 ops for a given
//   weight shape, never both (see the lazy-mode note below).
//
// Lazy-mode legacy custom ops are cached by input signature, not by GUID: a second op with the same input shapes in
// one process returns the first op's result. Within each variant the ops differ in tensor count or shapes, and each
// variant's library holds only its own ops (build.sh checks that), so one process never mixes variants.
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <mutex>

namespace {
using namespace habana::custom_op;
struct MoeParams { int32_t topk; int32_t e_min; float alpha; float limit; float beta; float ascale; };

int64_t r512(int64_t n) { return (n + 511) / 512 * 512; }

std::vector<InputDesc> inputs(int ntensor, int nscalar) {
    std::vector<InputDesc> in;
    for (int i = 0; i < ntensor; i++) in.push_back(InputDesc{input_type::TENSOR, (unsigned)i});
    for (int i = 0; i < nscalar; i++) in.push_back(InputDesc{input_type::SCALAR, (unsigned)(ntensor + i)});
    return in;
}

at::Tensor run(const char* op, std::vector<c10::IValue> in) {
    auto d = HabanaCustomOpDescriptor::getCustomOpDescriptor(op);
    return d.execute(in)[0];
}

// Node params of the MoE kernels: the scalars that follow the T tensor inputs (w13: topk, e_min, alpha, limit,
// beta, ascale; w2: topk, e_min).
#define W13_PARAMS(T) [](const at::Stack& st, size_t& size) {                                              \
        HPU_PARAMS_STUB(MoeParams);                                                                         \
        *params = MoeParams{(int32_t)st[T].toInt(), (int32_t)st[T + 1].toInt(), (float)st[T + 2].toDouble(), \
                            (float)st[T + 3].toDouble(), (float)st[T + 4].toDouble(), (float)st[T + 5].toDouble()}; \
        return params; }
#define W2_PARAMS(T) [](const at::Stack& st, size_t& size) {                                               \
        HPU_PARAMS_STUB(MoeParams);                                                                         \
        *params = MoeParams{(int32_t)st[T].toInt(), (int32_t)st[T + 1].toInt(), 1.f, 3.0e38f, 0.f, 1.f};    \
        return params; }

// ap [Q, Ip]: Q from ids (input 1), Ip from w (input 2, [E, 2I, K/2])
std::vector<OutputDesc> w13Out() {
    return {OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
        return std::vector<int64_t>{st[1].toTensor().size(0), r512(st[2].toTensor().size(1) / 2)}; }}};
}

#if defined(MXK_HAS_mxk_w13x) || defined(MXK_HAS_mxk_w13xi) || defined(MXK_HAS_mxk_w13xt)
at::Tensor w13x(const at::Tensor& xp, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, int64_t topk,
                int64_t e_min, double alpha, double limit, double beta, double ascale, bool interleaved) {
    // the stack must match the registered schema (the legacy bridge looks the op up by name for alias analysis)
    std::vector<c10::IValue> in{xp, ids, w, s, topk, e_min, alpha, limit, beta, ascale, interleaved};
    if (interleaved) {
#ifdef MXK_HAS_mxk_w13xi
        static std::once_flag f;
        std::call_once(f, []() {
            auto desc = inputs(4, 7);  // the bool `interleaved` is the 7th scalar
            auto outs = w13Out();
            auto pf = W13_PARAMS(4);
            REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w13xi", "mxk_w13xi", desc, outs, pf);
        });
        return run("custom_op::mxk_w13xi", in);
#else
        TORCH_CHECK(false, "mxk_w13x: this build has no interleaved w13 kernel (mxk_w13xi)");
#endif
    }
#if defined(MXK_HAS_mxk_w13x)
    static std::once_flag f;
    std::call_once(f, []() {
        auto desc = inputs(4, 7);  // the bool `interleaved` is the 7th scalar
        auto outs = w13Out();
        auto pf = W13_PARAMS(4);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w13x", "mxk_w13x", desc, outs, pf);
    });
    return run("custom_op::mxk_w13x", in);
#elif defined(MXK_HAS_mxk_w13xt)
    static std::once_flag f;
    std::call_once(f, []() {
        auto desc = inputs(4, 7);  // the bool `interleaved` is the 7th scalar
        auto outs = w13Out();
        auto pf = W13_PARAMS(4);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w13xt", "mxk_w13xt", desc, outs, pf);
    });
    return run("custom_op::mxk_w13xt", in);
#else
    TORCH_CHECK(false, "mxk_w13x: this build has only the interleaved w13 kernel (interleaved=True)");
#endif
}
#endif

#ifdef MXK_HAS_mxk_w2x
at::Tensor w2x(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& tw, const at::Tensor& w,
               const at::Tensor& s, int64_t topk, int64_t e_min) {
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            return std::vector<int64_t>{st[1].toTensor().size(0) / st[5].toInt(), st[3].toTensor().size(1)}; }}};
        auto desc = inputs(5, 2);
        auto pf = W2_PARAMS(5);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w2x", "mxk_w2x", desc, out, pf);
    });
    return run("custom_op::mxk_w2x", {ap, ids, tw, w, s, topk, e_min});
}
#endif

#ifdef MXK_HAS_mxk_w2xr
at::Tensor w2xr(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, int64_t topk,
                int64_t e_min) {
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            return std::vector<int64_t>{st[1].toTensor().size(0), st[2].toTensor().size(1)}; }}};
        auto desc = inputs(4, 2);
        auto pf = W2_PARAMS(4);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w2xr", "mxk_w2xr", desc, out, pf);
    });
    return run("custom_op::mxk_w2xr", {ap, ids, w, s, topk, e_min});
}
#endif

#ifdef MXK_HAS_mxk_w13xib
at::Tensor w13xib(const at::Tensor& xp, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s,
                  const at::Tensor& b, int64_t topk, int64_t e_min, double alpha, double limit, double beta,
                  double ascale) {
    static std::once_flag f;
    std::call_once(f, []() {
        auto desc = inputs(5, 6);
        auto outs = w13Out();
        auto pf = W13_PARAMS(5);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w13xib", "mxk_w13xib", desc, outs, pf);
    });
    return run("custom_op::mxk_w13xib", {xp, ids, w, s, b, topk, e_min, alpha, limit, beta, ascale});
}
#endif

#ifdef MXK_HAS_mxk_w2xb
at::Tensor w2xb(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& tw, const at::Tensor& w,
                const at::Tensor& s, const at::Tensor& b, int64_t topk, int64_t e_min) {
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            return std::vector<int64_t>{st[1].toTensor().size(0) / st[6].toInt(), st[3].toTensor().size(1)}; }}};
        auto desc = inputs(6, 2);
        auto pf = W2_PARAMS(6);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w2xb", "mxk_w2xb", desc, out, pf);
    });
    return run("custom_op::mxk_w2xb", {ap, ids, tw, w, s, b, topk, e_min});
}
#endif

#ifdef MXK_HAS_mxk_deq
at::Tensor deq(const at::Tensor& w, const at::Tensor& s, int64_t rowperm) {
    TORCH_CHECK(rowperm == 0, "mxk_deq: only rowperm=0 (natural layout) is built");
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            auto sz = st[0].toTensor().sizes().vec();
            sz.back() *= 2;
            return sz; }}};
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(int32_t);
            *params = (int32_t)st[2].toInt();
            return params;
        };
        auto desc = inputs(2, 1);
                REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_deq", "mxk_deq", desc, out, pf);
    });
    return run("custom_op::mxk_deq", {w, s, rowperm});
}
#endif
#if defined(MXK_HAS_mxk_w13xn) || defined(MXK_HAS_mxk_w13xk) || defined(MXK_HAS_mxk_w13xin)
// one GUID per op, mxk_w13x's schema; each kernel is built for one row layout (IL)
#define MXK_W13_V10(FN, GUID, IL)                                                                                \
    at::Tensor FN(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, int64_t topk, \
                  int64_t e_min, double alpha, double limit, double beta, double ascale, bool interleaved) {    \
        TORCH_CHECK(interleaved == IL, GUID ": built for interleaved=" #IL " only");                            \
        static std::once_flag f;                                                                                \
        std::call_once(f, []() {                                                                                \
            auto desc = inputs(4, 7);                                                                           \
            auto outs = w13Out();                                                                               \
            auto pf = W13_PARAMS(4);                                                                            \
            REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::" GUID, GUID, desc, outs, pf);                           \
        });                                                                                                     \
        return run("custom_op::" GUID, {x, ids, w, s, topk, e_min, alpha, limit, beta, ascale, interleaved});   \
    }
#endif
#ifdef MXK_HAS_mxk_w13xn
MXK_W13_V10(w13xn, "mxk_w13xn", false)
#endif
#ifdef MXK_HAS_mxk_w13xk
MXK_W13_V10(w13xk, "mxk_w13xk", false)
#endif
#ifdef MXK_HAS_mxk_w13xin
MXK_W13_V10(w13xin, "mxk_w13xin", true)
#endif

#ifdef MXK_HAS_mxk_w13xibn
at::Tensor w13xibn(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s,
                   const at::Tensor& b, int64_t topk, int64_t e_min, double alpha, double limit, double beta,
                   double ascale) {
    static std::once_flag f;
    std::call_once(f, []() {
        auto desc = inputs(5, 6);
        auto outs = w13Out();
        auto pf = W13_PARAMS(5);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w13xibn", "mxk_w13xibn", desc, outs, pf);
    });
    return run("custom_op::mxk_w13xibn", {x, ids, w, s, b, topk, e_min, alpha, limit, beta, ascale});
}
#endif

#ifdef MXK_HAS_mxk_w2xc
at::Tensor w2xc(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& tw, const at::Tensor& w,
                const at::Tensor& s, int64_t topk, int64_t e_min) {
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            return std::vector<int64_t>{st[1].toTensor().size(0) / st[5].toInt(), st[3].toTensor().size(1)}; }}};
        auto desc = inputs(5, 2);
        auto pf = W2_PARAMS(5);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w2xc", "mxk_w2xc", desc, out, pf);
    });
    return run("custom_op::mxk_w2xc", {ap, ids, tw, w, s, topk, e_min});
}
#endif

#if defined(MXK_HAS_mxk_w2gp) || defined(MXK_HAS_mxk_w2gn) || defined(MXK_HAS_mxk_w2g)
struct GroupParams { int32_t e_min; int32_t gmax; };
#define MXK_W2G(FN, GUID)                                                                                        \
    at::Tensor FN(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s,         \
                  int64_t e_min, int64_t gmax) {                                                                \
        TORCH_CHECK(ids.dim() == 2, GUID ": ids must be [T, topk]");                                            \
        static std::once_flag f;                                                                                \
        std::call_once(f, []() {                                                                                \
            std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::Float, [](const at::Stack& st) {         \
                return std::vector<int64_t>{st[0].toTensor().size(0), st[2].toTensor().size(1)}; }}};           \
            auto pf = [](const at::Stack& st, size_t& size) {                                                   \
                HPU_PARAMS_STUB(GroupParams);                                                                   \
                *params = GroupParams{(int32_t)st[4].toInt(), (int32_t)st[5].toInt()};                          \
                return params;                                                                                  \
            };                                                                                                  \
            auto desc = inputs(4, 2);                                                                           \
            REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::" GUID, GUID, desc, out, pf);                            \
        });                                                                                                     \
        return run("custom_op::" GUID, {ap, ids, w, s, e_min, gmax});                                          \
    }
#endif
#ifdef MXK_HAS_mxk_w2gp
MXK_W2G(w2gp, "mxk_w2gp")
#endif
#ifdef MXK_HAS_mxk_w2g
MXK_W2G(w2g, "mxk_w2g")
#endif

#ifdef MXK_HAS_mxk_w13g
struct W13gParams { int32_t e_min; int32_t gmax; float alpha; float limit; float beta; float ascale; };
at::Tensor w13g(const at::Tensor& xp, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, int64_t e_min,
                int64_t gmax, double alpha, double limit, double beta, double ascale) {
    TORCH_CHECK(ids.dim() == 2, "mxk_w13g: ids must be [T, topk]");
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            auto i = st[1].toTensor();
            return std::vector<int64_t>{i.size(0) * i.size(1), r512(st[2].toTensor().size(1) / 2)}; }}};
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(W13gParams);
            *params = W13gParams{(int32_t)st[4].toInt(), (int32_t)st[5].toInt(), (float)st[6].toDouble(),
                                 (float)st[7].toDouble(), (float)st[8].toDouble(), (float)st[9].toDouble()};
            return params;
        };
        auto desc = inputs(4, 6);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w13g", "mxk_w13g", desc, out, pf);
    });
    return run("custom_op::mxk_w13g", {xp, ids, w, s, e_min, gmax, alpha, limit, beta, ascale});
}
#endif
#ifdef MXK_HAS_mxk_w2gn
MXK_W2G(w2gn, "mxk_w2gn")
#endif

#if defined(MXK_HAS_mxk_w2c) || defined(MXK_HAS_mxk_w2cb)
struct CombineParams { int32_t e_min; int32_t e_local; };
#endif
#ifdef MXK_HAS_mxk_w2c
at::Tensor w2c(const at::Tensor& part, const at::Tensor& ids, const at::Tensor& tw, int64_t e_min, int64_t e_local) {
    TORCH_CHECK(ids.dim() == 2 && tw.dim() == 2, "mxk_w2c: ids and tw must be [T, topk]");
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            return std::vector<int64_t>{st[1].toTensor().size(0), st[0].toTensor().size(1)}; }}};
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(CombineParams);
            *params = CombineParams{(int32_t)st[3].toInt(), (int32_t)st[4].toInt()};
            return params;
        };
        auto desc = inputs(3, 2);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w2c", "mxk_w2c", desc, out, pf);
    });
    return run("custom_op::mxk_w2c", {part, ids, tw, e_min, e_local});
}
#endif

#ifdef MXK_HAS_mxk_w2cb
at::Tensor w2cb(const at::Tensor& part, const at::Tensor& ids, const at::Tensor& tw, const at::Tensor& b, int64_t e_min,
                int64_t e_local) {
    TORCH_CHECK(ids.dim() == 2 && tw.dim() == 2, "mxk_w2cb: ids and tw must be [T, topk]");
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16, [](const at::Stack& st) {
            return std::vector<int64_t>{st[1].toTensor().size(0), st[0].toTensor().size(1)}; }}};
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(CombineParams);
            *params = CombineParams{(int32_t)st[4].toInt(), (int32_t)st[5].toInt()};
            return params;
        };
        auto desc = inputs(4, 2);
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w2cb", "mxk_w2cb", desc, out, pf);
    });
    return run("custom_op::mxk_w2cb", {part, ids, tw, b, e_min, e_local});
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
#ifdef MXK_HAS_mxk_w13xib
    m.def("mxk_w13xib(Tensor xp, Tensor ids, Tensor w, Tensor s, Tensor b, int topk, int e_min, float alpha, "
          "float limit, float beta, float ascale) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2xb
    m.def("mxk_w2xb(Tensor ap, Tensor ids, Tensor tw, Tensor w, Tensor s, Tensor b, int topk, int e_min) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_deq
    m.def("mxk_deq(Tensor w, Tensor s, int rowperm) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w13xn
    m.def("mxk_w13xn(Tensor x, Tensor ids, Tensor w, Tensor s, int topk, int e_min, float alpha, float limit, "
          "float beta, float ascale, bool interleaved) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w13xk
    m.def("mxk_w13xk(Tensor xp, Tensor ids, Tensor w, Tensor s, int topk, int e_min, float alpha, float limit, "
          "float beta, float ascale, bool interleaved) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2xc
    m.def("mxk_w2xc(Tensor ap, Tensor ids, Tensor tw, Tensor w, Tensor s, int topk, int e_min) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2gp
    m.def("mxk_w2gp(Tensor ap, Tensor ids, Tensor w, Tensor s, int e_min, int gmax) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w13g
    m.def("mxk_w13g(Tensor xp, Tensor ids, Tensor w, Tensor s, int e_min, int gmax, float alpha, float limit, "
          "float beta, float ascale) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2g
    m.def("mxk_w2g(Tensor ap, Tensor ids, Tensor w, Tensor s, int e_min, int gmax) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2c
    m.def("mxk_w2c(Tensor part, Tensor ids, Tensor tw, int e_min, int e_local) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w13xin
    m.def("mxk_w13xin(Tensor x, Tensor ids, Tensor w, Tensor s, int topk, int e_min, float alpha, float limit, "
          "float beta, float ascale, bool interleaved) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w13xibn
    m.def("mxk_w13xibn(Tensor x, Tensor ids, Tensor w, Tensor s, Tensor b, int topk, int e_min, float alpha, "
          "float limit, float beta, float ascale) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2gn
    m.def("mxk_w2gn(Tensor ap, Tensor ids, Tensor w, Tensor s, int e_min, int gmax) -> Tensor");
#endif
#ifdef MXK_HAS_mxk_w2cb
    m.def("mxk_w2cb(Tensor part, Tensor ids, Tensor tw, Tensor b, int e_min, int e_local) -> Tensor");
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
#ifdef MXK_HAS_mxk_w13xib
    m.impl("mxk_w13xib", w13xib);
#endif
#ifdef MXK_HAS_mxk_w2xb
    m.impl("mxk_w2xb", w2xb);
#endif
#ifdef MXK_HAS_mxk_deq
    m.impl("mxk_deq", deq);
#endif
#ifdef MXK_HAS_mxk_w13xn
    m.impl("mxk_w13xn", w13xn);
#endif
#ifdef MXK_HAS_mxk_w13xk
    m.impl("mxk_w13xk", w13xk);
#endif
#ifdef MXK_HAS_mxk_w2xc
    m.impl("mxk_w2xc", w2xc);
#endif
#ifdef MXK_HAS_mxk_w2gp
    m.impl("mxk_w2gp", w2gp);
#endif
#ifdef MXK_HAS_mxk_w13g
    m.impl("mxk_w13g", w13g);
#endif
#ifdef MXK_HAS_mxk_w2g
    m.impl("mxk_w2g", w2g);
#endif
#ifdef MXK_HAS_mxk_w2c
    m.impl("mxk_w2c", w2c);
#endif
#ifdef MXK_HAS_mxk_w13xin
    m.impl("mxk_w13xin", w13xin);
#endif
#ifdef MXK_HAS_mxk_w13xibn
    m.impl("mxk_w13xibn", w13xibn);
#endif
#ifdef MXK_HAS_mxk_w2gn
    m.impl("mxk_w2gn", w2gn);
#endif
#ifdef MXK_HAS_mxk_w2cb
    m.impl("mxk_w2cb", w2cb);
#endif
}
