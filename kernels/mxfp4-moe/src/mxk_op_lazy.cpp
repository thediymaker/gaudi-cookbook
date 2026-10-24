// SPDX-License-Identifier: Apache-2.0
//
// libmxk_op_lazy.so: PyTorch ops torch.ops.custom_op.mxk_w13xib / mxk_w2xb for Gaudi lazy mode
// (PT_HPU_LAZY_MODE=1). Uses the legacy hpu_custom_op.h API from the installed habana-torch-plugin (not part of this
// repository): the PT2 custom-op API fails in lazy mode when the op's input is produced by a native op in the same
// lazy graph. The kernels themselves live in libmxk_tpc.so (mxk_tpc_lib.cpp), which must be the graph compiler's
// GC_KERNEL_PATH. Build against the torch and habana-torch-plugin of the serving image (build.sh).
//
//   mxk_w13xib(xp [T, Kp] bf16 (zero-padded to Kp = K rounded up to 512, permuted per 512-k chunk), ids [Q] i32,
//              w [E, 2I, K/2] u8 interleaved gate/up, s [E, 2I, K/32] u8, b [E, 4, Ip/2] f32 (kernel-order bias,
//              mxfp4_tpc_moe_rt.bias13), topk, e_min, alpha, limit, beta, ascale)
//              -> ap bf16 [Q, Ip]   (Ip = I rounded up to 512; the SwiGLU-OAI activation, bias first)
//   mxk_w2xb  (ap, ids, tw [Q] bf16, w [E, K, I/2] u8, s [E, K, I/32] u8, b [E, 2, K/2] f32 (even / odd rows,
//              mxfp4_tpc_moe_rt.bias2), topk, e_min)
//              -> bf16 [Q/topk, K]: sum over the token's local routes of tw * (a . W2 + b)
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <mutex>

namespace {
using namespace habana::custom_op;
struct MoeV7Params { int32_t topk; int32_t e_min; float alpha; float limit; float beta; float ascale; };

int64_t r512(int64_t n) { return (n + 511) / 512 * 512; }

at::Tensor w13xib(const at::Tensor& xp, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s,
                  const at::Tensor& b, int64_t topk, int64_t e_min, double alpha, double limit, double beta,
                  double ascale) {
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<InputDesc> in{InputDesc{input_type::TENSOR, 0}, InputDesc{input_type::TENSOR, 1},
                                  InputDesc{input_type::TENSOR, 2}, InputDesc{input_type::TENSOR, 3},
                                  InputDesc{input_type::TENSOR, 4},
                                  InputDesc{input_type::SCALAR, 5}, InputDesc{input_type::SCALAR, 6},
                                  InputDesc{input_type::SCALAR, 7}, InputDesc{input_type::SCALAR, 8},
                                  InputDesc{input_type::SCALAR, 9}, InputDesc{input_type::SCALAR, 10}};
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16,
            [](const at::Stack& st) {
                return std::vector<int64_t>{st[1].toTensor().size(0), r512(st[2].toTensor().size(1) / 2)}; }}};
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(MoeV7Params);
            params->topk = (int32_t)st[5].toInt();
            params->e_min = (int32_t)st[6].toInt();
            params->alpha = (float)st[7].toDouble();
            params->limit = (float)st[8].toDouble();
            params->beta = (float)st[9].toDouble();
            params->ascale = (float)st[10].toDouble();
            return params;
        };
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w13xib", "mxk_w13xib", in, out, pf);
    });
    auto d = HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::mxk_w13xib");
    std::vector<c10::IValue> in{xp, ids, w, s, b, topk, e_min, alpha, limit, beta, ascale};
    return d.execute(in)[0];
}

at::Tensor w2xb(const at::Tensor& ap, const at::Tensor& ids, const at::Tensor& tw, const at::Tensor& w,
                const at::Tensor& s, const at::Tensor& b, int64_t topk, int64_t e_min) {
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<InputDesc> in{InputDesc{input_type::TENSOR, 0}, InputDesc{input_type::TENSOR, 1},
                                  InputDesc{input_type::TENSOR, 2}, InputDesc{input_type::TENSOR, 3},
                                  InputDesc{input_type::TENSOR, 4}, InputDesc{input_type::TENSOR, 5},
                                  InputDesc{input_type::SCALAR, 6}, InputDesc{input_type::SCALAR, 7}};
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::BFloat16,
            [](const at::Stack& st) {
                return std::vector<int64_t>{st[1].toTensor().size(0) / st[6].toInt(), st[3].toTensor().size(1)}; }}};
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(MoeV7Params);
            params->topk = (int32_t)st[6].toInt();
            params->e_min = (int32_t)st[7].toInt();
            params->alpha = 1.f; params->limit = 3.0e38f; params->beta = 0.f; params->ascale = 1.f;
            return params;
        };
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::mxk_w2xb", "mxk_w2xb", in, out, pf);
    });
    auto d = HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::mxk_w2xb");
    std::vector<c10::IValue> in{ap, ids, tw, w, s, b, topk, e_min};
    return d.execute(in)[0];
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("mxk_w13xib(Tensor xp, Tensor ids, Tensor w, Tensor s, Tensor b, int topk, int e_min, float alpha, "
          "float limit, float beta, float ascale) -> Tensor");
    m.def("mxk_w2xb(Tensor ap, Tensor ids, Tensor tw, Tensor w, Tensor s, Tensor b, int topk, int e_min) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("mxk_w13xib", w13xib);
    m.impl("mxk_w2xb", w2xb);
}
