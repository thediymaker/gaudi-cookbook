// SPDX-License-Identifier: Apache-2.0
// torch.ops.custom_op.glm_mhc_sinkhorn for lazy mode (legacy hpu_custom_op.h API), linked into libmxk_op_lazy.so by
// kernels/mxfp4-moe/build.sh --extra-kernel.
//   glm_mhc_sinkhorn(c [16, N] f32, iters, eps) -> f32 [16, N]
//   c: one 4x4 matrix per column (element i*4 + j), softmax + eps applied;
//   out: c /= colsum(c) + eps, then iters - 1 times { c /= rowsum(c) + eps; c /= colsum(c) + eps }
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <mutex>

namespace {
using namespace habana::custom_op;
struct GlmMhcSinkParams { int32_t iters; float eps; };

at::Tensor glm_mhc_sinkhorn(const at::Tensor& c, int64_t iters, double eps) {
    static std::once_flag f;
    std::call_once(f, []() {
        std::vector<InputDesc> in{InputDesc{input_type::TENSOR, 0}, InputDesc{input_type::SCALAR, 1},
                                  InputDesc{input_type::SCALAR, 2}};
        std::vector<OutputDesc> out{OutputDesc{0, c10::ScalarType::Float,
            [](const at::Stack& st) { return st[0].toTensor().sizes().vec(); }}};
        auto pf = [](const at::Stack& st, size_t& size) {
            HPU_PARAMS_STUB(GlmMhcSinkParams);
            params->iters = (int32_t)st[1].toInt();
            params->eps = (float)st[2].toDouble();
            return params;
        };
        REGISTER_CUSTOM_OP_ATTRIBUTES("custom_op::glm_mhc_sinkhorn", "glm_mhc_sinkhorn", in, out, pf);
    });
    auto d = HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::glm_mhc_sinkhorn");
    std::vector<c10::IValue> in{c, iters, eps};
    return d.execute(in)[0];
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("glm_mhc_sinkhorn(Tensor c, int iters, float eps) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("glm_mhc_sinkhorn", glm_mhc_sinkhorn);
}
