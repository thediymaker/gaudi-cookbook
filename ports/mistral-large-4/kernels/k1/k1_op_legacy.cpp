// Legacy lazy-mode CustomOp registration. Int args go to the kernel as scalar
// params (struct of 8 int32) and may size outputs.
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <mutex>
#include <map>
namespace {
using namespace habana::custom_op;
struct P8 { int32_t p[8]; };
using ShapeFn = std::function<std::vector<int64_t>(const at::Stack&)>;
struct OutSpec { c10::ScalarType dt; ShapeFn shape; };
// nt tensors first, then ns int scalars on the stack
static std::vector<at::Tensor> run(const std::string& schema, const std::string& guid, int nt, int ns,
                                   const std::vector<OutSpec>& outs, std::vector<c10::IValue> in) {
    static std::mutex mu;
    static std::map<std::string, bool> done;
    {
        std::lock_guard<std::mutex> g(mu);
        if (!done[schema]) {
            std::vector<InputDesc> ids;
            for (int i = 0; i < nt; i++) ids.push_back(InputDesc{input_type::TENSOR, i});
            for (int i = 0; i < ns; i++) ids.push_back(InputDesc{input_type::SCALAR, nt + i});
            std::vector<OutputDesc> ods;
            for (size_t o = 0; o < outs.size(); o++) ods.push_back(OutputDesc{(unsigned)o, outs[o].dt, outs[o].shape});
            allocate_user_params_func pf = nullptr;
            if (ns > 0) pf = [nt, ns](const at::Stack& s, size_t& size) {
                HPU_PARAMS_STUB(P8);
                for (int i = 0; i < 8; i++) params->p[i] = i < ns ? (int32_t)s[nt + i].toInt() : 0;
                return params;
            };
            REGISTER_CUSTOM_OP_ATTRIBUTES(schema, guid, ids, ods, pf);
            done[schema] = true;
        }
    }
    auto d = HabanaCustomOpDescriptor::getCustomOpDescriptor(schema);
    return d.execute(in);
}
#define T(i) s[i].toTensor()
#define I(i) s[i].toInt()
#include "k1_ops.inc"
}  // namespace
