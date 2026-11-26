// r6 ops (lazy legacy custom ops). Inside the anonymous namespace on entry and on exit.
at::Tensor r6_ffk(const at::Tensor& zi, const at::Tensor& x, int64_t mode) {
    return run("custom_op::r6_ffk", "r6_ffk", 2, 1, {{at::kChar, SH(T(1).size(0), 132)}}, {zi, x, mode})[0]; }
at::Tensor r6_ffk_meta(const at::Tensor& zi, const at::Tensor& x, int64_t) { return at::empty({x.size(0), 132}, zi.options()); }
at::Tensor r6_ffkf(const at::Tensor& zi, const at::Tensor& x, int64_t mode) {
    return run("custom_op::r6_ffkf", "r6_ffkf", 2, 1, {{at::kChar, SH(T(1).size(0), 132)}}, {zi, x, mode})[0]; }
at::Tensor r6_ffkf_meta(const at::Tensor& zi, const at::Tensor& x, int64_t) { return at::empty({x.size(0), 132}, zi.options()); }
std::tuple<at::Tensor, at::Tensor> r6_rmsd(const at::Tensor& r, const at::Tensor& x, const at::Tensor& w) {
    auto o = run("custom_op::r6_rmsd", "r6_rmsd", 3, 0, {{at::kFloat, SH(T(1).size(0), T(2).size(0) / 2)},
                 {at::kFloat, SH(T(1).size(0), T(2).size(0))}}, {r, x, w});
    return {o[0], o[1]};
}
std::tuple<at::Tensor, at::Tensor> r6_rmsd_meta(const at::Tensor& r, const at::Tensor& x, const at::Tensor& w) {
    return {at::empty({x.size(0), w.size(0) / 2}, r.options()), at::empty({x.size(0), w.size(0)}, r.options())}; }
}  // namespace
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("r6_ffk(Tensor zi, Tensor x, int mode) -> Tensor");
    m.def("r6_ffkf(Tensor zi, Tensor x, int mode) -> Tensor");
    m.def("r6_rmsd(Tensor r, Tensor x, Tensor w) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("r6_ffk", r6_ffk); m.impl("r6_ffkf", r6_ffkf); m.impl("r6_rmsd", r6_rmsd);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("r6_ffk", r6_ffk_meta); m.impl("r6_ffkf", r6_ffkf_meta); m.impl("r6_rmsd", r6_rmsd_meta);
}
namespace {
