// r5 ops (lazy legacy custom ops). Inside the anonymous namespace on entry and on exit.
at::Tensor r5_rms(const at::Tensor& x, const at::Tensor& r, const at::Tensor& w) {
    return run("custom_op::r5_rms", "r5_rms", 3, 0, {{at::kBFloat16, SH(T(0).size(0), T(2).size(0))}}, {x, r, w})[0];
}
at::Tensor r5_rms_meta(const at::Tensor& x, const at::Tensor&, const at::Tensor& w) {
    return at::empty({x.size(0), w.size(0)}, x.options()); }
at::Tensor r5_rope(const at::Tensor& x, const at::Tensor& c, const at::Tensor& s, int64_t off) {
    return run("custom_op::r5_rope", "r5_rope", 3, 1, {{at::kBFloat16, SH(T(0).size(0), T(0).size(1), T(0).size(2))}},
               {x, c, s, off})[0];
}
at::Tensor r5_rope_meta(const at::Tensor& x, const at::Tensor&, const at::Tensor&, int64_t) { return at::empty_like(x); }
std::tuple<at::Tensor, at::Tensor> r5_rt(const at::Tensor& z, const at::Tensor& p, int64_t mode) {
    auto r = run("custom_op::r5_rt", "r5_rt", 2, 1, {{at::kInt, SH(T(1).size(0), 8)}, {at::kInt, SH(T(1).size(0), 4)}},
                 {z, p, mode});
    return {r[0], r[1]};
}
std::tuple<at::Tensor, at::Tensor> r5_rt_meta(const at::Tensor& z, const at::Tensor& p, int64_t) {
    return {at::empty({p.size(0), 8}, z.options()), at::empty({p.size(0), 4}, z.options())}; }
at::Tensor r5_ff8(const at::Tensor& x) {
    return run("custom_op::r5_ff8", "r5_ff8", 1, 0, {{at::kBFloat16, SH(T(0).size(0), 128)}}, {x})[0]; }
at::Tensor r5_ff8_meta(const at::Tensor& x) { return at::empty({x.size(0), 128}, x.options()); }
at::Tensor r5_ff8f(const at::Tensor& zb, const at::Tensor& x) {
    return run("custom_op::r5_ff8f", "r5_ff8f", 2, 0, {{at::kBFloat16, SH(T(1).size(0), 128)}}, {zb, x})[0]; }
at::Tensor r5_ff8f_meta(const at::Tensor& zb, const at::Tensor& x) { return at::empty({x.size(0), 128}, zb.options()); }
at::Tensor r5_ssl(const at::Tensor& s, const at::Tensor& b, const at::Tensor& lp) {
    return run("custom_op::r5_ssl", "r5_ssl", 3, 0, {{at::kBFloat16, SH(T(2).size(0), T(0).size(0))}}, {s, b, lp})[0]; }
at::Tensor r5_ssl_meta(const at::Tensor& s, const at::Tensor&, const at::Tensor& lp) {
    return at::empty({lp.size(0), s.size(0)}, s.options()); }
at::Tensor r5_sslf(const at::Tensor& lp, const at::Tensor& s, const at::Tensor& b) {
    return run("custom_op::r5_sslf", "r5_sslf", 3, 0, {{at::kFloat, SH(T(0).size(0), T(1).size(0))}}, {lp, s, b})[0]; }
at::Tensor r5_sslf_meta(const at::Tensor& lp, const at::Tensor& s, const at::Tensor&) {
    return at::empty({lp.size(0), s.size(0)}, lp.options()); }
at::Tensor r5_lat(const at::Tensor& zi, const at::Tensor& kv, const at::Tensor& r, const at::Tensor& w,
                  const at::Tensor& c, const at::Tensor& s, int64_t rope) {
    return run("custom_op::r5_lat", "r5_lat", 6, 1, {{at::kChar, SH(T(1).size(0), 656)}}, {zi, kv, r, w, c, s, rope})[0]; }
at::Tensor r5_lat_meta(const at::Tensor& zi, const at::Tensor& kv, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, int64_t) { return at::empty({kv.size(0), 656}, zi.options()); }
std::tuple<at::Tensor, at::Tensor> k2_g3(const at::Tensor& tags, const at::Tensor& data, const at::Tensor& tk,
                                         const at::Tensor& bt, const at::Tensor& ep, int64_t tagcheck, int64_t S) {
    auto r = run("custom_op::k2_g3", "k2_g3", 5, 2, {{at::kInt, SH(T(2).size(0), T(2).size(1), 576)},
                 {at::kInt, SH(T(2).size(0), T(2).size(1))}}, {tags, data, tk, bt, ep, tagcheck, S});
    return {r[0], r[1]};
}
std::tuple<at::Tensor, at::Tensor> k2_g3_meta(const at::Tensor& tags, const at::Tensor&, const at::Tensor& tk,
                                              const at::Tensor&, const at::Tensor&, int64_t, int64_t) {
    return {at::empty({tk.size(0), tk.size(1), 576}, tags.options()), at::empty({tk.size(0), tk.size(1)}, tags.options())}; }
}  // namespace
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("r5_rms(Tensor x, Tensor r, Tensor w) -> Tensor");
    m.def("r5_rope(Tensor x, Tensor c, Tensor s, int off) -> Tensor");
    m.def("r5_ff8(Tensor x) -> Tensor");
    m.def("r5_ff8f(Tensor zb, Tensor x) -> Tensor");
    m.def("r5_ssl(Tensor s, Tensor b, Tensor lp) -> Tensor");
    m.def("r5_sslf(Tensor lp, Tensor s, Tensor b) -> Tensor");
    m.def("r5_lat(Tensor zi, Tensor kv, Tensor r, Tensor w, Tensor c, Tensor s, int rope) -> Tensor");
    m.def("k2_g3(Tensor tags, Tensor data, Tensor tk, Tensor bt, Tensor ep, int tagcheck, int S) -> (Tensor, Tensor)");
    m.def("r5_rt(Tensor z, Tensor p, int mode) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("r5_rms", r5_rms); m.impl("r5_rope", r5_rope); m.impl("r5_rt", r5_rt);
    m.impl("r5_ff8", r5_ff8); m.impl("r5_ff8f", r5_ff8f); m.impl("r5_ssl", r5_ssl); m.impl("r5_sslf", r5_sslf); m.impl("r5_lat", r5_lat); m.impl("k2_g3", k2_g3);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("r5_rms", r5_rms_meta); m.impl("r5_rope", r5_rope_meta); m.impl("r5_rt", r5_rt_meta);
    m.impl("r5_ff8", r5_ff8_meta); m.impl("r5_ff8f", r5_ff8f_meta); m.impl("r5_ssl", r5_ssl_meta); m.impl("r5_sslf", r5_sslf_meta); m.impl("r5_lat", r5_lat_meta); m.impl("k2_g3", k2_g3_meta);
}
namespace {
