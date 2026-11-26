#define SH(...) [=](const at::Stack& s) { return std::vector<int64_t>{__VA_ARGS__}; }
// data u8 [nrows,132], tags i32 [nrows,2], bt i32 [B,NB], own i32 [B,NB] -> K f32 [B, NB*BS, 128], kv i32 [B, NB*BS]
at::Tensor k1_kg(const at::Tensor& tags, const at::Tensor& data, const at::Tensor& bt, const at::Tensor& own,
                 int64_t BS, int64_t tagcheck) {
    return run("custom_op::k1_kg", "k1_kg", 4, 2, {{at::kInt, SH(T(2).size(0), T(2).size(1) * I(4), 128)}},
               {tags, data, bt, own, BS, tagcheck})[0];
}
// d f32 [B, S*32, Lk], w f32 [B,S,32], pos i32 [B,S], kv i32 [B,Lk] -> keys i32 [B,S,Lk], cnt i32 [B,S,NCH,64]
std::tuple<at::Tensor, at::Tensor> k1_sc(const at::Tensor& kref, const at::Tensor& d, const at::Tensor& pos, const at::Tensor& w,
                                         int64_t NCH) {
    auto r = run("custom_op::k1_sc", "k1_sc", 4, 1,
               {{at::kInt, SH(T(2).size(0), T(2).size(1), T(1).size(2))}, {at::kInt, SH(T(2).size(0), T(2).size(1), I(4), 64)}},
               {kref, d, pos, w, NCH});
    return {r[0], r[1]};
}
at::Tensor k1_p2(const at::Tensor& keys, const at::Tensor& c1, int64_t NCH, int64_t K) {
    return run("custom_op::k1_p2", "k1_p2", 2, 2, {{at::kInt, SH(T(0).size(0), T(0).size(1), I(2), 64)}},
               {keys, c1, NCH, K})[0];
}
at::Tensor k1_p3(const at::Tensor& keys, const at::Tensor& c1, const at::Tensor& c2, int64_t NCH, int64_t K) {
    return run("custom_op::k1_p3", "k1_p3", 3, 2, {{at::kInt, SH(T(0).size(0), T(0).size(1), I(3), 64)}},
               {keys, c1, c2, NCH, K})[0];
}
at::Tensor k1_p4(const at::Tensor& keys, const at::Tensor& c1, const at::Tensor& c2, const at::Tensor& c3, int64_t NCH,
                 int64_t K) {
    return run("custom_op::k1_p4", "k1_p4", 4, 2, {{at::kInt, SH(T(0).size(0), T(0).size(1), I(4), 64)}},
               {keys, c1, c2, c3, NCH, K})[0];
}
at::Tensor k1_fin(const at::Tensor& c1, const at::Tensor& c2, const at::Tensor& c3, const at::Tensor& c4, int64_t NCH,
                  int64_t K, int64_t MW) {
    return run("custom_op::k1_fin", "k1_fin", 4, 3, {{at::kInt, SH(T(0).size(0), T(0).size(1), 256 + I(4) * I(6))}},
               {c1, c2, c3, c4, NCH, K, MW})[0];
}
static int64_t k1_mw(int64_t Lk, int64_t nch) { int64_t nv = Lk / 64, per = (nv + nch - 1) / nch; return (per + 15) / 16 * 64 + 64; }
at::Tensor k1_cm1(const at::Tensor& keys, const at::Tensor& pos, const at::Tensor& fin, int64_t NCH, int64_t K, int64_t MW) {
    return run("custom_op::k1_cm1", "k1_cm1", 3, 3, {{at::kInt, SH(T(0).size(0), T(0).size(1), I(3) * I(5))}},
               {keys, pos, fin, NCH, K, MW})[0];
}
at::Tensor k1_cm2(const at::Tensor& keys, const at::Tensor& pos, const at::Tensor& fin, const at::Tensor& mk, int64_t NCH,
                  int64_t K, int64_t MW) {
    return run("custom_op::k1_cm2", "k1_cm2", 4, 3, {{at::kInt, SH(T(0).size(0), T(0).size(1), I(5))}},
               {keys, pos, fin, mk, NCH, K, MW})[0];
}

// Meta (shape / dtype) impls: HPU graph capture infers outputs through these
static at::Tensor E(const at::Tensor& like, std::vector<int64_t> sh, at::ScalarType dt) {
    return at::empty(sh, like.options().dtype(dt)); }
at::Tensor k1_kg_meta(const at::Tensor& tags, const at::Tensor&, const at::Tensor& bt, const at::Tensor&, int64_t BS, int64_t) {
    return E(tags, {bt.size(0), bt.size(1) * BS, 128}, at::kInt); }
std::tuple<at::Tensor, at::Tensor> k1_sc_meta(const at::Tensor&, const at::Tensor& d, const at::Tensor& pos, const at::Tensor&,
                                              int64_t NCH) {
    return {E(d, {pos.size(0), pos.size(1), d.size(2)}, at::kInt), E(d, {pos.size(0), pos.size(1), NCH, 64}, at::kInt)}; }
at::Tensor k1_p2_meta(const at::Tensor& k, const at::Tensor&, int64_t NCH, int64_t) {
    return E(k, {k.size(0), k.size(1), NCH, 64}, at::kInt); }
at::Tensor k1_p3_meta(const at::Tensor& k, const at::Tensor&, const at::Tensor&, int64_t NCH, int64_t) {
    return E(k, {k.size(0), k.size(1), NCH, 64}, at::kInt); }
at::Tensor k1_p4_meta(const at::Tensor& k, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t NCH, int64_t) {
    return E(k, {k.size(0), k.size(1), NCH, 64}, at::kInt); }
at::Tensor k1_fin_meta(const at::Tensor& c, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t NCH, int64_t,
                       int64_t MW) {
    return E(c, {c.size(0), c.size(1), 256 + NCH * MW}, at::kInt); }
at::Tensor k1_cm1_meta(const at::Tensor& k, const at::Tensor&, const at::Tensor&, int64_t NCH, int64_t K, int64_t MW) {
    return E(k, {k.size(0), k.size(1), NCH * MW}, at::kInt); }
at::Tensor k1_cm2_meta(const at::Tensor& k, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t NCH, int64_t K,
                       int64_t MW) {
    return E(k, {k.size(0), k.size(1), K}, at::kInt); }
}  // namespace (reopened below)
// k2_g2 (ML4_ATTN_KERNEL): tags, data, idx i32 [R,K], prow i32 [R,K], pown i32 [R,K], tagcheck -> rows i32 [R,K,576], valid i32 [R,K]
std::tuple<at::Tensor, at::Tensor> k2_g2(const at::Tensor& tags, const at::Tensor& data, const at::Tensor& idx,
                                         const at::Tensor& prow, const at::Tensor& pown, int64_t tagcheck) {
    auto r = run("custom_op::k2_g2", "k2_g2", 5, 1, {{at::kInt, SH(T(2).size(0), T(2).size(1), 576)},
                 {at::kInt, SH(T(2).size(0), T(2).size(1))}}, {tags, data, idx, prow, pown, tagcheck});
    return {r[0], r[1]};
}
std::tuple<at::Tensor, at::Tensor> k2_g2_meta(const at::Tensor& tags, const at::Tensor&, const at::Tensor& idx,
                                              const at::Tensor&, const at::Tensor&, int64_t) {
    return {at::empty({idx.size(0), idx.size(1), 576}, tags.options().dtype(at::kInt)),
            at::empty({idx.size(0), idx.size(1)}, tags.options().dtype(at::kInt))}; }
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("k2_g2(Tensor tags, Tensor data, Tensor idx, Tensor prow, Tensor pown, int tagcheck) -> (Tensor, Tensor)");
    m.def("k1_kg(Tensor tags, Tensor data, Tensor bt, Tensor own, int BS, int tagcheck) -> Tensor");
    m.def("k1_sc(Tensor kref, Tensor d, Tensor pos, Tensor w, int NCH) -> (Tensor, Tensor)");
    m.def("k1_p2(Tensor keys, Tensor c1, int NCH, int K) -> Tensor");
    m.def("k1_p3(Tensor keys, Tensor c1, Tensor c2, int NCH, int K) -> Tensor");
    m.def("k1_p4(Tensor keys, Tensor c1, Tensor c2, Tensor c3, int NCH, int K) -> Tensor");
    m.def("k1_fin(Tensor c1, Tensor c2, Tensor c3, Tensor c4, int NCH, int K, int MW) -> Tensor");
    m.def("k1_cm1(Tensor keys, Tensor pos, Tensor fin, int NCH, int K, int MW) -> Tensor");
    m.def("k1_cm2(Tensor keys, Tensor pos, Tensor fin, Tensor mk, int NCH, int K, int MW) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("k1_kg", k1_kg); m.impl("k1_sc", k1_sc);
    m.impl("k1_p2", k1_p2); m.impl("k1_p3", k1_p3); m.impl("k1_p4", k1_p4); m.impl("k1_fin", k1_fin);
    m.impl("k1_cm1", k1_cm1); m.impl("k1_cm2", k1_cm2); m.impl("k2_g2", k2_g2);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("k1_kg", k1_kg_meta); m.impl("k1_sc", k1_sc_meta); m.impl("k1_p2", k1_p2_meta); m.impl("k1_p3", k1_p3_meta);
    m.impl("k1_p4", k1_p4_meta); m.impl("k1_fin", k1_fin_meta); m.impl("k1_cm1", k1_cm1_meta); m.impl("k1_cm2", k1_cm2_meta); m.impl("k2_g2", k2_g2_meta);
}
namespace {
#include "r5_ops.inc"
#include "r6_ops.inc"
