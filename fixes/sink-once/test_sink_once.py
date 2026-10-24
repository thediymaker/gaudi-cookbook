# SPDX-License-Identifier: Apache-2.0
"""HPU test for sink-once: stock and patched attention against an fp32 reference with a sink.

Needs a Gaudi device, habana_frameworks and vllm_gaudi installed; skipped otherwise. Run with:
pytest fixes/sink-once

The fix is applied to a temporary copy of vllm_gaudi/extension/ops.py (through --root), and both
copies are imported side by side. Decode covers sequences of 1 to 40 KV blocks plus padding
blocks; prefill (only when the installed layout gets the prefill part) covers a fresh prompt, a
prompt with prefix context and a windowed explicit mask.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("habana_frameworks.torch")
vllm_gaudi = pytest.importorskip("vllm_gaudi")

HERE = Path(__file__).resolve().parent
H, HKV, D, W, BS = 16, 2, 64, 128, 128
SCALE = D**-0.5


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ops_pair(tmp_path_factory):
    src = Path(vllm_gaudi.__file__).parent / "extension" / "ops.py"
    root = tmp_path_factory.mktemp("root")
    dst = root / "vllm_gaudi" / "extension" / "ops.py"
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    subprocess.run([sys.executable, str(HERE / "apply.py"), "--root", str(root)], check=True)
    return _load(src, "ops_stock"), _load(dst, "ops_fixed"), "_fsdpa_sink_extra_key" in dst.read_text()


def _ref(q, k, v, allowed, sinks, with_sink=True):
    """q [T,H,D], k/v [L,Hkv,D], allowed [T,L] -> [T,H,D] in fp32."""
    k = k.repeat_interleave(H // HKV, 1).permute(1, 2, 0)
    v = v.repeat_interleave(H // HKV, 1).transpose(0, 1)
    s = torch.einsum("thd,hdl->htl", q, k) * SCALE
    s = s.masked_fill(~allowed, float("-inf"))
    if not with_sink:
        return torch.einsum("htl,hld->thd", torch.softmax(s, -1), v)
    s = torch.cat([s, sinks.view(H, 1, 1).expand(H, s.shape[1], 1)], -1)
    return torch.einsum("htl,hld->thd", torch.softmax(s, -1)[..., :-1], v)


def _rel(a, b):
    return float((a.float() - b).norm() / b.norm())


def test_decode_sink_counted_once(ops_pair):
    stock, fixed, _ = ops_pair
    from vllm_gaudi.extension.utils import Matmul, VLLMKVCache

    torch.manual_seed(0)
    dev, bf = "hpu", torch.bfloat16
    sinks = torch.linspace(-1.0, 5.0, H)
    seq_blocks, n_pad = [1, 3, 17, 40], 8
    B, n_real = len(seq_blocks), sum(seq_blocks)
    num_blocks = n_real + n_pad + 4
    kc = torch.randn(num_blocks * BS, HKV, D)
    vc = torch.randn(num_blocks * BS, HKV, D)
    perm = torch.randperm(num_blocks)[:n_real]
    block_list, groups, usage, o = [], [], [], 0
    for b, n in enumerate(seq_blocks):
        block_list += perm[o : o + n].tolist()
        o += n
        groups += [b] * n
        usage += [BS] * (n - 1) + [int(torch.randint(1, BS + 1, (1,)))]
    block_list += [0] * n_pad
    groups += [B] * n_pad
    usage += [0] * n_pad
    q = torch.randn(B, H, D)
    refs = []
    for b in range(B):
        sel = [x for x, g in enumerate(groups) if g == b]
        ks = torch.cat([kc.view(-1, BS, HKV, D)[block_list[x], : usage[x]] for x in sel])
        vs = torch.cat([vc.view(-1, BS, HKV, D)[block_list[x], : usage[x]] for x in sel])
        refs.append(_ref(q[b : b + 1], ks, vs, torch.ones(1, ks.shape[0], dtype=torch.bool), sinks))
    ref = torch.cat(refs)
    mapping = torch.zeros(len(groups), B)
    for x, g in enumerate(groups):
        if g < B:
            mapping[x, g] = 1.0
    slot = torch.arange(BS).view(1, -1)
    bias = torch.zeros(len(usage), BS).masked_fill_(slot >= torch.tensor(usage).view(-1, 1), float("-inf"))
    fetch = VLLMKVCache()
    args = dict(
        query=q.reshape(B, 1, H * D).to(bf).to(dev),
        key_cache=kc.to(bf).to(dev),
        value_cache=vc.to(bf).to(dev),
        block_list=torch.tensor(block_list, device=dev),
        block_mapping=mapping.to(bf).to(dev),
        block_bias=bias.to(bf).to(dev),
        block_groups=torch.tensor(groups, device=dev),
        block_size=BS,
        scale=SCALE,
        matmul_qk_op=Matmul(),
        position_bias=None,
        matmul_av_op=Matmul(),
        batch2block_matmul_op=Matmul(),
        block2batch_matmul_op=Matmul(),
        keys_fetch_func=fetch.fetch_from_cache,
        values_fetch_func=fetch.fetch_from_cache,
        sinks=sinks.to(bf).to(dev),
        k_scales=None,
        v_scales=None,
    )
    out_stock = stock.flat_pa(**args).cpu().view(B, H, D)
    out_fixed = fixed.flat_pa(**args).cpu().view(B, H, D)
    for b, n in enumerate(seq_blocks):
        assert _rel(out_fixed[b : b + 1], ref[b : b + 1]) < 2e-2, f"{n}-block sequence"
    assert _rel(out_stock[-1:], ref[-1:]) > 5e-2, "stock decode was expected to over-count the sink"


@pytest.mark.parametrize("T, C, window", [(384, 0, False), (256, 640, False), (256, 640, True)])
def test_prefill_sink_applied(ops_pair, T, C, window):
    stock, fixed, has_prefill = ops_pair
    if not has_prefill:
        pytest.skip("the prefill part is not applied on this vllm-gaudi layout")
    import vllm_gaudi.extension.kernels as kernels
    from vllm_gaudi.extension.utils import ModuleFusedSDPA

    torch.manual_seed(0)
    dev, bf = "hpu", torch.bfloat16
    sinks = torch.linspace(-1.0, 5.0, H)
    fsdpa = ModuleFusedSDPA(kernels.fsdpa())
    L = C + T
    q, k, v = torch.randn(T, H, D), torch.randn(L, HKV, D), torch.randn(L, HKV, D)
    i = torch.arange(T).view(-1, 1) + C
    j = torch.arange(L).view(1, -1)
    allowed = j <= i
    if window:
        allowed &= j > i - W
    use_mask = C > 0 or window
    bias = torch.where(allowed, 0.0, float("-inf")).view(1, 1, T, L).to(bf).to(dev) if use_mask else None
    args = dict(
        query=q.unsqueeze(0).to(bf).to(dev),
        key=k.unsqueeze(0).to(bf).to(dev),
        value=v.unsqueeze(0).to(bf).to(dev),
        scale=SCALE,
        fsdpa_op=fsdpa,
        is_causal=True,
        attn_bias=bias,
        valid_seq_lengths=None if use_mask else torch.tensor([T], device=dev),
        sinks=sinks.to(bf).to(dev),
    )
    out_stock = stock._fsdpa_prompt_attention(**args).cpu()[0]
    out_fixed = fixed._fsdpa_prompt_attention(**args).cpu()[0]
    floor = _rel(out_stock, _ref(q, k, v, allowed, sinks, with_sink=False))
    assert _rel(out_fixed, _ref(q, k, v, allowed, sinks)) < max(2e-2, 1.5 * floor)
