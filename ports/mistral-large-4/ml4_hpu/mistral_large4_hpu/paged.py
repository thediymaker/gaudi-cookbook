"""Per-forward addressing of Mistral Large 4's paged caches, with ownership tags (from dsv41_hpu.paged).

Two persistent per-token caches, both one row per token, both in vLLM-allocated paged blocks addressed only through
the request's own block table:

    latent[l]  [576] bf16 = [kv_a_norm(c) (512) | k_pe (64)]   63 KV-owning layers (NoPE owners store k_pe = 0)
    index_k[l] [128] bf16 = fake-FP8 indexer key                44 full-indexer layers

    row(b, p) = block_table[b, p // BLOCK] * BLOCK + p % BLOCK

The 9 KV-shared layers (39, 43, ..., 71) own no latent cache: they read layer 35's rows by layer id, through the
same block table, after layer 35 has written this forward's tokens (layer order). Each row carries a tag (owner,
position), written with the row and checked on every read. With `block_epoch` (serving) the owner is the block's
allocation number (the runner bumps a physical block's number whenever vLLM hands it out fresh and keeps it on a
prefix-cache hit), so a row is valid only if it was written during the block's current allocation at this
position: stale rows of a previous owner, never-written rows and the null block are masked. Without
`block_epoch` (tests) the owner is the request uid. ML4_TAG_CHECK=count also counts masked rows, which is how the
isolation gate proves masking never fires in normal operation (nine readers of layer 35 make this matter more
than in DSV).

Physical block 0 is vLLM's null block and is never handed to a request, so rows 0..BLOCK-1 take the writes of
padding tokens (tag -1 never matches).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

TAG_MODE = os.environ.get("ML4_TAG_CHECK", "mask")  # mask | count | off
LATENT_W = 576
INDEX_W = 128
# FP8 latent KV, the fp8_ds_mla layout upstream serves this model with: per token the 512
# nope dims as e4m3 in 4 tiles of 128, one fp32 scale per tile (amax / 240, see E4M3_MAX), and the 64 RoPE dims in bf16 =
# 512 + 16 + 128 = 656 B (vs 1152). Stored in a bf16 cache of width 328 (the same bytes; the caches stay one
# uniform bf16 group for vLLM). Reads return bf16(q * scale), the value FlashMLA's sparse path dequantizes to.
#   ML4_KV_FP8=0 (default) bf16 latent; =1 FP8 storage; =fake bf16 storage of the FP8-rounded values (accuracy
#   A/B at the old pool size: exactly the values =1 reads back).
#   ML4_KV_CODEC (default e4m3-t128-f32, the layout above): <fmt>-t<tile>-<scale dtype>, fmt e4m3 | i8 (int8,
#   scale = amax / 127), tile 128 | 64 | 32, scale f32 | bf16. Bytes per token: 512 + (512 / tile) x (4 | 2) + 128;
#   e4m3-t128-f32, i8-t128-f32 and *-t64-bf16 keep 656 B. With =fake the control file key "kv_codec" picks the
#   codec at run time ("off": bf16), so one server A/Bs every codec on prompt rows.
KV_FP8 = os.environ.get("ML4_KV_FP8", "0")
KV_CODEC = os.environ.get("ML4_KV_CODEC", "e4m3-t128-f32")
# ML4_INDEX_U8=1: the index-K cache as e4m3 bytes + the row's power-of-two scale (132 B vs 256), lossless
# except values under ~7e-5 of the row's amax (index_u8_pack). =fake: bf16 storage of the round trip, on/off by
# the control file key "index_u8" (default on); the mismatching elements are counted (codec_stats()).
INDEX_U8_MODE = os.environ.get("ML4_INDEX_U8", "0")
INDEX_U8 = INDEX_U8_MODE == "1"
INDEX_U8_W = 66
# ML4_DCP=1: decode context parallel over the TP ranks with the 1/tp paged layout. Rank r keeps
# only the tokens at positions p with p % n == r: in each vLLM block of BLOCK tokens its rows hold BLOCK / n tokens,
#   row(b, p) = block_table[b, p // BLOCK] * (BLOCK / n) + (p % BLOCK) // n      (valid only if p % n == r)
# so a cache's per-token width as vLLM sees it is width / n (ML4KVCache: widths padded to a multiple of n) and the same
# memory holds n times the tokens. Prompt chunks read a cache's whole key range gathered from every rank (one
# all-gather per read, the same values as the replicated cache, so prompt rows are unchanged). Decode / verify rows
# (S <= ML4_DECODE_S) never gather the cache: the indexer scores this rank's keys only and merges the global top-k
# from every rank's candidates (indexer.py), and attention runs over this rank's selected keys for all heads with the
# partial outputs merged by LSE (attention.absorbed_dcp). At most ML4_DCP_KLOC selected keys per row and rank (2048 /
# n = 256 on the mean; past the cap keys are dropped and counted in dcp_overflow()).
DCP = os.environ.get("ML4_DCP", "0") == "1"
DCP_KLOC = int(os.environ.get("ML4_DCP_KLOC", "512"))
DCP_ROWS = int(os.environ.get("ML4_DECODE_S", "8"))
_DCP: dict = {}
# count mode: per (device, cache kind, prefill|decode) [violating rows, rows read], updated in place on the device
# so the counts keep accumulating inside HPU graph replays and the hot path never syncs (fp32: exact to 2^24,
# and a violation count only has to be exactly zero).
_counters: dict = {}


def _counter(device, label: str = "all") -> torch.Tensor:
    key = (str(device), label)
    c = _counters.get(key)
    if c is None:
        c = _counters[key] = torch.zeros(2, dtype=torch.float32, device=device)
    return c


def tag_violations(by_label: bool = False) -> dict:
    rows = reads = 0.0
    labels: dict = {}
    for (_, label), c in _counters.items():
        v = c.cpu().tolist()
        rows, reads = rows + v[0], reads + v[1]
        if v[0]:
            labels[label] = labels.get(label, 0) + int(v[0])
    out = {"rows": int(rows), "reads": int(reads)}
    if by_label:
        out["by_label"] = labels
    return out


def dcp_group():
    """(process group, rank, n) for the DCP collectives: vLLM's TP group in serving, the default group in tests."""
    if "g" not in _DCP:
        try:
            from vllm.distributed import get_tp_group

            g = get_tp_group()
            _DCP.update(g=g.device_group, rank=g.rank_in_group, n=g.world_size)
        except Exception:  # noqa: BLE001  (no vLLM, or its groups not initialised: plain torch.distributed)
            import torch.distributed as dist

            _DCP.update(g=None, rank=dist.get_rank(), n=dist.get_world_size())
    return _DCP["g"], _DCP["rank"], _DCP["n"]


def dcp_all_gather(x: torch.Tensor) -> torch.Tensor:
    """[...] -> [n, ...] from every rank of the DCP group."""
    import torch.distributed as dist

    group, _, n = dcp_group()
    x = x.contiguous()
    out = torch.empty((n, *x.shape), dtype=x.dtype, device=x.device)
    if x.device.type == "cpu":  # gloo (tests): no all_gather_into_tensor, no bf16 (moved as fp32, exact)
        xf = x.float()
        of = torch.empty((n, *x.shape), dtype=torch.float32)
        dist.all_gather(list(of.unbind(0)), xf, group=group)
        out.copy_(of)
    else:
        dist.all_gather_into_tensor(out, x, group=group)
    return out


def dcp_count_overflow(n_rows: torch.Tensor) -> None:
    c = _DCP.get("overflow")
    if c is None or c.device != n_rows.device:
        c = _DCP["overflow"] = torch.zeros((), dtype=torch.float32, device=n_rows.device)
    c.add_(n_rows.to(torch.float32))


def dcp_overflow() -> int:
    """Rows (summed over layers and steps) whose selected keys on this rank passed ML4_DCP_KLOC."""
    c = _DCP.get("overflow")
    return 0 if c is None else int(c.item())


def dcp_pad(width: int, n: int) -> int:
    return width + (-width % n)


def reset_tag_violations() -> None:
    with torch.inference_mode():
        for c in _counters.values():
            c.zero_()


@dataclass
class BatchCtx:
    """What one forward needs to address the caches. All [B, ...] with B the padded batch; padding requests have
    uid -1 and positions -1."""

    positions: torch.Tensor  # [B, S] long, -1 for padding tokens
    block_table: torch.Tensor  # [B, NB] long, physical block ids (-1 beyond the request)
    req_uid: torch.Tensor  # [B] int32, runner-assigned, never reused; -1 for padding
    block_size: int
    block_epoch: torch.Tensor | None = None  # [B, NB] int32 allocation number of each table entry
    memo: dict | None = None  # per-forward results shared by layers (decode graphs)
    host_max_pos: int | None = None  # largest position in the batch, known on the host (never .item()):
    # <= 2047 lets the indexer skip its scores (indexer.py)
    req_slot: torch.Tensor | None = None  # [B] long request slot (runner_patch), -1 padding: DSpark's rings

    def shared(self, key, make):
        """make() once per forward when a memo is set, else every call."""
        if self.memo is None:
            return make()
        if key not in self.memo:
            self.memo[key] = make()
        return self.memo[key]

    @property
    def valid(self) -> torch.Tensor:
        return self.positions >= 0

    @property
    def max_len(self) -> int:
        return self.block_table.shape[1] * self.block_size

    def rows(self, x: torch.Tensor, key=None, dcp=None):
        """Physical rows for positions x [B, ...]. Returns (rows, addressable, owner): not addressable (x < 0, past
        the table, or an unallocated block) maps to row 0; owner is the tag a valid row must carry. dcp (n, r): the
        1/n layout (DCP), where only positions p % n == r are addressable."""
        if key is not None and self.memo is not None:
            return self.shared(("rows", key, dcp), lambda: self._rows(x, dcp))
        return self._rows(x, dcp)

    def _rows(self, x: torch.Tensor, dcp=None):
        B, NB, BS = self.block_table.shape[0], self.block_table.shape[1], self.block_size
        flat = x.reshape(B, -1)
        from .numerics import NUMERICS_RW

        pow2 = NUMERICS_RW and BS & (BS - 1) == 0  # ML4_NUMERICS_RW: >> and & for the power-of-two block size
        blk_idx = (flat >> (BS.bit_length() - 1)) if pow2 else torch.div(flat, BS, rounding_mode="floor")
        inside = (flat >= 0) & (blk_idx < NB)
        bi = blk_idx.clamp(0, NB - 1)
        blk = torch.gather(self.block_table, 1, bi)
        ok = inside & (blk >= 0)
        rem = (flat & (BS - 1)) if pow2 else torch.remainder(flat, BS)
        if dcp is None:
            rows = torch.where(ok, blk * BS + rem, 0)
        else:
            n, r = dcp
            ok = ok & (torch.remainder(rem, n) == r)
            rows = torch.where(ok, blk * (BS // n) + torch.div(rem, n, rounding_mode="floor"), 0)
        if self.block_epoch is not None:
            owner = torch.gather(self.block_epoch, 1, bi)
        else:
            owner = self.req_uid.to(torch.int32).view(B, 1).expand_as(flat)
        return rows.reshape(x.shape), ok.reshape(x.shape), owner.reshape(x.shape)


class TaggedCache:
    """A paged cache view [num_rows, width] plus its tag table [num_rows, 2] int32."""

    def __init__(self, data: torch.Tensor, tags: torch.Tensor, label: str = "cache", layer: int = 0, dcp=None):
        assert data.shape[0] == tags.shape[0], (data.shape, tags.shape)
        self.data, self.tags, self.label, self.layer = data, tags, label, layer
        self.dump = 0  # where writes of tokens that keep nothing go (vLLM's null block)
        self.dcp = dcp  # (n, r): the DCP layout (this rank keeps positions p % n == r), None replicated

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def decode(self, rows: torch.Tensor) -> torch.Tensor:
        return rows

    def write(
        self, ctx: BatchCtx, x: torch.Tensor, values: torch.Tensor, keep: torch.Tensor, key=None, encoded: bool = False
    ) -> None:
        """Write values [B, S, W] at positions x [B, S] where keep [B, S] (encoded: already in the cache format)."""
        if ctx.memo is not None:  # memoised reads of this cache are stale now
            for k in [k for k in ctx.memo if k[0] == "read" and k[1] == id(self.data)]:
                del ctx.memo[k]
        rows, ok, owner = ctx.rows(x, key, self.dcp)
        keep = keep & ok
        rows = torch.where(keep, rows, self.dump).reshape(-1)
        tag = torch.stack([torch.where(keep, owner, -1), torch.where(keep, x.to(torch.int32), -1)], dim=-1)
        values = values if encoded else self.encode(values)
        if values.shape[-1] < self.data.shape[-1]:  # a width padded for the DCP layout (index-K U8: 66 -> 72)
            values = torch.nn.functional.pad(values, (0, self.data.shape[-1] - values.shape[-1]))
        self.data.index_copy_(0, rows, values.reshape(-1, values.shape[-1]).to(self.data.dtype))
        self.tags.index_copy_(0, rows, tag.reshape(-1, 2).to(torch.int32))

    def read(
        self, ctx: BatchCtx, x: torch.Tensor, want: torch.Tensor, key=None, share: bool = False, local: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather rows for positions x [B, ...] the caller expects to exist (want). Returns (values [B, ..., W],
        valid [B, ...]) where valid = want and the row's tag is exactly (owner, x). share: memoise the whole read
        per cache (layers reading the same cache at the same key get the same tensors). Under the DCP layout a read
        returns every rank's rows (_read_full) unless local (this rank's positions only; the others read invalid)."""
        if self.dcp is not None and not local:
            return self._read_full(ctx, x, want, key)
        if share and key is not None and ctx.memo is not None:
            return ctx.shared(("read", id(self.data), key), lambda: self._read(ctx, x, want, key))
        return self._read(ctx, x, want, key)

    def _gathered(self, ctx: BatchCtx, Lk: int):
        """DCP: the whole key range [B, Lk, W] (decoded values) and its validity [B, Lk], from every rank's rows."""
        n, r = self.dcp
        B, L = ctx.block_table.shape[0], Lk // n
        xl = (torch.arange(L, device=self.data.device) * n + r).view(1, L).expand(B, L)
        vals, valid = self._read(ctx, xl, xl <= ctx.positions.amax(-1, keepdim=True), key=("dcp_ent", Lk))
        W = vals.shape[-1]
        got = dcp_all_gather(torch.cat([vals, valid.unsqueeze(-1).to(vals.dtype)], -1))  # [n, B, L, W + 1]
        full = got.permute(1, 2, 0, 3).reshape(B, Lk, W + 1)  # position j * n + r
        return full[..., :W], full[..., W] > 0

    def _read_full(self, ctx: BatchCtx, x: torch.Tensor, want: torch.Tensor, key=None):
        Lk = ctx.max_len
        full, fvalid = ctx.shared(("read", id(self.data), ("dcp_full", Lk)), lambda: self._gathered(ctx, Lk))
        if key == ("ent", Lk):  # the whole range in position order: the gathered tensors themselves
            return full, fvalid & want
        B, W = full.shape[0], full.shape[-1]
        flat = x.reshape(B, -1)
        idx = flat.clamp(0, Lk - 1)
        vals = torch.gather(full, 1, idx.unsqueeze(-1).expand(B, idx.shape[1], W)).reshape(*x.shape, W)
        valid = torch.gather(fvalid, 1, idx) & (flat >= 0) & (flat < Lk)
        return vals, valid.reshape(x.shape) & want

    def _read(self, ctx: BatchCtx, x: torch.Tensor, want: torch.Tensor, key=None):
        rows, ok, owner = ctx.rows(x, key, self.dcp)
        flat = rows.reshape(-1)
        vals = self.decode(self.data.index_select(0, flat)).reshape(*x.shape, -1)
        tag = self.tags.index_select(0, flat).reshape(*x.shape, 2)
        want = want & ok
        valid = want & (tag[..., 0] == owner) & (tag[..., 1] == x.to(torch.int32))
        if TAG_MODE == "count":
            phase = "decode" if ctx.positions.shape[1] <= DCP_ROWS else "prefill"  # verify rows: decode
            _counter(vals.device, f"{self.label}/{phase}").add_(
                torch.stack([(want & ~valid).sum(), want.sum()]).float()
            )
        elif TAG_MODE == "off":
            valid = want
        return vals, valid


class Fp8LatentCache(TaggedCache):
    """The latent cache in an 8-bit codec (KV_FP8 = "1"), or bf16 rows rounded through it ("fake", codec from the
    control file)."""

    def __init__(self, data, tags, label="latent", layer=0, fake: bool = False, dcp=None):
        super().__init__(data, tags, label, layer, dcp)
        self.fake = fake

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if self.fake:
            from .control import get as ctl_get

            codec = ctl_get("kv_codec", KV_CODEC)
            return values if codec == "off" else fp8_latent_unpack(fp8_latent_pack(values, codec), codec)
        return fp8_latent_pack(values)

    def decode(self, rows: torch.Tensor) -> torch.Tensor:
        return rows if self.fake else fp8_latent_unpack(rows)


_codec_stats: dict = {}


def codec_stats() -> dict:
    """{label: [mismatching elements, elements]} of the fake index-K round trip, summed over devices."""
    out: dict = {}
    for (_, label), c in _codec_stats.items():
        v = c.cpu().tolist()
        o = out.setdefault(label, [0, 0])
        o[0] += int(v[0])
        o[1] += int(v[1])
    return out


class IndexU8Cache(TaggedCache):
    """The index-K cache as e4m3 bytes + scale (INDEX_U8); reads return the same bf16 values. fake: bf16 storage of
    the round trip, with the elements it changes counted."""

    def __init__(self, data, tags, label="index_k", layer=0, fake: bool = False, dcp=None):
        super().__init__(data, tags, label, layer, dcp)
        self.fake = fake

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if self.fake:
            from .control import get as ctl_get

            if not ctl_get("index_u8", True):
                return values
            rt = index_u8_unpack(index_u8_pack(values))
            key = (str(values.device), "index_u8")
            if key not in _codec_stats:
                _codec_stats[key] = torch.zeros(2, dtype=torch.float32, device=values.device)
            _codec_stats[key].add_(
                torch.stack(
                    [
                        (rt != values.to(rt.dtype)).sum().float(),
                        torch.tensor(float(values.numel()), device=values.device),
                    ]
                )
            )
            return rt
        return index_u8_pack(values)

    def decode(self, rows: torch.Tensor) -> torch.Tensor:
        return rows if self.fake else index_u8_unpack(rows)


# e4m3 through the device's own cast. Gaudi2's fp8 e4m3 tops out at 240 (exponent 15 is inf/NaN), where e4m3fn
# goes to 448; below 240 the two encodings are the same bytes. So every stored byte is on |v| <= 240 and the CPU's
# float8_e4m3fn reads it back exactly as the HPU does. (The first version decoded the fn bytes through a 256-entry
# table with integer bit work: ~25 small ops per read and per write, which made decode 2-3x slower.)
E4M3_MAX = 240.0


def _e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    """fp32 [..., n], |x| <= 240 -> uint8 [..., n] (round to nearest even, the device cast)."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def _e4m3_float(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def parse_codec(codec: str) -> tuple[str, int, str]:
    fmt, t, sd = codec.split("-")
    assert fmt in ("e4m3", "i8") and t[0] == "t" and int(t[1:]) in (32, 64, 128) and sd in ("f32", "bf16"), codec
    return fmt, int(t[1:]), sd


def codec_bytes(codec: str) -> int:
    fmt, tile, sd = parse_codec(codec)
    return 512 + (512 // tile) * (4 if sd == "f32" else 2) + 128


def fp8_latent_pack(values: torch.Tensor, codec: str | None = None) -> torch.Tensor:
    """[..., 576] bf16 -> [..., codec_bytes / 2] bf16: 512 codes (e4m3 or int8, tiles of `tile` dims, one scale per
    tile = amax / 240 or amax / 127, stored as fp32 or bf16), the scales, the 64 bf16 RoPE dims (128 bytes). The
    default codec is the fp8_ds_mla layout: 512 + 16 + 128 = 656 bytes."""
    fmt, tile, sd = parse_codec(codec or KV_CODEC)
    lead, nt = values.shape[:-1], 512 // tile
    top = E4M3_MAX if fmt == "e4m3" else 127.0
    c = values[..., :512].float().reshape(*lead, nt, tile)
    amax = c.abs().amax(-1, keepdim=True)
    scale = torch.where(amax > 0, amax / top, torch.ones_like(amax))
    if sd == "bf16":
        scale = scale.to(torch.bfloat16).float()  # the stored scale is the one used (codes clamp)
    x = c / scale
    if fmt == "e4m3":
        q = _e4m3_bytes(x)
    else:
        q = torch.round(x).clamp(-127, 127).to(torch.int8).view(torch.uint8)
    sdt = torch.float32 if sd == "f32" else torch.bfloat16
    sb = scale.reshape(*lead, nt).to(sdt).contiguous().view(torch.uint8)
    kb = values[..., 512:].to(torch.bfloat16).contiguous().view(torch.uint8)
    return torch.cat([q.reshape(*lead, 512), sb, kb], -1).view(torch.bfloat16)


def fp8_latent_unpack(rows: torch.Tensor, codec: str | None = None) -> torch.Tensor:
    """[..., codec_bytes / 2] (fp8_latent_pack) -> [..., 576] bf16 = [bf16(q * scale) | k_pe]."""
    fmt, tile, sd = parse_codec(codec or KV_CODEC)
    lead, nt = rows.shape[:-1], 512 // tile
    sbytes = nt * (4 if sd == "f32" else 2)
    b = rows.contiguous().view(torch.uint8)
    if fmt == "e4m3":
        q = _e4m3_float(b[..., :512])
    else:
        q = b[..., :512].contiguous().view(torch.int8).to(torch.float32)
    sdt = torch.float32 if sd == "f32" else torch.bfloat16
    scale = b[..., 512 : 512 + sbytes].contiguous().view(sdt).to(torch.float32).unsqueeze(-1)
    c = (q.reshape(*lead, nt, tile) * scale).reshape(*lead, 512).to(torch.bfloat16)
    k0 = (512 + sbytes) // 2
    return torch.cat([c, rows[..., k0 : k0 + 64]], -1)


def index_u8_pack(values: torch.Tensor) -> torch.Tensor:
    """Index-K rows [..., 128] bf16, already fake-FP8 with one power-of-two scale per token (indexer.qk: v / s on the
    e4m3fn grid, up to 448) -> [..., 66] bf16 holding 132 bytes: 128 e4m3 of v / (2 s) (<= 224, the device range)
    + the fp32 scale 2 s. Halving a power of two is exact except below e4m3's smallest normal (2^-6): values under
    ~7e-5 of the row's amax lose their last mantissa bit."""
    from .numerics import pow2_ceil

    v = values.float()
    amax = v.abs().amax(-1, keepdim=True)
    s = 2.0 * torch.where(
        amax > 0, pow2_ceil(torch.where(amax > 0, amax, torch.ones_like(amax)) / 448.0), torch.ones_like(amax)
    )
    return torch.cat([_e4m3_bytes(v / s), s.contiguous().view(torch.uint8)], -1).view(torch.bfloat16)


def index_u8_unpack(rows: torch.Tensor) -> torch.Tensor:
    b = rows.contiguous().view(torch.uint8)
    return (_e4m3_float(b[..., :128]) * b[..., 128:132].contiguous().view(torch.float32)).to(torch.bfloat16)


def latent_width() -> int:
    return codec_bytes(KV_CODEC) // 2 if KV_FP8 == "1" else LATENT_W


def latent_cache(data, tags, layer: int, dcp=None) -> TaggedCache:
    """The latent cache view for ML4_KV_FP8 (paged caches, the tiny engine and the tests); dcp (n, r): DCP layout."""
    if KV_FP8 in ("1", "fake"):
        return Fp8LatentCache(data, tags, "latent", layer, fake=KV_FP8 == "fake", dcp=dcp)
    return TaggedCache(data, tags, "latent", layer, dcp)


def index_width() -> int:
    return INDEX_U8_W if INDEX_U8 else INDEX_W


def index_cache(data, tags, layer: int, dcp=None) -> TaggedCache:
    if INDEX_U8_MODE in ("1", "fake"):
        return IndexU8Cache(data, tags, "index_k", layer, fake=INDEX_U8_MODE == "fake", dcp=dcp)
    return TaggedCache(data, tags, "index_k", layer, dcp)


def make_caches(roles, num_rows: int, device, dtype=torch.bfloat16) -> dict:
    """Tests / the tiny engine without vLLM's allocator: {layer: {"latent": TaggedCache, "index_k": TaggedCache}}
    for the layers that own them (roles from attention.layer_roles)."""
    caches: dict = {}
    for r in roles:
        c = {}
        if r.owns_kv:
            c["latent"] = latent_cache(
                torch.zeros(num_rows, latent_width(), dtype=dtype, device=device),
                torch.full((num_rows, 2), -1, dtype=torch.int32, device=device),
                r.layer_id,
            )
        if r.has_indexer:
            c["index_k"] = index_cache(
                torch.zeros(num_rows, index_width(), dtype=dtype, device=device),
                torch.full((num_rows, 2), -1, dtype=torch.int32, device=device),
                r.layer_id,
            )
        caches[r.layer_id] = c
    return caches


def cache_layout(roles) -> list[tuple[str, int, str, int]]:
    """The paged caches the model registers with vLLM, one spec each: (layer name, layer id, kind, width).
    63 latent caches (KV owners) + 44 index-K caches (full indexers) for the real model; KV-shared layers register
    none (they read their source's by layer id). All bf16, one block size, so vLLM merges them into ONE
    UniformTypeKVCacheSpecs group: every cache shares the request's block table (gate m3_cache_specs.py)."""
    out = []
    for r in roles:
        if r.owns_kv:
            out.append((f"model.layers.{r.layer_id}.attn.cache_latent", r.layer_id, "latent", latent_width()))
        if r.has_indexer:
            out.append((f"model.layers.{r.layer_id}.attn.cache_index_k", r.layer_id, "index_k", index_width()))
    return out
