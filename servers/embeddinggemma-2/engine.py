# SPDX-License-Identifier: Apache-2.0
"""EmbeddingGemma 2 encoder (text, image, audio, video) on one Gaudi card, with HPU graphs in lazy mode.

CPU side: the checkpoint's own processor turns each input into token ids (media become placeholder runs:
<boi> <|image|>... <eoi>, <boa> <|audio|>... <eoa>) and pixel or mel features, exactly as sentence-transformers does.

HPU side: every shape comes from a small fixed set, so each one is a single HPU graph compiled at startup.
  vision    image patches [n, 2520, 768] (280 soft tokens) or video frames [n, 1260, 768] (140), n padded to a
            batch bucket. The stock vision model strips padding with a boolean index (a dynamic shape); this path
            keeps the padded output and its mask instead.
  audio     mel frames [n, T, 128], T padded to a length bucket, the mask marking real frames.
  language  token ids [B, L] padded to (batch, length) buckets, plus a dense [B, L, hidden] tensor of media soft
            tokens and a mask saying which positions take them. Text-only batches pass zeros, so text and media
            share one set of graphs.
Pooling follows the checkpoint's sentence-transformers modules: masked mean over all tokens, then L2 normalize;
a Matryoshka width truncates and normalizes again.

Attention is "eager" everywhere: HPU SDPA returns wrong vectors on padded rows (and NaN at the 4k and 8k buckets)
with this model's bidirectional sliding-window mask.
"""

import os
import threading

import numpy as np
import torch
from torch import nn

MODEL = os.environ.get("EG2_MODEL", "google/embeddinggemma-2")
MAX_LEN = 8192
LEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
TOKENS_PER_BATCH = 8192  # per language forward: a 64-token bucket runs up to 128 rows, an 8192-token one runs 1
BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128]
IMAGE_BATCH_BUCKETS = [1, 2, 4, 8, 16]
FRAME_BATCH_BUCKETS = [1, 2, 4, 8, 16, 32]
# Mel frames are 10 ms and audio costs 25 tokens per second; 32768 frames (328 s) is about the 8192-token limit.
AUDIO_LEN_BUCKETS = [1024, 2048, 4096, 8192, 16384, 32768]
AUDIO_BATCH_BUCKETS = [1, 2, 4, 8, 16]
AUDIO_FRAMES_PER_BATCH = 16384
NATIVE_DIM = 768
SAMPLE_RATE = 16000
MODALITIES = tuple(m for m in os.environ.get("EG2_MODALITIES", "text,image,audio,video").split(",") if m)


def bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} is larger than the largest bucket {buckets[-1]}")


def block_context(hidden: torch.Tensor, past: int, future: int, chunk: int, context: int) -> torch.Tensor:
    """[B, L, H, D] -> [B, blocks, context, H, D]: the overlapping windows Gemma4 audio attention builds with
    Tensor.unfold, as an index gather. unfold has no HPU kernel and falls back to the CPU on every call."""
    hidden = torch.nn.functional.pad(hidden, (0, 0, 0, 0, past, future + chunk - 1))
    blocks = (hidden.shape[1] - context) // chunk + 1
    idx = torch.arange(blocks, device=hidden.device)[:, None] * chunk + torch.arange(context, device=hidden.device)
    return hidden[:, idx].contiguous()


def patch_audio_unfold() -> None:
    from transformers.models.gemma4 import modeling_gemma4 as g4

    def _extract_block_context(self, hidden_states):
        return block_context(
            hidden_states, self.max_past_horizon, self.max_future_horizon, self.chunk_size, self.context_size
        )

    g4.Gemma4AudioAttention._extract_block_context = _extract_block_context


class VisionPath(nn.Module):
    """Gemma4VisionModel.forward plus embed_vision, without the dynamic padding strip."""

    def __init__(self, model):
        super().__init__()
        self.vt = model.vision_tower
        self.embed = model.embed_vision

    def forward(self, pixel_values, position_ids):
        vt = self.vt
        k = vt.config.pooling_kernel_size
        padding = (position_ids == -1).all(dim=-1)
        x = vt.patch_embedder(pixel_values, position_ids, padding)
        out = vt.encoder(inputs_embeds=x, attention_mask=~padding, pixel_position_ids=position_ids)
        h, mask = vt.pooler(
            hidden_states=out.last_hidden_state,
            pixel_position_ids=position_ids,
            padding_positions=padding,
            output_length=pixel_values.shape[-2] // (k * k),
        )
        if vt.config.standardize:
            h = (h - vt.std_bias.float()) * vt.std_scale.float()
        return self.embed(inputs_embeds=h.to(x.dtype)), mask


class AudioPath(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.at = model.audio_tower
        self.embed = model.embed_audio

    def forward(self, features, mask):
        out = self.at(features, mask, return_dict=True)
        return self.embed(inputs_embeds=out.last_hidden_state), out.attention_mask


class LanguagePath(nn.Module):
    """Token embeddings, media soft tokens swapped in where `is_media`, the encoder, masked mean pooling."""

    def __init__(self, model):
        super().__init__()
        self.lm = model.language_model

    def forward(self, input_ids, attention_mask, media, is_media):
        from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

        x = self.lm.embed_tokens(input_ids)
        x = torch.where(is_media.unsqueeze(-1), media.to(x.dtype), x)
        # By default transformers drops the mask when attention_mask.all() (no padding). That is a Python branch on
        # tensor contents: an HPU graph captured on an unpadded batch would replay with no mask on padded ones.
        kw = dict(
            config=self.lm.config, inputs_embeds=x, attention_mask=attention_mask, allow_is_bidirectional_skip=False
        )
        masks = {
            "full_attention": create_bidirectional_mask(**kw),
            "sliding_attention": create_bidirectional_sliding_window_mask(**kw),
        }
        h = self.lm(inputs_embeds=x, attention_mask=masks).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1, dtype=torch.float32) / m.sum(1, dtype=torch.float32)


class Prepared:
    """One input after CPU processing."""

    __slots__ = ("ids", "images", "image_pos", "frames", "frame_pos", "audio", "audio_mask")

    def __init__(self, ids, images=None, image_pos=None, frames=None, frame_pos=None, audio=None, audio_mask=None):
        self.ids = ids  # list[int]
        self.images, self.image_pos = images, image_pos  # [n, 2520, 768], [n, 2520, 2]
        self.frames, self.frame_pos = frames, frame_pos  # [f, 1260, 768], [f, 1260, 2] (all videos' frames)
        self.audio, self.audio_mask = audio, audio_mask  # lists of [T, 128] and [T] bool

    @property
    def has_media(self):
        return self.images is not None or self.frames is not None or self.audio is not None


class Engine:
    def __init__(self, device="hpu", dtype=torch.bfloat16, hpu_graphs=True, modalities=MODALITIES):
        from transformers import AutoModel, AutoProcessor

        self.device, self.dtype = device, dtype
        self.modalities = set(modalities)
        if device == "hpu":
            import habana_frameworks.torch.core  # noqa: F401  registers the hpu device
        patch_audio_unfold()
        overrides = {}
        if not self.modalities & {"image", "video"}:
            overrides["vision_config"] = None
        if "audio" not in self.modalities:
            overrides["audio_config"] = None
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.tok = self.processor.tokenizer
        model = AutoModel.from_pretrained(MODEL, dtype=dtype, attn_implementation="eager", **overrides)
        model = model.to(device).eval()
        self.cfg = model.config
        self.hidden = model.config.text_config.hidden_size
        self.language = LanguagePath(model)
        self.vision = VisionPath(model) if model.vision_tower is not None else None
        self.audio_path = AudioPath(model) if model.audio_tower is not None else None
        if device == "hpu" and hpu_graphs:
            import habana_frameworks.torch.hpu.graphs as htgraphs

            self.language = htgraphs.wrap_in_hpu_graph(self.language)
            if self.vision is not None:
                self.vision = htgraphs.wrap_in_hpu_graph(self.vision)
            if self.audio_path is not None:
                self.audio_path = htgraphs.wrap_in_hpu_graph(self.audio_path)
        self.pad_id = self.tok.pad_token_id
        self.lock = threading.Lock()  # one batch on the card at a time

    # ---------------------------------------------------------------- CPU side
    def tokenize(self, texts, truncate=True):
        ids = self.tok(texts, add_special_tokens=True)["input_ids"]
        if truncate:  # keep the trailing EOS, as sentence-transformers does when it truncates
            eos = self.tok.eos_token_id
            ids = [r if len(r) <= MAX_LEN else r[: MAX_LEN - 1] + [eos] for r in ids]
        return ids

    def prepare(self, text, images=(), audios=(), videos=()):
        """`text` holds one <|image|> / <|audio|> / <|video|> placeholder per item, in order. images: PIL images;
        audios: float32 mono 16 kHz arrays; videos: uint8 [T, H, W, 3] frames already picked by
        sample_video_indices()."""
        kw = {}
        if images:
            kw["images"] = [list(images)]
        if audios:
            kw["audio"] = list(audios)
            kw["sampling_rate"] = SAMPLE_RATE
        if videos:
            kw["videos"] = list(videos)
            kw["videos_kwargs"] = {"do_sample_frames": False}
        f = self.processor(text=[text], return_tensors="pt", **kw)
        p = Prepared(f["input_ids"][0].tolist())
        if images:
            p.images, p.image_pos = f["pixel_values"], f["image_position_ids"]
        if videos:
            p.frames, p.frame_pos = f["pixel_values_videos"], f["video_position_ids"]
        if audios:
            m = f["input_features_mask"]
            p.audio = [f["input_features"][i][: int(m[i].sum())] for i in range(len(audios))]
            p.audio_mask = [m[i][: int(m[i].sum())] for i in range(len(audios))]
        return p

    def sample_video_indices(self, total_frames, fps, duration):
        """Frame indices the checkpoint's video processor keeps (1 fps, at most 32, uniform)."""
        from transformers.video_utils import VideoMetadata

        vp = self.processor.video_processor
        meta = VideoMetadata(total_num_frames=total_frames, fps=fps, duration=duration)
        return vp.sample_frames(
            meta, fps=vp.fps, max_frames=vp.max_frames, overflow_strategy=vp.overflow_strategy
        ).tolist()

    # ---------------------------------------------------------------- HPU side
    def _sync(self):
        if self.device == "hpu":
            import habana_frameworks.torch.core as htcore

            htcore.mark_step()

    def _vision(self, pv, pos, buckets):
        """[n, P, 768] patches -> per-input [k_i, hidden] soft tokens (CPU)."""
        out, i = [], 0
        while i < len(pv):
            n = min(len(pv) - i, buckets[-1])
            b = bucket(n, buckets)
            x, p = pv[i : i + n], pos[i : i + n]
            if b > n:  # pad the batch with copies of the first row (never all-padding rows)
                x = torch.cat([x, x[:1].expand(b - n, -1, -1)])
                p = torch.cat([p, p[:1].expand(b - n, -1, -1)])
            with torch.no_grad():
                feats, mask = self.vision(x.to(self.device, self.dtype), p.to(self.device))
            self._sync()
            feats, mask = feats.cpu(), mask.cpu()
            out.extend(feats[j][mask[j]] for j in range(n))
            i += n
        return out

    def _audio(self, feats, masks):
        """Lists of [T_i, 128] mel and [T_i] mask -> per-clip [k_i, hidden] soft tokens (CPU)."""
        out = [None] * len(feats)
        order = sorted(range(len(feats)), key=lambda i: len(feats[i]))
        i = 0
        while i < len(order):
            t = bucket(len(feats[order[i]]), AUDIO_LEN_BUCKETS)
            cap = max(1, AUDIO_FRAMES_PER_BATCH // t)
            group = [order[i]]
            j = i + 1
            while j < len(order) and len(group) < cap and len(feats[order[j]]) <= t:
                group.append(order[j])
                j += 1
            b = bucket(len(group), AUDIO_BATCH_BUCKETS)
            x = torch.zeros((b, t, feats[group[0]].shape[-1]), dtype=torch.float32)
            m = torch.zeros((b, t), dtype=torch.bool)
            for r, k in enumerate(group):
                x[r, : len(feats[k])] = feats[k]
                m[r, : len(feats[k])] = masks[k]
            for r in range(len(group), b):  # padding rows copy a real row, so no row is all-masked
                x[r], m[r] = x[0], m[0]
            with torch.no_grad():
                e, em = self.audio_path(x.to(self.device, self.dtype), m.to(self.device))
            self._sync()
            e, em = e.cpu(), em.cpu()
            for r, k in enumerate(group):
                out[k] = e[r][em[r]]
            i = j
        return out

    def _media_tokens(self, items):
        """Soft tokens for every media placeholder, per item and type, in placeholder order."""
        per_item = [{"image": [], "video": [], "audio": []} for _ in items]
        for kind, attr, pos_attr, buckets in (
            ("image", "images", "image_pos", IMAGE_BATCH_BUCKETS),
            ("video", "frames", "frame_pos", FRAME_BATCH_BUCKETS),
        ):
            have = [n for n, it in enumerate(items) if getattr(it, attr) is not None]
            if not have:
                continue
            toks = self._vision(
                torch.cat([getattr(items[n], attr) for n in have]),
                torch.cat([getattr(items[n], pos_attr) for n in have]),
                buckets,
            )
            k = 0
            for n in have:
                c = len(getattr(items[n], attr))
                per_item[n][kind].extend(toks[k : k + c])
                k += c
        aud = [(n, k) for n, it in enumerate(items) if it.audio is not None for k in range(len(it.audio))]
        if aud:
            toks = self._audio([items[n].audio[k] for n, k in aud], [items[n].audio_mask[k] for n, k in aud])
            for (n, _), t in zip(aud, toks, strict=True):
                per_item[n]["audio"].append(t)
        return per_item

    def _language(self, rows, media_rows, length):
        """rows: token-id lists (each <= length); media_rows: per row None or (positions, [k, hidden]).
        Returns fp32 CPU [len(rows), 768] (not yet normalized)."""
        bsz = bucket(len(rows), BATCH_BUCKETS)
        ids = torch.full((bsz, length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((bsz, length), dtype=torch.long)
        is_media = torch.zeros((bsz, length), dtype=torch.bool)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        # Padding rows get one real token so no row is fully masked (an all-masked softmax row is NaN); they are
        # dropped below.
        mask[len(rows) :, 0] = 1
        media = torch.zeros((bsz, length, self.hidden), dtype=self.dtype)
        for i, mr in enumerate(media_rows):
            if mr is None:
                continue
            positions, feats = mr
            positions = torch.tensor(positions)
            is_media[i, positions] = True
            ids[i, positions] = self.pad_id  # the stock model embeds placeholders as PAD before swapping
            media[i, positions] = feats.to(self.dtype)
        with torch.no_grad():
            pooled = self.language(
                ids.to(self.device), mask.to(self.device), media.to(self.device), is_media.to(self.device)
            )
        self._sync()
        return pooled.cpu()[: len(rows)].clone()

    def embed_items(self, items):
        """Prepared items -> L2-normalized fp32 [n, 768]."""
        for it in items:
            if len(it.ids) > MAX_LEN:
                raise ValueError(f"input has {len(it.ids)} tokens; the maximum is {MAX_LEN}")
        out = torch.empty((len(items), NATIVE_DIM), dtype=torch.float32)
        with self.lock:
            media_rows = [None] * len(items)
            mm = [n for n, it in enumerate(items) if it.has_media]
            if mm:
                for n, t in zip(mm, self._media_tokens([items[n] for n in mm]), strict=True):
                    ids = items[n].ids
                    feats, positions = [], []
                    for kind, tid in (
                        ("image", self.cfg.image_token_id),
                        ("video", self.cfg.video_token_id),
                        ("audio", self.cfg.audio_token_id),
                    ):
                        pos = [p for p, x in enumerate(ids) if x == tid]
                        got = sum(len(x) for x in t[kind])
                        if got != len(pos):
                            raise RuntimeError(f"{kind}: {len(pos)} placeholders but {got} soft tokens")
                        if pos:
                            positions.extend(pos)
                            feats.extend(t[kind])
                    media_rows[n] = (positions, torch.cat(feats))
            # Group by length bucket, longest-fitting rows together, up to TOKENS_PER_BATCH per forward.
            order = sorted(range(len(items)), key=lambda i: len(items[i].ids))
            i = 0
            while i < len(order):
                length = bucket(len(items[order[i]].ids), LEN_BUCKETS)
                cap = max(1, min(BATCH_BUCKETS[-1], TOKENS_PER_BATCH // length))
                group = [order[i]]
                j = i + 1
                while j < len(order) and len(group) < cap and len(items[order[j]].ids) <= length:
                    group.append(order[j])
                    j += 1
                out[group] = self._language([items[k].ids for k in group], [media_rows[k] for k in group], length)
                i = j
        return torch.nn.functional.normalize(out, dim=-1)

    def embed(self, texts, dim=None):
        e = self.embed_items([Prepared(r) for r in self.tokenize(texts)])
        if dim and dim != NATIVE_DIM:
            e = torch.nn.functional.normalize(e[:, :dim], dim=-1)
        return e

    def warmup(self, log=print):
        """Compile every shape the batcher can produce."""
        for length in LEN_BUCKETS:
            cap = max(1, min(BATCH_BUCKETS[-1], TOKENS_PER_BATCH // length))
            for bsz in (b for b in BATCH_BUCKETS if b <= cap):
                with self.lock:
                    self._language([[self.pad_id] * min(8, length)] * bsz, [None] * bsz, length)
            log(f"warmup language len {length} done (batches up to {cap})")
        if self.vision is not None:
            from PIL import Image

            if "image" in self.modalities:
                p = self.prepare("<|image|>", images=[Image.new("RGB", (640, 480), (120, 80, 40))])
                for b in IMAGE_BATCH_BUCKETS:
                    with self.lock:
                        self._vision(p.images.expand(b, -1, -1), p.image_pos.expand(b, -1, -1), IMAGE_BATCH_BUCKETS)
                log("warmup image done")
            if "video" in self.modalities:
                p = self.prepare("<|video|>", videos=[np.zeros((2, 240, 320, 3), dtype=np.uint8)])
                for b in FRAME_BATCH_BUCKETS:
                    with self.lock:
                        self._vision(
                            p.frames[:1].expand(b, -1, -1), p.frame_pos[:1].expand(b, -1, -1), FRAME_BATCH_BUCKETS
                        )
                log("warmup video done")
        if self.audio_path is not None:
            for t in AUDIO_LEN_BUCKETS:
                cap = max(1, AUDIO_FRAMES_PER_BATCH // t)
                wave = (np.random.default_rng(0).standard_normal(int(t * 0.9) * 160) * 0.01).astype(np.float32)
                p = self.prepare("<|audio|>", audios=[wave])
                for b in (b for b in AUDIO_BATCH_BUCKETS if b <= cap):
                    with self.lock:
                        self._audio(p.audio * b, p.audio_mask * b)
                log(f"warmup audio {t / 100:.0f}s done (batches up to {cap})")
