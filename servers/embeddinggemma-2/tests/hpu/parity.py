# SPDX-License-Identifier: Apache-2.0
"""EmbeddingGemma 2 parity on one card: the HPU engine against the stock model in fp32 on the CPU.

    python3 tests/hpu/parity.py [--min-cos 0.9999]

The reference is transformers' own EmbeddingGemma2Model (no padding, no graphs, fp32, one input at a time) on the
same token ids and media features, mean-pooled and normalized as the checkpoint's sentence-transformers modules do.
Gates, one PASS or FAIL line each:
  capture_order  each language graph is captured on an UNPADDED batch first, then replayed on padded rows
  text           a short query, a 240-token passage, a 3,000-token document, in one batch
  image          two aspect ratios, and two images plus text in one input
  audio          6 s and 35 s clips, and text plus audio
  video          five frames
  mixed_batch    all of the above in one call give the same vectors as one at a time
Each line also prints the stock model run in bf16 on the CPU against the same reference: that is the precision
floor of bf16 weights, and the HPU path should sit at it. The default bound, 0.9995, sits just under that floor on
these inputs and far above what the known bugs give (0.962 for a graph captured without its mask, about 0.6 for
HPU SDPA on padded rows).
Inputs are synthetic (patterns, tones, filler text): parity needs exact inputs, not meaningful ones.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import engine as eg  # noqa: E402

RESULTS: list[bool] = []


def gate(name, got, floor, min_cos):
    """`got`: HPU vs fp32 reference; `floor`: the stock model in bf16 on the CPU vs the same reference."""
    ok = bool(min(got) >= min_cos)
    RESULTS.append(ok)
    fmt = lambda xs: ", ".join(f"{c:.6f}" for c in xs)  # noqa: E731
    print(
        f"{'PASS' if ok else 'FAIL'} {name}: cosine {fmt(got)} (min {min_cos}); stock bf16 on CPU: {fmt(floor)}",
        flush=True,
    )


def image(w, h, seed):
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 128 // (w + h))], -1).astype(np.float32)
    a += rng.normal(0, 12, a.shape)
    for _ in range(6):  # a few solid shapes so the image has structure
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(20, 80)
        a[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = rng.integers(0, 255, 3)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def tones(seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * eg.SAMPLE_RATE)) / eg.SAMPLE_RATE
    f0 = 120 + 60 * np.sin(2 * np.pi * 0.7 * t)  # a gliding pitch with harmonics and a syllable-rate envelope
    phase = 2 * np.pi * np.cumsum(f0) / eg.SAMPLE_RATE
    x = sum(np.sin(k * phase) / k for k in range(1, 8)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    return (0.1 * x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


STOCK_BLOCK_CONTEXT = None  # the stock Tensor.unfold version, restored for the reference


def reference_vectors(ref, items):
    from transformers.models.gemma4 import modeling_gemma4 as g4

    out = []
    patched = g4.Gemma4AudioAttention._extract_block_context
    g4.Gemma4AudioAttention._extract_block_context = STOCK_BLOCK_CONTEXT
    with torch.no_grad():
        for p in items:
            dt = ref.dtype
            kw = {"input_ids": torch.tensor([p.ids])}
            if p.images is not None:
                kw.update(pixel_values=p.images.to(dt), image_position_ids=p.image_pos)
            if p.frames is not None:
                kw.update(
                    pixel_values_videos=p.frames.to(dt),
                    video_position_ids=p.frame_pos,
                    num_frames_per_video=torch.tensor([len(p.frames)]),
                )
            if p.audio is not None:
                t = max(len(a) for a in p.audio)
                feats = torch.zeros(len(p.audio), t, p.audio[0].shape[-1])
                mask = torch.zeros(len(p.audio), t, dtype=torch.bool)
                for i, (a, m) in enumerate(zip(p.audio, p.audio_mask, strict=True)):
                    feats[i, : len(a)], mask[i, : len(m)] = a, m
                kw.update(input_features=feats.to(dt), input_features_mask=mask)
            h = ref(**kw).last_hidden_state[0].float()
            out.append(torch.nn.functional.normalize(h.mean(0), dim=-1))
    g4.Gemma4AudioAttention._extract_block_context = patched
    return torch.stack(out)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cos", type=float, default=0.9995)
    a = ap.parse_args()
    global STOCK_BLOCK_CONTEXT
    from transformers import AutoModel
    from transformers.models.gemma4 import modeling_gemma4 as g4

    STOCK_BLOCK_CONTEXT = g4.Gemma4AudioAttention._extract_block_context
    hpu = eg.Engine()  # no warmup: the capture order below is the point
    ref = AutoModel.from_pretrained(eg.MODEL, dtype=torch.float32).eval()
    ref16 = AutoModel.from_pretrained(eg.MODEL, dtype=torch.bfloat16, attn_implementation="eager").eval()
    tok = hpu.tok

    def check(name, items, got=None):
        want = reference_vectors(ref, items)
        got = hpu.embed_items(items) if got is None else got
        gate(name, cos(got, want), cos(reference_vectors(ref16, items), want), a.min_cos)

    # capture_order: the graph for (1, L) is first captured on a row that fills the bucket exactly (no padding).
    word = tok("cluster", add_special_tokens=False)["input_ids"][0]
    padded, got = [], []
    for length in (64, 256, 1024):
        hpu.embed_items([eg.Prepared([tok.bos_token_id] + [word] * (length - 2) + [tok.eos_token_id])])
        row = hpu.tokenize([("Gaudi cards run lazy-mode graphs. " * (length // 8))])[0][: length * 2 // 3]
        padded.append(eg.Prepared(row))
        got.append(hpu.embed_items([padded[-1]]))
    check("capture_order", padded, torch.cat(got))

    filler = "The committee met in the old library to review the budget, the schedule and the new building plans. "
    texts = [
        "task: search result | query: how do I request a node",
        "title: none | text: " + filler * 12,
        "title: none | text: " + filler * 150,
    ]
    text_items = [eg.Prepared(r) for r in hpu.tokenize(texts)]
    print(f"     text lengths: {[len(p.ids) for p in text_items]} tokens", flush=True)
    check("text", text_items)

    img_items = [
        hpu.prepare("<|image|>", images=[image(640, 480, 1)]),
        hpu.prepare("<|image|>", images=[image(360, 720, 2)]),
        hpu.prepare("two pictures: <|image|> <|image|>", images=[image(640, 480, 3), image(480, 480, 4)]),
    ]
    check("image", img_items)

    aud_items = [
        hpu.prepare("<|audio|>", audios=[tones(6, 5)]),
        hpu.prepare("<|audio|>", audios=[tones(35, 6)]),
        hpu.prepare("a recording: <|audio|>", audios=[tones(8, 7)]),
    ]
    check("audio", aud_items)

    frames = np.stack([np.asarray(image(320, 240, 10 + i)) for i in range(5)])
    vid_items = [hpu.prepare("<|video|>", videos=[frames])]
    check("video", vid_items)

    every = text_items + img_items + aud_items + vid_items
    together = hpu.embed_items(every)
    alone = torch.cat([hpu.embed_items([p]) for p in every])
    ok = min(cos(together, alone)) >= a.min_cos
    RESULTS.append(ok)
    print(
        f"{'PASS' if ok else 'FAIL'} mixed_batch: min cosine {min(cos(together, alone)):.6f} over {len(every)} inputs"
    )
    print(f"{sum(RESULTS)}/{len(RESULTS)} gates passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
