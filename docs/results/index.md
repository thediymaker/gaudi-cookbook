# Results

Studies, features and findings from running a large Gaudi 2 fleet that are not tied to a single model port. Every
number says how it was measured.

| Page | What it is | Headline |
| --- | --- | --- |
| [Platform findings](platform-findings.md) | What Gaudi 2 gets wrong, and the workaround: NIC auto-negotiation stalls, FusedSDPA limits, fp32 trig, lazy-mode casts, sampling, runtime signal handling | Port-9 auto-negotiation fix: 7.56 to 15.45 tok/s on an affected node (same-node A/B) |
| [Prefix caching on Gated DeltaNet hybrids](gdn-prefix-caching.md) | An in-page state layout that makes prefix caching correct on Qwen3.5/3.6-style hybrids | Later-turn TTFT 19.8 to 2.6 s in an agent replay (same-pod A/B) |
| [Anatomy of a large-MoE decode step](moe-decode-step.md) | An 8-rank critical-path method for expert-parallel MoE decode, with the fixes it found | Expert-parallel stragglers explained; EP 4 x TP 2 layout +5% to 11%; 51.6 to 57.8 tok/s, bit-identical |
| [Two-node pipeline parallel](two-node-pipeline.md) | PP 2 x TP 8 across two nodes over the host NIC, a long-context tier | KV pool 709,888 tokens vs about 346K on one node; needles pass at 637K |
| [Gemma-4-E2B fast prefill](gemma-4-e2b-prefill.md) | KV-shared layers skip work in prefill, ported to HPU | 72.8k to about 170k tok/s prefill on one card (same card) |
| [Fast text-to-speech](tts-on-gaudi.md) | Qwen3-TTS, Chatterbox and Orpheus on Gaudi 2, with reusable patterns for HF generate loops | Qwen3-TTS about 8x real time with 50 ms to first audio; Chatterbox Turbo 10x to 13x |
| [EmbeddingGemma 2 without vLLM](embeddinggemma-2.md) | A custom server for a bidirectional multimodal embedder | About 145k to 235k tok/s per card for text; cosine of at least 0.9999 against a CPU fp32 reference |
