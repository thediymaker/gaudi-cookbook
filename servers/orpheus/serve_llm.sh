#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# vLLM for Orpheus 3B on one Gaudi 2 card, in the image built by servers/orpheus/Dockerfile (vllm-gaudi v0.26.0 with
# the v0.26.0 fix profile and penalty-shapes applied at build time).
#   serve_llm.sh [model dir or id]   (default canopylabs/orpheus-3b-0.1-ft; extra arguments go to vllm serve)
# Only the speech sidecar calls this server, so it listens on 8001 under its own name.
set -euo pipefail
MODEL=${1:-canopylabs/orpheus-3b-0.1-ft}
shift || true

export PT_HPU_LAZY_MODE=1
# Warm every bucket at start: a first-touch graph compile mid-sentence would stall a live audio stream for seconds.
# The bucket ranges below are small, so warmup takes about 150 s.
export VLLM_SKIP_WARMUP=false
# Buckets sized to --max-num-seqs 8, --max-model-len 4096 and --block-size 128; keep them in step. Prompts are one
# text chunk (about 100 tokens).
export VLLM_EXPONENTIAL_BUCKETING=True
export VLLM_PROMPT_BS_BUCKET_MIN=1 VLLM_PROMPT_BS_BUCKET_STEP=1 VLLM_PROMPT_BS_BUCKET_MAX=4
export VLLM_DECODE_BS_BUCKET_MIN=1 VLLM_DECODE_BS_BUCKET_STEP=1 VLLM_DECODE_BS_BUCKET_MAX=8
export VLLM_PROMPT_SEQ_BUCKET_MIN=128 VLLM_PROMPT_SEQ_BUCKET_STEP=128 VLLM_PROMPT_SEQ_BUCKET_MAX=2048
export VLLM_DECODE_BLOCK_BUCKET_MIN=16 VLLM_DECODE_BLOCK_BUCKET_STEP=64 VLLM_DECODE_BLOCK_BUCKET_MAX=256

# --dtype bfloat16: the checkpoint is fp32.
# --no-async-scheduling: with v0.26.0's async scheduler (the default) sampled ids never reach a request's output
#   history, so repetition, presence and frequency penalties do nothing, and Orpheus needs its repetition penalty to
#   end clips instead of trailing on in noise. Sync scheduling makes the penalty tensors change shape every step;
#   the penalty-shapes fix pads them to buckets so the HPU stops recompiling.
# --max-num-seqs 8: with the penalty on, a 16-wide decode falls below real time per stream (cause not found).
# No prefix caching: prompts are short and differ per chunk.
exec vllm serve "$MODEL" \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 8001 \
  --served-model-name orpheus-3b-llm \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  "$@"
