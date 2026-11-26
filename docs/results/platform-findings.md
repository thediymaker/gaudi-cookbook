# What Gaudi 2 gets wrong, and the workaround

Defects found below vLLM while porting and serving models on Gaudi 2, in the Habana PyTorch bridge, the
FusedSDPA kernel, the runtime and the driver. Most are silent. Each section gives the symptom, cause, who is
affected, the workaround and how to check. Where the cookbook ships a fix, the section links it.

!!! summary "At a glance"
    - **Stack:** Gaudi 2, Gaudi software 1.24.1, lazy mode, vllm-gaudi v0.26.0 and releases/v0.30.0.
    - **Silent wrong results:** FusedSDPA sinks, windows and fast softmax; fp32 sin/cos; int64 cat with an int32
      cast; `torch.multinomial`; `torch.unique(dim=0)`.
    - **Crashes:** seeded `exponential_`, SIGPIPE, the FusedSDPA bias workspace, "tensor_data is empty" after a
      device allocation failure, order-9 host allocations, cards that hang at their first collective, an allocation
      failure on one rank that looks like an all-rank hang.
    - **Reproducibility:** speculative decoding ignores `seed`; compiled samplers recompile per seeded request;
      a single HPU request is not a reference (only fp32 on the CPU is).
    - **Benchmarks:** an HPU graph of only custom ops replays nothing.
    - **Slow FP8:** a view on a block-FP8 linear's output splits the decode graph at TP > 1.
    - **Throughput:** port-9 re-training. A driver reload took TP8 decode on one node from 7.56 to 15.45 tok/s
      (same-node A/B).

## Port-9 re-training stalls TP8 decode

- **Symptom.** Multi-card decode pauses for 11 to 21 s about every 30 s; the kernel log shows `Port 9 link
  down` / `link up`.
- **Cause.** Each OAM pair of cards (0 and 4, 1 and 5, 2 and 6, 3 and 7) uses internal scale-up NIC port 9 as
  one of its links. That channel is electrically marginal, and driver auto-negotiation re-trains it about every
  10 s under live HCCL traffic.
- **Who.** 8-card jobs, and 2- or 4-card jobs holding both cards of a pair. The device plugin does not allocate
  by topology, so treat every node as affected.
- **Workaround.** Turn auto-negotiation off on port 9 only (`nic_auto_neg_mask=0xFFFDFF`) with an in-place
  driver reload on a drained node:
    1. Stop everything holding `/dev/accel*` (device plugin, exporters).
    2. `modprobe -r habanalabs_compat habanalabs_ib habanalabs_en habanalabs`, then `modprobe -r habanalabs_cn`.
    3. `modprobe habanalabs_cn nic_auto_neg_mask=0xFFFDFF`, `modprobe habanalabs nic_auto_neg_mask=0xFFFDFF`,
       then `habanalabs_en` and `habanalabs_ib`.
    4. Re-apply the scale-out IP addresses; the reload recreates the per-card network devices and drops them.
    5. Restore the card order. The reload probes cards in parallel, so the `accelN` to module mapping comes out
       arbitrary and 2- or 4-card pods fail with "Device not found". Unbind all cards and bind them back one at
       a time in the module order a cold boot gives.
    6. Restart the device plugin.

    The mask on the kernel command line (`habanalabs.nic_auto_neg_mask`, `habanalabs_cn.nic_auto_neg_mask`)
    persists the setting but is **not enough**: a node booted with it still stalled (4.24 tok/s, 7 stalls) until
    a reload (14.70 tok/s, 0 stalls; same-node A/B, same TP8 benchmark). **Reload after every boot.** Do not
    drop the port with `nic_ports_mask`: HCL expects three scale-up NICs per pair and TP8 hangs at init.
- **How to check.** `/sys/module/habanalabs_cn/parameters/nic_auto_neg_mask` shows the mask. Port 9 is
  InfiniBand port 10, so `/sys/class/infiniband/hbl_*/ports/10/state` should be `ACTIVE` and
  `ports/10/hw_counters/pcs_link_toggles` should not move over 15 s under load. A node can pass this and still
  stall, so also watch per-stream inter-token gaps. A node that keeps toggling after a reload is marginal
  hardware; keep multi-card jobs off it.

## FusedSDPA ignores `sinks` and `window_size`

- **Symptom.** Nothing visible; quality drops on long or structured output.
- **Cause.** On 1.24.1 FusedSDPA accepts both arguments and returns attention without them, in every call mode
  tried.
- **Who.** gpt-oss, MiMo-V2 and any model whose prefill passes sinks or a window to FusedSDPA.
- **Workaround.** A sink is one extra key whose logit is the sink and whose value is zero; a window needs an
  explicit mask or a banded kernel. [sink-once](../fixes/sink-once.md) fixes sinks in v0.26.0 prefill.
- **How to check.** On small random shapes, compare FusedSDPA with and without the arguments against a CPU
  reference.

## FusedSDPA fast softmax is wrong when a few keys dominate

- **Symptom.** Long-context retrieval fails (garbage, loops, missed needles); short prompts look fine.
- **Cause.** vllm-gaudi runs prompt attention with `softmax_mode="fast"` unless fp32 softmax is on (qwen2-family
  only by default). On deep-layer tensors of a large MoE at 64K context, the worst fast-mode row had cosine
  0.086 and -0.021 against fp32; the accurate bf16 mode (`"None"`) stayed above 0.9997.
- **Who.** Long-context models with sharp attention in deep layers.
- **Workaround.** Use `"None"` for prompt attention (a small patch; there is no switch upstream). One
  8192-query chunk against 70K keys: fast 27.6 ms, `"None"` 33.5 ms, fp32 125 ms (per-call, one card). Dense
  attention in `"None"` passed every needle from 16K to 128K; fast failed from 32K.
- **How to check.** Recompute a deep layer's attention in fp32 on the CPU and compare per row, or run needles
  past 32K.

## FusedSDPA with an explicit bias keeps a large workspace

- **Symptom.** Allocation failures or NaN in long prefills with an explicit bias.
- **Cause.** FusedSDPA keeps a workspace proportional to query rows x keys. Separately, byte offsets wrap at 2^31
  bytes per bias plane (vllm-gaudi now tiles around that).
- **Who.** Models needing explicit prefill masks (windows, sparse attention) at long context.
- **Workaround.** Tile query rows; tiled and single calls agree to bf16 rounding.
- **How to check.** Log `torch.hpu.memory_stats()` across a long prefill. No standalone repro yet.

## fp32 sin and cos are inaccurate for large arguments

- **Symptom.** Recall fails past roughly 150K to 200K positions.
- **Cause.** Device fp32 `sin` / `cos` lose accuracy above about 1.5e5 to 2e5. At positions 196K to 205K the
  fastest rotary frequency had random error (median pi/2); the others were within about 5e-4 rad.
- **Who.** Models that build RoPE tables on the device.
- **Workaround.** Build tables on the CPU in float64: [rope-cpu-cache](../fixes/rope-cpu-cache.md) (upstream in
  v0.30.0). On one model, seeded needles at 207.5K and 242.3K went from 7/12 to 12/12. Pass `device="cpu"`
  to every tensor the table is built from: vLLM constructs the model under `torch.device("hpu")`, and a float64
  tensor created there is computed in fp32 (`cos(200000)` came back as 0.0).
- **How to check.** Compare `torch.cos(torch.arange(N).float().to("hpu"))` with CPU float64.

## int64 cat then an int32 cast reads back corrupted

- **Symptom.** `top_logprobs` ids wrong or out of range under concurrency; ranks disagree.
- **Cause.** In lazy mode, `torch.cat` of int64 tensors then `.to(torch.int32)` reads back wrong under load.
  Casting before the cat is correct.
- **Who.** Lazy-mode deployments serving logprobs.
- **Workaround.** [gather-logprobs-int32](../fixes/gather-logprobs-int32.md).
- **How to check.** Reproduced only inside vLLM so far: check returned ids against the vocabulary.

## Seeded sampling segfaults with lazy collectives

- **Symptom.** A worker segfaults in `exponential_` under requests with `seed` and temperature above 0; about
  one death per 40 to 60 seeded pod-minutes at saturation.
- **Cause.** Per-row `exponential_` with an HPU `torch.Generator` races in the bridge's collective handling when
  `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` (vllm-gaudi's lazy-mode default).
- **Who.** Lazy-mode multi-card generation deployments.
- **Workaround.** [seeded-sampling](../fixes/seeded-sampling.md); output stays deterministic but changes.
- **How to check.** Load test with seeded requests.

## `torch.multinomial` samples a distorted distribution

- **Symptom.** Total-variation distance about 0.10 from the target, where sampling noise is about 0.012.
- **Who.** Custom generate loops (speech, image, research code). vLLM's sampler does not call it.
- **Workaround.** Gumbel-max on the device, `argmax(logits + Gumbel noise)`, which is exact.
- **How to check.** Histogram many draws from a fixed distribution.

## SIGPIPE is treated as a device fatality

- **Symptom.** `synStatus 31 [Synapse is being terminated due to fatality]` after a client hang-up; every later
  request fails.
- **Cause.** The runtime's own SIGPIPE handler treats the signal as fatal.
- **Who.** Services driving a card directly.
- **Workaround.** Block SIGPIPE before importing torch:
  [Fatal device errors](../kubernetes/health-and-failure.md#fatal-device-errors).
- **How to check.** Write to a closed pipe after HPU init; expect `BrokenPipeError`.

## Opening a card needs an order-9 host allocation

- **Symptom.** `Device acquire failed`, with `page allocation failure: order:9` in the kernel log.
- **Cause.** Card open needs a contiguous 2 MiB host block; page cache from weight loads fragments memory. The
  card is healthy.
- **Who.** Busy nodes, especially under bin-packing.
- **Workaround and check.**
  [Host memory fragmentation](../kubernetes/health-and-failure.md#host-memory-fragmentation).

## "tensor_data is empty" after a device allocation failure

- **Symptom.** `PT_LAZY ValidateSyncInputTensors: tensor_data is empty` (often after `PT_BRIDGE Exception in Launch
  thread`) kills the engine. The Python stack points at whatever host sync came next (`sample_tokens`,
  `take_draft_token_ids`, a `.cpu()` in model code), so deaths seem to come from different places.
- **Cause.** A graph launch could not allocate device memory, so its outputs were never produced. `HABANA_LOGS`
  shows `PT_DEVMEM Allocation failed for size N MB` just before. On one model, three deaths on different ranks all
  failed the same 12 MiB allocation at 24% KV use; the bridge's workspace had grown to 4.13 GiB per card after
  long prompts and never shrank. The same message also appeared once in four fresh processes on the first
  compile of a prefill shape near 190K context; that case's runtime logs were not kept.
- **Who.** TP models serving long prompts, especially with large prefill chunks.
- **Workaround.** Keep `HABANA_LOGS` on a volume, log memory statistics, and shrink the largest single allocation:
  [Crash forensics](../kubernetes/health-and-failure.md#crash-forensics).
- **How to check.** Search `HABANA_LOGS` for `Allocation failed` right before the death.

## `torch.unique(dim=0)` drops rows above about 500K

- **Symptom.** Data missing after deduplication; 1.4M coordinate rows lost about 8%.
- **Who.** Sparse 3D pipelines and any large on-device dedup.
- **Workaround.** Run large `unique` calls on the CPU.
- **How to check.** Compare row counts with CPU `torch.unique`.

## Speculative decoding ignores the request `seed`

- **Symptom.** The same prompt with the same `seed` and temperature above 0, sent twice, gives different text on
  models served with MTP or EAGLE drafts: 0 of 6 pairs identical on three such models, against 6 of 6 on models
  without speculative decoding.
- **Cause.** On the stock HPU path, the rejection sampler draws its accept tests and recovery tokens from the
  global RNG. The per-request generator only reaches the sampler that picks the bonus token.
- **Who.** Speculative decoding with sampling, for clients that rely on `seed` for reproducible output.
- **Workaround.** Draw the seeded rows' accept and recovery noise from a counter keyed by (seed, step), as
  [seeded-sampling](../fixes/seeded-sampling.md) does for normal sampling. With that change, 6 of 6 repeats
  matched on each of two pods. The cookbook does not ship it yet; until then, document that `seed` is not
  honoured on speculative models.
- **How to check.** Send the same seeded request six times at low load and compare the texts. Matches between
  different pods, or at different concurrency, are not expected (see the next section).

## A single HPU request is not a reference

- **Symptom.** The same greedy prompt gives different tokens at C=1 and C=8, or with different prefill chunk
  sizes, and the difference looks like a bug in batching or chunking.
- **Cause.** bf16 rounding depends on the shape of each matmul, and in MoE models also on which other tokens
  share an expert in the step. Where two candidate tokens are nearly tied, any change in batch composition can
  flip the choice. A single-request, single-chunk, eager HPU run is one more bf16 sample, not ground truth.
- **Evidence.** On one MoE model with a looped layer stack, every HPU variant (eager, graphs, chunk sizes 512 to
  2048, single request) sat 0.06 to 0.6 nats from an fp32 CPU reference at the token where they diverged, the
  same band as the model's own bf16 CPU implementation (0.035 to 0.21). Which variant missed changed from prompt
  to prompt. Eight copies of one request at the exact C=8 shape stayed in the single-request band; the larger gap
  at real C=8 came from the other requests' tokens through MoE expert grouping. fp32 residuals and an fp32 router
  did not close the gap.
- **Who.** Anyone judging a port or a batching change by comparing HPU outputs with each other.
- **Workaround.** Compare against fp32 on the CPU. If the HPU differs from fp32 by no more than a bf16 CPU run
  does, and only at near-ties (a small logit margin between the top two tokens), it is precision, not a bug.
- **How to check.** Record the top-2 margin at the first differing token; a real bug shows up as a divergence
  with a large margin or as a wrong answer, not as a flip between two near-equal tokens.

## An HPU graph of only custom ops replays nothing

- **Symptom.** A microbenchmark that captures a custom TPC op in an HPU graph reports impossibly fast replays, and
  the outputs are stale.
- **Cause.** A captured graph whose only nodes are custom ops replays as empty.
- **Who.** Kernel benchmarks and parity tests that wrap custom ops in HPU graphs.
- **Workaround.** End every captured function with a stock op on the output, for example `z * 1.0`.
- **How to check.** Compare the replayed output with an eager run on fresh inputs.

## Compiled samplers recompile for every seeded request

- **Symptom.** In compile mode (`PT_HPU_LAZY_MODE=0`), each new request with a `seed` adds about 0.2 s to
  TTFT (seeded 0.42 s against unseeded 0.19 s on one model), and the log eventually reports the recompile
  limit for the sampler.
- **Cause.** Dynamo guards on each request's `torch.Generator` object, so every new seeded request is a new guard
  and a recompile, until the limit is hit and that function runs eagerly.
- **Who.** Compile-mode models with seeded requests. Seen on one model on releases/v0.30.0; two compile-mode
  models on v0.26.0 showed no difference. Lazy mode is not affected, because vllm-gaudi turns `torch.compile`
  into a no-op there (which also hides the bug from a lazy-mode test).
- **Workaround.** Run batches that contain seeded rows through the uncompiled sampler (`sampler._orig_mod`). The
  sampler is a small part of a step: decode speed stayed at 22 to 23 tok/s, and seeded TTFT came back to
  0.102 s against 0.101 s unseeded.
- **How to check.** Compare seeded and unseeded TTFT on an idle pod, and search the log for `recompile_limit`.

## Some cards hang at their first collective

- **Symptom.** Every pod placed on the same cards of one node (for example cards 4 to 7) hangs at its first
  collective and restarts, while the node's other cards and other nodes serve the same model normally. One such
  node caused 119 restarts of one model in a day; two nodes showed it within a week.
- **Cause.** A card or baseboard fault on those slots. Single-card tests can pass.
- **Who.** Any multi-card pod that lands on those cards.
- **Workaround.** Cordon the node, record the slots (PCI bus ids), and report it as a hardware fault:
  [Node fault patterns](../kubernetes/health-and-failure.md#node-fault-patterns).
- **How to check.** Group restarts by node and card set; a crash loop that follows the cards, not the model or the
  image, is this.

## Reading the DFA dump of an all-rank "No progress" hang

- **Symptom.** Every rank of a TP job reports "No progress" and times out after 600 s; bare HCCL tests pass.
- **Method.** Synapse writes a device failure analysis into `HABANA_LOGS` (`dfa_log.txt`, `dfa_nic_info.txt`,
  `scal_log.txt`).
    1. Point `HABANA_LOGS` at a volume that outlives the container (check that your launcher does not override
       it); otherwise the next restart replaces the dump.
    2. Healthy cards have compute-stream submissions done equal to sent and wait in the next collective, with
       queue pairs showing producer index ahead of consumer index. The culprit is one graph behind
       (done = sent - 1) with no pending network work.
    3. Check its engines. In the case found this way, one MME had drained its queue but never went idle: stuck
       mid-graph, it never posted its sends.
    4. Rule out the network: no queue-pair errors, link flaps, ECC or driver events.
    5. A second hang on the same card and engine, in a different graph, makes it a card fault. Cordon and file
       the dumps with `hl-smi -q`.
- Read the dump before reloading drivers or blaming links, and do not condemn a node on one dump.

## A device allocation failure on one rank looks like a hang

- **Symptom.** Every rank of a TP server reports "Synapse detected a device critical error ... No progress error"
  about 10 minutes after the last step, as for a card fault. Restarted, the server runs for hours before it happens
  again, more often under long contexts and many concurrent requests.
- **Cause.** One rank failed a small device allocation in an HPU graph replay ("PT_DEVMEM Allocation failed", a few
  MB). That rank stops; the others are already in the next all-reduce and wait until the device times out. In the
  case found, the memory went to decode HPU graphs: with `VLLM_SKIP_WARMUP=true` every new decode shape is captured at
  first use, the graph cache never drops one, and each graph keeps memory in proportion to its block list (on a 65B
  dense model with 80 layers at TP 4, 0.31 MiB per block, 1.25 MiB per sliding-window block and 0.2 MiB per row: one
  graph for 32 requests at 64K of context kept about 5.5 GiB). Pad buckets make many more shapes than exponential ones.
- **Who.** Lazy-mode servers that capture graphs at run time, with long contexts or many shapes.
- **Workaround.** Bound the graphs. [hpu-graph-lru](../fixes/hpu-graph-lru.md) caps their number, which is enough
  when graph sizes are similar; where they span orders of magnitude, bound their predicted size instead, as the
  [ripple-65b port](../ports/ripple-65b.md#decode-graph-memory) does. More `VLLM_GRAPH_RESERVED_MEM` only delays it.
- **How to check.** Search every rank's log, not only rank 0's, for "PT_DEVMEM" in the minutes before the first "No
  progress"; log `torch.hpu.memory_stats()` per step and watch in-use memory climb with new decode shapes.

## A block-FP8 linear view splits the decode graph at TP greater than 1

- **Symptom.** A dense model with block-FP8 weights decodes slower than its BF16 checkpoint at TP > 1 (20 against 30
  tok/s per stream on a 65B model at TP 4), while a device profile shows FP8 doing less work (12.7 against 18.4 ms per
  step).
- **Cause.** vllm-gaudi's block-FP8 linear, with its default per-channel requantization, returns
  `out.to(input.dtype).view(*input.shape[:-1], -1)`. For 2-D BF16 inputs both are no-ops, but the view is all-reduced
  in place after o_proj and down_proj and splits the HPU graph around each collective: 7 recipes per layer instead of
  3, each followed by a host event sync.
- **Who.** Block-FP8 dense models at TP > 1 on `releases/v0.30.0`. Checked on one model only.
- **Workaround.** Return the GEMM output itself for 2-D inputs already in the output dtype (the
  [ripple-65b port](../ports/ripple-65b.md) does this behind `RIPPLE_FP8_VIEW_FIX`); FP8 then decodes as fast as BF16.
- **How to check.** Count recipe launches per decode step with and without the change: on two cards, 4 layers took 28
  recipes against 12.
