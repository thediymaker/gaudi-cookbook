# What Gaudi 2 gets wrong, and the workaround

Defects found below vLLM while porting and serving models on Gaudi 2, in the Habana PyTorch bridge, the
FusedSDPA kernel, the runtime and the driver. Most are silent. Each section gives the symptom, cause, who is
affected, the workaround and how to check. Where the cookbook ships a fix, the section links it.

!!! summary "At a glance"
    - **Stack:** Gaudi 2, Gaudi software 1.24.1, lazy mode, vllm-gaudi v0.26.0 and releases/v0.30.0.
    - **Silent wrong results:** FusedSDPA sinks, windows and fast softmax; fp32 sin/cos; int64 cat with an int32
      cast; `torch.multinomial`; `torch.unique(dim=0)`.
    - **Crashes:** seeded `exponential_`, SIGPIPE, the FusedSDPA bias workspace, "tensor_data is empty",
      order-9 host allocations.
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
  v0.30.0). On one model, seeded needles at 207.5K and 242.3K went from 7/12 to 12/12.
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

## "tensor_data is empty" on a first compile

- **Symptom.** `PT_LAZY ValidateSyncInputTensors: tensor_data is empty` kills the engine on the first compile of
  a large prefill shape in a TP all-reduce; once in four fresh processes in one test near 190K context.
- **Who.** TP models compiling long prefill shapes on live traffic.
- **Workaround.** None confirmed; a restart recovers. Compiling those shapes at warm-up is untested.
- **How to check.** Search logs for the string after long prompts on fresh pods.

## `torch.unique(dim=0)` drops rows above about 500K

- **Symptom.** Data missing after deduplication; 1.4M coordinate rows lost about 8%.
- **Who.** Sparse 3D pipelines and any large on-device dedup.
- **Workaround.** Run large `unique` calls on the CPU.
- **How to check.** Compare row counts with CPU `torch.unique`.

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
