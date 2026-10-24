# pipeline-parallel

Source: [`fixes/pipeline-parallel/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/pipeline-parallel)

Make `--pipeline-parallel-size` greater than 1 work on the HPU runner, with the stages on different nodes linked
over the ordinary host network. Opt-in feature, inert at pipeline size 1.

| | |
| --- | --- |
| Affects | anyone who needs a model split by layers across nodes, for example a checkpoint that does not fit one 8-card node |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm_gaudi/v1/worker/hpu_model_runner.py`, `vllm_gaudi/v1/worker/hpu_worker.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/model_executor/model_loader/weight_utils.py` |
| Upstream | not filed yet (vllm-gaudi has no pipeline-parallel path in either release) |
| Test | `test_pipeline_parallel.py` (stdlib; the source tests need a release's sources) |

## Problem

The model classes on both releases already split their layers by pipeline rank and return
`IntermediateTensors` on a non-last stage, and the HPU runner's `_update_states` already takes the scheduler's
token ids on non-last ranks. What is missing is the runner moving the hidden states between stages:
`_execute_model_generic` calls the model without `intermediate_tensors`, and a non-last stage would try to
compute logits from `IntermediateTensors`. Three smaller things block it as well: Habana's post-forward module
hook loops forever on `IntermediateTensors` (no `__iter__`), the worker's start-up check runs an HCCL all-reduce
across all ranks (which needs RoCE scale-out links between the nodes), and the scheduler hands every running
request to one step, so the stages take turns instead of overlapping.

## Symptom

With `--pipeline-parallel-size 2` the engine stops on the first request of a non-first stage with
`assert intermediate_tensors is not None` in the model's forward, or hangs at 100% CPU in the first forward of the
first stage, or fails the start-up all-reduce when the nodes have no RoCE scale-out links.

## Who is affected

Anyone who tries pipeline parallelism on these releases. Tensor and expert parallelism on one node are not
affected, and with `--pipeline-parallel-size 1` the fix calls the original methods.

## The fix

- **Runner.** A non-last stage's `sample_tokens` prepares the step with the same helpers, in the same order, as
  the full method, runs the same padded forwards and sends each forward's `IntermediateTensors` to the next
  stage; it computes no logits and samples nothing (the scheduler returns the sampled tokens, which needs
  `--no-async-scheduling`). A non-first stage receives the previous stage's tensors before each forward. When the
  [prompt-order](prompt-order.md) fix is applied, the non-last stages run its prompt sort too, so every
  stage sees the same batch order.
- **Transfer.** CPU tensors over gloo on the pipeline group's CPU group (set `GLOO_SOCKET_IFNAME` to the host
  NIC). Each forward sends a fixed 64-integer header (magic, sequence number, dtypes, shapes) and one payload.
  Payloads of 1 MiB or more (prefill chunks) are split: each tensor-parallel rank sends its share over its own
  connection and the receiving stage all-gathers the pieces over its local HCCL group. The receiver checks the
  sequence number and that the token count equals its own padded batch, so a pipeline that falls out of step
  stops instead of decoding another batch's hidden states. It also waits for the received tensors on the device
  before queueing the forward.
- **HPU graphs.** `IntermediateTensors` gets an `__iter__`, and a non-last stage's outputs are cloned inside the
  captured graph (with tensor parallelism they come from in-place collectives, which a replayed graph with the
  tensor cache disabled frees).
- **Worker.** The start-up world check runs over gloo; HCCL is checked on the node-local tensor-parallel group.
  Set `VLLM_HPU_PP_HCCL_WORLD_CHECK=1` to keep the original HCCL check on nodes with scale-out links.
- **Scheduler.** With pipeline parallelism and no async scheduling, one step schedules at most
  `ceil(running / pp)` running requests, so the requests form microbatches that follow each other through the
  stages. `--max-num-seqs` stays the total over all microbatches.
- **Loader.** Each stage skips the checkpoint tensors of the decoder layers it does not own before reading them
  (stock vLLM reads every tensor and drops the other stage's layers afterwards). Text-only configurations only.

`VLLM_HPU_PP_STATS_EVERY=N` (default 0, off) logs the transfer counters of tensor-parallel rank 0 of each stage
every N messages.

## Serving

Two nodes, one vLLM instance, vLLM's multiprocessing executor across nodes (no Ray). The node with rank 0 runs the
API server and pipeline stage 0; the other node is headless:

```bash
python3 fixes/apply_all.py --profile v0.30.0 pipeline-parallel || exit 1
export GLOO_SOCKET_IFNAME=<host NIC> VLLM_HOST_IP=<this node's IP on it>
# node 0
vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2 --no-async-scheduling \
  --distributed-executor-backend mp --nnodes 2 --node-rank 0 --master-addr <node 0 IP> --master-port 29555 ...
# node 1: the same command with --node-rank 1 --headless
```

`VLLM_PP_LAYER_PARTITION` (a vLLM setting) moves the stage boundary, for example `42,36` for 78 layers. Models
whose layers reuse state from an earlier layer (the shared indexer layers of DeepSeek sparse attention) must start
each stage on a layer that computes that state.

## Verify

```bash
python3 fixes/pipeline-parallel/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/pipeline-parallel
```

On hardware: serve the same model with the same flags at `--pipeline-parallel-size 1` and 2 and compare greedy
answers and teacher-forced prompt logprobs. With Qwen3-30B-A3B-Instruct-2507 at TP 2 on each of two nodes (pod
network, `--no-async-scheduling`), both releases gave identical greedy answers (12 of 12) and identical
teacher-forced top-1 tokens over 8,441 positions (NLL 1.4425 both) against TP 2 on one node.
