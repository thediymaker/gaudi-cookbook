# Routing

How requests reach model pods matters as much as how many pods there are. A plain
Kubernetes Service balances **connections**, not requests, and knows nothing about queues
or caches. This page covers model-aware routing, what it cannot see, and the chain of
timeouts every request passes.

## Route through a model-aware gateway

Send every client through one gateway that picks a pod per request (for example an
Envoy-based inference gateway with a routing plugin, such as AIBrix or the Gateway API
inference extension). Clients that call a model's Service directly bypass it:

- kube-proxy balances per connection, so a client with a keep-alive pool sticks to the pods
  that existed when it connected. In one burst of several thousand requests, the pods the
  autoscaler added received nothing, because every connection had been opened before they
  existed. A direct client defeats both routing and autoscaling.
- Restrict direct access to model Services with a NetworkPolicy that admits only the
  gateway (and Prometheus).

## Prefix-aware routing

vLLM's prefix caching (`--enable-prefix-caching`) skips prefill for a prompt prefix that
is already in a pod's KV cache. It only helps if requests sharing a prefix reach the same
pod. A prefix-aware router hashes prompt prefixes and prefers the pod that served them.

- **Use prefix-aware routing for models with prefix caching on,** and least-request for
  models with it off. The router remembers its own routing decisions, not the real cache
  state, so on a model without caching it just packs shared-prompt requests onto one pod.
- **Measure the hit rate** per model: `vllm:prefix_cache_hits_total /
  vllm:prefix_cache_queries_total`. On hybrid models (linear attention or state-space
  layers mixed with attention), `1 - computed prefill tokens / prompt tokens` is the more
  honest number.
- Prompts shorter than one cache block never hit; hybrid models can have large blocks.

## Saturation-aware routing

Pure prefix affinity sends a request to the pod with the best cache match even when that
pod is full and its siblings are idle. Treat a pod as **full** and skip it when:

- it has requests waiting (`num_requests_waiting >= 1`), or
- its running requests reached `--max-num-seqs`, or
- for long-prompt models, its projected KV use (current KV use plus prompts already sent but
  not yet visible in its metrics, plus this prompt) would pass a threshold.

The KV test matters for long prompts. A pod can accept several long requests with free
slots and no queue, then run out of KV blocks as they decode. vLLM preempts some of them,
and a preempted request can only resume when its whole context fits again, so it waits for
another request to finish while sibling pods sit idle. Slot-based saturation checks do not
see this.

When every pod is full, fall back to the shortest queue.

### Check what the router believes

A saturation filter is only as good as its inputs. Two ways it went wrong on a live fleet,
both silent:

- **max-num-seqs read from the wrong place.** The router took each model's
  `--max-num-seqs` from the pod's container command, and on two models it matched a flag
  inside a shell comment (`2`) instead of the real argument (`4`). Every pod with two
  requests running looked full, so 52% of that model's routing decisions found every pod
  saturated and fell back to least-request, with no prefix awareness at all. Give the
  router the value explicitly (for example a pod label or annotation rendered from the
  same variable as the flag) rather than parsing the command line; when you must parse
  it, strip comments and let the last occurrence win, as the argument parser does. Check
  it against the `max_num_seqs` the engine prints in its startup arguments.
- **A lifetime in-flight counter that drifts.** The router's own count of in-flight
  requests per pod sat at 37 to 41 on one model while vLLM reported at most 8 running.
  Every pod read as full, about 90% of decisions went to the "all saturated" fallback, and
  the pod with the lowest count (a newly started one) took nearly all traffic, with 16
  running and 20 or more waiting for hours while older pods idled. Restarting the gateway
  reset the counts. Base the filter on scraped `num_requests_running` and
  `num_requests_waiting`, plus a short-lived count of requests routed since the last
  scrape, rather than a counter that only ever adds and subtracts.

Log the router's decision type per request (prefix hit, filtered as full, all saturated)
and alert when "all saturated" becomes a large share while the pods themselves report free
slots.

### One prefix index per gateway replica

A prefix-aware router remembers which pod served which prefix, but each gateway replica
keeps its own index unless they share state. A conversation whose second turn reaches a
different gateway replica is routed as if new, often to a pod without its cache. On one
long-prompt model, turn 1 went through one gateway replica and turn 2 through another, to
a different pod, and the turn prefilled cold (75 s).

To check, compare the pod chosen for consecutive turns of one conversation in the gateway
logs before suspecting vLLM's cache. Fixes: share the index between gateway replicas, or
route on a stable conversation key that every replica computes the same way (a session
header, or a hash of the system prompt and first user message) with rendezvous hashing over
the ready pods, and spill to another pod only when the home pod is full. In a replay of real
agent traffic on one model this raised the cache hit rate from 0.44 to 0.79; on a test
deployment, 0.52 to 0.80, with the same pod for 99% of turns (was 33%) and median TTFT 4.3
to 3.4 s.

**When capacity is the limit, routing does not help.** Two weeks later the same model ran
at its replica ceiling with about half its pods at max-num-seqs. A canary of the anchored
router kept only 31% of turns on their home pod (the rest spilled because home was full),
and the hit rate was unchanged at about 0.5. Check occupancy first: if most pods are full,
add capacity before tuning the router.

### Cap runaway generations

A client that sends no `max_tokens` (or a huge one) to a model whose generation config has
no cap can generate until `--max-model-len`. On one model, prompts of about 1.1K tokens
produced 18K to 130K tokens each; the longest ran 55 minutes. Each such request held up to
64% of a pod's KV cache, made vLLM preempt the other requests on that pod (about 100
preemptions in 30 minutes on one pod), and queued its users for 27 to 57 minutes.

How it shows: KV use high while running requests are few, preemptions climbing on one pod,
and a 99th percentile of generation tokens far above the median
(`vllm:request_generation_tokens`).

Cap it in the engine, per model:

```bash
--override-generation-config '{"max_new_tokens": 32768}'
```

In vLLM 0.26 this cap is applied as a hard minimum over the client's `max_tokens`: a
request asking for 100000 tokens got 32768. After the cap rolled, running requests on that
model fell from about 50 to 2 and its autoscaler went from 12 replicas to 4. Pick the cap
from the longest legitimate outputs (reasoning models need more), and say it in the
model's documentation, since clients see a `length` finish at the cap.

## Bulk traffic

Batch and bulk clients compete with interactive users in a way that a gateway queue does
not show. A model's slots fill with long prompts, their prefill chunks interleave with
every stream's decode, and everyone's tokens per second drop while nobody waits at the
gateway. On one model, a single client sending about 100,000 requests a day of about 13K
tokens each took the median speed of long replies from 20 to 30 tok/s down to about 7,
and interactive chat on the same model to 6 to 10 tok/s.

Admission ideas that apply to any gateway in front of vLLM:

- **Cap the share of slots** (`pods x max-num-seqs`) that background work may hold, for
  example half.
- **Weight background requests by prompt size**: hold `max(1, ceil(prompt_tokens / 4096))`
  slots per request, clamped to the background cap, so a 40K-token prompt counts as ten.
  With 24 prompts of about 10K tokens sent as a batch against a cap of 12 slots, 4 ran at a
  time instead of 12, the rest queued, all completed, and an interactive request sent
  meanwhile answered in 0.4 s.
- **Bulk traffic sent as interactive** (a script on a normal API key) bypasses both. Spot
  it per key: request rate, prompt size and one-request sessions. Give such keys a
  background class or a per-key concurrency limit.
- Dashboards that count requests rather than weighted slots will show a queue of
  background work next to "idle" capacity. That is the weighting working; label it.

## What least-request cannot see

A least-request router counts only the requests **it** has in flight per pod.

- Load sent from inside a pod (warm-up or benchmarks through `kubectl exec`) is invisible,
  so users keep being routed onto a pod that is busy compiling or prefilling.
- Freshly started pods look idle and are preferred. If they still have to compile their
  first shapes, the first users on them stall. Warm pods before they are Ready
  ([Start-up and warm-up](startup-and-warmup.md#warm-up)).
- Gateway metrics scraped through a Service come from one gateway replica at a time, so
  totals read from them are unreliable. Scrape gateway pods individually.

## Long and short context

`--max-model-len` by itself reserves no KV memory: the KV pool is fixed (by
`--gpu-memory-utilization` or `--num-gpu-blocks-override`) and paged attention allocates
blocks per sequence on demand. On Gaudi a larger window does enlarge bucket ranges, and so
compile time and graph memory, but not the KV reservation.

The real cost of mixing very long and short requests on one pod is latency: with chunked
prefill, a long prefill mostly interleaves with short requests, but the short request that
arrives while the long one is still in its first chunks can wait for most of the prefill.
Measured on one model, median short-request latency barely moved while the worst case went
from a fraction of a second to about a minute, roughly one stalled short request per long
prompt.

If that tail matters, split by context: a pool with a smaller window for normal traffic,
and one or two replicas with the large window. Give them different served model names and
different router labels (a gateway treats one label as one scalable pool), and let the API
proxy retry on the context-length error: the small pool rejects an oversized request
before the engine does any work, and the proxy resends it to the large pool. This is
per request, so a conversation moves to the large pool on the exact turn that first
exceeds the small window. Measure the stall on your model before doing this; it scales with
prefill cost.

## The timeout chain

A request passes several timers, and the shortest one wins. Every one of them must agree.

| Hop | Typical setting | Common trap |
| --- | --- | --- |
| Client | read timeout | Non-streaming clients wait for the whole generation |
| Edge load balancer | backend or tunnel timeout | Shorter than the gateway |
| API proxy (keys, quotas) | per-request timeout, retries | Default retries re-run a whole long generation after a timeout, multiplying wasted decode |
| Gateway route | route timeout | Default route timeouts can be seconds |
| Gateway stream idle | idle timeout | Envoy's default stream idle timeout is 5 minutes: a non-streamed request with no bytes for 300 s is cut with a 504 |
| vLLM | none by default | |

- **Set one limit for the whole chain** (for example 60 minutes) and keep per-model
  overrides empty, so it holds everywhere.
- **No automatic retries for long generations.** A retry after a timeout runs the whole
  generation again. On one model, most generated tokens were being thrown away by
  timeout-and-retry cycles, and the retries alone inflated its replica count.
- **Clients that read with a timeout:** some HTTP client libraries apply the read timeout
  to the wait for response headers too, which for a non-streamed request is the whole
  generation.
- **End a cut stream with an error event,** not with `finish_reason: "stop"` and `[DONE]`:
  a client cannot tell a truncated answer from a complete one otherwise.
- **The gateway's own shutdown grace** bounds requests in flight during a gateway rollout
  or scale-down, just like a model pod's grace.
- **Request body limits:** large base64 images can exceed a gateway's buffer limit and
  return 413, which is not a capacity problem.

When queues grow while requests "time out", search the gateway logs for idle-timeout cuts
and the proxy logs for timeouts and retries before adding capacity.
