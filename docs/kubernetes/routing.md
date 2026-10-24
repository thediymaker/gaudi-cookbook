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
