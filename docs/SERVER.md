# Server API

`undertow serve` exposes an OpenAI-compatible surface over the loaded
model. Anything that speaks the OpenAI chat API (Open WebUI, client SDKs
pointed at a custom base URL) works against it.

```sh
undertow serve --model /models/some-moe-int4 --port 8080 \
    --api-key sekrit --timeout-secs 300 --max-queue 16 \
    --cors-origin "https://my-frontend.example"
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness; never requires auth. |
| GET | `/metrics` | Prometheus text: requests, tokens, cancellations, timeouts, queue rejections, expert-store hits/misses/bytes, prefetch counters, cache hit ratio, memory gauges, evictions, disk read latency histogram, and decode rate. |
| GET | `/v1/models` | The loaded model. |
| POST | `/v1/completions` | Raw text completion. |
| POST | `/v1/chat/completions` | Chat; `"stream": true` for SSE. |

Supported request fields: `messages` or `prompt`, `max_tokens` (alias
`max_completion_tokens`), `temperature`, `top_p`, `seed`, `stop` (string
or array), `stream`. Prompts longer than the context window are clamped
to their tail, OpenAI-style.

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "authorization: Bearer sekrit" -H "content-type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 128, "temperature": 0.7, "stream": true,
    "stop": ["\n\n"]
  }'
```

Streaming responses are `chat.completion.chunk` SSE events ending with
`data: [DONE]`. Stop strings are held back until they either match (cut,
never emitted) or cannot complete anymore (flushed), so clients never see
a partial stop marker.

## Operational behavior

- One generation runs at a time; up to `--max-queue` requests wait, and
  the rest get 429 with a `retry-after` header. One CPU-saturating
  forward pass beats several thrashing the expert cache.
- Every generation carries a deadline (`--timeout-secs`); when it passes,
  the request answers with what was produced and `finish_reason: length`.
- A client hanging up on a stream cancels the forward pass at the next
  token; the `undertow_generations_cancelled_total` counter tracks it.
- With `--api-key` (or `UNDERTOW_API_KEY`), every endpoint except
  `/health` requires `Authorization: Bearer <key>`.
- Ctrl-c stops accepting connections and lets in-flight requests finish
  inside their deadline.

## Prometheus Metrics

The `/metrics` endpoint exposes runtime telemetry in Prometheus text exposition format:

- **Gauges**:
  - `undertow_inflight`: Active generations currently executing.
  - `undertow_queued`: Pending requests waiting for generation slot.
  - `undertow_tokens_per_second`: Instantaneous token generation rate from the last completed decode.
  - `undertow_expert_cache_hit_ratio`: Ratio of expert cache hits to total lookups (`hits / (hits + misses)`).
  - `undertow_expert_cache_bytes_used`: Total bytes currently held in the expert cache.
  - `undertow_expert_cache_budget_bytes`: Configured byte budget for the expert cache.
- **Counters**:
  - `undertow_requests_total`, `undertow_responses_4xx_total`, `undertow_responses_5xx_total`
  - `undertow_tokens_generated_total`, `undertow_generations_cancelled_total`, `undertow_generations_timed_out_total`, `undertow_queue_rejections_total`
  - `undertow_expert_store_hits_total`, `undertow_expert_store_misses_total`, `undertow_expert_store_bytes_read_total`
  - `undertow_prefetch_issued_total`, `undertow_prefetch_dropped_total`
  - `undertow_expert_evictions_total`: Expert weights evicted under memory pressure.
- **Histogram**:
  - `undertow_disk_read_duration_seconds`: Latency of synchronous `pread` expert fetch calls (`_bucket`, `_sum`, `_count`).

