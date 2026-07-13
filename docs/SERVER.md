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
| GET | `/metrics` | Prometheus text: requests, tokens, cancellations, timeouts, queue rejections, expert-store hits/misses/bytes, prefetch counters. |
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
