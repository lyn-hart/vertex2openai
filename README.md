# vertex2openai

An adapter that exposes **Google Vertex AI Gemini models through the Express API** behind two client-facing protocols:

- **OpenAI-compatible** — `GET /v1/models`, `POST /v1/chat/completions`
- **Anthropic Messages** — `POST /v1/messages`, `POST /v1/messages/count_tokens` (works with Claude Code and the Anthropic SDKs)

There is exactly one upstream auth method: **Vertex Express API keys**. Service accounts, credential files, and the OpenAI-direct channel have been removed.

## Getting started

1. Get a Vertex AI Express API key: https://cloud.google.com/vertex-ai/generative-ai/docs/express-mode/overview
2. Run with Docker Compose:

```bash
git clone https://github.com/lyn-hart/vertex2openai.git
cd vertex2openai
# edit docker-compose.yml: set PROXY_API_KEY and EXPRESS_API_KEYS
docker compose up -d
```

The server listens on port **8050**. It refuses to start when `PROXY_API_KEY` or `EXPRESS_API_KEYS` is missing — there are no default credentials.

## Environment variables

### Required

| Variable | Meaning |
|---|---|
| `PROXY_API_KEY` | API key clients must present (`Authorization: Bearer …` or `x-api-key: …`) |
| `EXPRESS_API_KEYS` | Comma-separated Vertex Express API keys |

### Optional

| Variable | Default | Meaning |
|---|---|---|
| `KEY_ROTATION` | `random` | `random` or `roundrobin` selection across express keys |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `FAKE_STREAMING` | `false` | Emit non-streamed responses as SSE chunks |
| `FAKE_STREAMING_INTERVAL` | `1.0` | Keep-alive interval (seconds) for fake streaming |
| `RETRY_COUNT` | `30` | Upstream 429 retries after the first attempt |
| `RETRY_INTERVAL_MS` | `1000` | Fixed interval between 429 retries |
| `MODELS_CONFIG_URL` | upstream JSON | Remote model list; must contain `vertex_express_models` |
| `PROXY_URL` | — | Outbound proxy for Google API calls (http/https/socks5) |
| `SSL_CERT_FILE` | — | Custom CA bundle for upstream TLS verification |

### Renamed variables (migration)

| Old name | New name |
|---|---|
| `VERTEX_EXPRESS_API_KEY` | `EXPRESS_API_KEYS` |
| `API_KEY` (default `123456`) | `PROXY_API_KEY` (no default; required) |
| `ROUNDROBIN=true/false` | `KEY_ROTATION=random\|roundrobin` |

Removed entirely: `GOOGLE_CREDENTIALS_JSON`, `CREDENTIALS_DIR`, `HUGGINGFACE`, `HUGGINGFACE_API_KEY`, `SAFETY_SCORE`.

## Models and suffixes

The model list comes from `MODELS_CONFIG_URL` (or the built-in fallback in [`app/catalog.py`](app/catalog.py)). Append capability suffixes to any listed model:

| Suffix | Effect |
|---|---|
| `-search` | Enable Google Search grounding |
| `-nothinking` | Disable thinking (budget 0 / 128) |
| `-max` | Max thinking budget (24576 / 32768) |
| `-2k`, `-4k` | Image output resolution (gemini-3-pro-image only) |

The `[PAY]`, `[EXPRESS]` prefixes and `-encrypt`, `-encrypt-full`, `-auto`, `-openai`, `-openaisearch` suffixes are no longer recognized.

**Grounded search without the suffix** — pass one of these request-body params on `/v1/chat/completions`:

```json
{"model": "gemini-2.5-flash", "search": true, "messages": [...]}
```

`web_search_options: {...}` (OpenAI's parameter) also enables it. On `/v1/messages`, include an Anthropic `web_search_*` tool in the `tools` array. When search grounding runs, the response text is appended with a markdown **Sources:** list of the retrieved pages.

**Known upstream limitation:** the Vertex endpoint does not serve Google Search grounding together with function calling — when a request carries both, the search tool is dropped and the model only sees the functions (a WARNING is logged server-side). Ask for search only in requests without function tools.

## Usage

### OpenAI clients

```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-pro-preview",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Anthropic clients / Claude Code

```bash
curl http://localhost:8050/v1/messages \
  -H "x-api-key: $PROXY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-pro-preview",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Claude Code: point `ANTHROPIC_BASE_URL` at the adapter and set `ANTHROPIC_AUTH_TOKEN` (or `ANTHROPIC_API_KEY`) to `PROXY_API_KEY`. Claude Code effort levels map to Gemini `thinking_budget`.

## How it works

```
client ──▶ routes/          protocol handling (OpenAI / Anthropic)
       ──▶ translate_*      request/response conversion
       ──▶ streaming.py     stream orchestration, 429 retries, fake streaming
       ──▶ client.py        Express client construction + project-id fallback
       ──▶ keys.py          Express key rotation (random / roundrobin)
       ──▶ catalog.py       model list fetching + built-in fallback
```

When an upstream call returns 404/400, the adapter discovers the numeric project id behind the Express key once and retries against the project-scoped endpoint (needed by some models). 429 responses are retried with a fixed interval; other errors surface to the client.

## License

See [LICENSE](LICENSE).
