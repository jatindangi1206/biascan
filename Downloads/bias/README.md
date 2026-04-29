# BiasScan

Open-source agentic cognitive bias detection for AI-generated systematic review synthesis text. Five specialised agents — **ARGUS** (confirmation), **LIBRA** (certainty inflation), **LENS** (overgeneralisation), **QUILL** (framing), **VIGIL** (causal inference) — run in parallel against your chosen LLM.

This implements the architecture in `biasscan_plan.html` and the prompt-engineering design in `biasscan_prompts_v1.html`.

## Bring your own model

BiasScan does not ship credentials and does not own a quota. You pick the provider:

| Provider | Notes |
| --- | --- |
| **Ollama** (default) | Local, fully open source. Recommended starting point — no key, no spend. Works with Qwen, Llama, Mistral, Gemma, Phi, etc. |
| **Anthropic** | Claude Haiku / Sonnet / Opus. Highest accuracy in our internal tests. |
| **OpenAI** | GPT-4o-mini / GPT-4o. |
| **Google Gemini** | Gemini 2.0 Flash / 1.5 Pro. |

### Why no login system

This is a self-hosted open-source tool. Adding accounts would mean:

- A user database, password handling, and session management to maintain.
- A custodianship problem for API keys you store on behalf of users.
- An attack surface — login systems are where credentials leak.

None of that earns its keep when every install is the user's own machine. Instead:

- **Keys never live on the server.** They are entered in the frontend and stored only in your browser's `localStorage`. The Settings panel has a "clear" button that wipes them.
- **Each request carries its own key.** The backend forwards it to the provider you chose and forgets it the moment the response comes back. Nothing is logged, cached, or persisted.
- **The backend reads no API keys from environment variables.** The only env vars it reads are tuning knobs (confidence floor, request timeout).

If you ever want a hosted, multi-tenant deployment, that's a separate concern and would need its own auth layer — but the core tool stays self-hosted-first.

## Quickstart with Ollama (recommended)

### 1. Install and pull a model

```bash
# install: https://ollama.com/download
ollama pull qwen2.5:7b      # 4.7 GB — solid JSON compliance
# or any of: llama3.1:8b · mistral:7b · gemma2:9b · phi3.5:3.8b
ollama serve &              # starts the local server on :11434
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

`GET http://localhost:8000/api/health` should return `{"status":"ok", "key_storage": "none — ..."}`.

### 3. Frontend

```bash
cd frontend
npm install && npm run dev
```

Open `http://localhost:5173`. The Settings panel will be pre-filled for Ollama at `localhost:11434`. Click **Test connection** to verify, then paste synthesis text and click Analyse.

## Switching providers

In the frontend Settings panel, click the provider card. Defaults are pre-filled per provider:

- **Ollama** → `qwen2.5:7b`, `http://localhost:11434`, no key.
- **Anthropic** → `claude-haiku-4-5-20251001`, paste your key.
- **OpenAI** → `gpt-4o-mini`, paste your key.
- **Gemini** → `gemini-2.0-flash`, paste your key.

You can override the model field with any model id the provider exposes.

## Agent selection

The Agents panel lets you run any subset of the 5 detectors. Use cases:

- "I only care about causal claims" → run **VIGIL** alone.
- "Quick confirmation-bias check" → run **ARGUS** alone.
- "Full audit" → all five.

Latency is bounded by the slowest agent regardless of how many you select (they run in parallel via `asyncio.gather`).

## Repository layout

```
backend/
  app/
    agents/            # base + 5 agents + orchestrator + meta-evaluator
    prompts/v1/        # versioned system prompts (~7-10 kB each)
    providers/         # ollama, anthropic, openai_compat, gemini
    rag/               # stub vector store + reference fetcher (premium, v0.2)
    config.py
    main.py            # FastAPI entry
    schemas.py
  requirements.txt
frontend/
  src/
    components/        # InputPanel, SettingsPanel, AgentPicker, AnnotatedOutput, ResultsPanel
    api.ts, types.ts, App.tsx, main.tsx, styles.css
  package.json
  vite.config.ts
```

## API

### `GET /api/providers`
Returns the list of supported providers and their default models / URLs.

### `POST /api/ping-provider`
```json
{ "provider": { "provider": "ollama", "model": "qwen2.5:7b", "base_url": "http://localhost:11434" } }
```
Verifies connectivity with a tiny round-trip. Used by the **Test connection** button.

### `POST /api/analyze`
```json
{
  "text": "The synthesis section text...",
  "references": "Optional reference list, newline-separated.",
  "mode": "lite",
  "provider": {
    "provider": "ollama",
    "model": "qwen2.5:7b",
    "api_key": null,
    "base_url": "http://localhost:11434"
  },
  "agents": ["ARGUS", "VIGIL"]
}
```

`agents` is optional — omit or pass `null` to run all five.

Response shape (abbreviated):
```json
{
  "document_id": "doc_a1b2c3d4e5",
  "mode": "lite",
  "overall_bias_score": 0.42,
  "annotations": [ { "bias_type": "causal_inference_error", "span_start": 412, "span_end": 489, "confidence": 0.88, "severity": "high", "agent_name": "VIGIL", "...": "..." } ],
  "agents": [ { "agent": "ARGUS", "raw_count": 2, "kept_count": 1, "...": "..." } ],
  "warnings": [],
  "provider": { "provider": "ollama", "model": "qwen2.5:7b", "base_url": "http://localhost:11434" }
}
```

The provider echo never includes `api_key`.

## Tuning (env vars, all optional)

| Var | Default | Notes |
| --- | --- | --- |
| `BIASSCAN_CONFIDENCE_FLOOR` | `0.5` | Annotations below this are dropped. |
| `BIASSCAN_MAX_TOKENS` | `4096` | Output cap per agent call. |
| `BIASSCAN_PROVIDER_TIMEOUT` | `180` | Seconds before a provider call times out. |

## Privacy / security notes

- The backend never reads, stores, logs, or caches API keys.
- The `provider` echo in `/api/analyze` responses excludes the key field by construction (`ProviderConfig.model_dump_safe()`).
- For local-only use, run uvicorn with `--host 127.0.0.1` (the default) — do not bind to `0.0.0.0` unless you actually want the backend reachable from elsewhere on your network.
- Frontend stores provider config in `localStorage` under the key `biasscan.provider`. Click "clear" in the Settings panel to wipe the API key, or run `localStorage.removeItem('biasscan.provider')` in your browser console to wipe everything.

## Roadmap

- **v0.2 — RAG**: reference resolver (DOI / PubMed / Semantic Scholar / CrossRef), chunker, sentence-transformer embeddings, ChromaDB vector store, premium-mode tool-use loop.
- **v0.3 — Eval**: harness against CoBBLEr, BABE, MBIC, BRU, danthareja/cognitive-distortion, CausalBank, COPA. Per-agent recall ≥ 0.85 target.
- **v0.4 — UI**: severity colouring, conflict visualisation polish, head-to-head comparison mode (BiasScan vs. single-LLM baseline).
- **v1.0 — OSS release**: GitHub release, bioRxiv preprint, HuggingFace model card.
