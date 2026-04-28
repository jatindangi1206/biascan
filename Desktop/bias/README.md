# BiasScan

Agentic cognitive bias detection for AI-generated systematic review synthesis text. Five specialised agents — **ARGUS** (confirmation), **LIBRA** (certainty inflation), **LENS** (overgeneralisation), **QUILL** (framing), **VIGIL** (causal inference) — run in parallel, followed by a meta-evaluator pass that flags span conflicts and aggregates a document-level score.

This repository implements the architecture described in `biasscan_plan.html` and the prompt-engineering design in `biasscan_prompts_v1.html`.

## What's in v0.1

- **Lite mode** (text-only): fully functional end-to-end via the Anthropic API.
- **5 agents** as separate Python classes, each loading a versioned prompt from `backend/app/prompts/v1/`.
- **Orchestrator** dispatches all 5 agents in parallel via `asyncio.gather`.
- **Meta-evaluator** detects overlapping multi-agent flags and marks them as conflicts (preserved, never silently dropped).
- **Confidence floor** filter (configurable via `BIASSCAN_CONFIDENCE_FLOOR`).
- **React + Vite frontend** with character-span highlight overlay, per-bias colour coding, hover tooltips, conflict dotted outlines, agent-run table, and JSON detail view.
- **Premium / Adaptive modes** are scaffolded but emit warnings; reference fetching, embedding, and vector retrieval are stubbed in `backend/app/rag/` and not wired in.

## Repository layout

```
backend/
  app/
    agents/            # base, 5 concrete agents, orchestrator, meta-evaluator
    prompts/v1/        # versioned system prompts (~2-3 kB each)
    rag/               # stub vector store + reference fetcher (premium)
    config.py
    main.py            # FastAPI entry
    schemas.py         # pydantic models
  requirements.txt
  .env.example
frontend/
  src/
    components/        # InputPanel, AnnotatedOutput, ResultsPanel
    api.ts, types.ts, App.tsx, main.tsx, styles.css
  package.json
  vite.config.ts
```

## Run it

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

`GET http://localhost:8000/api/health` should return `{"status":"ok", "anthropic_key_set": true, ...}`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/*` to `localhost:8000`.

## API

### `POST /api/analyze`

```json
{
  "text": "The synthesis section text to analyse...",
  "references": "Optional reference list, newline-separated.",
  "mode": "lite"
}
```

Response:

```json
{
  "document_id": "doc_a1b2c3d4e5",
  "mode": "lite",
  "overall_bias_score": 0.42,
  "annotations": [
    {
      "bias_type": "causal_inference_error",
      "span_start": 412,
      "span_end": 489,
      "flagged_text": "increased physical activity leads to improved mental health",
      "confidence": 0.88,
      "severity": "high",
      "clean_alternative": "...",
      "false_positive_check": "...",
      "rag_check_needed": true,
      "rag_query": "study design of [Smith 2020]",
      "conflict": false,
      "agent_name": "VIGIL",
      "prompt_version": "v1",
      "extras": { "evidence_design": "cross-sectional", "...": "..." }
    }
  ],
  "agents": [ { "agent": "ARGUS", "raw_count": 2, "kept_count": 1, "...": "..." } ],
  "warnings": []
}
```

## Configuration

| Env var | Default | Notes |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | (required) | If unset, agents return empty results and a warning is emitted. |
| `BIASSCAN_MODEL` | `claude-haiku-4-5-20251001` | Any current Claude model id. Sonnet 4.6 (`claude-sonnet-4-6`) is recommended for higher recall in production. |
| `BIASSCAN_MAX_TOKENS` | `4096` | Output cap per agent. |
| `BIASSCAN_CONFIDENCE_FLOOR` | `0.5` | Annotations below this confidence are discarded. |
| `BIASSCAN_ADAPTIVE_ESCALATE_BELOW` | `0.6` | Reserved for adaptive mode (not yet implemented). |

System prompts use Anthropic prompt caching (`cache_control: ephemeral`), so the large per-agent system prompts are only billed in full on the first call within the cache TTL.

## Roadmap

- **v0.2 — RAG**: Reference resolver (DOI / PubMed / Semantic Scholar / CrossRef), chunker, sentence-transformer embeddings, ChromaDB vector store, premium-mode tool use loop.
- **v0.3 — Eval**: Harness against CoBBLEr, BABE, MBIC, BRU, danthareja/cognitive-distortion, CausalBank, COPA. Per-agent recall ≥ 0.85 target.
- **v0.4 — UI**: Severity colouring, conflict visualisation polish, comparison mode (BiasScan vs. single-LLM baseline).
- **v1.0 — OSS**: GitHub release, bioRxiv preprint, HuggingFace card.
