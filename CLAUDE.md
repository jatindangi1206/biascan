# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

Backend (FastAPI, Python 3.10+):
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Health probe: `curl http://localhost:8000/api/health`.

Frontend (Vite + React + TS):
```bash
cd frontend
npm install
npm run dev        # dev server on :5173
npm run build      # tsc -b && vite build
npm run preview
```

There is no test suite, linter, or formatter configured. Type-check the frontend via `npm run build` (runs `tsc -b`). The root `run_test.py` is stale — it imports `backend.models.llm_client` / `backend.orchestrator.pipeline`, which no longer exist; do not rely on it.

The RAG layer needs `numpy`, which is present in the root `requirements.txt` (used by the Vercel build) but **missing from `backend/requirements.txt`**. When installing the backend locally, `pip install numpy` as well, or the orchestrator import fails.

For local LLM dev, the default provider is Ollama on `localhost:11434`; `ollama serve &` plus `ollama pull qwen2.5:7b` is the zero-cost path.

## Architecture

BiasScan is a "bring your own model" agentic detector for cognitive bias in AI-generated systematic-review synthesis text. Five specialised agents (ARGUS, LIBRA, LENS, QUILL, VIGIL) run in parallel against a user-chosen LLM provider.

### Key invariant: no server-side credential state

API keys are accepted per request, forwarded once to the chosen provider, and never logged, cached, or persisted. The backend reads zero credential env vars (only tuning knobs from `app/config.py`). `ProviderConfig.model_dump_safe()` strips `api_key` from every response echo. Preserve this invariant when adding endpoints, logging, or middleware.

### Backend request flow

1. `app/main.py` (FastAPI) exposes `/api/health`, `/api/providers`, `/api/agents`, `/api/ping-provider`, `/api/analyze`, and `/api/analyze/stream` (SSE — emits one `agent_done` event per agent, then a final `complete` event). `/api/analyze` rejects `text` longer than `INPUT_CHAR_HARD_LIMIT` (200,000 chars; sanity ceiling). The real cap is per-provider: `_truncate_to_word_cap` trims input to the chosen provider's `word_cap` (defined in `providers/base.py`'s `SUPPORTED_PROVIDERS` — 8K for ollama, up to 20K for gemini) and surfaces a warning. The cap exists for cost reasons and is meant to be raised by users with higher API limits. A single lazily-built `Orchestrator` is reused per process — agents cache only their system prompts, no client state.
2. `Orchestrator.analyze` (`app/agents/orchestrator.py`) builds the provider via `providers.build_provider(ProviderConfig)`, selects the requested agent subset (default = all five), and runs them under an `asyncio.Semaphore(3)` cap (so at most 3 provider calls in flight even when all 5 agents are selected). `analyze_stream` runs the same flow sequentially and yields per-agent results as they complete.
3. **Input RAG** chunks the document (`InputRAG.index_document` → `assemble_all`). `_RAG_CHAR_THRESHOLD = 0`, so it runs on every request, but `assemble_all` deliberately returns the *whole* document in reading order (agents must see full context to judge bias), so today it acts as chunk-and-reassemble, not retrieval filtering. The per-agent retrieval routing (`retrieve_for_agent`) exists but is **not** wired into the orchestrator yet.
4. Each `BaseAgent` (`app/agents/base.py`) loads its versioned system prompt from `app/prompts/{PROMPT_VERSION}/{agent}_v1.0.txt`, builds a structured user message with mode + synthesis text + reference list, calls `provider.complete(...)`, then parses the returned JSON into `Annotation` objects. Parsing is defensive: it tolerates code-fenced JSON, bare arrays, and re-anchors `flagged_text` to `source_text` when offsets are wrong. Annotations whose `flagged_text` cannot be located are dropped.
5. Annotations below `CONFIDENCE_FLOOR` (default 0.5, env-overridable) are filtered out per-agent.
6. **Evidence RAG cross-check** (`_evidence_crosscheck`): each surviving annotation's `flagged_text` is queried against the evidence index; matches labelled `biased` nudge confidence up, `neutral`/`control` nudge it down, total adjustment clamped to ±0.15 so the cross-check refines but never overrides the agent.
7. `meta_evaluate` (`app/agents/meta_evaluator.py`) marks `conflict=True` on overlapping spans of *different* bias types using IoU ≥ 0.3. Nothing is silently dropped at this stage.
8. `_overall_score` weights `severity × confidence`, normalised by a doc-length-aware saturation constant, clamped to `[0, 1]`.

### Mode handling

`mode` is one of `lite | premium | adaptive`. The agent path is the same for all three: `_resolve_mode` collapses `premium` and `adaptive` to `lite` and appends a warning. RAG is **not** mode-gated — both Input RAG and the Evidence RAG cross-check run on every request regardless of mode. The only premium-specific piece is the LLM `reranker` (`app/rag/reranker.py`), which is implemented but not yet called from the orchestrator.

### RAG layer

`app/rag/` is fully implemented and active (it is no longer the v0.1 stub). It is a dependency-light, pure-Python design — no embedding model or external vector DB.

- `vector_store.py` — hybrid retrieval: TF-IDF dense cosine + BM25 sparse, fused with Reciprocal Rank Fusion (RRF), numpy only.
- `chunker.py` — splits documents into ~500-token overlapping chunks; holds `AGENT_QUERIES` (per-agent focus queries).
- `input_rag.py` (`InputRAG`) — facade over chunker + vector store for the user's document; supports per-agent retrieval and ±N adjacent-chunk expansion, but the orchestrator currently calls `assemble_all` (whole document).
- `evidence_rag.py` (`EvidenceRAG`) — built once per process; indexes 11 hand-written `BIAS_PATTERNS` (B01–B11, the domain ontology) **plus** real biased/control exemplars loaded at runtime from `eval/output/samples.json`. Without exemplars it has only ~33 generic chunks and rarely fires. `check_annotation` returns scored matches consumed by `_evidence_crosscheck`.
- `reranker.py` — optional LLM reranker (premium); `reference_fetcher.py` — stub for reference grounding.

Note: this couples the backend to an eval artifact at runtime — `eval/output/samples.json` (generated by `python -m eval.build_corpus`). It is optional; a missing/failed build is caught and logged, and the cross-check is skipped (`is_built` is False).

### Provider layer

`app/providers/` holds a `Protocol` (`LLMProvider`) and nine implementations: `ollama.py`, `groq.py`, `together.py`, `nvidia.py`, `anthropic.py`, `openai_compat.py` (provider name `openai`), `gemini.py`, `lightning.py`, `mistral.py`. Each accepts a `ProviderConfig`, performs JSON-mode hinting where the provider supports it, and surfaces failures as `LLMError` (whose message is considered safe to return to the user). The OpenAI-compatible providers (Groq, Together, NVIDIA, Lightning, Mistral) reuse `openai_compat`-style request shapes but are kept as separate classes for default base-URL and quirk handling. `SUPPORTED_PROVIDERS` in `providers/base.py` is the canonical list of providers + UI defaults consumed by `/api/providers`; `ProviderName` is the matching `Literal` type.

To add a provider: implement `complete(system_prompt, user_message, max_tokens) -> str`, register in `build_provider`, and append a metadata entry to `SUPPORTED_PROVIDERS`.

### Adding an agent

1. Drop a system prompt at `app/prompts/v1/<name>_v1.0.txt`.
2. Subclass `BaseAgent` in `app/agents/<name>.py` setting `name`, `bias_type` (must be in `BiasType` literal in `schemas.py`), and `prompt_filename`.
3. Add the class to `ALL_AGENTS` in `app/agents/__init__.py`. The orchestrator picks it up automatically.

### Frontend

Vite + React 18 + TypeScript, no state library. `frontend/src/App.tsx` owns provider config (persisted in `localStorage` under `biasscan.provider`), agent selection, and request lifecycle. `api.ts` wraps `fetch` to the backend (default `http://localhost:8000`) and consumes the SSE stream from `/api/analyze/stream`. Components: `SettingsPanel` (provider + key + Test connection), `AgentPicker`, `InputPanel`, `ProgressPanel` (live per-agent status driven by SSE events), `AnnotatedOutput` (renders span-level highlights), `ResultsPanel`. The frontend is the only place API keys live; the Settings panel "clear" button wipes them.

### Tuning knobs (env, optional)

`BIASSCAN_CONFIDENCE_FLOOR` (0.5), `BIASSCAN_MAX_TOKENS` (4096), `BIASSCAN_PROVIDER_TIMEOUT` (180s), `BIASSCAN_OLLAMA_NUM_CTX` (16384 — Ollama's default `num_ctx` is 2048, which silently truncates our ~3K-token system prompt plus user paste; we override it). These are the *only* env vars the backend reads. See `backend/.env.example` for a template.

### Deployment (Vercel)

`vercel.json` builds the frontend as a static site (`cd frontend && npm install && npm run build` → `frontend/dist`) and runs the backend as a single Python serverless function. `api/index.py` is the entry point: it adds `backend/` to `sys.path` and re-exports `app.main:app`. All `/api/*` requests are rewritten to that function (`maxDuration` 60s, 1024 MB). The function installs from the **root** `requirements.txt` (which includes `numpy`), not `backend/requirements.txt`.

### Evaluation harness

`eval/` is a standalone benchmarking package (not imported by the backend). Key scripts:
- `pipeline.py` — drives a full analysis run for eval purposes
- `auto_eval.py` / `manual_eval.py` — automated and human scoring of agent output
- `benchmark.py` — comparison across provider/model combinations
- `build_corpus.py` — assembles the test corpus from `eval/datasets/`; its `eval/output/samples.json` output is also read at runtime by the backend's Evidence RAG (see RAG layer above)
- `generate_report.py` / `generate_leaderboard_html.py` — output formatting

Results land in `eval/results/` and `eval/reports/`. The eval package has its own `eval_config.py` for thresholds; it is separate from `app/config.py`.
