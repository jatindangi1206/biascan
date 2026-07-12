# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

Backend (FastAPI, Python 3.10+):
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install numpy   # numpy missing from backend/requirements.txt
uvicorn app.main:app --reload --port 8000
```
Health probe: `curl http://localhost:8000/api/health`.

Frontend (Vite + React + TS):
```bash
cd frontend
npm install
npm run dev        # dev server on :5173
npm run build      # tsc -b && vite build (type-check gate)
npm run preview
```

There is no test suite, linter, or formatter configured. The root `run_test.py` is stale — do not rely on it.

For local LLM dev, the default provider is Ollama on `localhost:11434`; `ollama serve &` plus `ollama pull qwen2.5:7b` is the zero-cost path.

Eval scripts (standalone, not imported by backend):
```bash
python -m eval.build_corpus          # regenerates eval/output/samples.json (eval-only artifact)
python -m eval.sentence_eval         # sentence-level multi-model validation (needs OPENROUTER_API_KEY)
python -m eval.probe_models          # raw JSON probe against ARGUS for a single sentence
python -m eval.benchmark             # cross-provider/model benchmark
```

## Architecture

BiasScan is a "bring your own model" agentic detector for cognitive bias in AI-generated text. Five primary agents (ARGUS, LIBRA, LENS, QUILL, VIGIL) plus one conflict-resolution agent (AEGIS) run against a user-chosen LLM provider.

### Key invariant: no server-side credential state

API keys are accepted per request, forwarded once to the chosen provider, and never logged, cached, or persisted. The backend reads zero credential env vars (only tuning knobs from `app/config.py`). `ProviderConfig.model_dump_safe()` strips `api_key` from every response echo. Preserve this invariant when adding endpoints, logging, or middleware.

### Analysis modes and prompt versioning

The request field `analysis_mode` selects which prompt version is loaded — it is independent of the `mode` field (lite/premium/adaptive):

| `analysis_mode` | prompt version | directory |
|---|---|---|
| `systematic_review` (default, **the product**) | `v1` | `app/prompts/v1/` |
| `general_research` | `v2` | `app/prompts/v2/` |

`systematic_review` (v1) is the only user-facing mode: the frontend no longer exposes a mode toggle, and `DEFAULT_ANALYSIS_MODE` / the `AnalyzeRequest.analysis_mode` default are both `systematic_review`. The `general_research` (v2) path and its prompt dir are retained only for the eval harness (e.g. `sentence_eval_v2.py`) — do not surface v2 in the product UI.

`app/config.py` exposes three resolver functions used throughout the backend:
- `resolve_prompt_version(analysis_mode)` → `"v1"` or `"v2"`
- `resolve_prompt_filename(prompt_filename, prompt_version)` → replaces the version suffix in the filename
- `resolve_prompt_path(prompt_filename, analysis_mode)` → full `Path` to the prompt file

`PROMPT_VERSION` (the legacy single-string constant) still exists for code paths that haven't migrated; new code should use the resolver functions.

### Backend request flow

1. `app/main.py` (FastAPI) exposes `/api/health`, `/api/providers`, `/api/agents`, `/api/ping-provider`, `/api/analyze`, and `/api/analyze/stream` (SSE). `/api/analyze` rejects `text` longer than `INPUT_CHAR_HARD_LIMIT` (200,000 chars). The real cap is per-provider via `word_cap` in `SUPPORTED_PROVIDERS`; `_truncate_to_word_cap` trims and surfaces a warning. A single lazily-built `Orchestrator` is reused per process.
2. `Orchestrator.analyze` (`app/agents/orchestrator.py`) builds the provider via `providers.build_provider(ProviderConfig)`, selects the requested agent subset, and runs them under `asyncio.Semaphore(MAX_CONCURRENCY)` (default 3; lower to 1 for rate-limited providers). `analyze_stream` yields per-agent results as they complete.
3. **Input RAG** chunks the document, but only when `len(text) > _RAG_CHAR_THRESHOLD` (in `orchestrator.py`; currently `0`, so RAG is active for any non-empty doc). When active, `InputRAG.index_document` → `assemble_all` returns the whole document in reading order (agents must see full context); per-agent retrieval routing (`retrieve_for_agent`) exists but is **not** wired into the orchestrator. Below the threshold the raw `text` is passed through unchanged.
4. Each `BaseAgent` (`app/agents/base.py`) loads its versioned system prompt via `resolve_prompt_path`, builds a structured user message with synthesis text + references, calls `provider.complete(...)`, then parses the JSON response. Agent prompts now output `certainty` (string tier) instead of `confidence` (float); `CERTAINTY_MAP` in `base.py` converts: `certain→1.0`, `probable→0.8`, `suspected→0.6`, `weak→0.4`. The legacy float `confidence` field is still accepted as a fallback.
5. Annotations below `CONFIDENCE_FLOOR` (default 0.5) are filtered out. (There is no longer any Evidence RAG confidence cross-check — that step and `evidence_rag.py` were removed from the pipeline.)
6. **AEGIS conflict resolution** lives in `app/agents/meta_evaluator.py`, not in `aegis.py`. `find_conflict_clusters(annotations, iou_threshold=0.5)` groups overlapping spans of *different* bias types via union-find (so transitive A↔B↔C overlaps collapse into one cluster); `aegis_resolve_clusters` then runs `AegisAgent` on each cluster in parallel. Non-conflicting annotations pass through untouched. `AegisAgent` (`app/agents/aegis.py`) is **not** a subclass of `BaseAgent` (its `bias_type` comes from the model output). `AnalyzeRequest` accepts an optional `aegis_provider` to run AEGIS on a different model from the primary agents.
7. `_overall_score` weights `severity × confidence`, normalised by a doc-length-aware saturation constant, clamped to `[0, 1]`.

### Mode handling

`mode` (`lite | premium | adaptive`) is independent of `analysis_mode`. `_resolve_mode` collapses `premium` and `adaptive` to `lite` and appends a warning. The LLM reranker (`app/rag/reranker.py`) is the only premium-specific piece and is not yet called from the orchestrator.

### RAG layer

`app/rag/` — dependency-light, pure-Python (numpy only, no embedding model or external vector DB).

- `vector_store.py` — hybrid TF-IDF cosine + BM25, fused with Reciprocal Rank Fusion (RRF).
- `chunker.py` — ~500-token overlapping chunks; holds `AGENT_QUERIES` per-agent focus queries.
- `input_rag.py` (`InputRAG`) — facade over chunker + vector store for the user's document. `assemble_all()` returns the full document in reading order — the orchestrator always calls this so agents receive the complete context.
- `reranker.py` — optional LLM reranker (premium, not yet wired in).
- `reference_fetcher.py` — stub for premium reference resolution (DOI/PubMed/etc.); `fetch_full_text` always returns `None`. Not implemented.
- `evidence_rag.py` has been **deleted**. The Evidence RAG cross-check (confidence calibration via exemplar matching) is gone from the pipeline; `build_corpus.py`'s `samples.json` is now only an eval artifact, not a runtime input.

### Provider layer

`app/providers/` — a `Protocol` (`LLMProvider`) and nine implementations. `SUPPORTED_PROVIDERS` in `providers/base.py` is the canonical list; `ProviderName` is the matching `Literal` type.

| name | label | notable |
|---|---|---|
| `ollama` | Ollama (local, free) | `needs_base_url`, word_cap 8K |
| `groq` | Groq (free tier) | word_cap 12K |
| `nvidia` | NVIDIA NIM | word_cap 12K |
| `qwen` | Qwen (Alibaba Model Studio) | DashScope, word_cap 15K |
| `openrouter` | OpenRouter (all models) | single key, any model; default for eval scripts |
| `anthropic` | Anthropic (Claude) | native SDK, word_cap 15K |
| `openai` | OpenAI (GPT) | word_cap 15K |
| `gemini` | Google Gemini | word_cap 20K |
| `mistral` | Mistral La Plateforme | rate-limited free plan → set `BIASSCAN_MAX_CONCURRENCY=1` |

Each provider implements `complete(system_prompt, user_message, max_tokens) -> str` and surfaces failures as `LLMError` (safe to return to the user). `get_word_cap(provider_name)` returns the cap for a given provider.

To add a provider: implement `complete(...)`, add to `build_provider`, and append a metadata entry (including `word_cap`) to `SUPPORTED_PROVIDERS`.

### Adding an agent

1. Drop system prompts at `app/prompts/v1/<name>_v1.0.txt` and/or `app/prompts/v2/<name>_v2.0.txt`.
2. Subclass `BaseAgent` in `app/agents/<name>.py` — set `name`, `bias_type` (must be in `BiasType` literal in `schemas.py`), and `prompt_filename`.
3. Add the class to `ALL_AGENTS` in `app/agents/__init__.py`. The orchestrator picks it up automatically.

### Frontend

Vite + React 18 + TypeScript, no state library and no router. `App.tsx` owns provider config (persisted in `localStorage` under `biasscan.provider`), theme preference (`biasscan.theme` — light/dark), agent selection, and request lifecycle. Full-page views (`HowItWorks`, `Leaderboard`) are boolean-toggled in `App.tsx` rather than routed. `analysisMode` is hardcoded to `systematic_review` (the `ANALYSIS_MODE` constant) — the research-mode selector was removed from `SettingsPanel`. `api.ts` wraps `fetch` to the backend (default `http://localhost:8000`) and consumes the SSE stream. Components: `SettingsPanel` (provider + key + Test connection), `AgentPicker`, `InputPanel`, `ProgressPanel` (live SSE-driven status), `AnnotatedOutput` (span highlights), `ResultsPanel`, `Leaderboard` (top-right nav; renders the ranked table from the checked-in `src/leaderboard.json`, generated from the systematic-review eval run). Dark mode is toggled via `document.documentElement.dataset.theme` and CSS custom property overrides in `styles.css`.

### Tuning knobs (env, optional)

| env var | default | notes |
|---|---|---|
| `BIASSCAN_CONFIDENCE_FLOOR` | `0.5` | annotations below this are dropped |
| `BIASSCAN_MAX_TOKENS` | `4096` | provider `max_tokens` per call |
| `BIASSCAN_PROVIDER_TIMEOUT` | `180` | seconds per provider HTTP call |
| `BIASSCAN_OLLAMA_NUM_CTX` | `16384` | overrides Ollama's 2K default to fit ~3K system prompt |
| `BIASSCAN_MAX_CONCURRENCY` | `3` | semaphore cap on simultaneous provider calls; set to `1` for rate-limited providers like Mistral free |

See `backend/.env.example` for a template.

### Deployment (Vercel)

`vercel.json` builds the frontend (`cd frontend && npm install && npm run build` → `frontend/dist`) and runs the backend as a single Python serverless function (`api/index.py`, `maxDuration` 60s, 1024 MB). All `/api/*` requests are rewritten to that function. The function installs from the **root** `requirements.txt` (which includes `numpy`), not `backend/requirements.txt`.

### Evaluation harness

`eval/` is a standalone package (not imported by the backend). Key scripts:

There are two eval tracks — **sentence-level** (Phase 1) and **document-level** (Phase 2, the leaderboard).

| script | purpose |
|---|---|
| `build_corpus.py` | assembles `eval/output/samples.json` (eval-only artifact since Evidence RAG was removed) |
| `sentence_eval.py` | sentence-level multi-model validation across 5 models via OpenRouter; writes `eval/output/sentence_eval_*.csv/json` |
| `sentence_eval_v2.py` | same as above but for v2 (general_research) prompts |
| `multi_model_benchmark.py` | document-level benchmark (`analysis_mode="systematic_review"`); produces `eval/output/leaderboard.json` (per-model × per-document runs) |
| `probe_models.py` | raw JSON probe against ARGUS for a single sentence — quick sanity check for new models |
| `benchmark.py` | cross-provider/model comparison |
| `auto_eval.py` / `manual_eval.py` | automated and human scoring of agent output |
| `generate_report.py` / `generate_leaderboard_html.py` | output formatting |

**Document-level scoring is recomputed, not read from the JSON.** `leaderboard.json`'s stored `score` field predates the current formula. `_doc_metrics.py` is the single source of truth: it re-derives coverage / impact / score from each run's raw flags per `docs/SCORING.md` (substring-matching each flag's `flagged_text` against the source synthesis `.txt`). Both the figure scripts and `edit_significance.py` (exact paired permutation + Wilcoxon significance tests, dependency-free — scipy is unavailable) import from it. When changing the scoring formula, update `_doc_metrics.py` and `docs/SCORING.md` together. The frontend's `src/leaderboard.json` is generated from these recomputed metrics (systematic-review, working models only).

`eval/datasets/suites/` holds named evaluation suites used by `sentence_eval.py`. Results land in `eval/output/` and `eval/results/`. The eval package has its own `eval_config.py` separate from `app/config.py`. The `*_viz.py` scripts (`leaderboard_viz.py`, `leaderboard_viz_extra.py`, `leaderboard_table_viz.py`, `leaderboard_scale_viz.py`, `sentence_eval_viz.py`, `sentence_eval_v1v2_viz.py`) are matplotlib figure generators that read the result JSON/CSV files and write PNGs under `eval/output/*_figs/`.
