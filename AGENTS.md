# AGENTS

Project-specific guidance for AI coding agents. Keep this file concise and link to existing docs instead of duplicating them.

## Where to look first

- Project overview, API shapes, and quickstart: [README.md](README.md)
- Detailed architecture and invariants: [CLAUDE.md](CLAUDE.md)

## Non-negotiable invariants

- No server-side credential state. API keys must not be stored, logged, cached, or read from env vars on the backend.
- Provider echoes must strip `api_key` via `ProviderConfig.model_dump_safe()`.

## Common commands

- Backend dev server: `uvicorn app.main:app --reload --port 8000` (after venv + `pip install -r requirements.txt`)
- Frontend dev server: `npm run dev` (after `npm install`)
- Frontend type-check/build: `npm run build`

## Codebase conventions

- Agents live in `backend/app/agents/` and load prompts from `backend/app/prompts/v1/`.
- Providers live in `backend/app/providers/` with the canonical list in `providers/base.py` (`SUPPORTED_PROVIDERS`).
- The orchestrator limits provider concurrency with a semaphore; do not remove the cap.
- `premium` and `adaptive` modes currently downgrade to `lite`; do not assume RAG works (see `backend/app/rag/`).

## When adding a provider

- Implement `complete(system_prompt, user_message, max_tokens) -> str` in a new provider class.
- Register it in `build_provider` and `SUPPORTED_PROVIDERS`.
- Keep error messages safe to return to users (see `LLMError`).

## When adding an agent

- Add a prompt file `app/prompts/v1/<name>_v1.0.txt`.
- Subclass `BaseAgent` and set `name`, `bias_type`, and `prompt_filename`.
- Register in `ALL_AGENTS` (see `backend/app/agents/__init__.py`).

## Testing

- There is no automated test suite or formatter configured; prefer small, targeted manual checks.
