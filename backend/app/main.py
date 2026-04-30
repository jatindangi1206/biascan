from __future__ import annotations
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agents.orchestrator import Orchestrator
from .config import PROMPT_VERSION
from .providers import LLMError, SUPPORTED_PROVIDERS, build_provider, get_word_cap
from .schemas import AnalyzeRequest, AnalyzeResponse, PingRequest

INPUT_CHAR_HARD_LIMIT = 200_000  # sanity ceiling; per-provider word_cap is the real limit


def _truncate_to_word_cap(text: str, cap: int) -> tuple[str, int, bool]:
    """Return (possibly-truncated text, original word count, was_truncated)."""
    matches = list(re.finditer(r"\S+", text))
    word_count = len(matches)
    if word_count <= cap:
        return text, word_count, False
    cut = matches[cap].start()
    return text[:cut].rstrip(), word_count, True

app = FastAPI(
    title="BiasScan",
    version="0.2.0",
    description=(
        "Open source. Bring your own model. API keys are accepted per-request, "
        "forwarded to the provider you choose, and never stored, logged, or cached."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "prompt_version": PROMPT_VERSION,
        "key_storage": "none — keys are accepted per-request and never persisted",
    }


@app.get("/api/providers")
async def list_providers() -> dict:
    return {"providers": SUPPORTED_PROVIDERS}


@app.get("/api/agents")
async def list_agents() -> dict:
    orch = get_orchestrator()
    return {
        "agents": [
            {
                "name": a.name,
                "bias_type": a.bias_type,
                "prompt_version": a.prompt_version,
                "prompt_filename": a.prompt_filename,
            }
            for a in orch.agents
        ]
    }


@app.post("/api/ping-provider")
async def ping_provider(req: PingRequest) -> dict:
    """Verify the supplied provider config can do a 1-token round-trip.

    Useful for the frontend's 'Test connection' button. Keys are never logged.
    """
    try:
        provider = build_provider(req.provider)
    except LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        text = await provider.complete(
            system_prompt='Reply with the single token: OK',
            user_message="ping",
            max_tokens=8,
        )
    except LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "sample": text[:64]}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` is required and cannot be empty.")
    if len(req.text) > INPUT_CHAR_HARD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"`text` exceeds {INPUT_CHAR_HARD_LIMIT:,} character limit.",
        )

    cap = get_word_cap(req.provider.provider)
    text, original_words, truncated = _truncate_to_word_cap(req.text, cap)
    extra_warnings: list[str] = []
    if truncated:
        extra_warnings.append(
            f"Input was {original_words:,} words; only the first {cap:,} were analysed "
            f"(per-provider cap for {req.provider.provider}). Raise word_cap in providers/base.py to lift this."
        )

    orch = get_orchestrator()
    return await orch.analyze(
        text=text,
        references=req.references,
        mode=req.mode,
        provider_config=req.provider,
        agents=req.agents,
        extra_warnings=extra_warnings,
    )


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeRequest) -> StreamingResponse:
    """SSE endpoint — yields one event per agent as it completes, then a final 'complete' event."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` is required and cannot be empty.")
    if len(req.text) > INPUT_CHAR_HARD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"`text` exceeds {INPUT_CHAR_HARD_LIMIT:,} character limit.",
        )

    cap = get_word_cap(req.provider.provider)
    text, original_words, truncated = _truncate_to_word_cap(req.text, cap)
    extra_warnings: list[str] = []
    if truncated:
        extra_warnings.append(
            f"Input was {original_words:,} words; only the first {cap:,} were analysed "
            f"(per-provider cap for {req.provider.provider}). Raise word_cap in providers/base.py to lift this."
        )

    orch = get_orchestrator()

    async def event_generator():
        async for payload in orch.analyze_stream(
            text=text,
            references=req.references,
            mode=req.mode,
            provider_config=req.provider,
            agents=req.agents,
            extra_warnings=extra_warnings,
        ):
            event_type = payload.pop("event")
            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
