from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agents.orchestrator import Orchestrator
from .config import ANTHROPIC_API_KEY, MODEL, PROMPT_VERSION
from .schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="BiasScan", version="0.1.0")

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
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
    }


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


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` is required and cannot be empty.")
    if len(req.text) > 50_000:
        raise HTTPException(status_code=400, detail="`text` exceeds 50,000 character limit.")

    orch = get_orchestrator()
    return await orch.analyze(req.text, req.references, req.mode)
