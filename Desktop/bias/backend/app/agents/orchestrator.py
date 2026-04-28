from __future__ import annotations
import asyncio
import uuid
from typing import Iterable

from anthropic import AsyncAnthropic

from ..config import ANTHROPIC_API_KEY, CONFIDENCE_FLOOR
from ..schemas import (
    AgentRunInfo,
    AnalyzeResponse,
    Annotation,
    Mode,
)
from . import ALL_AGENTS
from .base import BaseAgent
from .meta_evaluator import meta_evaluate


class Orchestrator:
    def __init__(self) -> None:
        self._client: AsyncAnthropic | None = (
            AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        )
        self._agents: list[BaseAgent] = [cls(client=self._client) for cls in ALL_AGENTS]

    @property
    def agents(self) -> list[BaseAgent]:
        return self._agents

    async def analyze(
        self,
        text: str,
        references: str | None,
        mode: Mode,
    ) -> AnalyzeResponse:
        warnings: list[str] = []
        if not ANTHROPIC_API_KEY:
            warnings.append(
                "ANTHROPIC_API_KEY is not set on the server. Agents will return empty results. "
                "Set the env var and restart the backend to enable detection."
            )
        if mode == "premium":
            warnings.append(
                "Premium mode requested. RAG / reference-fetching is not yet implemented; "
                "agents are running in text-only Lite mode for this request."
            )
            effective_mode: Mode = "lite"
        elif mode == "adaptive":
            warnings.append(
                "Adaptive mode requested. Lite-only first pass is used; per-span escalation "
                "to Premium is not yet implemented."
            )
            effective_mode = "lite"
        else:
            effective_mode = mode

        results = await asyncio.gather(
            *(agent.run(text, references, effective_mode) for agent in self._agents),
            return_exceptions=False,
        )

        infos: list[AgentRunInfo] = []
        all_annotations: list[Annotation] = []
        for agent, (anns, err) in zip(self._agents, results):
            kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]
            infos.append(
                AgentRunInfo(
                    agent=agent.name,
                    bias_type=agent.bias_type,
                    prompt_version=agent.prompt_version,
                    raw_count=len(anns),
                    kept_count=len(kept),
                    error=err,
                )
            )
            all_annotations.extend(kept)
            if err:
                warnings.append(f"{agent.name}: {err}")

        merged = meta_evaluate(all_annotations)
        score = _overall_score(merged, len(text))

        return AnalyzeResponse(
            document_id=f"doc_{uuid.uuid4().hex[:10]}",
            mode=mode,
            overall_bias_score=score,
            annotations=merged,
            agents=infos,
            warnings=warnings,
        )


def _overall_score(annotations: Iterable[Annotation], doc_len: int) -> float:
    if doc_len <= 0:
        return 0.0
    weight = {"low": 1.0, "medium": 2.0, "high": 3.0}
    total = sum(weight[a.severity] * a.confidence for a in annotations)
    # Normalise: ~3 high-confidence high-severity flags per ~1500 chars saturates to ~1.0.
    saturation = 9.0 * max(1.0, doc_len / 1500.0)
    return round(min(1.0, total / saturation), 3)
