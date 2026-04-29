from __future__ import annotations
import asyncio
import uuid
from typing import AsyncIterator, Iterable

from ..config import CONFIDENCE_FLOOR
from ..providers import build_provider, LLMError, ProviderConfig
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
    """Stateless. One instance per process is fine; agents are reused for their
    cached prompts only — no API client state."""

    def __init__(self) -> None:
        self._agents: list[BaseAgent] = [cls() for cls in ALL_AGENTS]

    @property
    def agents(self) -> list[BaseAgent]:
        return self._agents

    def select_agents(self, names: list[str] | None) -> list[BaseAgent]:
        if not names:
            return self._agents
        wanted = {n.upper() for n in names}
        chosen = [a for a in self._agents if a.name in wanted]
        return chosen or self._agents

    async def analyze(
        self,
        *,
        text: str,
        references: str | None,
        mode: Mode,
        provider_config: ProviderConfig,
        agents: list[str] | None,
    ) -> AnalyzeResponse:
        warnings: list[str] = []

        try:
            provider = build_provider(provider_config)
        except LLMError as e:
            return AnalyzeResponse(
                document_id=f"doc_{uuid.uuid4().hex[:10]}",
                mode=mode,
                overall_bias_score=0.0,
                annotations=[],
                agents=[],
                warnings=[str(e)],
                provider=provider_config.model_dump_safe(),
            )

        effective_mode, warnings = _resolve_mode(mode, warnings)
        chosen = self.select_agents(agents)
        if agents and len(chosen) < len({a.upper() for a in agents}):
            warnings.append(
                f"Some requested agents were unknown and skipped. Running: "
                f"{', '.join(a.name for a in chosen)}."
            )

        sem = asyncio.Semaphore(3)

        async def _run_with_sem(agent: BaseAgent):
            async with sem:
                return await agent.run(
                    text=text, references=references,
                    mode=effective_mode, provider=provider,
                )

        results = await asyncio.gather(
            *(_run_with_sem(agent) for agent in chosen),
            return_exceptions=False,
        )

        infos: list[AgentRunInfo] = []
        all_annotations: list[Annotation] = []
        for agent, (anns, err) in zip(chosen, results):
            kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]
            infos.append(AgentRunInfo(
                agent=agent.name, bias_type=agent.bias_type,
                prompt_version=agent.prompt_version,
                raw_count=len(anns), kept_count=len(kept), error=err,
            ))
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
            provider=provider_config.model_dump_safe(),
        )

    async def analyze_stream(
        self,
        *,
        text: str,
        references: str | None,
        mode: Mode,
        provider_config: ProviderConfig,
        agents: list[str] | None,
    ) -> AsyncIterator[dict]:
        """Async generator — yields SSE-ready dicts as each agent completes."""
        warnings: list[str] = []

        try:
            provider = build_provider(provider_config)
        except LLMError as e:
            yield {"event": "error", "message": str(e)}
            return

        effective_mode, warnings = _resolve_mode(mode, warnings)
        chosen = self.select_agents(agents)
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"

        yield {
            "event": "start",
            "document_id": doc_id,
            "total_agents": len(chosen),
            "agent_names": [a.name for a in chosen],
        }

        agent_infos: dict[str, dict] = {}
        all_kept: list[Annotation] = []

        for agent in chosen:
            anns, err = await agent.run(
                text=text, references=references,
                mode=effective_mode, provider=provider,
            )
            kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]
            all_kept.extend(kept)
            if err:
                warnings.append(f"{agent.name}: {err}")
            info = {
                "agent": agent.name,
                "bias_type": agent.bias_type,
                "prompt_version": agent.prompt_version,
                "raw_count": len(anns),
                "kept_count": len(kept),
                "error": err,
            }
            agent_infos[agent.name] = info
            yield {
                "event": "agent_done",
                **info,
                "annotations": [a.model_dump() for a in kept],
            }

        merged = meta_evaluate(all_kept)
        score = _overall_score(merged, len(text))

        yield {
            "event": "complete",
            "document_id": doc_id,
            "overall_bias_score": score,
            "annotations": [a.model_dump() for a in merged],
            "agents": list(agent_infos.values()),
            "mode": mode,
            "warnings": warnings,
            "provider": provider_config.model_dump_safe(),
        }


def _resolve_mode(mode: Mode, warnings: list[str]) -> tuple[Mode, list[str]]:
    if mode == "premium":
        warnings.append(
            "Premium mode requested; RAG not yet implemented. Running Lite."
        )
        return "lite", warnings
    if mode == "adaptive":
        warnings.append("Adaptive mode requested; running Lite first pass.")
        return "lite", warnings
    return mode, warnings


def _overall_score(annotations: Iterable[Annotation], doc_len: int) -> float:
    if doc_len <= 0:
        return 0.0
    weight = {"low": 1.0, "medium": 2.0, "high": 3.0}
    total = sum(weight[a.severity] * a.confidence for a in annotations)
    saturation = 9.0 * max(1.0, doc_len / 1500.0)
    return round(min(1.0, total / saturation), 3)
