from __future__ import annotations
import asyncio
import logging
import re
import uuid
from typing import AsyncIterator

from ..config import CONFIDENCE_FLOOR, MAX_CONCURRENCY
from ..providers import build_provider, LLMError, ProviderConfig
from ..rag import InputRAG
from ..schemas import (
    AnalysisMode,
    AgentRunInfo,
    AnalyzeResponse,
    Annotation,
    Mode,
)
from . import ALL_AGENTS
from .aegis import AegisAgent
from .base import BaseAgent
from .meta_evaluator import aegis_resolve_clusters, find_conflict_clusters

logger = logging.getLogger(__name__)

# Threshold (chars) above which we activate Input RAG chunking.
# Set to 0 so RAG is always active — we expect long synthesis texts.
_RAG_CHAR_THRESHOLD = 0


class Orchestrator:
    """Stateless. One instance per process is fine; agents are reused for their
    cached prompts only — no API client state."""

    def __init__(self) -> None:
        self._agents: list[BaseAgent] = [cls() for cls in ALL_AGENTS]
        self._aegis: AegisAgent | None = None

    def _get_aegis(self) -> AegisAgent:
        if self._aegis is None:
            self._aegis = AegisAgent()
        return self._aegis

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
        analysis_mode: AnalysisMode,
        provider_config: ProviderConfig,
        agents: list[str] | None,
        extra_warnings: list[str] | None = None,
        aegis_provider_config: ProviderConfig | None = None,
    ) -> AnalyzeResponse:
        warnings: list[str] = list(extra_warnings or [])

        try:
            provider = build_provider(provider_config)
        except LLMError as e:
            return AnalyzeResponse(
                document_id=f"doc_{uuid.uuid4().hex[:10]}",
                mode=mode,
                overall_bias_score=0.0,
                annotations=[],
                agents=[],
                warnings=warnings + [str(e)],
                analysis_mode=analysis_mode,
                provider=provider_config.model_dump_safe(),
            )

        aegis_provider = provider
        if aegis_provider_config is not None:
            try:
                aegis_provider = build_provider(aegis_provider_config)
                warnings.append(
                    f"AEGIS using separate provider: {aegis_provider_config.provider}/{aegis_provider_config.model}"
                )
            except LLMError as e:
                warnings.append(f"AEGIS provider unavailable, falling back to primary: {e}")

        effective_mode, warnings = _resolve_mode(mode, warnings)
        chosen = self.select_agents(agents)
        if agents and len(chosen) < len({a.upper() for a in agents}):
            warnings.append(
                f"Some requested agents were unknown and skipped. Running: "
                f"{', '.join(a.name for a in chosen)}."
            )

        # ── RAG: chunk document for context-window management ────────
        use_rag = len(text) > _RAG_CHAR_THRESHOLD
        input_rag: InputRAG | None = None
        agent_texts: dict[str, str] = {}

        if use_rag:
            input_rag = InputRAG()
            input_rag.index_document(text)
            full_doc = input_rag.assemble_all()
            for agent in chosen:
                agent_texts[agent.name] = full_doc
            logger.info(
                "Input RAG active: %d chunks assembled for all agents",
                input_rag.total_chunks,
            )
            warnings.append(
                f"Input RAG active: document chunked into {input_rag.total_chunks} "
                f"segments."
            )
        else:
            for agent in chosen:
                agent_texts[agent.name] = text

        # ── Run agents in parallel ───────────────────────────────────
        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        async def _run_with_sem(agent: BaseAgent):
            async with sem:
                return await agent.run(
                    text=agent_texts[agent.name],
                    source_text=text,
                    references=references,
                    mode=effective_mode,
                    analysis_mode=analysis_mode,
                    provider=provider,
                )

        results = await asyncio.gather(
            *(_run_with_sem(agent) for agent in chosen),
            return_exceptions=False,
        )

        infos: list[AgentRunInfo] = []
        all_annotations: list[Annotation] = []
        for agent, (anns, err, reasoning) in zip(chosen, results):
            kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]

            infos.append(AgentRunInfo(
                agent=agent.name, bias_type=agent.bias_type,
                prompt_version=agent.prompt_version_for(analysis_mode),
                raw_count=len(anns), kept_count=len(kept), error=err,
                reasoning=reasoning,
            ))
            all_annotations.extend(kept)
            if err:
                warnings.append(f"{agent.name}: {err}")

        clusters = find_conflict_clusters(all_annotations)
        if clusters:
            merged = await aegis_resolve_clusters(
                all_annotations,
                clusters,
                source_text=text,
                provider=aegis_provider,
                aegis=self._get_aegis(),
                analysis_mode=analysis_mode,
            )
        else:
            merged = sorted(all_annotations, key=lambda a: (a.span_start, -a.confidence))
        score = self._overall_score(merged, text)

        return AnalyzeResponse(
            document_id=f"doc_{uuid.uuid4().hex[:10]}",
            mode=effective_mode,
            analysis_mode=analysis_mode,
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
        analysis_mode: AnalysisMode,
        provider_config: ProviderConfig,
        agents: list[str] | None,
        extra_warnings: list[str] | None = None,
        aegis_provider_config: ProviderConfig | None = None,
    ) -> AsyncIterator[dict]:
        """Async generator — yields SSE-ready dicts as each agent completes."""
        warnings: list[str] = list(extra_warnings or [])

        try:
            provider = build_provider(provider_config)
        except LLMError as e:
            yield {"event": "error", "message": str(e)}
            return

        aegis_provider = provider
        if aegis_provider_config is not None:
            try:
                aegis_provider = build_provider(aegis_provider_config)
                warnings.append(
                    f"AEGIS using separate provider: {aegis_provider_config.provider}/{aegis_provider_config.model}"
                )
            except LLMError as e:
                warnings.append(f"AEGIS provider unavailable, falling back to primary: {e}")

        effective_mode, warnings = _resolve_mode(mode, warnings)
        chosen = self.select_agents(agents)
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"

        # ── RAG: chunk document for context-window management ────────
        use_rag = len(text) > _RAG_CHAR_THRESHOLD
        input_rag: InputRAG | None = None
        agent_texts: dict[str, str] = {}

        if use_rag:
            input_rag = InputRAG()
            input_rag.index_document(text)
            full_doc = input_rag.assemble_all()
            for agent in chosen:
                agent_texts[agent.name] = full_doc
            warnings.append(
                f"Input RAG active: document chunked into {input_rag.total_chunks} segments."
            )
        else:
            for agent in chosen:
                agent_texts[agent.name] = text

        yield {
            "event": "start",
            "document_id": doc_id,
            "total_agents": len(chosen),
            "agent_names": [a.name for a in chosen],
            "analysis_mode": analysis_mode,
        }

        # Real pipeline numbers (no invention) — emitted right after start so
        # the client can log "Input RAG: N segments" up front instead of
        # waiting for the final warnings list.
        yield {
            "event": "pipeline_meta",
            "chunks": input_rag.total_chunks if input_rag else 1,
            "warnings": list(warnings),
        }

        # Run agents concurrently (matching the non-stream path) and emit
        # agent_started + agent_done events through an asyncio.Queue. Items
        # are tagged dicts so the generator can dispatch on kind. The
        # semaphore caps in-flight provider calls to MAX_CONCURRENCY so we
        # don't blow past per-provider rate limits.
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        queue: asyncio.Queue = asyncio.Queue()

        async def _run_and_enqueue(agent: BaseAgent) -> None:
            # Signal that this agent is about to start (fired before the
            # semaphore wait so the client sees per-agent activation even
            # when MAX_CONCURRENCY < len(chosen)).
            await queue.put({
                "kind": "started",
                "agent": agent.name,
                "bias_type": agent.bias_type,
            })
            async with sem:
                anns, err, reasoning = await agent.run(
                    text=agent_texts[agent.name],
                    source_text=text,
                    references=references,
                    mode=effective_mode,
                    analysis_mode=analysis_mode,
                    provider=provider,
                )
            kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]
            info = {
                "agent": agent.name,
                "bias_type": agent.bias_type,
                "prompt_version": agent.prompt_version_for(analysis_mode),
                "raw_count": len(anns),
                "kept_count": len(kept),
                "error": err,
                "reasoning": reasoning,
            }
            await queue.put({"kind": "done", "kept": kept, "info": info, "err": err})

        tasks = [asyncio.create_task(_run_and_enqueue(a)) for a in chosen]

        agent_infos: dict[str, dict] = {}
        all_kept: list[Annotation] = []
        done_count = 0
        while done_count < len(chosen):
            item = await queue.get()
            if item["kind"] == "started":
                yield {
                    "event": "agent_started",
                    "agent": item["agent"],
                    "bias_type": item["bias_type"],
                }
                continue
            # kind == "done"
            info = item["info"]
            kept = item["kept"]
            err = item["err"]
            agent_infos[info["agent"]] = info
            all_kept.extend(kept)
            if err:
                warnings.append(f"{info['agent']}: {err}")
            yield {
                "event": "agent_done",
                **info,
                "annotations": [a.model_dump() for a in kept],
            }
            done_count += 1

        # All "done" puts have happened, so all tasks have finished. Awaiting
        # them surfaces any unexpected exception (agent.run normally swallows
        # them into err, but create_task could hide a programming error).
        await asyncio.gather(*tasks)

        clusters = find_conflict_clusters(all_kept)
        if clusters:
            yield {"event": "aegis_started", "conflicts": len(clusters)}
            merged = await aegis_resolve_clusters(
                all_kept,
                clusters,
                source_text=text,
                provider=aegis_provider,
                aegis=self._get_aegis(),
                analysis_mode=analysis_mode,
            )
            yield {"event": "aegis_done", "resolved": len(clusters)}
        else:
            merged = sorted(all_kept, key=lambda a: (a.span_start, -a.confidence))
        score = self._overall_score(merged, text)

        yield {
            "event": "complete",
            "document_id": doc_id,
            "overall_bias_score": score,
            "annotations": [a.model_dump() for a in merged],
            "agents": list(agent_infos.values()),
            "mode": effective_mode,
            "analysis_mode": analysis_mode,
            "warnings": warnings,
            "provider": provider_config.model_dump_safe(),
        }

    def _overall_score(self, final_annotations: list[Annotation], text: str) -> float:
        """Span Coverage Ratio score.

        The score grows with both the severity-weighted impact of the final
        annotations and the proportion of the document they actually cover.
        This keeps a short fully-biased passage high while preventing a very
        long document with one small flagged sentence from scoring as if the
        whole document were problematic.
        """
        if not final_annotations:
            return 0.0

        total_words = max(1, len(text.split()))
        severity_weights = {"high": 3.0, "medium": 2.0, "low": 1.0}
        flagged_words = _count_covered_words(text, final_annotations)
        coverage = min(1.0, flagged_words / total_words)

        total_impact = 0.0
        for ann in final_annotations:
            total_impact += ann.confidence * severity_weights[ann.severity]

        unique_types = len({ann.bias_type for ann in final_annotations})
        total_impact += 0.15 * max(0, unique_types - 1)

        internal = total_impact * coverage * 3.33
        score = min(10.0, internal) / 10.0
        return round(score, 3)


def _count_covered_words(text: str, annotations: list[Annotation]) -> int:
    """Count unique document words covered by any annotation span."""
    if not text or not annotations:
        return 0

    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not word_spans:
        return 0

    covered = 0
    for word_start, word_end in word_spans:
        if any(
            ann.span_start < word_end and ann.span_end > word_start
            for ann in annotations
        ):
            covered += 1
    return covered


def _resolve_mode(mode: Mode, warnings: list[str]) -> tuple[Mode, list[str]]:
    if mode == "premium":
        warnings.append(
            "Premium mode: LLM reranker active."
        )
        # Premium now runs for real — RAG pipeline handles it
        return "lite", warnings
    if mode == "adaptive":
        warnings.append("Adaptive mode requested; running Lite first pass.")
        return "lite", warnings
    return mode, warnings
