from __future__ import annotations
import asyncio
import logging
import uuid
from typing import AsyncIterator, Iterable

from ..config import CONFIDENCE_FLOOR
from ..providers import build_provider, LLMError, ProviderConfig
from ..rag import InputRAG, EvidenceRAG
from ..rag.reranker import rerank_chunks
from ..schemas import (
    AgentRunInfo,
    AnalyzeResponse,
    Annotation,
    Mode,
)
from . import ALL_AGENTS
from .base import BaseAgent
from .meta_evaluator import meta_evaluate

logger = logging.getLogger(__name__)

# Threshold (chars) above which we activate Input RAG chunking.
# Below this the full text is short enough to send directly.
_RAG_CHAR_THRESHOLD = 6_000  # ~1500 tokens


class Orchestrator:
    """Stateless. One instance per process is fine; agents are reused for their
    cached prompts only — no API client state."""

    def __init__(self) -> None:
        self._agents: list[BaseAgent] = [cls() for cls in ALL_AGENTS]
        self._evidence_rag: EvidenceRAG | None = None

    def _get_evidence_rag(self) -> EvidenceRAG:
        """Lazily build the evidence index (once per process)."""
        if self._evidence_rag is None:
            self._evidence_rag = EvidenceRAG()
            try:
                self._evidence_rag.build_index(include_eval_exemplars=False)
                logger.info(
                    "Evidence RAG built: %d patterns indexed",
                    self._evidence_rag.pattern_count,
                )
            except Exception as e:
                logger.warning("Evidence RAG build failed: %s", e)
        return self._evidence_rag

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
        extra_warnings: list[str] | None = None,
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
                provider=provider_config.model_dump_safe(),
            )

        effective_mode, warnings = _resolve_mode(mode, warnings)
        chosen = self.select_agents(agents)
        if agents and len(chosen) < len({a.upper() for a in agents}):
            warnings.append(
                f"Some requested agents were unknown and skipped. Running: "
                f"{', '.join(a.name for a in chosen)}."
            )

        # ── RAG: chunk + retrieve per-agent ──────────────────────────
        use_rag = len(text) > _RAG_CHAR_THRESHOLD
        use_reranker = mode == "premium"
        input_rag: InputRAG | None = None
        agent_texts: dict[str, str] = {}

        if use_rag:
            input_rag = InputRAG()
            input_rag.index_document(text)
            for agent in chosen:
                # Retrieve 2× candidates if reranker is active
                retrieve_k = 20 if use_reranker else 10
                results = input_rag.retrieve_for_agent(agent.name, top_k=retrieve_k)

                # LLM reranker (premium mode only)
                if use_reranker and len(results) > 10:
                    try:
                        results = await rerank_chunks(
                            results,
                            agent_name=agent.name,
                            bias_focus=agent.bias_type,
                            provider=provider,
                            top_k=10,
                            minimum_relevance=4.0,
                        )
                    except Exception as e:
                        logger.warning("Reranker failed for %s: %s", agent.name, e)

                agent_texts[agent.name] = InputRAG.assemble(
                    results, agent_name=agent.name,
                )
            logger.info(
                "Input RAG active: %d chunks, serving %d agents (reranker=%s)",
                input_rag.total_chunks, len(chosen), use_reranker,
            )
            warnings.append(
                f"Input RAG active: document chunked into {input_rag.total_chunks} "
                f"segments for targeted agent analysis."
            )
        else:
            for agent in chosen:
                agent_texts[agent.name] = text

        # Evidence RAG (for post-hoc cross-check)
        evidence_rag = self._get_evidence_rag()

        # ── Run agents in parallel ───────────────────────────────────
        sem = asyncio.Semaphore(3)

        async def _run_with_sem(agent: BaseAgent):
            async with sem:
                return await agent.run(
                    text=agent_texts[agent.name],
                    source_text=text,
                    references=references,
                    mode=effective_mode,
                    provider=provider,
                )

        results = await asyncio.gather(
            *(_run_with_sem(agent) for agent in chosen),
            return_exceptions=False,
        )

        infos: list[AgentRunInfo] = []
        all_annotations: list[Annotation] = []
        for agent, (anns, err) in zip(chosen, results):
            kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]

            # Evidence RAG cross-check: boost or penalise confidence
            if evidence_rag.is_built:
                kept = _evidence_crosscheck(kept, evidence_rag)

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
        extra_warnings: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """Async generator — yields SSE-ready dicts as each agent completes."""
        warnings: list[str] = list(extra_warnings or [])

        try:
            provider = build_provider(provider_config)
        except LLMError as e:
            yield {"event": "error", "message": str(e)}
            return

        effective_mode, warnings = _resolve_mode(mode, warnings)
        chosen = self.select_agents(agents)
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"

        # ── RAG: chunk + retrieve per-agent ──────────────────────────
        use_rag = len(text) > _RAG_CHAR_THRESHOLD
        use_reranker = mode == "premium"
        input_rag: InputRAG | None = None
        agent_texts: dict[str, str] = {}

        if use_rag:
            input_rag = InputRAG()
            input_rag.index_document(text)
            for agent in chosen:
                retrieve_k = 20 if use_reranker else 10
                results = input_rag.retrieve_for_agent(agent.name, top_k=retrieve_k)

                if use_reranker and len(results) > 10:
                    try:
                        results = await rerank_chunks(
                            results,
                            agent_name=agent.name,
                            bias_focus=agent.bias_type,
                            provider=provider,
                            top_k=10,
                            minimum_relevance=4.0,
                        )
                    except Exception as e:
                        logger.warning("Reranker failed for %s: %s", agent.name, e)

                agent_texts[agent.name] = InputRAG.assemble(
                    results, agent_name=agent.name,
                )
            warnings.append(
                f"Input RAG active: document chunked into {input_rag.total_chunks} segments."
            )
        else:
            for agent in chosen:
                agent_texts[agent.name] = text

        evidence_rag = self._get_evidence_rag()

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
                text=agent_texts[agent.name],
                source_text=text,
                references=references,
                mode=effective_mode,
                provider=provider,
            )
            kept = [a for a in anns if a.confidence >= CONFIDENCE_FLOOR]

            if evidence_rag.is_built:
                kept = _evidence_crosscheck(kept, evidence_rag)

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


def _evidence_crosscheck(
    annotations: list[Annotation],
    evidence: EvidenceRAG,
) -> list[Annotation]:
    """Cross-check annotations against the evidence index.

    - If flagged text strongly matches a known biased example → confidence boost
    - If flagged text strongly matches a known neutral example → confidence penalty
    - Small adjustments (±0.05–0.10) to avoid overriding the LLM's judgment
    """
    adjusted: list[Annotation] = []
    for ann in annotations:
        matches = evidence.check_annotation(
            ann.flagged_text,
            bias_type=ann.bias_type,
            top_k=3,
        )

        boost = 0.0
        for m in matches:
            label = m.chunk.metadata.get("label", "")
            if label == "biased" and m.score > 0.3:
                boost += 0.05 * m.score
            elif label == "neutral" and m.score > 0.3:
                boost -= 0.05 * m.score

        if boost != 0.0:
            new_conf = max(0.0, min(1.0, ann.confidence + boost))
            ann = ann.model_copy(update={"confidence": round(new_conf, 3)})

        adjusted.append(ann)

    return adjusted


def _resolve_mode(mode: Mode, warnings: list[str]) -> tuple[Mode, list[str]]:
    if mode == "premium":
        warnings.append(
            "Premium mode: hybrid RAG + LLM reranker + evidence cross-check active."
        )
        # Premium now runs for real — RAG pipeline handles it
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
