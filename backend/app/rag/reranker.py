"""
LLM-based reranker for retrieved chunks.

After hybrid retrieval returns 2×top_k candidates, the reranker asks the
same user-selected LLM to score each chunk's relevance to the agent's
bias-detection focus on a 0–10 scale. Chunks below minimum_relevance
are filtered out.

Inspired by LlmReranker.cs from matt-bentley/LLM-RAG-Architecture, adapted
for BiasScan's multi-agent bias detection use case.

This is optional — activated only in premium mode or when explicitly enabled.
In lite mode the hybrid retrieval results are used directly.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .vector_store import SearchResult

logger = logging.getLogger(__name__)

# System prompt for the reranker LLM call
_RERANKER_SYSTEM = """You are a relevance scorer for a bias detection system.
You will receive a FOCUS description (what kind of bias to look for) and a list
of text chunks. Score each chunk from 0 to 10 based on how likely it contains
the described bias pattern or is relevant to detecting that bias.

Scoring guide:
  0-2: Irrelevant — factual, neutral, no bias signals
  3-4: Marginally relevant — some related language but unlikely biased
  5-6: Moderately relevant — contains language worth examining for bias
  7-8: Highly relevant — strong bias signals or patterns present
  9-10: Critical — clear bias that an agent should definitely analyze

Return ONLY a JSON array of objects: [{"index": 0, "score": 7}, ...]
No explanation. Just the JSON array."""


@dataclass
class RerankedResult:
    """A search result with reranking metadata."""
    chunk_result: SearchResult
    relevance_score: float     # 0-10 from LLM
    original_rank: int
    new_rank: int


async def rerank_chunks(
    results: list[SearchResult],
    agent_name: str,
    bias_focus: str,
    provider,  # LLMProvider
    top_k: int = 5,
    minimum_relevance: float = 4.0,
    max_tokens: int = 1024,
) -> list[SearchResult]:
    """Rerank retrieved chunks using the LLM provider.

    Args:
        results: Retrieved chunks from hybrid search.
        agent_name: Name of the agent (ARGUS, LIBRA, etc.)
        bias_focus: Description of what bias pattern this agent detects.
        provider: The user's LLM provider instance.
        top_k: Maximum results to return after reranking.
        minimum_relevance: Filter threshold (0-10).
        max_tokens: Max tokens for the reranker LLM call.

    Returns:
        Filtered and reranked SearchResult list.
    """
    if not results:
        return []

    if len(results) <= top_k:
        # Not enough candidates to justify an LLM call
        return results

    # Build the scoring request
    chunk_texts = []
    for i, r in enumerate(results):
        text_preview = r.chunk.text[:400]  # keep it short for the reranker
        chunk_texts.append(f"[{i}] {text_preview}")

    user_message = (
        f"FOCUS: {agent_name} agent — detecting {bias_focus}\n\n"
        f"CHUNKS TO SCORE:\n" + "\n\n".join(chunk_texts)
    )

    try:
        raw = await provider.complete(
            system_prompt=_RERANKER_SYSTEM,
            user_message=user_message,
            max_tokens=max_tokens,
        )
        scores = _parse_scores(raw, len(results))
    except Exception as e:
        logger.warning("Reranker LLM call failed (%s), using original ranking", e)
        return results[:top_k]

    # Build reranked results
    scored = []
    for i, (result, score) in enumerate(zip(results, scores)):
        if score >= minimum_relevance:
            scored.append(RerankedResult(
                chunk_result=result,
                relevance_score=score,
                original_rank=i + 1,
                new_rank=0,
            ))

    # Sort by relevance score descending
    scored.sort(key=lambda r: r.relevance_score, reverse=True)
    for i, r in enumerate(scored):
        r.new_rank = i + 1

    # Return as SearchResult list (top_k only)
    return [
        SearchResult(
            chunk=r.chunk_result.chunk,
            score=r.relevance_score / 10.0,  # normalise to 0-1
        )
        for r in scored[:top_k]
    ]


def _parse_scores(raw: str, expected_count: int) -> list[float]:
    """Parse LLM response into a list of scores.

    Handles:
      - Clean JSON array: [{"index": 0, "score": 7}, ...]
      - Code-fenced JSON
      - Fallback: declining scores if parsing fails
    """
    raw = raw.strip()

    # Strip code fences
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)

    # Find the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json.loads(raw[start:end + 1])
            if isinstance(arr, list):
                scores = [0.0] * expected_count
                for item in arr:
                    if isinstance(item, dict):
                        idx = item.get("index", -1)
                        sc = item.get("score", 0)
                        if 0 <= idx < expected_count:
                            scores[idx] = float(sc)
                    elif isinstance(item, (int, float)):
                        # Simple list of scores
                        idx = arr.index(item)
                        if idx < expected_count:
                            scores[idx] = float(item)
                return scores
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: declining scores (preserve original order)
    logger.warning("Reranker response unparseable, using fallback scores")
    return [max(0.0, 8.0 - i * 0.5) for i in range(expected_count)]
