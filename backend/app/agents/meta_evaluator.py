"""Conflict resolution between primary agents.

When two or more primary agents flag overlapping text spans (IoU >= threshold)
for different bias_types, that's a conflict. AEGIS resolves each cluster of
conflicting annotations down to a single consolidated annotation.

This module provides two primitives:
  - find_conflict_clusters(): pure, synchronous IoU + union-find grouping.
  - aegis_resolve_clusters(): async; calls AEGIS on each cluster in parallel
    and returns the merged annotation list with un-conflicting annotations
    untouched.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from ..providers import LLMProvider
from ..schemas import AnalysisMode, Annotation
from .aegis import AegisAgent

logger = logging.getLogger(__name__)


def find_conflict_clusters(
    annotations: list[Annotation],
    iou_threshold: float = 0.5,
) -> list[list[int]]:
    """Return clusters (lists of annotation indices) where every cluster
    contains 2+ annotations that pairwise overlap (IoU >= threshold) with
    different bias_types. Singleton (non-conflicting) annotations are NOT
    returned — only the clusters AEGIS needs to resolve.

    Uses union-find so transitively-overlapping triples (A↔B, B↔C) collapse
    into one cluster, not two pairwise clusters.

    Default threshold 0.5: only spans whose intersection is at least half
    their union qualify as conflicting. Lower thresholds (e.g. 0.3) pull in
    brushing spans that share only a few words — those are usually two
    distinct findings on adjacent phrases, not a real conflict.
    """
    n = len(annotations)
    if n < 2:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            a, b = annotations[i], annotations[j]
            if a.bias_type == b.bias_type:
                continue
            if _iou(a, b) >= iou_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return [members for members in groups.values() if len(members) >= 2]


async def aegis_resolve_clusters(
    annotations: list[Annotation],
    clusters: list[list[int]],
    source_text: str,
    provider: LLMProvider,
    aegis: AegisAgent,
    analysis_mode: AnalysisMode,
    *,
    context_pad: int = 80,
) -> list[Annotation]:
    """Run AEGIS on each cluster in parallel, then return the merged list:
    every non-conflicting annotation preserved as-is, plus one annotation per
    cluster (AEGIS's resolution, or the highest-confidence original on
    AEGIS failure).
    """
    if not clusters:
        return _sorted(annotations)

    async def _resolve(indices: list[int]) -> tuple[list[int], list[Annotation]]:
        cluster = [annotations[i] for i in indices]
        ctx_lo = max(0, min(a.span_start for a in cluster) - context_pad)
        ctx_hi = min(len(source_text), max(a.span_end for a in cluster) + context_pad)
        span_text = source_text[ctx_lo:ctx_hi]
        resolved = await aegis.resolve(
            span_text=span_text,
            candidates=cluster,
            source_text=source_text,
            analysis_mode=analysis_mode,
            provider=provider,
        )
        return indices, resolved

    results = await asyncio.gather(*(_resolve(c) for c in clusters))

    in_cluster = {i for c in clusters for i in c}
    merged: list[Annotation] = [
        a for i, a in enumerate(annotations) if i not in in_cluster
    ]
    for indices, resolved in results:
        if resolved:
            # AEGIS may return 1 annotation (single root cause) or 2+
            # (genuinely co-occurring distinct biases).
            merged.extend(resolved)
        else:
            # AEGIS failed for this cluster — keep the highest-confidence
            # original so we don't silently drop the flag entirely.
            logger.warning(
                "AEGIS returned no annotations for cluster of %d", len(indices),
            )
            fallback = max(
                (annotations[i] for i in indices),
                key=lambda a: a.confidence,
            )
            merged.append(fallback)

    return _sorted(merged)


def _iou(a: Annotation, b: Annotation) -> float:
    inter_start = max(a.span_start, b.span_start)
    inter_end = min(a.span_end, b.span_end)
    inter = max(0, inter_end - inter_start)
    if inter == 0:
        return 0.0
    union_len = (a.span_end - a.span_start) + (b.span_end - b.span_start) - inter
    return inter / union_len if union_len > 0 else 0.0


def _sorted(annotations: Iterable[Annotation]) -> list[Annotation]:
    return sorted(annotations, key=lambda a: (a.span_start, -a.confidence))


__all__ = ["find_conflict_clusters", "aegis_resolve_clusters"]
