from __future__ import annotations

from ..schemas import Annotation


def meta_evaluate(annotations: list[Annotation]) -> list[Annotation]:
    """Mark conflicts when two agents flag overlapping spans for different biases.

    Preserves all annotations (no silent drops). Sorts by span_start, then by
    descending confidence within the same span.
    """
    if not annotations:
        return []

    flagged = [a.model_copy() for a in annotations]
    flagged.sort(key=lambda a: (a.span_start, a.span_end))

    n = len(flagged)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = flagged[i], flagged[j]
            if b.span_start >= a.span_end:
                continue  # no overlap, list is sorted
            if a.bias_type == b.bias_type:
                continue
            if _iou(a, b) >= 0.3:
                a.conflict = True
                b.conflict = True

    flagged.sort(key=lambda a: (a.span_start, -a.confidence))
    return flagged


def _iou(a: Annotation, b: Annotation) -> float:
    inter_start = max(a.span_start, b.span_start)
    inter_end = min(a.span_end, b.span_end)
    inter = max(0, inter_end - inter_start)
    if inter == 0:
        return 0.0
    union = (a.span_end - a.span_start) + (b.span_end - b.span_start) - inter
    return inter / union if union > 0 else 0.0
