"""AEGIS — Appellate Arbiter / Conflict Resolution Agent.

Triggered only when two or more primary agents flag overlapping spans with
different bias_types. Reads the source span, the conflicting annotations
(including each agent's chain_of_thought), and returns a single consolidated
annotation per the Resolution Hierarchy spelled out in aegis_v1.0.txt.

AEGIS is a sibling of BaseAgent, not a subclass — its I/O contract differs:
its bias_type comes from the model output (the "winning" bias), not from a
fixed class attribute, and it returns at most one annotation.
"""
from __future__ import annotations

import json
from typing import Any

from ..config import (
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_MAX_TOKENS,
    resolve_prompt_filename,
    resolve_prompt_path,
    resolve_prompt_version,
)
from ..providers import LLMError, LLMProvider
from ..schemas import AnalysisMode, Annotation, BiasType
from .base import _extract_json, _find_in_source, _resolve_confidence

_VALID_BIAS_TYPES = {
    "confirmation_bias",
    "certainty_inflation",
    "overgeneralisation",
    "framing_effect",
    "causal_inference_error",
}


class AegisAgent:
    name = "AEGIS"
    prompt_filename = "aegis_v1.0.txt"

    def __init__(self) -> None:
        self._system_prompts: dict[str, str] = {}

    @property
    def prompt_version(self) -> str:
        return self.prompt_version_for(DEFAULT_ANALYSIS_MODE)

    def prompt_version_for(self, analysis_mode: AnalysisMode) -> str:
        return resolve_prompt_version(analysis_mode)

    def prompt_filename_for(self, analysis_mode: AnalysisMode) -> str:
        return resolve_prompt_filename(
            self.prompt_filename,
            self.prompt_version_for(analysis_mode),
        )

    def load_prompt(self, analysis_mode: AnalysisMode) -> str:
        cache_key = self.prompt_version_for(analysis_mode)
        cached = self._system_prompts.get(cache_key)
        if cached is not None:
            return cached
        prompt = resolve_prompt_path(self.prompt_filename, analysis_mode).read_text(
            encoding="utf-8"
        )
        self._system_prompts[cache_key] = prompt
        return prompt

    def build_user_message(self, span_text: str, candidates: list[Annotation]) -> str:
        cand_data = [_annotation_for_aegis(a) for a in candidates]
        return (
            f"[SOURCE TEXT SPAN]\n{span_text}\n\n"
            f"[CONFLICTING ANNOTATIONS]\n{json.dumps(cand_data, indent=2, ensure_ascii=False)}\n\n"
            f"Apply your reasoning protocol from BLOCK 5. Return ONLY the JSON object."
        )

    async def resolve(
        self,
        *,
        span_text: str,
        candidates: list[Annotation],
        source_text: str,
        analysis_mode: AnalysisMode,
        provider: LLMProvider,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> list[Annotation]:
        """Returns one or more consolidated Annotations (one per distinct
        cognitive error AEGIS identifies on the span), or [] on failure.
        Caller is responsible for fallback behaviour when the list is empty."""
        try:
            prompt_version = self.prompt_version_for(analysis_mode)
            raw = await provider.complete(
                system_prompt=self.load_prompt(analysis_mode),
                user_message=self.build_user_message(span_text, candidates),
                max_tokens=max_tokens,
            )
        except LLMError:
            return []
        except Exception:
            return []
        return self._parse(raw, source_text, candidates, prompt_version)

    def _parse(
        self,
        raw: str,
        source_text: str,
        candidates: list[Annotation],
        prompt_version: str,
    ) -> list[Annotation]:
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return []
        items = data.get("annotations")
        if not isinstance(items, list) or not items:
            return []
        out: list[Annotation] = []
        seen_types: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                ann = self._coerce(item, source_text, data, candidates, prompt_version)
            except Exception:
                continue
            if ann is None:
                continue
            # Defensive dedup: AEGIS shouldn't emit two annotations of the
            # same bias_type on the same cluster — drop the duplicate.
            if ann.bias_type in seen_types:
                continue
            seen_types.add(ann.bias_type)
            out.append(ann)
        return out

    def _coerce(
        self,
        item: dict[str, Any],
        source_text: str,
        wrapper: dict[str, Any],
        candidates: list[Annotation],
        prompt_version: str,
    ) -> Annotation | None:
        # bias_type comes from the AEGIS output, validated against the literal.
        bias_type = item.get("bias_type")
        if bias_type not in _VALID_BIAS_TYPES:
            return None

        flagged = (item.get("flagged_text") or "").strip()
        span_start = item.get("span_start")
        span_end = item.get("span_end")

        # Same re-anchoring logic as BaseAgent._coerce.
        if (
            isinstance(span_start, int)
            and isinstance(span_end, int)
            and 0 <= span_start < span_end <= len(source_text)
            and source_text[span_start:span_end] == flagged
        ):
            pass
        elif flagged:
            idx = _find_in_source(flagged, source_text)
            if idx >= 0:
                span_start, span_end = idx, idx + len(flagged)
            elif (
                isinstance(span_start, int)
                and isinstance(span_end, int)
                and 0 <= span_start < span_end <= len(source_text)
            ):
                pass
            else:
                # Fall back to the union of the candidates' spans.
                span_start = min(c.span_start for c in candidates)
                span_end = max(c.span_end for c in candidates)
        else:
            span_start = min(c.span_start for c in candidates)
            span_end = max(c.span_end for c in candidates)

        # Default to "probable" (0.8) if neither certainty nor confidence is provided —
        # AEGIS only fires on real conflicts, so absence implies a moderately confident merge.
        confidence = _resolve_confidence(item) or 0.8
        severity = item.get("severity", "medium")
        if severity not in ("low", "medium", "high"):
            severity = "medium"

        extras: dict[str, Any] = {
            "aegis_chain_of_thought": wrapper.get("chain_of_thought"),
            "mechanism": item.get("mechanism"),
            "resolved_from": [
                {"agent": c.agent_name, "bias_type": c.bias_type}
                for c in candidates
            ],
        }

        return Annotation(
            bias_type=bias_type,  # type: ignore[arg-type]
            span_start=int(span_start),
            span_end=int(span_end),
            flagged_text=source_text[int(span_start) : int(span_end)],
            confidence=max(0.0, min(1.0, confidence)),
            severity=severity,  # type: ignore[arg-type]
            clean_alternative=item.get("clean_alternative"),
            agent_name=self.name,
            prompt_version=prompt_version,
            extras=extras,
        )


def _annotation_for_aegis(a: Annotation) -> dict[str, Any]:
    """Serialise an annotation for the AEGIS user message. Pulls
    chain_of_thought out of extras (set by BaseAgent._parse) if present."""
    cot = a.extras.get("chain_of_thought") if a.extras else None
    return {
        "agent": a.agent_name,
        "bias_type": a.bias_type,
        "span_start": a.span_start,
        "span_end": a.span_end,
        "flagged_text": a.flagged_text,
        "confidence": a.confidence,
        "severity": a.severity,
        "clean_alternative": a.clean_alternative,
        "false_positive_check": a.false_positive_check,
        "chain_of_thought": cot,
    }


__all__ = ["AegisAgent"]
