from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from ..config import DEFAULT_MAX_TOKENS, PROMPT_VERSION, PROMPTS_DIR
from ..providers import LLMError, LLMProvider
from ..schemas import Annotation, BiasType


class BaseAgent:
    """A single bias-detection agent. Stateless: takes a provider on each run."""

    name: str = ""
    bias_type: BiasType = "confirmation_bias"
    prompt_filename: str = ""

    def __init__(self) -> None:
        self._system_prompt: str | None = None

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    def load_prompt(self) -> str:
        if self._system_prompt is not None:
            return self._system_prompt
        path: Path = PROMPTS_DIR / PROMPT_VERSION / self.prompt_filename
        self._system_prompt = path.read_text(encoding="utf-8")
        return self._system_prompt

    def build_user_message(self, text: str, references: str | None, mode: str) -> str:
        ref_block = (references or "").strip() or "[No reference list provided.]"
        return (
            f"=== MODE === {mode}\n\n"
            f"=== SYNTHESIS TEXT (analyse this) ===\n{text}\n\n"
            f"=== REFERENCE LIST (for context) ===\n{ref_block}\n\n"
            f"Apply your reasoning protocol. Return ONLY the JSON object with key "
            f"'annotations'. Character offsets must index into the SYNTHESIS TEXT exactly as given."
        )

    async def run(
        self,
        *,
        text: str,
        references: str | None,
        mode: str,
        provider: LLMProvider,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[list[Annotation], str | None]:
        """Returns (annotations, error_message). error is None on success."""
        try:
            raw = await provider.complete(
                system_prompt=self.load_prompt(),
                user_message=self.build_user_message(text, references, mode),
                max_tokens=max_tokens,
            )
        except LLMError as e:
            return [], str(e)
        except Exception as e:  # defensive
            return [], f"{type(e).__name__}: {e}"
        return self._parse(raw, text), None

    def _parse(self, raw: str, source_text: str) -> list[Annotation]:
        data = _extract_json(raw)
        if data is None:
            return []
        items = data.get("annotations") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []

        out: list[Annotation] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                ann = self._coerce(item, source_text)
            except Exception:
                continue
            if ann is not None:
                out.append(ann)
        return out

    def _coerce(self, item: dict[str, Any], source_text: str) -> Annotation | None:
        flagged = (item.get("flagged_text") or "").strip()
        span_start = item.get("span_start")
        span_end = item.get("span_end")

        if (
            isinstance(span_start, int)
            and isinstance(span_end, int)
            and 0 <= span_start < span_end <= len(source_text)
            and source_text[span_start:span_end] == flagged
        ):
            pass  # exact offset match — best case
        elif flagged:
            idx = _find_in_source(flagged, source_text)
            if idx >= 0:
                span_start, span_end = idx, idx + len(flagged)
            else:
                # last resort: use provided offsets if they're in range
                if (
                    isinstance(span_start, int)
                    and isinstance(span_end, int)
                    and 0 <= span_start < span_end <= len(source_text)
                ):
                    pass  # accept offset even if text doesn't match character-for-character
                else:
                    return None
        else:
            return None

        confidence = float(item.get("confidence", 0.0))
        severity = item.get("severity", "low")
        if severity not in ("low", "medium", "high"):
            severity = "low"

        known = {
            "bias_type",
            "span_start",
            "span_end",
            "flagged_text",
            "confidence",
            "severity",
            "clean_alternative",
            "false_positive_check",
            "rag_check_needed",
            "rag_query",
        }
        extras = {k: v for k, v in item.items() if k not in known}

        return Annotation(
            bias_type=self.bias_type,
            span_start=int(span_start),
            span_end=int(span_end),
            flagged_text=source_text[int(span_start) : int(span_end)],
            confidence=max(0.0, min(1.0, confidence)),
            severity=severity,  # type: ignore[arg-type]
            clean_alternative=item.get("clean_alternative"),
            false_positive_check=item.get("false_positive_check"),
            rag_check_needed=bool(item.get("rag_check_needed", False)),
            rag_query=item.get("rag_query"),
            agent_name=self.name,
            prompt_version=self.prompt_version,
            extras=extras,
        )


def _normalise(s: str) -> str:
    """Collapse unicode dashes/quotes to ASCII equivalents and normalise whitespace.
    Used for fuzzy span-matching when a model normalises special chars in its output."""
    s = unicodedata.normalize("NFKD", s)
    # Common unicode → ascii substitutions LLMs make
    for src, dst in [
        ("–", "-"), ("—", "-"), ("‒", "-"),  # dashes
        ("‘", "'"), ("’", "'"),                    # single quotes
        ("“", '"'), ("”", '"'),                    # double quotes
        ("…", "..."),                                   # ellipsis
        (" ", " "),                                     # non-breaking space
    ]:
        s = s.replace(src, dst)
    return " ".join(s.split())  # collapse whitespace


_SOURCE_NORM_CACHE: dict[int, str] = {}


def _find_in_source(flagged: str, source: str) -> int:
    """Return the start index of flagged in source, trying several normalisation
    strategies. Returns -1 if no match found."""
    # 1. Exact
    idx = source.find(flagged)
    if idx >= 0:
        return idx

    # 2. Strip trailing ellipsis the model sometimes appends
    stripped = re.sub(r"\.{2,}$", "", flagged).rstrip()
    if len(stripped) >= 20:
        idx = source.find(stripped)
        if idx >= 0:
            return idx

    # 3. Normalised comparison (unicode dashes → hyphens, collapsed whitespace)
    src_key = id(source)
    if src_key not in _SOURCE_NORM_CACHE:
        _SOURCE_NORM_CACHE[src_key] = _normalise(source)
    norm_source = _SOURCE_NORM_CACHE[src_key]
    norm_flagged = _normalise(flagged)
    idx = norm_source.find(norm_flagged)
    if idx >= 0:
        # Map back to original index (approximate — norm may have shifted chars)
        # Walk forward in original to re-anchor
        original_idx = source.lower().find(norm_flagged.lower()[:40])
        if original_idx >= 0:
            return original_idx
        return idx  # best effort

    # 4. Prefix match — find first 50 chars of flagged in source
    prefix = flagged[:50].strip()
    if len(prefix) >= 20:
        idx = source.find(prefix)
        if idx >= 0:
            return idx
        norm_prefix = _normalise(prefix)
        idx = norm_source.find(norm_prefix)
        if idx >= 0:
            return source.lower().find(prefix.lower()[:30]) if source.lower().find(prefix.lower()[:30]) >= 0 else idx

    return -1


_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    m = _FENCED.search(raw)
    candidate = m.group(1) if m else raw
    candidate_stripped = candidate.strip()
    if candidate_stripped.startswith("["):
        candidate_stripped = '{"annotations": ' + candidate_stripped + "}"
    try:
        return json.loads(candidate_stripped)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
