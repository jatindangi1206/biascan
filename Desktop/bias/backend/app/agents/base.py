from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from ..config import (
    ANTHROPIC_API_KEY,
    MODEL,
    MAX_TOKENS,
    PROMPT_VERSION,
    PROMPTS_DIR,
)
from ..schemas import Annotation, BiasType


class BaseAgent:
    """Base class for a single bias-detection agent.

    Subclasses set: name, bias_type, prompt_filename.
    """

    name: str = ""
    bias_type: BiasType = "confirmation_bias"
    prompt_filename: str = ""

    def __init__(self, client: AsyncAnthropic | None = None):
        self._client = client
        self._system_prompt: str | None = None

    @property
    def prompt_version(self) -> str:
        return PROMPT_VERSION

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        return self._client

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

    async def run(self, text: str, references: str | None, mode: str) -> tuple[list[Annotation], str | None]:
        """Returns (annotations, error_message). error_message is None on success."""
        if not ANTHROPIC_API_KEY:
            return [], "ANTHROPIC_API_KEY not set on server."

        system_prompt = self.load_prompt()
        user_msg = self.build_user_message(text, references, mode)
        try:
            resp = await self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:  # network / api errors
            return [], f"{type(e).__name__}: {e}"

        raw = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
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

        # Repair offsets by finding the flagged_text in source if offsets are off.
        if (
            isinstance(span_start, int)
            and isinstance(span_end, int)
            and 0 <= span_start < span_end <= len(source_text)
            and source_text[span_start:span_end] == flagged
        ):
            pass
        elif flagged:
            idx = source_text.find(flagged)
            if idx >= 0:
                span_start, span_end = idx, idx + len(flagged)
            else:
                # Best effort: keep original ints if sane.
                if not (isinstance(span_start, int) and isinstance(span_end, int)):
                    return None
        else:
            return None

        confidence = float(item.get("confidence", 0.0))
        severity = item.get("severity", "low")
        if severity not in ("low", "medium", "high"):
            severity = "low"

        # Strip known top-level keys; the rest become "extras" (mechanism, etc.)
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


_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    m = _FENCED.search(raw)
    candidate = m.group(1) if m else raw
    # If the model returned a bare list, wrap it.
    candidate_stripped = candidate.strip()
    if candidate_stripped.startswith("["):
        candidate_stripped = '{"annotations": ' + candidate_stripped + "}"
    try:
        return json.loads(candidate_stripped)
    except json.JSONDecodeError:
        # Try to locate the first { ... } object.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
