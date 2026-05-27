"""Live smoke test for steps #1–#8. Runs the analyze pipeline once on
a deliberately-clean text and once on a deliberately-biased text, then
prints a compact diff so we can verify the new prompts + scoring.

USAGE
-----
    export BIASSCAN_API_KEY=sk-or-v1-...
    export BIASSCAN_MODEL=openai/gpt-4o-mini   # or any OpenRouter model
    python -m eval.smoke_test
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

CLEAN_TEXT = (
    "The evidence for intervention X is mixed: three randomised trials show a "
    "moderate benefit (effect size 0.3–0.5), while one well-powered observational "
    "study reported a null result. The literature does not yet support a "
    "definitive conclusion, and further trials are warranted."
)

BIASED_TEXT = (
    "These findings clearly demonstrate that intervention X causes substantial "
    "reduction in anxiety across all populations. Three randomised trials "
    "definitively confirm the benefit. While one observational study reported "
    "null results, methodological limitations preclude drawing strong conclusions "
    "from that work. The intervention unequivocally reduces risk of relapse by "
    "40% and is broadly applicable to all patients seeking treatment."
)


def fmt_anns(label: str, score: float, anns) -> str:
    lines = [f"\n=== {label} ===", f"  score: {score*10:.2f}/10"]
    if not anns:
        lines.append("  (no flags)")
        return "\n".join(lines)
    lines.append(f"  flags: {len(anns)}")
    by_sev = {"high": 0, "medium": 0, "low": 0}
    confidences: list[float] = []
    types: set[str] = set()
    for a in anns:
        by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
        confidences.append(a.confidence)
        types.add(a.bias_type)
    lines.append(f"  severity: {by_sev}")
    lines.append(f"  bias types: {sorted(types)}")
    lines.append(f"  confidences: {[round(c, 2) for c in confidences]}")
    # The discrete tier values we expect to see (within ±0.01 from RAG nudges)
    expected_tiers = {1.0, 0.8, 0.6, 0.4}
    for c in confidences:
        nearest = min(expected_tiers, key=lambda t: abs(t - c))
        delta = c - nearest
        marker = "✓" if abs(delta) <= 0.16 else "?"
        lines.append(f"    {c:.3f} → nearest tier {nearest:.1f} (Δ {delta:+.3f}) {marker}")
    for i, a in enumerate(anns, 1):
        lines.append(
            f"  flag {i}: {a.agent_name} · {a.bias_type} · {a.severity} · conf {a.confidence:.2f}"
        )
        lines.append(f"    text: {a.flagged_text[:90]!r}")
    return "\n".join(lines)


async def main() -> int:
    api_key = os.environ.get("BIASSCAN_API_KEY")
    model = os.environ.get("BIASSCAN_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        print("ERROR: set BIASSCAN_API_KEY (OpenRouter key).", file=sys.stderr)
        return 1

    cfg = ProviderConfig(
        provider="openrouter",
        model=model,
        api_key=api_key,
        base_url=None,
    )
    print(f"Using provider=openrouter model={model}")

    orch = Orchestrator()
    print("\n[1/2] Analysing clean fixture (expected: 0–1 low-severity flags)…")
    clean_resp = await orch.analyze(
        text=CLEAN_TEXT, references=None, mode="lite",
        provider_config=cfg, agents=None,
    )
    print(fmt_anns("CLEAN", clean_resp.overall_bias_score, clean_resp.annotations))

    print("\n[2/2] Analysing biased fixture (expected: 3–5 flags, ≥1 high-severity)…")
    biased_resp = await orch.analyze(
        text=BIASED_TEXT, references=None, mode="lite",
        provider_config=cfg, agents=None,
    )
    print(fmt_anns("BIASED", biased_resp.overall_bias_score, biased_resp.annotations))

    print("\n=== DIAGNOSTIC ===")
    print(f"clean score: {clean_resp.overall_bias_score*10:.2f}/10")
    print(f"biased score: {biased_resp.overall_bias_score*10:.2f}/10")
    ranked = biased_resp.overall_bias_score > clean_resp.overall_bias_score
    print(f"biased > clean? {ranked}  {'✓' if ranked else '✗ FAIL'}")
    return 0 if ranked else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
