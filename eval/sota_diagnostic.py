"""Diagnostic: why does GPT-5.4 return 0 flags on the systematic review?

Test 1: SAME systematic review text across multiple models.
Test 2: KNOWN biased synthetic text on GPT-5.4 (proves the model can flag bias).

Picks model from BIASSCAN_MODEL, picks fixture from BIASSCAN_TEXT in {user, biased}.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

USER_TEXT = """Summary of Studies
The studies included in this systematic review were published between 2016 and 2022. Three studies (50%) were conducted in the USA, while the remaining 3 studies were carried out in Serbia (n=1, 16.7%), the UK (n=1, 16.7%), and the Netherlands (n=1, 16.7%). The sample size of the RCTs included ranged from 12 to 66, for a total of 201 healthy older adults included in the systematic review.
The mean age of the subjects included ranged from 67.0±1.0 years to 76.0±5.6 years. It should be noticed that Yoshino et al. did not characterize the study sample for gender differences. As a result, based on the remaining 181 participants, the sample of the present study was composed of 83 males and 98 females, with a body composition assessed by BMI ranging from 25.3±1.3 kg/ m2 to 28.6±3.9 kg/m2. Interestingly, 4 studies did not report any standardization in terms of diet and physical activity during the study protocol, while both Zanini et al. and Yoshino et al. instructed study participants to maintain usual diet and physical activity.
Control groups were characterized by placebo administration in all the studies included in the present review (n=6, 100%), indistinguishable from the nutraceutical treatment administered in the intervention group.
Nutraceutical intervention
The intervention was characterized by the administration of several nutraceuticals based on the study included. More in detail, the experimental treatment included: sodium nitrite, fish oil-derived n-3 polyunsaturated fatty acids (PUFA), hydrogen-rich water, nicotinamide riboside (NR), urolithin A, and whey protein powder.
Main findings in terms of muscle mitochondrial modifications
Yoshino et al. assessed mitochondrial function by microarray analyses. The study showed significant improvements (p<0.05) in respiratory electron transport ATP synthesis. On the other hand, Liu et al. investigated the effect of urolithin A supplementation on the mean change in mitochondrial oxidative phosphorylation capacity, without reporting significant differences (p=NS). The study by Elhassan et al. assessed the high-resolution respirometry on muscle without reporting significant improvement in skeletal muscle oxidative phosphorylation (p=NS). Accordingly, the RCT by Connell et al. assessed mitochondrial oxidative capacity by ex vivo high-resolution respirometry. However, no significant differences between groups were underlined (p=0.882). Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (-35%; P<0.05) after sodium nitrite supplementation. Liu et al. assessed the effect of urolithin A reporting a significant reduction in plasma acylcarnitine levels (p<0.05). Elhassan et al. found significant improvement in the NAM methylation clearance pathways. Yoshino et al. showed significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05) after n-3 PUFA therapy compared with baseline. In contrast, no significant improvements were reported in PPARGC1A, PPARA, PDHA1, CPT1B, CS, UQCRC2, COX4I1, and COX5B gene expression."""

BIASED_TEXT = (
    "These findings clearly demonstrate that intervention X causes substantial "
    "reduction in anxiety across all populations. Three randomised trials "
    "definitively confirm the benefit. While one observational study reported "
    "null results, methodological limitations preclude drawing strong conclusions "
    "from that work. The intervention unequivocally reduces risk of relapse by "
    "40% and is broadly applicable to all patients seeking treatment."
)


def fmt(label: str, model: str, score: float, anns, warnings) -> str:
    lines = [f"\n{'='*70}", f"  {label}  ·  model: {model}", "="*70]
    lines.append(f"  score: {score*10:.2f}/10")
    if not anns:
        lines.append("  flags: 0  (model returned no flags)")
    else:
        lines.append(f"  flags: {len(anns)}")
        for i, a in enumerate(anns, 1):
            lines.append(f"  {i}. {a.agent_name} · {a.bias_type} · {a.severity} · conf {a.confidence:.2f}")
            lines.append(f"     {a.flagged_text[:100]!r}")
    err_warnings = [w for w in warnings if any(x in w for x in (": OpenRouter", ": error", "retry failed"))]
    if err_warnings:
        lines.append(f"  errors: {len(err_warnings)}")
        for w in err_warnings:
            lines.append(f"     {w[:120]}")
    return "\n".join(lines)


async def main() -> int:
    api_key = os.environ["BIASSCAN_API_KEY"]
    model = os.environ.get("BIASSCAN_MODEL", "openai/gpt-5.4-mini")
    fixture = os.environ.get("BIASSCAN_TEXT", "user").lower()
    text = USER_TEXT if fixture == "user" else BIASED_TEXT
    label = "SYSTEMATIC REVIEW (user's text)" if fixture == "user" else "SYNTHETIC BIASED FIXTURE"
    cfg = ProviderConfig(provider="openrouter", model=model, api_key=api_key, base_url=None)
    orch = Orchestrator()
    resp = await orch.analyze(
        text=text, references=None, mode="lite",
        analysis_mode="systematic_review",
        provider_config=cfg, agents=None,
    )
    print(fmt(label, model, resp.overall_bias_score, resp.annotations, resp.warnings))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
