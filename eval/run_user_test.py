"""One-off: run the user's nutraceutical systematic review through analyze()."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

TEXT = """Summary of Studies
The studies included in this systematic review were published between 2016 and 2022. Three studies (50%) were conducted in the USA, while the remaining 3 studies were carried out in Serbia (n=1, 16.7%), the UK (n=1, 16.7%), and the Netherlands (n=1, 16.7%). The sample size of the RCTs included ranged from 12 to 66, for a total of 201 healthy older adults included in the systematic review.
The mean age of the subjects included ranged from 67.0±1.0 years to 76.0±5.6 years. It should be noticed that Yoshino et al. did not characterize the study sample for gender differences. As a result, based on the remaining 181 participants, the sample of the present study was composed of 83 males and 98 females, with a body composition assessed by BMI ranging from 25.3±1.3 kg/ m2 to 28.6±3.9 kg/m2. Interestingly, 4 studies did not report any standardization in terms of diet and physical activity during the study protocol, while both Zanini et al. and Yoshino et al. instructed study participants to maintain usual diet and physical activity.
Control groups were characterized by placebo administration in all the studies included in the present review (n=6, 100%), indistinguishable from the nutraceutical treatment administered in the intervention group. The characteristics of the studies included are presented in detail in Table 1.
Nutraceutical intervention
The intervention was characterized by the administration of several nutraceuticals based on the study included. More in detail, the experimental treatment included: sodium nitrite, fish oil-derived n-3 polyunsaturated fatty acids (PUFA), hydrogen-rich water, nicotinamide riboside (NR), urolithin A, and whey protein powder.
Main findings in terms of muscle mitochondrial modifications
The effects of nutraceuticals on muscle mitochondrial modifications were assessed in 6 terms of: mitochondrial oxidative capacity; mitochondrial antioxidant capacity; mitochondrial volume; mitochondrial bioenergetic capacity; mitochondrial transcriptome. Yoshino et al. assessed mitochondrial function by microarray analyses showing significant improvements (p<0.05) in respiratory electron transport ATP synthesis. On the other hand, Liu et al. investigated the effect of urolithin A supplementation without reporting significant differences (p=NS). The study by Elhassan et al. assessed the high-resolution respirometry on muscle without reporting significant improvement (p=NS). Rossman et al. assessed mitochondrial-specific ROS-bioactivity showing a significant decrease (–35%; P<0.05) after sodium nitrite supplementation. Liu et al. assessed the effect of urolithin A reporting a significant reduction in plasma acylcarnitine levels (p<0.05). Yoshino et al. showed significant improvements in gene expression of UCP3 (~30%, p<0.05) and UQCRC1 (~20%, p<0.05). In contrast, no significant improvements were reported in PPARGC1A, PPARA, PDHA1, CPT1B, CS, UQCRC2, COX4I1, and COX5B gene expression."""


def fmt(label: str, score: float, anns) -> str:
    lines = [f"\n=== {label} ===", f"  score: {score*10:.2f}/10"]
    if not anns:
        lines.append("  (no flags)")
        return "\n".join(lines)
    lines.append(f"  flags: {len(anns)}")
    by_sev = {"high": 0, "medium": 0, "low": 0}
    types: set[str] = set()
    for a in anns:
        by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
        types.add(a.bias_type)
    lines.append(f"  severity: {by_sev}")
    lines.append(f"  bias types: {sorted(types)}")
    for i, a in enumerate(anns, 1):
        lines.append(
            f"  flag {i}: {a.agent_name} · {a.bias_type} · {a.severity} · conf {a.confidence:.2f}"
        )
        lines.append(f"    text: {a.flagged_text[:110]!r}")
    return "\n".join(lines)


async def main() -> int:
    api_key = os.environ["BIASSCAN_API_KEY"]
    model = os.environ.get("BIASSCAN_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    cfg = ProviderConfig(provider="openrouter", model=model, api_key=api_key, base_url=None)
    print(f"Using model: {model}")
    print(f"Text length: {len(TEXT)} chars, {len(TEXT.split())} words")
    orch = Orchestrator()
    print("\nRunning analyze()…")
    resp = await orch.analyze(
        text=TEXT, references=None, mode="lite",
        analysis_mode="systematic_review",
        provider_config=cfg, agents=None,
    )
    print(fmt("USER TEXT", resp.overall_bias_score, resp.annotations))
    print(f"\nwarnings: {resp.warnings}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
