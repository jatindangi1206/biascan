"""Smoke test: MS_GUT × 3 variants × gpt-4.1-mini × 1 run, v1 prompts."""
from __future__ import annotations
import asyncio, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig      # noqa: E402

SYNTHESIS_DIR = ROOT / "backend" / "app" / "synthesis"
VARIANTS = {
    "OG":       "MS_GUT_OG.txt",
    "GPT_Made": "MS_GUT_GPT_Made.txt",
    "GPT_Edit": "MS_GUT_GPT_Edit.txt",
}
MODEL = "openai/gpt-4.1-mini"


async def main() -> None:
    api_key = os.environ["OPENROUTER_API_KEY"]
    cfg = ProviderConfig(provider="openrouter", model=MODEL, api_key=api_key, base_url=None)
    orch = Orchestrator()

    for label, fname in VARIANTS.items():
        text = (SYNTHESIS_DIR / fname).read_text(encoding="utf-8")
        print(f"\n{'='*70}\nMS_GUT — {label}  ({fname})  [{len(text.split())} words]")
        resp = await orch.analyze(
            text=text, references=None, mode="lite",
            analysis_mode="systematic_review",
            provider_config=cfg, agents=None,
        )
        print(f"Overall bias score : {resp.overall_bias_score:.3f}  ({resp.overall_bias_score*10:.1f}/10)")
        print(f"Flags kept         : {len(resp.annotations)}")
        for a in resp.annotations:
            print(f"  [{a.bias_type:30}] conf={a.confidence:.2f}  sev={a.severity}")
            print(f"    \"{a.flagged_text[:100]}\"")
        if resp.warnings:
            print("Warnings:", "; ".join(resp.warnings[:3]))

asyncio.run(main())
