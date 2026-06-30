"""Full leaderboard benchmark: every candidate model × every synthesis variant.
Saves raw results to eval/output/leaderboard.json so the website can render them
without re-running.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

SYNTHESIS_DIR = ROOT / "backend" / "app" / "synthesis"
OUT_PATH = ROOT / "eval" / "output" / "leaderboard.json"
PAPERS = [
    "MS_GUT_OG", "MS_GUT_GPT_Made", "MS_GUT_GPT_Edit",
    "Nu_OG",     "Nu_GPT_OG",       "Nu_GPT_Edit",
]
N_RUNS = 5

# Ordered by tier so cheap models run first; if something explodes we don't
# waste State-of-the-Art budget. Tier metadata is preserved in output.
MODELS: list[dict] = [
    # TIER 1 — Budget
    {"id": "meta-llama/llama-4-maverick",        "tier": "Budget"},
    {"id": "x-ai/grok-4.3",                      "tier": "Budget"},
    {"id": "openai/gpt-4.1-mini",                "tier": "Budget"},
    {"id": "google/gemini-2.5-flash",            "tier": "Budget"},
    # TIER 2 — Decent
    {"id": "google/gemini-2.5-pro",              "tier": "Decent"},
    {"id": "openai/gpt-4.1",                     "tier": "Decent"},
    {"id": "anthropic/claude-haiku-4.5",         "tier": "Decent"},
    {"id": "deepseek/deepseek-r1",               "tier": "Decent"},
    # TIER 3 — State of the Art
    {"id": "anthropic/claude-sonnet-4-5",        "tier": "State of the Art"},
    {"id": "openai/gpt-5.5",                     "tier": "State of the Art"},
    {"id": "google/gemini-2.5-pro-preview",      "tier": "State of the Art"},
    {"id": "z-ai/glm-5.2",                       "tier": "State of the Art"},
]


async def one_run(orch: Orchestrator, text: str, cfg: ProviderConfig) -> dict:
    try:
        resp = await orch.analyze(
            text=text, references=None, mode="lite",
            analysis_mode="systematic_review",
            provider_config=cfg, agents=None,
        )
        flags = [
            {"agent": a.agent_name, "bias_type": a.bias_type, "severity": a.severity,
             "confidence": a.confidence, "flagged_text": a.flagged_text[:140]}
            for a in resp.annotations
        ]
        return {"ok": True, "score": resp.overall_bias_score * 10,
                "n_flags": len(flags), "flags": flags}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "score": None, "n_flags": 0, "flags": []}


async def benchmark_model(orch: Orchestrator, model_meta: dict, api_key: str) -> dict:
    model = model_meta["id"]
    cfg = ProviderConfig(provider="openrouter", model=model, api_key=api_key, base_url=None)
    print(f"\n{'#'*80}\n# [{model_meta['tier']:>20}] {model}\n{'#'*80}", flush=True)
    result = {"model": model, "tier": model_meta["tier"], "papers": {}}
    for paper in PAPERS:
        path = SYNTHESIS_DIR / f"{paper}.txt"
        if not path.is_file():
            print(f"  MISSING: {path}", flush=True)
            continue
        text = path.read_text(encoding="utf-8")
        t0 = time.time()
        runs = await asyncio.gather(*(one_run(orch, text, cfg) for _ in range(N_RUNS)))
        elapsed = time.time() - t0
        scores = [r["score"] for r in runs if r["ok"] and r["score"] is not None]
        m = statistics.mean(scores) if scores else 0.0
        sd = statistics.stdev(scores) if len(scores) >= 2 else 0.0
        result["papers"][paper] = {
            "runs": runs, "mean": m, "sd": sd,
            "valid": len(scores), "elapsed_s": round(elapsed, 1),
        }
        run_str = " ".join(
            f"{r['score']:>5.2f}" if r['ok'] and r['score'] is not None else " FAIL"
            for r in runs
        )
        print(f"  {paper:<20} runs=[{run_str}]  mean={m:>5.2f}  sd={sd:>4.2f}  ({elapsed:.0f}s)",
              flush=True)
    return result


async def main() -> int:
    api_key = os.environ["BIASSCAN_API_KEY"]
    orch = Orchestrator()
    all_results: list[dict] = []
    started = time.time()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, mm in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] starting model {mm['id']} (elapsed so far {time.time()-started:.0f}s)",
              flush=True)
        try:
            r = await benchmark_model(orch, mm, api_key)
        except Exception as e:
            print(f"  ! benchmark for {mm['id']} crashed: {type(e).__name__}: {e}", flush=True)
            r = {"model": mm["id"], "tier": mm["tier"], "papers": {},
                 "crashed": f"{type(e).__name__}: {e}"}
        all_results.append(r)
        # Write incrementally so a crash doesn't lose finished models.
        OUT_PATH.write_text(json.dumps({
            "papers": PAPERS, "n_runs": N_RUNS,
            "results": all_results,
            "completed": i, "total": len(MODELS),
        }, indent=2))
        print(f"  → saved progress to {OUT_PATH} ({i}/{len(MODELS)})", flush=True)

    col_w = 14
    header_papers = "  ".join(f"{p[:col_w]:<{col_w}}" for p in PAPERS)
    sep = "=" * (2 + 1 + 42 + 2 + 22 + 2 + col_w * len(PAPERS) + 2 * len(PAPERS))
    print(f"\n{sep}")
    print("LEADERBOARD — mean ± SD per (model, paper)")
    print(sep)
    print(f"{'#':>2}  {'Model':<42}  {'Tier':<22}  {header_papers}")
    print("-" * len(sep))

    def sortkey(r):
        scores = [v.get("mean", -1) for v in r.get("papers", {}).values() if v.get("mean") is not None]
        return -(sum(scores) / len(scores)) if scores else 1

    for rank, r in enumerate(sorted(all_results, key=sortkey), 1):
        short = r["model"].split("/")[-1][:40]
        cells = []
        for p in PAPERS:
            d = r.get("papers", {}).get(p, {})
            cells.append(f"{d['mean']:>5.2f}±{d['sd']:<4.2f}" if d else f"{'—':>{col_w}}")
        print(f"{rank:>2}  {short:<42}  {r['tier']:<22}  {'  '.join(cells)}")
    print(sep)
    print(f"\nDone. Total elapsed: {time.time()-started:.0f}s. Raw JSON: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
