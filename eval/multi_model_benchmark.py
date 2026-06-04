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
PAPERS = ["Nu-OG", "Nu-OG-GPT", "Nu-Edit", "Nu-bias-injected"]
N_RUNS = 2

# Ordered by tier so cheap models run first; if something explodes we don't
# waste TIER 4 money. Tier metadata is preserved in output for the leaderboard.
MODELS: list[dict] = [
    # TIER 1 — Open Source
    {"id": "qwen/qwen-2.5-72b-instruct",                 "tier": "Open Source"},
    {"id": "meta-llama/llama-4-maverick",                "tier": "Open Source"},
    {"id": "deepseek/deepseek-v4-flash",                 "tier": "Open Source"},
    {"id": "qwen/qwen3-235b-a22b",                       "tier": "Open Source"},
    {"id": "mistralai/mistral-small-3.1-24b-instruct",   "tier": "Open Source"},
    # TIER 2 — Budget Proprietary
    {"id": "openai/gpt-4o-mini",                         "tier": "Budget Proprietary"},
    {"id": "openai/gpt-4.1-nano",                        "tier": "Budget Proprietary"},
    {"id": "openai/gpt-4.1-mini",                        "tier": "Budget Proprietary"},
    {"id": "google/gemini-2.5-flash-lite",               "tier": "Budget Proprietary"},
    {"id": "google/gemini-3-flash",                      "tier": "Budget Proprietary"},
    # TIER 3 — Decent
    {"id": "google/gemini-2.5-flash",                    "tier": "Decent"},
    {"id": "anthropic/claude-haiku-4.5",                 "tier": "Decent"},
    {"id": "google/gemini-2.5-pro",                      "tier": "Decent"},
    {"id": "openai/gpt-4o",                              "tier": "Decent"},
    # TIER 4 — State of the Art
    {"id": "google/gemini-3.1-pro",                      "tier": "State of the Art"},
    {"id": "openai/gpt-5.4",                             "tier": "State of the Art"},
    {"id": "anthropic/claude-sonnet-4.6",                "tier": "State of the Art"},
    {"id": "anthropic/claude-opus-4.6",                  "tier": "State of the Art"},
]


async def one_run(orch: Orchestrator, text: str, cfg: ProviderConfig) -> dict:
    try:
        resp = await orch.analyze(
            text=text, references=None, mode="lite",
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

    # Leaderboard table (sorted by Nu-bias-injected mean descending — truth test)
    print("\n" + "=" * 130)
    print("LEADERBOARD — mean ± SD per (model, paper)")
    print("=" * 130)
    print(f"{'#':>2}  {'Model':<42}  {'Tier':<22}  {'Nu-OG':>11}  {'Nu-OG-GPT':>11}  {'Nu-Edit':>11}  {'Nu-injected':>11}")
    print("-" * 130)

    def sortkey(r):
        ni = r.get("papers", {}).get("Nu-bias-injected", {})
        return -(ni.get("mean", -1))

    sorted_results = sorted(all_results, key=sortkey)
    for rank, r in enumerate(sorted_results, 1):
        short = r["model"].split("/")[-1][:40]
        cells = []
        for p in PAPERS:
            d = r.get("papers", {}).get(p, {})
            if d:
                cells.append(f"{d['mean']:>5.2f}±{d['sd']:<4.2f}")
            else:
                cells.append(f"{'—':>11}")
        print(f"{rank:>2}  {short:<42}  {r['tier']:<22}  {cells[0]}  {cells[1]}  {cells[2]}  {cells[3]}")
    print("=" * 130)
    print(f"\nDone. Total elapsed: {time.time()-started:.0f}s. Raw JSON: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
