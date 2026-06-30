"""Re-run the 5 models that produced suspect 0-scores in the main benchmark.

Differences vs multi_model_benchmark.py:
  * Captures `warnings` array from each AnalyzeResponse so we can distinguish
    "all agents returned API errors" (warnings full of "ARGUS: OpenRouter
    error 402: …") from "model thought hard and returned no flags"
    (warnings empty or only contain non-error notices).
  * Orders models cheapest first so we maximise data captured before any
    credit-exhaustion failure.
  * Saves to leaderboard_rerun.json — merges into leaderboard.json at the end.
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
LEADERBOARD = ROOT / "eval" / "output" / "leaderboard.json"
OUT_PATH = ROOT / "eval" / "output" / "leaderboard_rerun.json"
PAPERS = ["Nu-OG", "Nu-OG-GPT", "Nu-Edit", "Nu-bias-injected"]
N_RUNS = 2

MODELS: list[dict] = [
    # cheapest first
    {"id": "qwen/qwen-2.5-72b-instruct",  "tier": "Open Source"},
    {"id": "google/gemini-3-flash",       "tier": "Budget Proprietary"},
    {"id": "google/gemini-3.1-pro",       "tier": "State of the Art"},
    {"id": "anthropic/claude-sonnet-4.6", "tier": "State of the Art"},
    {"id": "anthropic/claude-opus-4.6",   "tier": "State of the Art"},
]


def _is_error_warning(w: str) -> bool:
    """True if this warning line looks like an LLM provider error."""
    lowered = w.lower()
    return any(needle in lowered for needle in
               ("openrouter error", "error 4", "error 5", "rate-limit", "timeout"))


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
        agent_errors = [w for w in resp.warnings if _is_error_warning(w)]
        # If >=3 of 5 agents errored, the score is not a real judgment.
        api_failure = len(agent_errors) >= 3
        return {"ok": True, "score": resp.overall_bias_score * 10,
                "n_flags": len(flags), "flags": flags,
                "api_failure": api_failure,
                "n_agent_errors": len(agent_errors),
                "warnings": resp.warnings[:8]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "score": None, "n_flags": 0, "flags": [],
                "api_failure": True, "n_agent_errors": 5, "warnings": []}


async def benchmark_model(orch: Orchestrator, mm: dict, api_key: str) -> dict:
    model = mm["id"]
    cfg = ProviderConfig(provider="openrouter", model=model, api_key=api_key, base_url=None)
    print(f"\n{'#'*80}\n# [{mm['tier']:>20}] {model}\n{'#'*80}", flush=True)
    result = {"model": model, "tier": mm["tier"], "papers": {}}
    for paper in PAPERS:
        path = SYNTHESIS_DIR / f"{paper}.txt"
        text = path.read_text(encoding="utf-8")
        t0 = time.time()
        runs = await asyncio.gather(*(one_run(orch, text, cfg) for _ in range(N_RUNS)))
        elapsed = time.time() - t0
        real_scores = [r["score"] for r in runs
                       if r["ok"] and r["score"] is not None and not r["api_failure"]]
        api_fails = sum(1 for r in runs if r.get("api_failure"))
        m = statistics.mean(real_scores) if real_scores else 0.0
        sd = statistics.stdev(real_scores) if len(real_scores) >= 2 else 0.0
        result["papers"][paper] = {
            "runs": runs, "mean": m, "sd": sd,
            "valid": len(real_scores), "api_failures": api_fails,
            "elapsed_s": round(elapsed, 1),
        }
        fail_marker = f"  ({api_fails}/{N_RUNS} API-fail)" if api_fails else ""
        run_str = " ".join(
            f"{r['score']:>5.2f}" if r['ok'] and r['score'] is not None else " FAIL"
            for r in runs
        )
        print(f"  {paper:<20} runs=[{run_str}]  mean={m:>5.2f}  sd={sd:>4.2f}  "
              f"({elapsed:.0f}s){fail_marker}", flush=True)
        # Surface a few warnings so we can see WHY when it fails
        if api_fails:
            sample = runs[0].get("warnings", [])[:2]
            for w in sample:
                print(f"      ! {w[:150]}", flush=True)
    return result


async def main() -> int:
    api_key = os.environ["BIASSCAN_API_KEY"]
    orch = Orchestrator()
    rerun: list[dict] = []
    started = time.time()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, mm in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] starting {mm['id']} (elapsed {time.time()-started:.0f}s)",
              flush=True)
        try:
            r = await benchmark_model(orch, mm, api_key)
        except Exception as e:
            print(f"  ! benchmark crashed: {type(e).__name__}: {e}", flush=True)
            r = {"model": mm["id"], "tier": mm["tier"], "papers": {},
                 "crashed": f"{type(e).__name__}: {e}"}
        rerun.append(r)
        OUT_PATH.write_text(json.dumps({
            "papers": PAPERS, "n_runs": N_RUNS,
            "results": rerun, "completed": i, "total": len(MODELS),
        }, indent=2))

    # Merge into the main leaderboard.json, replacing the suspect rows
    main_data = json.load(LEADERBOARD.open())
    rerun_models = {r["model"] for r in rerun}
    merged = [r for r in main_data["results"] if r["model"] not in rerun_models] + rerun
    main_data["results"] = merged
    LEADERBOARD.write_text(json.dumps(main_data, indent=2))
    print(f"\nMerged {len(rerun)} re-run entries into {LEADERBOARD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
