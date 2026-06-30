"""Rerun only the models that failed/were invalid in the first pass.
Merges results back into leaderboard.json, replacing the bad entries.
"""
from __future__ import annotations
import asyncio, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator
from app.providers.base import ProviderConfig

SYNTHESIS_DIR = ROOT / "backend" / "app" / "synthesis"
OUT_PATH = ROOT / "eval" / "output" / "leaderboard.json"
PAPERS = [
    "MS_GUT_OG", "MS_GUT_GPT_Made", "MS_GUT_GPT_Edit",
    "Nu_OG",     "Nu_GPT_OG",       "Nu_GPT_Edit",
]
N_RUNS = 5

# Only the corrected / previously-broken models
RERUN_MODELS: list[dict] = [
    {"id": "x-ai/grok-4.3",                 "tier": "Budget",           "replaces": "x-ai/grok-4-fast"},
    {"id": "google/gemini-2.5-pro",          "tier": "Decent",           "replaces": "google/gemini-3-flash"},
    {"id": "deepseek/deepseek-r1",           "tier": "Decent",           "replaces": "deepseek/deepseek-v4"},
    {"id": "openai/gpt-5.5",                 "tier": "State of the Art", "replaces": None},   # rerun — parse failures
    {"id": "google/gemini-2.5-pro-preview",  "tier": "State of the Art", "replaces": "google/gemini-3.5-pro"},
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
    started = time.time()

    # Load existing results
    existing = json.loads(OUT_PATH.read_text()) if OUT_PATH.is_file() else {"results": []}
    results_by_model: dict[str, dict] = {r["model"]: r for r in existing.get("results", [])}

    for i, mm in enumerate(RERUN_MODELS, 1):
        print(f"\n[{i}/{len(RERUN_MODELS)}] {mm['id']}", flush=True)
        try:
            r = await benchmark_model(orch, mm, api_key)
        except Exception as e:
            print(f"  ! crashed: {e}", flush=True)
            r = {"model": mm["id"], "tier": mm["tier"], "papers": {},
                 "crashed": str(e)}

        # Remove the old bad entry (either same ID or the one it replaces)
        old_id = mm.get("replaces") or mm["id"]
        results_by_model.pop(old_id, None)
        results_by_model[mm["id"]] = r

        # Save after every model
        OUT_PATH.write_text(json.dumps({
            "papers": PAPERS, "n_runs": N_RUNS,
            "results": list(results_by_model.values()),
            "completed": i, "total": len(RERUN_MODELS),
        }, indent=2))
        print(f"  → saved ({i}/{len(RERUN_MODELS)})", flush=True)

    # Final leaderboard
    all_results = list(results_by_model.values())
    col_w = 14
    header_papers = "  ".join(f"{p[:col_w]:<{col_w}}" for p in PAPERS)
    sep = "=" * 140
    print(f"\n{sep}\nLEADERBOARD\n{sep}")
    print(f"{'#':>2}  {'Model':<42}  {'Tier':<22}  {header_papers}")
    print("-" * 140)

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
    print(f"\nDone. Elapsed: {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
