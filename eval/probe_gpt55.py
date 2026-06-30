"""Probe gpt-5.5 — 1 run on the highest-bias paper first.
If score == 0 or LIBRA fails again, stop immediately."""
from __future__ import annotations
import asyncio, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator
from app.providers.base import ProviderConfig

SYNTHESIS_DIR = ROOT / "backend" / "app" / "synthesis"
OUT_PATH      = ROOT / "eval" / "output" / "leaderboard.json"
MODEL         = "openai/gpt-5"
PAPERS        = ["MS_GUT_OG", "MS_GUT_GPT_Made", "MS_GUT_GPT_Edit",
                 "Nu_OG", "Nu_GPT_OG", "Nu_GPT_Edit"]
N_RUNS        = 5
# Paper most likely to score non-zero (based on other models)
PROBE_PAPER   = "MS_GUT_GPT_Made"


async def one_run(orch, text, cfg):
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
                "n_flags": len(flags), "flags": flags,
                "warnings": resp.warnings}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "score": None, "n_flags": 0,
                "flags": [], "warnings": []}


async def main():
    api_key = os.environ["BIASSCAN_API_KEY"]
    cfg = ProviderConfig(provider="openrouter", model=MODEL, api_key=api_key, base_url=None)
    orch = Orchestrator()

    print(f"=== PROBE: {MODEL} on {PROBE_PAPER} (1 run) ===")
    probe_text = (SYNTHESIS_DIR / f"{PROBE_PAPER}.txt").read_text()
    t0 = time.time()
    probe = await one_run(orch, probe_text, cfg)
    elapsed = time.time() - t0

    print(f"Score   : {probe['score']}")
    print(f"Flags   : {probe['n_flags']}")
    print(f"Elapsed : {elapsed:.0f}s")
    for w in probe.get("warnings", []):
        if "no parseable" in w.lower() or "error" in w.lower() or "failed" in w.lower():
            print(f"WARNING : {w}")

    parse_failures = sum(
        1 for w in probe.get("warnings", [])
        if "no parseable" in w.lower() or "retry also failed" in w.lower()
    )

    if probe["score"] == 0 or not probe["ok"]:
        print("\n⛔ Score is 0 or call failed — stopping. Credits saved.")
        return

    if parse_failures > 0:
        print(f"\n⚠️  {parse_failures} parse failure(s) but got score {probe['score']:.2f} — proceeding cautiously.")

    print(f"\n✅ Probe passed (score={probe['score']:.2f}) — running full 6 papers × {N_RUNS} runs...")

    result = {"model": MODEL, "tier": "State of the Art", "papers": {}}
    # Include probe run as run #1 for PROBE_PAPER
    # Then run remaining N_RUNS-1 for that paper, and full N_RUNS for others
    for paper in PAPERS:
        path = SYNTHESIS_DIR / f"{paper}.txt"
        text = path.read_text()
        t0 = time.time()
        if paper == PROBE_PAPER:
            extra = await asyncio.gather(*(one_run(orch, text, cfg) for _ in range(N_RUNS - 1)))
            runs = [probe] + list(extra)
        else:
            runs = list(await asyncio.gather(*(one_run(orch, text, cfg) for _ in range(N_RUNS))))
        elapsed = time.time() - t0
        scores = [r["score"] for r in runs if r["ok"] and r["score"] is not None]
        m  = statistics.mean(scores) if scores else 0.0
        sd = statistics.stdev(scores) if len(scores) >= 2 else 0.0
        result["papers"][paper] = {"runs": runs, "mean": m, "sd": sd,
                                   "valid": len(scores), "elapsed_s": round(elapsed, 1)}
        run_str = " ".join(f"{r['score']:>5.2f}" if r["ok"] and r["score"] is not None else " FAIL" for r in runs)
        print(f"  {paper:<20} runs=[{run_str}]  mean={m:>5.2f}  sd={sd:>4.2f}  ({elapsed:.0f}s)")

    # Merge into leaderboard.json
    data = json.loads(OUT_PATH.read_text())
    results = [r for r in data["results"] if r["model"] != MODEL]
    results.append(result)
    data["results"] = results
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n✅ Saved to {OUT_PATH}")


asyncio.run(main())
