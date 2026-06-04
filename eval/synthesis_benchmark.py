"""Run multiple synthesis variants against one model N times each,
report Mean, SD per variant + Separation (OG - Edit)."""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402

SYNTHESIS_DIR = ROOT / "backend" / "app" / "synthesis"
PAPERS = ["Nu-OG", "Nu-Edit", "Nu-OG-GPT"]
N_RUNS = 5


async def one_run(orch: Orchestrator, text: str, cfg: ProviderConfig) -> tuple[float, int]:
    """Return (score_out_of_10, flag_count). On failure returns (-1, 0)."""
    try:
        resp = await orch.analyze(
            text=text, references=None, mode="lite",
            provider_config=cfg, agents=None,
        )
        return resp.overall_bias_score * 10, len(resp.annotations)
    except Exception as e:
        print(f"  ! run failed: {type(e).__name__}: {e}", flush=True)
        return -1.0, 0


async def main() -> int:
    api_key = os.environ["BIASSCAN_API_KEY"]
    model = os.environ.get("BIASSCAN_MODEL", "google/gemini-2.5-pro")
    cfg = ProviderConfig(provider="openrouter", model=model, api_key=api_key, base_url=None)
    print(f"Model: {model}")
    print(f"Runs per paper: {N_RUNS}")
    print()

    orch = Orchestrator()
    results: dict[str, list[float]] = {}
    flags: dict[str, list[int]] = {}

    for paper in PAPERS:
        path = SYNTHESIS_DIR / f"{paper}.txt"
        if not path.is_file():
            print(f"MISSING: {path}", flush=True)
            results[paper] = [-1.0] * N_RUNS
            flags[paper] = [0] * N_RUNS
            continue
        text = path.read_text(encoding="utf-8")
        print(f"=== {paper} ({len(text)} chars) ===", flush=True)
        # Run N times in parallel (Gemini 2.5 Pro is a paid tier; rate limits are generous)
        scores_with_flags = await asyncio.gather(
            *(one_run(orch, text, cfg) for _ in range(N_RUNS))
        )
        scores = [s for s, _ in scores_with_flags]
        flag_counts = [f for _, f in scores_with_flags]
        results[paper] = scores
        flags[paper] = flag_counts
        for i, (s, f) in enumerate(scores_with_flags, 1):
            print(f"  Run {i}: score={s:.2f}/10  flags={f}", flush=True)
        print(flush=True)

    # Summary table
    print("=" * 90)
    print(f"{'Paper':<14} {'Run 1':>7} {'Run 2':>7} {'Run 3':>7} {'Run 4':>7} {'Run 5':>7}   {'Mean':>6} {'SD':>5}  {'Notes'}")
    print("-" * 90)
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    for paper in PAPERS:
        scores = results[paper]
        valid = [s for s in scores if s >= 0]
        if valid:
            m = statistics.mean(valid)
            s = statistics.stdev(valid) if len(valid) >= 2 else 0.0
            means[paper] = m
            sds[paper] = s
            run_cells = " ".join(f"{x:>7.2f}" for x in scores)
            note = "" if len(valid) == N_RUNS else f"({N_RUNS - len(valid)} failed)"
            print(f"{paper:<14} {run_cells}   {m:>6.2f} {s:>5.2f}  {note}")
        else:
            print(f"{paper:<14}  all runs failed")
    print("=" * 90)

    if "Nu-OG" in means and "Nu-Edit" in means:
        sep = means["Nu-OG"] - means["Nu-Edit"]
        avg_sd = statistics.mean(sds.values()) if sds else 0
        print()
        print(f"Separation (OG - Edit): {sep:+.2f}  ({'edit reduced bias' if sep > 0 else 'no reduction'})")
        print(f"Avg SD across papers:   {avg_sd:.2f}  ({'consistent' if avg_sd < 0.5 else 'noisy'})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
