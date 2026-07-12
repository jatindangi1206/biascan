"""Calibrate the _overall_score weights against eval/output/samples.json.

The samples.json corpus is keyed by `BIAS_*_dataset` with `biased` and
`control` arrays per dataset (see `eval/EVAL_DATASETS.md`). This script:

  1. Samples N biased + N control documents (default N=30 each).
  2. Runs the full analyze() pipeline on each using the user's provider.
  3. Computes ROC AUC for "biased scores > control scores" using the
     current formula in orchestrator._overall_score.
  4. Grid-searches over the score formula constants to find the
     combination that maximises discrimination AUC on a 70/30
     train/holdout split.
  5. Prints the recommended new constants and the AUC improvement.

This script is NOT run automatically. It costs roughly N × 2 × 5 LLM
calls (default 300) which costs ~$2–8 on OpenRouter depending on the
chosen model. Run it manually only when you want to recalibrate.

USAGE
-----
    cd backend && source .venv/bin/activate
    export BIASSCAN_PROVIDER=openrouter
    export BIASSCAN_API_KEY=sk-or-v1-...
    export BIASSCAN_MODEL=openai/gpt-4o-mini
    cd ..
    python -m eval.calibrate_score --n-per-class 30

OUTPUT
------
The script prints:
  - Baseline AUC (current formula on holdout)
  - Best-found AUC (after grid search)
  - The five constants in the recommended formula
You can then manually update orchestrator._overall_score and the
matching frontend ResultsPanel.tsx mirror, plus the table in
docs/SCORING.md.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import random
import sys
from pathlib import Path

# Make `app.*` importable without changing cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.orchestrator import Orchestrator  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402
from app.schemas import Annotation  # noqa: E402

logger = logging.getLogger("calibrate")

SAMPLES_JSON = ROOT / "eval" / "output" / "samples.json"


# ───────────────────────── corpus loading ──────────────────────────


def load_corpus(samples_path: Path) -> tuple[list[str], list[str]]:
    """Return (biased_texts, control_texts) flattened across all datasets."""
    data = json.loads(samples_path.read_text())
    biased: list[str] = []
    control: list[str] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        for item in value.get("biased", []):
            txt = item.get("input_text") if isinstance(item, dict) else None
            if isinstance(txt, str) and txt.strip():
                biased.append(txt)
        for item in value.get("control", []):
            txt = item.get("input_text") if isinstance(item, dict) else None
            if isinstance(txt, str) and txt.strip():
                control.append(txt)
    return biased, control


# ───────────────────────── pipeline runner ──────────────────────────


async def score_one(orch: Orchestrator, text: str, provider_cfg: ProviderConfig) -> tuple[float, list[Annotation]]:
    """Run analyze() on a single text and return (score, annotations).
    Caller swallows exceptions — we return (0.0, []) on failure."""
    try:
        resp = await orch.analyze(
            text=text,
            references=None,
            mode="lite",
            analysis_mode="systematic_review",
            provider_config=provider_cfg,
            agents=None,  # all 5
        )
        return resp.overall_bias_score, resp.annotations
    except Exception as e:
        logger.warning("scoring failed: %s", e)
        return 0.0, []


async def collect_scores(
    orch: Orchestrator,
    biased: list[str],
    control: list[str],
    provider_cfg: ProviderConfig,
    n_per_class: int,
) -> tuple[list[float], list[float], list[list[Annotation]], list[list[Annotation]]]:
    """Sample N biased + N control documents and run analyze() on each.
    Returns four parallel lists: biased_scores, control_scores, biased_anns, control_anns.
    Annotations are kept so the grid search can recompute scores under
    different formula constants without re-calling the LLM.
    """
    rng = random.Random(42)
    biased_sample = rng.sample(biased, min(n_per_class, len(biased)))
    control_sample = rng.sample(control, min(n_per_class, len(control)))

    b_scores: list[float] = []
    b_anns: list[list[Annotation]] = []
    for i, txt in enumerate(biased_sample):
        logger.info("biased %d/%d", i + 1, len(biased_sample))
        s, a = await score_one(orch, txt, provider_cfg)
        b_scores.append(s)
        b_anns.append(a)

    c_scores: list[float] = []
    c_anns: list[list[Annotation]] = []
    for i, txt in enumerate(control_sample):
        logger.info("control %d/%d", i + 1, len(control_sample))
        s, a = await score_one(orch, txt, provider_cfg)
        c_scores.append(s)
        c_anns.append(a)

    return b_scores, c_scores, b_anns, c_anns


# ───────────────────────── scoring + AUC ──────────────────────────


def score_with_constants(
    anns: list[Annotation],
    *,
    base_cap: float = 8.0,
    base_mult: float = 12.0,
    base_denom: float = 5.0,
    sev_high: float = 0.4,
    sev_medium: float = 0.15,
    sev_cap: float = 1.5,
    diversity_step: float = 0.15,
) -> float:
    """Recompute the overall score under arbitrary constants — used by
    the grid search so we don't have to re-call the LLM."""
    n = len(anns)
    if n == 0:
        return 0.0
    base = min(base_cap, base_mult * n / (base_denom + n))
    high_sum = sum(a.confidence for a in anns if a.severity == "high")
    med_sum = sum(a.confidence for a in anns if a.severity == "medium")
    severity = min(sev_cap, sev_high * high_sum + sev_medium * med_sum)
    unique_types = len({a.bias_type for a in anns})
    diversity = diversity_step * max(0, unique_types - 1)
    return min(10.0, base + severity + diversity) / 10.0


def auc(biased_scores: list[float], control_scores: list[float]) -> float:
    """ROC AUC for "biased > control". Pure-python, no sklearn dependency.
    Equivalent to Mann-Whitney U / (n_b × n_c)."""
    n_b, n_c = len(biased_scores), len(control_scores)
    if n_b == 0 or n_c == 0:
        return 0.5
    wins = 0.0
    for b in biased_scores:
        for c in control_scores:
            if b > c:
                wins += 1.0
            elif b == c:
                wins += 0.5
    return wins / (n_b * n_c)


# ───────────────────────── grid search ──────────────────────────


def grid_search(
    b_anns: list[list[Annotation]],
    c_anns: list[list[Annotation]],
    holdout_frac: float = 0.3,
) -> tuple[dict, float, float]:
    """Search a small grid over the formula constants. Returns
    (best_params, baseline_auc_on_holdout, best_auc_on_holdout)."""
    rng = random.Random(7)
    n_b = len(b_anns)
    n_c = len(c_anns)
    b_idx = list(range(n_b))
    c_idx = list(range(n_c))
    rng.shuffle(b_idx)
    rng.shuffle(c_idx)
    b_holdout_n = max(1, int(n_b * holdout_frac))
    c_holdout_n = max(1, int(n_c * holdout_frac))
    b_train = [b_anns[i] for i in b_idx[b_holdout_n:]]
    b_hold = [b_anns[i] for i in b_idx[:b_holdout_n]]
    c_train = [c_anns[i] for i in c_idx[c_holdout_n:]]
    c_hold = [c_anns[i] for i in c_idx[:c_holdout_n]]

    # Baseline (current production constants)
    baseline = {
        "base_cap": 8.0, "base_mult": 12.0, "base_denom": 5.0,
        "sev_high": 0.4, "sev_medium": 0.15, "sev_cap": 1.5,
        "diversity_step": 0.15,
    }
    b_hold_scores = [score_with_constants(a, **baseline) for a in b_hold]
    c_hold_scores = [score_with_constants(a, **baseline) for a in c_hold]
    baseline_auc = auc(b_hold_scores, c_hold_scores)

    # Search grid — kept small on purpose. Add more axes only if these
    # don't move the needle.
    grid = {
        "base_mult":    [10.0, 12.0, 14.0],
        "base_denom":   [3.0, 5.0, 7.0],
        "sev_high":     [0.3, 0.4, 0.5],
        "sev_medium":   [0.1, 0.15, 0.2],
        "diversity_step": [0.10, 0.15, 0.20],
    }
    fixed = {"base_cap": 8.0, "sev_cap": 1.5}

    best_auc = -1.0
    best_params: dict = baseline
    for combo in itertools.product(*grid.values()):
        params = dict(fixed)
        for key, value in zip(grid.keys(), combo):
            params[key] = value
        b_train_scores = [score_with_constants(a, **params) for a in b_train]
        c_train_scores = [score_with_constants(a, **params) for a in c_train]
        train_auc = auc(b_train_scores, c_train_scores)
        if train_auc > best_auc:
            best_auc = train_auc
            best_params = params

    # Final report uses holdout AUC for the winning params
    b_hold_scores = [score_with_constants(a, **best_params) for a in b_hold]
    c_hold_scores = [score_with_constants(a, **best_params) for a in c_hold]
    best_holdout_auc = auc(b_hold_scores, c_hold_scores)
    return best_params, baseline_auc, best_holdout_auc


# ───────────────────────── entry point ──────────────────────────


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=30, help="biased + control sample size each")
    parser.add_argument("--samples", type=Path, default=SAMPLES_JSON)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    provider_name = os.environ.get("BIASSCAN_PROVIDER")
    model = os.environ.get("BIASSCAN_MODEL")
    api_key = os.environ.get("BIASSCAN_API_KEY")
    base_url = os.environ.get("BIASSCAN_BASE_URL")
    if not provider_name or not model:
        print("ERROR: set BIASSCAN_PROVIDER and BIASSCAN_MODEL in env.", file=sys.stderr)
        sys.exit(1)

    provider_cfg = ProviderConfig(
        provider=provider_name, model=model, api_key=api_key, base_url=base_url,
    )

    print(f"Loading corpus from {args.samples} …")
    biased, control = load_corpus(args.samples)
    print(f"  {len(biased)} biased, {len(control)} control samples available.")
    print(f"  Sampling {args.n_per_class} of each.")
    expected_calls = args.n_per_class * 2 * 5
    print(f"  Estimated LLM calls: {expected_calls}  (5 agents × {args.n_per_class*2} docs)")

    orch = Orchestrator()
    b_scores, c_scores, b_anns, c_anns = await collect_scores(
        orch, biased, control, provider_cfg, args.n_per_class,
    )

    print(f"\nMean biased score : {sum(b_scores)/len(b_scores):.3f}")
    print(f"Mean control score: {sum(c_scores)/len(c_scores):.3f}")
    print(f"Full-corpus AUC   : {auc(b_scores, c_scores):.3f}")

    print("\nGrid-searching formula constants …")
    best_params, baseline_auc, best_auc = grid_search(b_anns, c_anns)
    print(f"\nBaseline AUC on holdout : {baseline_auc:.3f}")
    print(f"Best AUC on holdout     : {best_auc:.3f}   (Δ {best_auc - baseline_auc:+.3f})")
    print("\nRecommended constants:")
    for key, value in best_params.items():
        print(f"  {key:18s} = {value}")
    print(
        "\nTo apply: edit backend/app/agents/orchestrator.py::_overall_score\n"
        "          and the matching mirror in frontend/src/components/ResultsPanel.tsx.\n"
        "          Update the lookup tables in docs/SCORING.md from the new values."
    )


if __name__ == "__main__":
    asyncio.run(main())
