"""
Evaluation Pipeline — STEPS 01 through 05.

STEP 01: Run transforms → generate biased_samples.json + control_samples.json
STEP 02: Validation checkpoint — preview samples for manual approval
STEP 03: Run models and collect outputs
STEP 04: Compute metrics (accuracy, precision, recall, F1, FP, FN)
STEP 05: Rank models into leaderboard

Usage:
  # Generate samples (STEP 01)
  python -m eval.pipeline generate --max-samples 20

  # Preview samples for approval (STEP 02)
  python -m eval.pipeline preview --count 5

  # Run evaluation (STEP 03-05) — requires STEP 01 output
  python -m eval.pipeline run --provider groq --max-samples 10

  # Full pipeline (generate → preview → approve → run → report)
  python -m eval.pipeline full --provider groq --max-samples 10 --auto-approve
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Add parent to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.pipeline")

# Output directory for generated samples and results
EVAL_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 01: Generate samples from transforms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step01_generate(max_samples: int = 20, seed: int = 42, dataset_ids: list[str] | None = None) -> dict:
    """
    Run all registered transforms and generate sample files.

    Returns dict of dataset_id → {biased: [...], control: [...]}
    """
    # Import triggers registration of all transforms
    from eval.transforms import get_all_specs

    specs = get_all_specs()
    logger.info(f"STEP 01: Generating samples from {len(specs)} registered transforms...")

    if dataset_ids:
        specs = [s for s in specs if s.dataset_id in dataset_ids]
        logger.info(f"  Filtered to {len(specs)} datasets: {[s.dataset_id for s in specs]}")

    all_results = {}
    total_biased = 0
    total_control = 0
    failed = []

    for spec in specs:
        logger.info(f"  Running transform: {spec.dataset_id} ({spec.dataset_name})...")
        try:
            samples = spec.transform_fn(max_samples=max_samples, seed=seed)
            biased = [s for s in samples if s.bias_present]
            control = [s for s in samples if not s.bias_present]

            all_results[spec.dataset_id] = {
                "spec": {
                    "dataset_id": spec.dataset_id,
                    "bias_type": spec.bias_type,
                    "dataset_name": spec.dataset_name,
                    "agent": spec.agent,
                    "how_to_use": spec.how_to_use,
                },
                "biased": [asdict(s) for s in biased],
                "control": [asdict(s) for s in control],
                "stats": {
                    "total": len(samples),
                    "biased": len(biased),
                    "control": len(control),
                },
            }
            total_biased += len(biased)
            total_control += len(control)
            logger.info(f"    ✓ {len(samples)} samples ({len(biased)} biased, {len(control)} control)")

        except Exception as e:
            logger.error(f"    ✗ FAILED: {e}")
            failed.append((spec.dataset_id, str(e)))

    # Save to files
    output_path = EVAL_OUTPUT_DIR / "generated_samples.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Also save per-dataset files
    per_dataset_dir = EVAL_OUTPUT_DIR / "per_dataset"
    per_dataset_dir.mkdir(exist_ok=True)
    for did, data in all_results.items():
        with open(per_dataset_dir / f"{did}.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    logger.info(f"\nSTEP 01 COMPLETE:")
    logger.info(f"  Datasets: {len(all_results)} succeeded, {len(failed)} failed")
    logger.info(f"  Total samples: {total_biased + total_control} ({total_biased} biased, {total_control} control)")
    logger.info(f"  Output: {output_path}")

    if failed:
        logger.warning(f"  Failed datasets: {failed}")

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 02: Validation checkpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step02_preview(count: int = 5) -> dict:
    """
    Show sample previews for manual inspection.
    Returns the preview data for display.
    """
    samples_path = EVAL_OUTPUT_DIR / "generated_samples.json"
    if not samples_path.exists():
        logger.error("No generated samples found. Run STEP 01 first.")
        return {}

    with open(samples_path) as f:
        all_results = json.load(f)

    preview = {}
    print("\n" + "=" * 80)
    print("STEP 02: VALIDATION CHECKPOINT — Sample Previews")
    print("=" * 80)

    for did, data in all_results.items():
        spec = data["spec"]
        biased = data["biased"]
        control = data["control"]

        print(f"\n{'─' * 60}")
        print(f"Dataset: {spec['dataset_name']} ({did})")
        print(f"Bias Type: {spec['bias_type']} | Agent: {spec['agent']}")
        print(f"Samples: {len(biased)} biased + {len(control)} control")
        print(f"How to Use: {spec['how_to_use']}")

        samples_to_show = []

        # Show first N/2 biased and N/2 control
        for s in biased[:max(1, count // 2)]:
            samples_to_show.append(("BIASED", s))
        for s in control[:max(1, count // 2)]:
            samples_to_show.append(("CONTROL", s))

        for label_tag, sample in samples_to_show:
            print(f"\n  [{label_tag}] {sample['id']}")
            print(f"  Transform: {sample['transform_description']}")
            input_preview = sample['input_text'][:200]
            if len(sample['input_text']) > 200:
                input_preview += "..."
            print(f"  Input: {input_preview}")
            if sample.get('original_fields'):
                orig_keys = list(sample['original_fields'].keys())[:3]
                print(f"  Original fields: {', '.join(orig_keys)}")

        preview[did] = {
            "dataset_name": spec["dataset_name"],
            "biased_count": len(biased),
            "control_count": len(control),
            "previewed": len(samples_to_show),
        }

    print(f"\n{'=' * 80}")
    print(f"Total: {sum(d['biased_count'] + d['control_count'] for d in preview.values())} samples "
          f"across {len(preview)} datasets")
    print("=" * 80)

    return preview


def step02_approve() -> bool:
    """Interactive approval checkpoint."""
    print("\n>>> Do you approve these samples for model evaluation? [y/n]: ", end="")
    response = input().strip().lower()
    approved = response in ("y", "yes")
    if approved:
        # Write approval marker
        marker = EVAL_OUTPUT_DIR / ".approved"
        marker.write_text(f"Approved at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info("✓ Samples APPROVED for evaluation.")
    else:
        logger.info("✗ Samples NOT approved. Please review and re-generate.")
    return approved


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 03: Run models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ModelPrediction:
    """A single model prediction on an eval sample."""
    sample_id: str
    dataset_id: str
    bias_type: str
    expected_label: str          # "biased" or "control"
    expected_bias_present: bool
    predicted_bias_present: bool  # Did the model detect bias?
    composite_score: float        # Pipeline composite score
    agent_scores: dict = field(default_factory=dict)  # {agent: score}
    findings_count: int = 0
    convergence_iterations: int = 0
    model_name: str = ""
    elapsed_seconds: float = 0.0
    token_usage: dict = field(default_factory=dict)
    error: str = ""


async def step03_run_models(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    max_samples: int = 10,
    dataset_ids: list[str] | None = None,
    bias_threshold: float = 0.3,
) -> list[ModelPrediction]:
    """
    Run the BiasScan pipeline on generated samples and collect predictions.

    bias_threshold: composite score above this → model predicts "biased"
    """
    from backend.models.llm_client import LLMClient
    from backend.orchestrator.pipeline import Pipeline

    samples_path = EVAL_OUTPUT_DIR / "generated_samples.json"
    if not samples_path.exists():
        logger.error("No generated samples. Run STEP 01 first.")
        return []

    with open(samples_path) as f:
        all_results = json.load(f)

    # Collect all samples to evaluate
    eval_samples = []
    for did, data in all_results.items():
        if dataset_ids and did not in dataset_ids:
            continue
        for s in data["biased"][:max_samples // 2]:
            eval_samples.append(s)
        for s in data["control"][:max_samples // 2]:
            eval_samples.append(s)

    if not eval_samples:
        logger.error("No samples to evaluate.")
        return []

    logger.info(f"STEP 03: Running {provider}/{model or 'default'} on {len(eval_samples)} samples...")

    # Build pipeline
    llm = LLMClient(provider=provider, model=model, api_key=api_key)
    pipeline = Pipeline(llm)

    predictions = []

    for i, sample in enumerate(eval_samples):
        sample_id = sample["id"]
        logger.info(f"  [{i+1}/{len(eval_samples)}] {sample_id} (expected: {sample['label']})...")

        start = time.time()
        try:
            result = await pipeline.run(
                raw_text=sample["input_text"],
                max_iterations=2,
                patience=2,
                threshold=0.05,
            )
            elapsed = time.time() - start

            # Extract scores
            agent_scores = {}
            findings_count = 0
            if result.iterations:
                last_it = result.iterations[-1]
                agent_scores = {
                    "argus": last_it.argus_output.score,
                    "libra": last_it.libra_output.score,
                    "lens": last_it.lens_output.score,
                }
                findings_count = (
                    len(last_it.argus_output.findings)
                    + len(last_it.libra_output.findings)
                    + len(last_it.lens_output.findings)
                )

            # Token usage
            token_usage = {}
            if result.iterations and result.iterations[-1].token_usage:
                token_usage = result.iterations[-1].token_usage.get("total", {})

            predicted_biased = result.final_composite > bias_threshold

            predictions.append(ModelPrediction(
                sample_id=sample_id,
                dataset_id=sample["dataset_id"],
                bias_type=sample["bias_type"],
                expected_label=sample["label"],
                expected_bias_present=sample["bias_present"],
                predicted_bias_present=predicted_biased,
                composite_score=result.final_composite,
                agent_scores=agent_scores,
                findings_count=findings_count,
                convergence_iterations=len(result.iterations),
                model_name=result.model_used or f"{provider}/{model}",
                elapsed_seconds=elapsed,
                token_usage=token_usage,
            ))

            status = "✓" if predicted_biased == sample["bias_present"] else "✗"
            logger.info(
                f"    {status} composite={result.final_composite:.2f} "
                f"predicted={'biased' if predicted_biased else 'control'} "
                f"(expected: {sample['label']}) [{elapsed:.1f}s]"
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"    ✗ ERROR: {e} [{elapsed:.1f}s]")
            predictions.append(ModelPrediction(
                sample_id=sample_id,
                dataset_id=sample["dataset_id"],
                bias_type=sample["bias_type"],
                expected_label=sample["label"],
                expected_bias_present=sample["bias_present"],
                predicted_bias_present=False,
                composite_score=0.0,
                model_name=f"{provider}/{model}",
                elapsed_seconds=elapsed,
                error=str(e),
            ))

    # Save predictions
    preds_path = EVAL_OUTPUT_DIR / f"predictions_{provider}_{model or 'default'}_{int(time.time())}.json"
    with open(preds_path, "w") as f:
        json.dump([asdict(p) for p in predictions], f, indent=2, default=str)
    logger.info(f"  Predictions saved: {preds_path}")

    return predictions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 04: Compute metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DatasetMetrics:
    """Metrics for one dataset."""
    dataset_id: str
    dataset_name: str
    bias_type: str
    total_samples: int = 0
    true_positives: int = 0   # Biased correctly detected
    true_negatives: int = 0   # Control correctly passed
    false_positives: int = 0  # Control incorrectly flagged
    false_negatives: int = 0  # Biased missed

    @property
    def accuracy(self) -> float:
        total = self.total_samples
        return (self.true_positives + self.true_negatives) / total if total else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalMetrics:
    """Overall evaluation metrics."""
    model_name: str
    provider: str
    per_dataset: dict[str, DatasetMetrics] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    mean_composite_biased: float = 0.0
    mean_composite_control: float = 0.0
    total_samples: int = 0
    total_errors: int = 0
    mean_latency_seconds: float = 0.0


def step04_compute_metrics(predictions: list[ModelPrediction]) -> EvalMetrics:
    """Compute precision, recall, F1, accuracy per dataset and overall."""
    if not predictions:
        return EvalMetrics(model_name="", provider="")

    model_name = predictions[0].model_name
    provider = model_name.split("/")[0] if "/" in model_name else model_name

    # Group by dataset
    by_dataset: dict[str, list[ModelPrediction]] = {}
    for p in predictions:
        by_dataset.setdefault(p.dataset_id, []).append(p)

    per_dataset = {}
    total_tp = total_tn = total_fp = total_fn = 0
    composite_biased = []
    composite_control = []
    errors = 0
    latencies = []

    for did, preds in by_dataset.items():
        dm = DatasetMetrics(
            dataset_id=did,
            dataset_name=did,  # Will be enriched from spec
            bias_type=preds[0].bias_type if preds else "",
            total_samples=len(preds),
        )

        for p in preds:
            if p.error:
                errors += 1
                continue

            latencies.append(p.elapsed_seconds)

            if p.expected_bias_present:
                composite_biased.append(p.composite_score)
                if p.predicted_bias_present:
                    dm.true_positives += 1
                else:
                    dm.false_negatives += 1
            else:
                composite_control.append(p.composite_score)
                if not p.predicted_bias_present:
                    dm.true_negatives += 1
                else:
                    dm.false_positives += 1

        total_tp += dm.true_positives
        total_tn += dm.true_negatives
        total_fp += dm.false_positives
        total_fn += dm.false_negatives
        per_dataset[did] = dm

    # Overall metrics
    total = total_tp + total_tn + total_fp + total_fn
    overall_acc = (total_tp + total_tn) / total if total else 0.0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) else 0.0

    metrics = EvalMetrics(
        model_name=model_name,
        provider=provider,
        per_dataset=per_dataset,
        overall_accuracy=overall_acc,
        overall_precision=overall_prec,
        overall_recall=overall_rec,
        overall_f1=overall_f1,
        mean_composite_biased=sum(composite_biased) / len(composite_biased) if composite_biased else 0.0,
        mean_composite_control=sum(composite_control) / len(composite_control) if composite_control else 0.0,
        total_samples=len(predictions),
        total_errors=errors,
        mean_latency_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
    )

    # Print report
    print(f"\n{'=' * 70}")
    print(f"STEP 04: METRICS — {model_name}")
    print(f"{'=' * 70}")

    for did, dm in per_dataset.items():
        print(f"\n  {did}:")
        print(f"    Samples: {dm.total_samples} | TP={dm.true_positives} TN={dm.true_negatives} "
              f"FP={dm.false_positives} FN={dm.false_negatives}")
        print(f"    Accuracy={dm.accuracy:.3f} Precision={dm.precision:.3f} "
              f"Recall={dm.recall:.3f} F1={dm.f1:.3f}")

    print(f"\n  OVERALL:")
    print(f"    Accuracy={overall_acc:.3f} Precision={overall_prec:.3f} "
          f"Recall={overall_rec:.3f} F1={overall_f1:.3f}")
    print(f"    Mean composite (biased): {metrics.mean_composite_biased:.3f}")
    print(f"    Mean composite (control): {metrics.mean_composite_control:.3f}")
    print(f"    Separation: {metrics.mean_composite_biased - metrics.mean_composite_control:.3f}")
    print(f"    Mean latency: {metrics.mean_latency_seconds:.1f}s")
    print(f"    Errors: {errors}/{len(predictions)}")
    print(f"{'=' * 70}")

    # Save metrics
    metrics_path = EVAL_OUTPUT_DIR / f"metrics_{provider}_{int(time.time())}.json"
    metrics_dict = asdict(metrics)
    # Convert DatasetMetrics to dicts
    metrics_dict["per_dataset"] = {
        k: {**asdict(v), "accuracy": v.accuracy, "precision": v.precision,
            "recall": v.recall, "f1": v.f1}
        for k, v in per_dataset.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    logger.info(f"  Metrics saved: {metrics_path}")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 05: Leaderboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step05_leaderboard(all_metrics: list[EvalMetrics]) -> str:
    """
    Rank models and generate leaderboard.
    Returns formatted leaderboard string and saves to file.
    """
    if not all_metrics:
        return "No metrics to rank."

    # Sort by F1 descending
    ranked = sorted(all_metrics, key=lambda m: m.overall_f1, reverse=True)

    lines = []
    lines.append("=" * 90)
    lines.append("STEP 05: MODEL LEADERBOARD")
    lines.append("=" * 90)
    lines.append(f"{'Rank':<5} {'Model':<35} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Acc':>6} {'Sep':>6} {'Lat':>7}")
    lines.append("-" * 90)

    for i, m in enumerate(ranked):
        sep = m.mean_composite_biased - m.mean_composite_control
        lines.append(
            f"{i+1:<5} {m.model_name[:34]:<35} "
            f"{m.overall_f1:>6.3f} {m.overall_precision:>6.3f} {m.overall_recall:>6.3f} "
            f"{m.overall_accuracy:>6.3f} {sep:>6.3f} {m.mean_latency_seconds:>6.1f}s"
        )

    lines.append("-" * 90)
    lines.append(f"Sep = Mean composite score separation (biased - control). Higher = better discrimination.")
    lines.append(f"Lat = Mean latency per sample.")
    lines.append("=" * 90)

    leaderboard = "\n".join(lines)
    print(leaderboard)

    # Save
    lb_path = EVAL_OUTPUT_DIR / f"leaderboard_{int(time.time())}.txt"
    with open(lb_path, "w") as f:
        f.write(leaderboard)
    logger.info(f"Leaderboard saved: {lb_path}")

    # Also save as JSON
    lb_json_path = EVAL_OUTPUT_DIR / f"leaderboard_{int(time.time())}.json"
    lb_data = []
    for i, m in enumerate(ranked):
        lb_data.append({
            "rank": i + 1,
            "model_name": m.model_name,
            "provider": m.provider,
            "f1": m.overall_f1,
            "precision": m.overall_precision,
            "recall": m.overall_recall,
            "accuracy": m.overall_accuracy,
            "separation": m.mean_composite_biased - m.mean_composite_control,
            "mean_latency": m.mean_latency_seconds,
            "total_samples": m.total_samples,
            "errors": m.total_errors,
        })
    with open(lb_json_path, "w") as f:
        json.dump(lb_data, f, indent=2)

    return leaderboard


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="BiasScan Evaluation Pipeline")
    sub = parser.add_subparsers(dest="command")

    # STEP 01: generate
    gen = sub.add_parser("generate", help="STEP 01: Generate samples from transforms")
    gen.add_argument("--max-samples", type=int, default=20)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--datasets", nargs="*", help="Specific dataset IDs to generate")

    # STEP 02: preview
    prev = sub.add_parser("preview", help="STEP 02: Preview samples for approval")
    prev.add_argument("--count", type=int, default=5, help="Samples to preview per dataset")

    # STEP 03-05: run
    run = sub.add_parser("run", help="STEP 03-05: Run model eval + metrics + leaderboard")
    run.add_argument("--provider", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--max-samples", type=int, default=10)
    run.add_argument("--datasets", nargs="*", help="Specific dataset IDs")
    run.add_argument("--threshold", type=float, default=0.3, help="Bias detection threshold")

    # Full pipeline
    full = sub.add_parser("full", help="Run full pipeline (generate → preview → run)")
    full.add_argument("--provider", required=True)
    full.add_argument("--model", default=None)
    full.add_argument("--api-key", default=None)
    full.add_argument("--max-samples", type=int, default=10)
    full.add_argument("--seed", type=int, default=42)
    full.add_argument("--datasets", nargs="*")
    full.add_argument("--threshold", type=float, default=0.3)
    full.add_argument("--auto-approve", action="store_true", help="Skip approval checkpoint")

    args = parser.parse_args()

    if args.command == "generate":
        step01_generate(max_samples=args.max_samples, seed=args.seed, dataset_ids=args.datasets)

    elif args.command == "preview":
        step02_preview(count=args.count)

    elif args.command == "run":
        predictions = asyncio.run(step03_run_models(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            max_samples=args.max_samples,
            dataset_ids=args.datasets,
            bias_threshold=args.threshold,
        ))
        metrics = step04_compute_metrics(predictions)
        step05_leaderboard([metrics])

    elif args.command == "full":
        # STEP 01
        step01_generate(max_samples=args.max_samples, seed=args.seed, dataset_ids=args.datasets)

        # STEP 02
        step02_preview(count=5)
        if not args.auto_approve:
            if not step02_approve():
                logger.info("Pipeline stopped at approval checkpoint.")
                return
        else:
            logger.info("Auto-approve enabled — skipping approval checkpoint.")
            marker = EVAL_OUTPUT_DIR / ".approved"
            marker.write_text(f"Auto-approved at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # STEP 03-05
        predictions = asyncio.run(step03_run_models(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            max_samples=args.max_samples,
            dataset_ids=args.datasets,
            bias_threshold=args.threshold,
        ))
        metrics = step04_compute_metrics(predictions)
        step05_leaderboard([metrics])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
