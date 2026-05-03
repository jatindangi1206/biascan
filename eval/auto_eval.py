#!/usr/bin/env python3
"""
Automated Evaluation Runner — runs BiasScan agents against labeled datasets
and computes precision/recall/F1 metrics for bias detection accuracy.

Usage:
    python -m eval.auto_eval [--provider groq] [--model llama-3.3-70b-versatile]
                             [--dataset SciFact] [--samples 50]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.models.llm_client import LLMClient
from backend.orchestrator.pipeline import Pipeline
from eval.datasets.loader import (
    TestCase, load_dataset_by_name, load_all_datasets, LOADER_REGISTRY,
)
from eval.eval_config import EVAL_PIPELINE_SETTINGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto_eval")


# ── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Result of running one test case through the pipeline."""
    test_case_id: str
    dataset_name: str
    expected_bias_present: bool
    expected_bias_types: list[str]
    detected_bias_present: bool
    detected_bias_types: list[str]
    detected_findings_count: int
    composite_score: float
    elapsed_sec: float
    tokens_used: int
    error: str = ""


@dataclass
class DatasetMetrics:
    """Aggregated metrics for one dataset."""
    dataset_name: str
    model_used: str
    total_cases: int = 0
    true_positives: int = 0        # bias expected AND detected
    false_positives: int = 0       # bias NOT expected but detected
    true_negatives: int = 0        # no bias expected, none detected
    false_negatives: int = 0       # bias expected but NOT detected
    bias_type_hits: dict = field(default_factory=dict)   # per-bias-code TP counts
    bias_type_misses: dict = field(default_factory=dict)  # per-bias-code FN counts
    avg_composite_score: float = 0.0
    avg_elapsed_sec: float = 0.0
    total_tokens: int = 0
    errors: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total_cases if self.total_cases > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["precision"] = round(self.precision, 4)
        d["recall"] = round(self.recall, 4)
        d["f1"] = round(self.f1, 4)
        d["accuracy"] = round(self.accuracy, 4)
        return d


@dataclass
class EvalReport:
    """Complete evaluation report across all datasets."""
    model_used: str
    provider: str
    timestamp: str = ""
    total_elapsed_sec: float = 0.0
    dataset_metrics: list[DatasetMetrics] = field(default_factory=list)
    individual_results: list[DetectionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_used": self.model_used,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "total_elapsed_sec": round(self.total_elapsed_sec, 1),
            "summary": {
                "avg_precision": round(sum(m.precision for m in self.dataset_metrics) / len(self.dataset_metrics), 4) if self.dataset_metrics else 0,
                "avg_recall": round(sum(m.recall for m in self.dataset_metrics) / len(self.dataset_metrics), 4) if self.dataset_metrics else 0,
                "avg_f1": round(sum(m.f1 for m in self.dataset_metrics) / len(self.dataset_metrics), 4) if self.dataset_metrics else 0,
            },
            "datasets": [m.to_dict() for m in self.dataset_metrics],
            "individual_results": [asdict(r) for r in self.individual_results],
        }


# ── Runner ──────────────────────────────────────────────────────────────────

BIAS_DETECTION_THRESHOLD = 3.0  # composite score > 3.0 means "bias detected"


async def evaluate_single_case(
    pipeline: Pipeline,
    test_case: TestCase,
) -> DetectionResult:
    """Run one test case through the pipeline and compare to ground truth."""
    start = time.time()
    try:
        result = await pipeline.run(
            raw_text=test_case.input_text,
            max_iterations=EVAL_PIPELINE_SETTINGS["max_iterations"],
            patience=EVAL_PIPELINE_SETTINGS["patience"],
            threshold=EVAL_PIPELINE_SETTINGS["threshold"],
        )

        # Determine if bias was detected
        detected_bias = result.final_composite > BIAS_DETECTION_THRESHOLD and not result.no_bias_detected

        # Collect all detected bias types
        detected_types = set()
        for it in result.iterations:
            for f in it.argus_output.findings:
                detected_types.add(f.bias_type)
            for f in it.libra_output.findings:
                detected_types.add(f.bias_type)
            for f in it.lens_output.findings:
                detected_types.add(f.bias_type)

        # Count total findings
        total_findings = sum(
            len(it.argus_output.findings) + len(it.libra_output.findings) + len(it.lens_output.findings)
            for it in result.iterations
        )

        # Token usage
        tokens = 0
        for it in result.iterations:
            if it.token_usage:
                tokens += it.token_usage.get("total", {}).get("total_tokens", 0)

        elapsed = time.time() - start
        return DetectionResult(
            test_case_id=test_case.id,
            dataset_name=test_case.dataset_name,
            expected_bias_present=test_case.bias_present,
            expected_bias_types=test_case.expected_bias_types,
            detected_bias_present=detected_bias,
            detected_bias_types=list(detected_types),
            detected_findings_count=total_findings,
            composite_score=result.final_composite,
            elapsed_sec=elapsed,
            tokens_used=tokens,
        )

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Error evaluating {test_case.id}: {e}")
        return DetectionResult(
            test_case_id=test_case.id,
            dataset_name=test_case.dataset_name,
            expected_bias_present=test_case.bias_present,
            expected_bias_types=test_case.expected_bias_types,
            detected_bias_present=False,
            detected_bias_types=[],
            detected_findings_count=0,
            composite_score=0.0,
            elapsed_sec=elapsed,
            tokens_used=0,
            error=str(e),
        )


async def evaluate_dataset(
    pipeline: Pipeline,
    dataset_name: str,
    model_label: str,
    sample_size: int = 50,
) -> tuple[DatasetMetrics, list[DetectionResult]]:
    """Evaluate pipeline on a single dataset. Returns metrics + individual results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating on {dataset_name} ({sample_size} samples)...")

    test_cases = load_dataset_by_name(dataset_name, sample_size=sample_size)
    if not test_cases:
        logger.warning(f"No test cases loaded for {dataset_name}")
        return DatasetMetrics(dataset_name=dataset_name, model_used=model_label), []

    results = []
    for idx, tc in enumerate(test_cases):
        logger.info(f"  [{idx+1}/{len(test_cases)}] {tc.id} (expected_bias={tc.bias_present})")
        result = await evaluate_single_case(pipeline, tc)
        results.append(result)

        # Brief progress
        status = "TP" if result.expected_bias_present and result.detected_bias_present else \
                 "TN" if not result.expected_bias_present and not result.detected_bias_present else \
                 "FP" if not result.expected_bias_present and result.detected_bias_present else "FN"
        logger.info(f"    → {status} | composite={result.composite_score:.2f} | "
                     f"findings={result.detected_findings_count} | {result.elapsed_sec:.1f}s")

    # Aggregate metrics
    metrics = DatasetMetrics(dataset_name=dataset_name, model_used=model_label, total_cases=len(results))
    composite_scores = []
    elapsed_times = []

    for r in results:
        if r.error:
            metrics.errors += 1
            continue

        composite_scores.append(r.composite_score)
        elapsed_times.append(r.elapsed_sec)
        metrics.total_tokens += r.tokens_used

        if r.expected_bias_present and r.detected_bias_present:
            metrics.true_positives += 1
            for bt in r.expected_bias_types:
                metrics.bias_type_hits[bt] = metrics.bias_type_hits.get(bt, 0) + 1
        elif r.expected_bias_present and not r.detected_bias_present:
            metrics.false_negatives += 1
            for bt in r.expected_bias_types:
                metrics.bias_type_misses[bt] = metrics.bias_type_misses.get(bt, 0) + 1
        elif not r.expected_bias_present and r.detected_bias_present:
            metrics.false_positives += 1
        else:
            metrics.true_negatives += 1

    metrics.avg_composite_score = sum(composite_scores) / len(composite_scores) if composite_scores else 0
    metrics.avg_elapsed_sec = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0

    logger.info(f"\n  {dataset_name} Results:")
    logger.info(f"    Precision={metrics.precision:.3f}  Recall={metrics.recall:.3f}  F1={metrics.f1:.3f}")
    logger.info(f"    TP={metrics.true_positives} FP={metrics.false_positives} "
                f"TN={metrics.true_negatives} FN={metrics.false_negatives} Errors={metrics.errors}")

    return metrics, results


async def run_full_eval(
    provider: str,
    model: Optional[str] = None,
    datasets: Optional[list[str]] = None,
    sample_size: int = 50,
    api_key: Optional[str] = None,
) -> EvalReport:
    """Run the full automated evaluation across datasets."""
    import datetime

    llm = LLMClient(provider=provider, model=model, api_key=api_key)
    pipeline = Pipeline(llm)
    model_label = llm.display_name

    logger.info(f"Starting automated evaluation with {model_label}")
    start_time = time.time()

    dataset_names = datasets or list(LOADER_REGISTRY.keys())

    report = EvalReport(
        model_used=model_label,
        provider=provider,
        timestamp=datetime.datetime.now().isoformat(),
    )

    for ds_name in dataset_names:
        try:
            metrics, results = await evaluate_dataset(pipeline, ds_name, model_label, sample_size)
            report.dataset_metrics.append(metrics)
            report.individual_results.extend(results)
        except Exception as e:
            logger.error(f"Failed to evaluate {ds_name}: {e}")

    report.total_elapsed_sec = time.time() - start_time

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    safe_name = model_label.replace("/", "_").replace(" ", "_")
    result_path = results_dir / f"auto_eval_{safe_name}_{int(time.time())}.json"
    with open(result_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION COMPLETE — {model_label}")
    logger.info(f"  Total time: {report.total_elapsed_sec:.1f}s")
    logger.info(f"  Results saved to: {result_path}")
    for m in report.dataset_metrics:
        logger.info(f"  {m.dataset_name}: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")

    return report


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BiasScan Automated Evaluation")
    parser.add_argument("--provider", default="mock", help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name (uses provider default if omitted)")
    parser.add_argument("--dataset", default=None, help="Specific dataset name (runs all if omitted)")
    parser.add_argument("--samples", type=int, default=10, help="Samples per dataset (default 10 for smoke test)")
    parser.add_argument("--api-key", default=None, help="API key override (BYOK)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else None

    report = asyncio.run(run_full_eval(
        provider=args.provider,
        model=args.model,
        datasets=datasets,
        sample_size=args.samples,
        api_key=args.api_key,
    ))

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Dataset':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}")
    print(f"{'-'*70}")
    for m in report.dataset_metrics:
        print(f"{m.dataset_name:<25} {m.precision:>10.3f} {m.recall:>8.3f} {m.f1:>8.3f} {m.accuracy:>10.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
