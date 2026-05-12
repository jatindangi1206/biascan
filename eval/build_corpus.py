#!/usr/bin/env python3
"""
Build the full evaluation corpus — downloads real datasets from HuggingFace
where available, uses synthetic fallbacks where not.

Outputs:
  eval/output/samples.json  — full corpus with all samples
  eval/output/corpus_stats.json — statistics about the generated corpus

Usage:
  cd /path/to/bias
  python -m eval.build_corpus [--max-samples 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.transforms import (
    TRANSFORM_REGISTRY,
    get_all_specs,
    EvalSample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("build_corpus")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build_corpus(max_samples: int = 50, seed: int = 42) -> dict:
    """Run all registered transforms and collect samples."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    specs = get_all_specs()
    logger.info(f"Found {len(specs)} registered dataset transforms")

    for spec in specs:
        dataset_id = spec.dataset_id
        logger.info(f"--- Processing {dataset_id} ({spec.dataset_name}) ---")

        if spec.transform_fn is None:
            logger.warning(f"  No transform function for {dataset_id}, skipping")
            continue

        try:
            samples: list[EvalSample] = spec.transform_fn(
                max_samples=max_samples, seed=seed,
            )
        except Exception as e:
            logger.error(f"  Transform failed for {dataset_id}: {e}")
            samples = []

        if not samples:
            logger.warning(f"  No samples generated for {dataset_id}")
            continue

        # Split into biased and control
        biased = [s for s in samples if s.bias_present]
        control = [s for s in samples if not s.bias_present]

        corpus[dataset_id] = {
            "spec": {
                "dataset_id": spec.dataset_id,
                "bias_type": spec.bias_type,
                "dataset_name": spec.dataset_name,
                "agent": spec.agent,
                "how_to_use": spec.how_to_use,
            },
            "biased": [_sample_to_dict(s) for s in biased],
            "control": [_sample_to_dict(s) for s in control],
        }

        stats[dataset_id] = {
            "dataset_name": spec.dataset_name,
            "bias_type": spec.bias_type,
            "agent": spec.agent,
            "total": len(samples),
            "biased": len(biased),
            "control": len(control),
            "has_synthetic": any(
                s.original_fields.get("synthetic", False) for s in samples
            ),
        }

        logger.info(
            f"  {dataset_id}: {len(biased)} biased + {len(control)} control "
            f"= {len(samples)} total"
        )

    # Write outputs
    samples_path = OUTPUT_DIR / "samples.json"
    stats_path = OUTPUT_DIR / "corpus_stats.json"

    with open(samples_path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    total_samples = sum(s["total"] for s in stats.values())
    total_biased = sum(s["biased"] for s in stats.values())
    total_control = sum(s["control"] for s in stats.values())

    summary = {
        "total_datasets": len(stats),
        "total_samples": total_samples,
        "total_biased": total_biased,
        "total_control": total_control,
        "bias_types_covered": sorted(set(s["bias_type"] for s in stats.values())),
        "agents_covered": sorted(set(s["agent"] for s in stats.values())),
        "datasets": stats,
    }

    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Corpus built: {total_samples} samples across {len(stats)} datasets")
    logger.info(f"  Biased: {total_biased}, Control: {total_control}")
    logger.info(f"  Bias types: {summary['bias_types_covered']}")
    logger.info(f"  Output: {samples_path}")
    logger.info(f"  Stats:  {stats_path}")

    return summary


def _sample_to_dict(sample: EvalSample) -> dict:
    """Convert an EvalSample to a serializable dict."""
    return {
        "id": sample.id,
        "dataset_id": sample.dataset_id,
        "bias_type": sample.bias_type,
        "agent": sample.agent,
        "input_text": sample.input_text,
        "label": sample.label,
        "bias_present": sample.bias_present,
        "original_fields": sample.original_fields,
        "transform_description": sample.transform_description,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BiasScan evaluation corpus")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per dataset (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    build_corpus(max_samples=args.max_samples, seed=args.seed)
