"""
B08 — Causal Inference Error transforms.

Datasets: Corr2Cause
Transform logic from EVAL_DATASETS:
  - Corr2Cause: Correlational statements → valid/invalid causal inferences.
    Directly tests whether the system can detect when correlation is
    misrepresented as causation. The gold standard for B08.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)


def transform_corr2cause(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    Corr2Cause B08 transform.

    HOW TO USE: Correlational statements → valid/invalid causal inferences.
    Directly tests whether the system can detect when correlation is
    misrepresented as causation.

    BIASED (label=0): Invalid causal inference — correlation presented as causation.
    CONTROL (label=1): Valid causal inference — properly supported causal claim.
    """
    from datasets import load_dataset

    logger.info("Loading Corr2Cause dataset...")
    ds = load_dataset("causal-nlp/corr2cause")

    # Use test split if available, fall back to validation, then train
    split_name = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
    split = ds[split_name]

    rng = random.Random(seed)
    items = list(split)
    rng.shuffle(items)

    samples = []
    biased_count = 0
    control_count = 0

    for row in items:
        if len(samples) >= max_samples:
            break

        premise = row.get("input", "")
        label = row.get("label", -1)

        if not premise or len(str(premise)) < 20:
            continue

        if label == 0 and biased_count < max_samples // 2:
            # BIASED: Invalid causal inference — correlation ≠ causation
            biased_text = (
                f"Evidence synthesis: {premise} "
                f"Based on this evidence, the causal relationship appears "
                f"well-established and supports targeted intervention."
            )

            samples.append(EvalSample(
                id=f"B08_corr2cause_{len(samples)}",
                dataset_id="B08_corr2cause",
                bias_type="B08",
                agent="argus",
                input_text=biased_text,
                label="biased",
                bias_present=True,
                original_fields={
                    "premise": premise,
                    "label": label,
                    "num_variables": row.get("num_variables", ""),
                    "template": row.get("template", ""),
                },
                transform_description=(
                    "Invalid causal inference: correlational evidence presented "
                    "as established causation."
                ),
            ))
            biased_count += 1

        elif label == 1 and control_count < max_samples // 2:
            # CONTROL: Valid causal inference
            control_text = (
                f"Evidence synthesis: {premise} "
                f"The causal direction is supported by the study design."
            )

            samples.append(EvalSample(
                id=f"B08_corr2cause_{len(samples)}",
                dataset_id="B08_corr2cause",
                bias_type="B08",
                agent="argus",
                input_text=control_text,
                label="control",
                bias_present=False,
                original_fields={
                    "premise": premise,
                    "label": label,
                    "num_variables": row.get("num_variables", ""),
                    "template": row.get("template", ""),
                },
                transform_description="Valid causal inference supported by study design.",
            ))
            control_count += 1

    logger.info(f"Corr2Cause: {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


# ── Register ──────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="B08_corr2cause",
    bias_type="B08",
    dataset_name="Corr2Cause",
    agent="argus",
    how_to_use="Correlational statements with valid/invalid causal inferences. Gold standard for B08.",
    transform_fn=transform_corr2cause,
))
