"""
LENS Agent — Discourse Analysis transforms.

Datasets: SciCite
Transform logic from EVAL_DATASETS:
  - SciCite: Citations classified as BACKGROUND/METHOD/RESULT.
    Test LENS's ability to detect citation function and whether
    support/contradict direction is correctly identified.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)


def transform_scicite(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    SciCite LENS transform.

    HOW TO USE: Citations classified as BACKGROUND/METHOD/RESULT. Test LENS's
    ability to detect citation function and whether support/contradict direction
    is correctly identified.

    This is a CLASSIFICATION task rather than bias detection — samples test
    whether LENS correctly identifies citation intent in scientific text.

    We construct evaluation samples by:
    - Taking citation contexts and embedding them in synthesis-like paragraphs
    - BIASED: Citations used misleadingly (e.g., background citation presented
      as result-supporting evidence, or method citation used as evidence claim)
    - CONTROL: Citations used appropriately for their actual intent
    """
    logger.info("Loading SciCite dataset...")

    # Try HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/scicite")
        split = ds.get("test", ds.get("validation", ds.get("train")))
        items = list(split)
    except Exception as e:
        logger.warning(f"SciCite HF load failed: {e}. Using synthetic fallback.")
        return _scicite_synthetic_fallback(max_samples, seed)

    rng = random.Random(seed)
    rng.shuffle(items)

    # Label map: 0=background, 1=method, 2=result
    label_map = {0: "background", 1: "method", 2: "result"}

    samples = []
    biased_count = 0
    control_count = 0

    for row in items:
        if len(samples) >= max_samples:
            break

        text = row.get("string", row.get("text", ""))
        label = row.get("label", -1)
        section = row.get("sectionName", "")

        if not text or len(text) < 30:
            continue

        intent = label_map.get(label, str(label)) if isinstance(label, int) else str(label)

        if intent == "background" and biased_count < max_samples // 2:
            # BIASED: Background citation misrepresented as direct result evidence
            biased_text = (
                f"Our results are directly supported by the finding that {text.lower()} "
                f"This evidence confirms the effectiveness of the proposed approach."
            )

            samples.append(EvalSample(
                id=f"LENS_scicite_{len(samples)}",
                dataset_id="LENS_scicite",
                bias_type="B01",  # Misrepresenting citation intent → confirmation bias signal
                agent="lens",
                input_text=biased_text,
                label="biased",
                bias_present=True,
                original_fields={
                    "original_text": text,
                    "true_intent": intent,
                    "presented_as": "result",
                    "section": section,
                },
                transform_description=(
                    f"BACKGROUND citation misrepresented as RESULT evidence. "
                    f"True intent: {intent}."
                ),
            ))
            biased_count += 1

        elif intent == "result" and control_count < max_samples // 2:
            # CONTROL: Result citation used appropriately as evidence
            control_text = (
                f"Evidence from related work shows that {text.lower()} "
                f"This result is consistent with the broader literature."
            )

            samples.append(EvalSample(
                id=f"LENS_scicite_{len(samples)}",
                dataset_id="LENS_scicite",
                bias_type="B01",
                agent="lens",
                input_text=control_text,
                label="control",
                bias_present=False,
                original_fields={
                    "original_text": text,
                    "true_intent": intent,
                    "section": section,
                },
                transform_description=f"RESULT citation used appropriately as evidence.",
            ))
            control_count += 1

        elif intent == "method" and biased_count < max_samples // 2:
            # BIASED: Method citation presented as evidence for claims
            biased_text = (
                f"Strong evidence demonstrates that {text.lower()} "
                f"These findings provide conclusive support for the hypothesis."
            )

            samples.append(EvalSample(
                id=f"LENS_scicite_{len(samples)}",
                dataset_id="LENS_scicite",
                bias_type="B01",
                agent="lens",
                input_text=biased_text,
                label="biased",
                bias_present=True,
                original_fields={
                    "original_text": text,
                    "true_intent": intent,
                    "presented_as": "result",
                    "section": section,
                },
                transform_description=(
                    f"METHOD citation misrepresented as conclusive evidence. "
                    f"True intent: {intent}."
                ),
            ))
            biased_count += 1

    logger.info(f"SciCite LENS: {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


def _scicite_synthetic_fallback(max_samples: int, seed: int) -> list[EvalSample]:
    """Fallback when SciCite can't load from HuggingFace."""
    rng = random.Random(seed)

    biased_examples = [
        {
            "text": "Neural networks have been widely used for text classification [1].",
            "true_intent": "background",
            "biased_framing": (
                "Our results are confirmed by the established finding that neural networks "
                "have been widely used for text classification [1]. This evidence validates "
                "our approach."
            ),
        },
        {
            "text": "We follow the preprocessing pipeline of Chen et al. [2].",
            "true_intent": "method",
            "biased_framing": (
                "Strong evidence from Chen et al. [2] demonstrates the superiority of this "
                "preprocessing approach, conclusively supporting our methodology."
            ),
        },
        {
            "text": "Attention mechanisms were introduced by Vaswani et al. [3].",
            "true_intent": "background",
            "biased_framing": (
                "The results of Vaswani et al. [3] directly prove that attention mechanisms "
                "outperform all alternatives, confirming our findings."
            ),
        },
        {
            "text": "Data augmentation was performed using the rotation method of Brown [4].",
            "true_intent": "method",
            "biased_framing": (
                "Brown [4] demonstrated that rotation-based data augmentation produces "
                "superior results, providing evidence for our claims."
            ),
        },
        {
            "text": "Deep learning has transformed natural language processing [5].",
            "true_intent": "background",
            "biased_framing": (
                "The findings show that deep learning has proven to transform NLP [5], "
                "directly supporting the conclusions of our study."
            ),
        },
    ]

    control_examples = [
        {
            "text": "Our model achieved 94.2% accuracy, exceeding the 91.5% of prior work [6].",
            "true_intent": "result",
            "control_framing": (
                "Compared to prior results, our model achieved 94.2% accuracy versus "
                "91.5% reported by prior work [6]."
            ),
        },
        {
            "text": "Similar improvements were reported by Wang [7] using larger datasets.",
            "true_intent": "result",
            "control_framing": (
                "Consistent with our findings, Wang [7] reported similar improvements "
                "when using larger datasets."
            ),
        },
        {
            "text": "The error rate decreased from 8.1% to 3.2% with the proposed method [8].",
            "true_intent": "result",
            "control_framing": (
                "Evidence from [8] shows a reduction in error rate from 8.1% to 3.2% "
                "with the proposed method, supporting the effectiveness of this approach."
            ),
        },
        {
            "text": "Performance gains were significant across all evaluation metrics [9].",
            "true_intent": "result",
            "control_framing": (
                "The results in [9] demonstrate significant performance gains across "
                "all evaluation metrics."
            ),
        },
        {
            "text": "The F1 score improved by 5.3 points over the baseline [10].",
            "true_intent": "result",
            "control_framing": (
                "Our results align with [10], which showed a 5.3-point F1 improvement "
                "over the baseline."
            ),
        },
    ]

    rng.shuffle(biased_examples)
    rng.shuffle(control_examples)

    samples = []

    for i, ex in enumerate(biased_examples[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"LENS_scicite_synth_{len(samples)}",
            dataset_id="LENS_scicite",
            bias_type="B01",
            agent="lens",
            input_text=ex["biased_framing"],
            label="biased",
            bias_present=True,
            original_fields={
                "original_text": ex["text"],
                "true_intent": ex["true_intent"],
                "synthetic": True,
            },
            transform_description=f"Synthetic: {ex['true_intent']} citation misrepresented as result evidence.",
        ))

    for i, ex in enumerate(control_examples[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"LENS_scicite_synth_{len(samples)}",
            dataset_id="LENS_scicite",
            bias_type="B01",
            agent="lens",
            input_text=ex["control_framing"],
            label="control",
            bias_present=False,
            original_fields={
                "original_text": ex["text"],
                "true_intent": ex["true_intent"],
                "synthetic": True,
            },
            transform_description=f"Synthetic: {ex['true_intent']} citation used appropriately.",
        ))

    return samples


# ── Register ──────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="LENS_scicite",
    bias_type="B01",
    dataset_name="SciCite (LENS)",
    agent="lens",
    how_to_use="Test LENS's citation intent detection and support/contradict direction.",
    transform_fn=transform_scicite,
))
