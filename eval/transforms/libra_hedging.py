"""
LIBRA Agent — Scope & Hedging Analysis transforms.

Datasets: BioScope (hedge detection), CoNLL-2010, Scientific Exaggeration
Transform logic from EVAL_DATASETS:
  - BioScope: Gold standard for biomedical hedging. Annotations mark hedge cues
    and their scope. Compare LIBRA's Hyland-based hedge counts against BioScope.
  - CoNLL-2010: Two domains (bio + Wikipedia). Sentence-level uncertain/certain
    labels + in-sentence hedge scope spans. Benchmark LIBRA's classification accuracy.
  - Scientific Exaggeration: Abstract→press release exaggeration captures
    temporal overclaiming, population overclaiming, and scope extension.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)


# ── BioScope for LIBRA ───────────────────────────────────────────────────

def transform_bioscope_hedge(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    BioScope LIBRA transform.

    HOW TO USE: Gold standard for biomedical hedging. Annotations mark hedge cues
    ("may", "possibly", "suggests") and their scope. Compare LIBRA's hedge detection
    against BioScope annotations.

    BIASED: Text where hedging is MISSING but should be present (speculative content
    presented with certainty). LIBRA should flag insufficient hedging.
    CONTROL: Text with appropriate hedging for speculative content.

    NOTE: BioScope requires manual download. Uses synthetic biomedical examples
    with known hedge patterns.
    """
    rng = random.Random(seed)

    # Pairs: (speculative text without hedging, same text with appropriate hedging)
    hedge_pairs = [
        (
            "The protein functions as a transcription factor in hepatocytes.",
            "The protein may function as a transcription factor in hepatocytes.",
            ["may"],
        ),
        (
            "This pathway regulates apoptosis through caspase-3 activation.",
            "This pathway appears to regulate apoptosis, possibly through caspase-3 activation.",
            ["appears to", "possibly"],
        ),
        (
            "The drug inhibits tumor growth by blocking angiogenesis.",
            "The drug might inhibit tumor growth, potentially by blocking angiogenesis.",
            ["might", "potentially"],
        ),
        (
            "Dietary fiber protects against colorectal cancer through microbiome modulation.",
            "Dietary fiber may protect against colorectal cancer, possibly through microbiome modulation.",
            ["may", "possibly"],
        ),
        (
            "The mutation causes resistance to first-line therapy in all patients.",
            "The mutation could contribute to resistance to first-line therapy in some patients.",
            ["could", "contribute to", "some"],
        ),
        (
            "Neuroinflammation drives cognitive decline in Alzheimer's disease.",
            "Neuroinflammation is thought to contribute to cognitive decline in Alzheimer's disease.",
            ["is thought to", "contribute to"],
        ),
        (
            "Sleep deprivation directly impairs immune function through cortisol elevation.",
            "Sleep deprivation likely impairs immune function, possibly through cortisol elevation.",
            ["likely", "possibly"],
        ),
        (
            "The biomarker indicates disease progression with 95% accuracy.",
            "The biomarker suggests disease progression, though its accuracy requires further validation.",
            ["suggests", "requires further validation"],
        ),
        (
            "This gene is responsible for drug metabolism variation across populations.",
            "This gene is believed to play a role in drug metabolism variation across populations.",
            ["is believed to", "play a role in"],
        ),
        (
            "Gut dysbiosis triggers autoimmune responses in genetically susceptible individuals.",
            "Gut dysbiosis may trigger autoimmune responses in genetically susceptible individuals.",
            ["may"],
        ),
        (
            "The compound eliminates biofilm formation in all tested bacterial strains.",
            "The compound appears to reduce biofilm formation in the tested bacterial strains.",
            ["appears to", "reduce" , "the tested"],
        ),
        (
            "Epigenetic changes fully explain the increased disease risk.",
            "Epigenetic changes might partially explain the increased disease risk.",
            ["might", "partially"],
        ),
    ]

    rng.shuffle(hedge_pairs)

    samples = []

    # BIASED: Missing hedging (speculative claims stated as certain)
    for i, (no_hedge, with_hedge, cues) in enumerate(hedge_pairs[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"LIBRA_bioscope_{len(samples)}",
            dataset_id="LIBRA_bioscope",
            bias_type="B03",  # Certainty inflation via missing hedging
            agent="libra",
            input_text=f"Research finding: {no_hedge}",
            label="biased",
            bias_present=True,
            original_fields={
                "without_hedging": no_hedge,
                "with_hedging": with_hedge,
                "expected_hedge_cues": cues,
                "synthetic": True,
            },
            transform_description=(
                f"Speculative claim stated without hedging. "
                f"Missing cues: {', '.join(cues)}."
            ),
        ))

    # CONTROL: Appropriate hedging present
    for i, (no_hedge, with_hedge, cues) in enumerate(hedge_pairs[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"LIBRA_bioscope_{len(samples)}",
            dataset_id="LIBRA_bioscope",
            bias_type="B03",
            agent="libra",
            input_text=f"Research finding: {with_hedge}",
            label="control",
            bias_present=False,
            original_fields={
                "with_hedging": with_hedge,
                "hedge_cues_present": cues,
                "synthetic": True,
            },
            transform_description=f"Appropriate hedging present: {', '.join(cues)}.",
        ))

    rng.shuffle(samples)
    logger.info(f"BioScope LIBRA (synthetic): {len(samples)} samples")
    return samples


# ── Scientific Exaggeration for LIBRA ────────────────────────────────────

def transform_sci_exag_libra(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    Scientific Exaggeration LIBRA transform.

    HOW TO USE: Abstract→press release exaggeration. The "exaggerates" label
    captures temporal overclaiming, population overclaiming, and scope extension —
    all LIBRA targets.

    BIASED: Press releases that overclaim vs abstract (scope/hedge mismatch).
    CONTROL: Press releases that faithfully represent the abstract's scope.
    """
    from datasets import load_dataset

    logger.info("Loading Scientific Exaggeration for LIBRA...")
    ds = load_dataset("copenlu/scientific-exaggeration-detection", split="train")

    rng = random.Random(seed)
    items = list(ds)
    rng.shuffle(items)

    samples = []
    biased_count = 0
    control_count = 0

    for row in items:
        if len(samples) >= max_samples:
            break

        abstract_conclusion = row.get("abstract_conclusion", "")
        press_conclusion = row.get("press_release_conclusion", "")
        label = str(row.get("exaggeration_label", "")).lower().strip()
        abstract_strength = str(row.get("abstract_strength", ""))
        press_strength = str(row.get("press_release_strength", ""))

        if not press_conclusion or not abstract_conclusion:
            continue

        if label == "exaggerates" and biased_count < max_samples // 2:
            # BIASED: Scope/hedge mismatch between abstract and press release
            biased_text = (
                f"Summary of findings: {press_conclusion}\n"
                f"Original abstract: {abstract_conclusion}\n"
                f"Assessment: The summary accurately represents the study findings."
            )

            samples.append(EvalSample(
                id=f"LIBRA_sciexag_{len(samples)}",
                dataset_id="LIBRA_sci_exag",
                bias_type="B03",
                agent="libra",
                input_text=biased_text,
                label="biased",
                bias_present=True,
                original_fields={
                    "abstract_conclusion": abstract_conclusion,
                    "press_conclusion": press_conclusion,
                    "abstract_strength": abstract_strength,
                    "press_strength": press_strength,
                    "exaggeration_label": label,
                },
                transform_description=(
                    f"Press release exaggerates abstract. "
                    f"Strength mismatch: abstract={abstract_strength}, press={press_strength}."
                ),
            ))
            biased_count += 1

        elif label == "same" and control_count < max_samples // 2:
            control_text = (
                f"Summary of findings: {press_conclusion}\n"
                f"Original abstract: {abstract_conclusion}\n"
                f"Assessment: The summary accurately represents the study findings."
            )

            samples.append(EvalSample(
                id=f"LIBRA_sciexag_{len(samples)}",
                dataset_id="LIBRA_sci_exag",
                bias_type="B03",
                agent="libra",
                input_text=control_text,
                label="control",
                bias_present=False,
                original_fields={
                    "abstract_conclusion": abstract_conclusion,
                    "press_conclusion": press_conclusion,
                    "exaggeration_label": label,
                },
                transform_description="Press release faithfully represents abstract scope (SAME).",
            ))
            control_count += 1

    logger.info(f"Sci Exaggeration LIBRA: {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


# ── Register ──────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="LIBRA_bioscope",
    bias_type="B03",
    dataset_name="BioScope (LIBRA)",
    agent="libra",
    how_to_use="Compare LIBRA's Hyland-based hedge counts against BioScope annotations.",
    transform_fn=transform_bioscope_hedge,
))

register_transform(DatasetSpec(
    dataset_id="LIBRA_sci_exag",
    bias_type="B03",
    dataset_name="Scientific Exaggeration (LIBRA)",
    agent="libra",
    how_to_use="Abstract→press release exaggeration captures scope extension — LIBRA targets.",
    transform_fn=transform_sci_exag_libra,
))
