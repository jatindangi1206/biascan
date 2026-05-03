"""
B04 — Overgeneralization transforms.

Datasets: PubHealth, ClaimBuster, Scientific Exaggeration (scope overclaiming)
Transform logic from EVAL_DATASETS:
  - PubHealth: Health claims labeled TRUE/FALSE/UNPROVEN/MIXTURE with explanations.
    "FALSE" claims often contain overgeneralizations. Extract explanation text as
    ground truth for overgeneralization detection.
  - ClaimBuster: Sentences classified by claim-worthiness (NFS/UFS/CFS).
    Check-worthy factual statements often contain broad claims.
  - Scientific Exaggeration: "Exaggerates" label captures scope overclaiming
    (e.g., "mice study" → "humans").
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)


# ── PubHealth ─────────────────────────────────────────────────────────────

def transform_pubhealth(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    PubHealth B04 transform.

    HOW TO USE: Health claims labeled TRUE/FALSE/UNPROVEN/MIXTURE with explanations.
    "FALSE" claims often contain overgeneralizations (e.g., "X cures all cancer").

    BIASED: FALSE/MIXTURE claims presented as established evidence syntheses.
    CONTROL: TRUE claims presented with their supporting explanations.
    """
    import json
    import urllib.request
    import tempfile
    import os

    logger.info("Loading PubHealth...")

    # Try HuggingFace first
    try:
        from datasets import load_dataset
        ds = load_dataset("ImperialCollegeLondon/health_fact")
        split = ds.get("test", ds.get("validation", ds.get("train")))
        items = list(split)
    except Exception as e:
        logger.warning(f"HF PubHealth failed: {e}. Trying GitHub fallback...")
        try:
            # Download from GitHub raw data
            url = "https://raw.githubusercontent.com/neemakot/Health-Fact-Checking/master/data/PUBHEALTH/test.tsv"
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "pubhealth_test.tsv")
            urllib.request.urlretrieve(url, path)

            items = []
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip().split("\t")
                for line in f:
                    fields = line.strip().split("\t")
                    if len(fields) >= len(header):
                        row = dict(zip(header, fields))
                        items.append(row)
        except Exception as e2:
            logger.warning(f"GitHub fallback also failed: {e2}. Using synthetic.")
            return _pubhealth_synthetic_fallback(max_samples, seed)

    rng = random.Random(seed)
    rng.shuffle(items)

    samples = []
    biased_count = 0
    control_count = 0

    for row in items:
        if len(samples) >= max_samples:
            break

        claim = str(row.get("claim", ""))
        explanation = str(row.get("explanation", ""))
        label = str(row.get("label", "")).lower().strip()

        if not claim or len(claim) < 20:
            continue

        if label in ("false", "mixture", "0", "2") and biased_count < max_samples // 2:
            # BIASED: FALSE/MIXTURE claim presented as synthesis
            biased_text = (
                f"Health evidence synthesis: {claim} "
                f"This finding has been consistently replicated across multiple studies "
                f"and populations, establishing it as a well-supported conclusion."
            )

            samples.append(EvalSample(
                id=f"B04_pubhealth_{len(samples)}",
                dataset_id="B04_pubhealth",
                bias_type="B04",
                agent="argus",
                input_text=biased_text,
                label="biased",
                bias_present=True,
                original_fields={
                    "claim": claim,
                    "explanation": explanation[:200],
                    "original_label": label,
                },
                transform_description=(
                    f"FALSE/MIXTURE claim ({label}) presented as established evidence. "
                    "Likely contains overgeneralization."
                ),
            ))
            biased_count += 1

        elif label in ("true", "1") and control_count < max_samples // 2:
            # CONTROL: TRUE claim with explanation
            control_text = (
                f"Health evidence synthesis: {claim} "
            )
            if explanation:
                control_text += f"Supporting evidence: {explanation[:300]}"

            samples.append(EvalSample(
                id=f"B04_pubhealth_{len(samples)}",
                dataset_id="B04_pubhealth",
                bias_type="B04",
                agent="argus",
                input_text=control_text,
                label="control",
                bias_present=False,
                original_fields={
                    "claim": claim,
                    "explanation": explanation[:200],
                    "original_label": label,
                },
                transform_description="TRUE claim with supporting explanation. No overgeneralization.",
            ))
            control_count += 1

    logger.info(f"PubHealth: {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


def _pubhealth_synthetic_fallback(max_samples: int, seed: int) -> list[EvalSample]:
    """Fallback when PubHealth can't load."""
    rng = random.Random(seed)
    biased_claims = [
        ("Drinking bleach cures COVID-19",
         "No scientific evidence supports this claim; bleach ingestion is dangerous."),
        ("Vaccines cause autism in all children",
         "Multiple large-scale studies have found no link between vaccines and autism."),
        ("Eating raw garlic eliminates all infections",
         "Garlic has some antimicrobial properties but cannot eliminate all infections."),
        ("5G towers cause cancer",
         "No credible scientific evidence links 5G radiation to cancer."),
        ("Essential oils cure diabetes",
         "Essential oils have no proven efficacy in treating or curing diabetes."),
        ("Homeopathy is more effective than antibiotics",
         "Systematic reviews show homeopathy performs no better than placebo."),
    ]
    control_claims = [
        ("Regular exercise reduces cardiovascular disease risk",
         "Meta-analyses consistently show exercise reduces CVD events by 20-35%."),
        ("Adequate sleep improves immune function",
         "Multiple RCTs demonstrate sleep deprivation impairs immune markers."),
        ("Fluoride in water is safe at recommended levels",
         "Decades of epidemiological data support safety at 0.7 mg/L."),
        ("Handwashing reduces transmission of respiratory infections",
         "Intervention studies show 20-40% reduction in transmission rates."),
        ("Smoking increases lung cancer risk",
         "Strong causal evidence from prospective cohort studies spanning 50+ years."),
    ]

    rng.shuffle(biased_claims)
    rng.shuffle(control_claims)

    samples = []
    for i, (claim, expl) in enumerate(biased_claims[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B04_pubhealth_synth_{i}",
            dataset_id="B04_pubhealth",
            bias_type="B04", agent="argus",
            input_text=(
                f"Health evidence synthesis: {claim}. "
                "This has been consistently replicated across studies."
            ),
            label="biased", bias_present=True,
            original_fields={"claim": claim, "explanation": expl, "synthetic": True},
            transform_description="Synthetic: FALSE health claim presented as established evidence.",
        ))
    for i, (claim, expl) in enumerate(control_claims[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B04_pubhealth_synth_ctrl_{i}",
            dataset_id="B04_pubhealth",
            bias_type="B04", agent="argus",
            input_text=f"Health evidence synthesis: {claim}. {expl}",
            label="control", bias_present=False,
            original_fields={"claim": claim, "explanation": expl, "synthetic": True},
            transform_description="Synthetic: TRUE health claim with supporting evidence.",
        ))
    return samples


# ── ClaimBuster ───────────────────────────────────────────────────────────
# ClaimBuster is on Zenodo, not HuggingFace. Use synthetic examples.

def transform_claimbuster(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    ClaimBuster B04 transform.

    HOW TO USE: Sentences classified by claim-worthiness (NFS/UFS/CFS).
    Check-worthy factual statements often contain broad claims — useful
    for testing scope detection.

    BIASED: Overgeneralized check-worthy claims (CFS that contain scope overreach).
    CONTROL: Non-claim factual sentences (NFS) or properly scoped claims.

    NOTE: ClaimBuster on Zenodo requires manual download. Uses synthetic examples
    modeled on U.S. presidential debate claim patterns.
    """
    rng = random.Random(seed)

    # Synthetic overgeneralized claims (modeled on ClaimBuster CFS patterns)
    overgeneralized_claims = [
        "Every single study has shown that this treatment is 100% effective.",
        "All scientists agree that climate change will destroy civilization within 10 years.",
        "This policy has never failed anywhere it has been implemented.",
        "Everyone who takes this medication experiences complete recovery.",
        "No country in history has ever succeeded with this economic model.",
        "All the evidence unanimously points to a single cause.",
        "This intervention works for every patient regardless of their condition.",
        "The entire scientific community supports this conclusion without exception.",
        "In every case studied, the treatment eliminated the disease entirely.",
        "All randomized trials have shown identical positive results.",
        "This finding applies universally across all populations and demographics.",
        "There is absolutely no evidence contradicting this hypothesis.",
        "Every expert in the field endorses this approach without reservation.",
        "The cure rate is 100% across all age groups and severity levels.",
        "No side effects have ever been reported in any study of this drug.",
    ]

    properly_scoped = [
        "In three of five randomized trials, the treatment showed modest improvement.",
        "The study found a statistically significant but small effect (d = 0.3) in adults aged 40-60.",
        "Among the 12 studies reviewed, results were mixed, with 7 showing benefit and 5 showing no effect.",
        "The intervention reduced symptoms in 45% of participants in the treatment group.",
        "While promising, results are limited to a single-center study with 120 participants.",
        "The meta-analysis found moderate heterogeneity (I² = 55%) across included studies.",
        "Benefits were observed primarily in the subgroup with severe baseline symptoms.",
        "The 95% confidence interval ranged from 0.8 to 2.4, suggesting uncertainty in the estimate.",
        "Long-term efficacy beyond 12 months has not been established.",
        "Generalizability is limited as participants were predominantly from urban academic centers.",
    ]

    rng.shuffle(overgeneralized_claims)
    rng.shuffle(properly_scoped)

    samples = []

    for i, claim in enumerate(overgeneralized_claims[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B04_claimbuster_{len(samples)}",
            dataset_id="B04_claimbuster",
            bias_type="B04",
            agent="argus",
            input_text=f"Evidence synthesis: {claim}",
            label="biased",
            bias_present=True,
            original_fields={"claim": claim, "synthetic": True},
            transform_description="Overgeneralized check-worthy claim (synthetic CFS pattern).",
        ))

    for i, claim in enumerate(properly_scoped[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B04_claimbuster_{len(samples)}",
            dataset_id="B04_claimbuster",
            bias_type="B04",
            agent="argus",
            input_text=f"Evidence synthesis: {claim}",
            label="control",
            bias_present=False,
            original_fields={"claim": claim, "synthetic": True},
            transform_description="Properly scoped factual statement.",
        ))

    rng.shuffle(samples)
    logger.info(f"ClaimBuster (synthetic): {len(samples)} samples")
    return samples


# ── Scientific Exaggeration for B04 (scope overclaiming) ─────────────────

def transform_sci_exaggeration_scope(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    Scientific Exaggeration B04 (scope) transform.

    HOW TO USE: "Exaggerates" label often captures scope overclaiming
    (e.g., "mice study" → "humans"). Extract these and test if ARGUS catches
    population/scope leaps.

    BIASED: Press releases that overclaim scope vs the abstract.
    CONTROL: Press releases that maintain appropriate scope (SAME label).
    """
    from datasets import load_dataset

    logger.info("Loading Scientific Exaggeration for B04 scope analysis...")
    ds = load_dataset("copenlu/scientific-exaggeration-detection", split="train")

    rng = random.Random(seed)
    items = list(ds)
    rng.shuffle(items)

    # Scope-related keywords that signal population/generalization leaps
    scope_keywords = [
        "all patients", "everyone", "humans", "people", "population",
        "universally", "always", "never", "completely", "entirely",
        "definitively", "proven", "established", "cure", "eliminat",
    ]

    samples = []
    biased_count = 0
    control_count = 0

    for row in items:
        if len(samples) >= max_samples:
            break

        abstract_conclusion = row.get("abstract_conclusion", "")
        press_conclusion = row.get("press_release_conclusion", "")
        label = str(row.get("exaggeration_label", "")).lower().strip()

        if not press_conclusion or not abstract_conclusion:
            continue

        if label == "exaggerates" and biased_count < max_samples // 2:
            # Present the press release as a scope-overclaiming synthesis
            biased_text = (
                f"Study findings: The research demonstrates that {press_conclusion} "
                f"These results are broadly applicable across patient populations."
            )

            samples.append(EvalSample(
                id=f"B04_sciexag_scope_{len(samples)}",
                dataset_id="B04_sci_exag_scope",
                bias_type="B04",
                agent="argus",
                input_text=biased_text,
                label="biased",
                bias_present=True,
                original_fields={
                    "abstract_conclusion": abstract_conclusion,
                    "press_conclusion": press_conclusion,
                    "exaggeration_label": label,
                },
                transform_description=(
                    "Press release overclaims scope vs abstract. "
                    "Broad applicability asserted without evidence support."
                ),
            ))
            biased_count += 1

        elif label == "same" and control_count < max_samples // 2:
            control_text = (
                f"Study findings: {press_conclusion} "
                f"Original abstract conclusion: {abstract_conclusion}"
            )

            samples.append(EvalSample(
                id=f"B04_sciexag_scope_{len(samples)}",
                dataset_id="B04_sci_exag_scope",
                bias_type="B04",
                agent="argus",
                input_text=control_text,
                label="control",
                bias_present=False,
                original_fields={
                    "abstract_conclusion": abstract_conclusion,
                    "press_conclusion": press_conclusion,
                    "exaggeration_label": label,
                },
                transform_description="Scope faithfully represented (SAME label).",
            ))
            control_count += 1

    logger.info(f"Sci Exaggeration Scope: {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


# ── Register ──────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="B04_pubhealth",
    bias_type="B04",
    dataset_name="PubHealth",
    agent="argus",
    how_to_use="FALSE/MIXTURE claims contain overgeneralizations. TRUE claims are controls.",
    transform_fn=transform_pubhealth,
))

register_transform(DatasetSpec(
    dataset_id="B04_claimbuster",
    bias_type="B04",
    dataset_name="ClaimBuster",
    agent="argus",
    how_to_use="Check-worthy factual statements with broad claims test scope detection.",
    transform_fn=transform_claimbuster,
))

register_transform(DatasetSpec(
    dataset_id="B04_sci_exag_scope",
    bias_type="B04",
    dataset_name="Scientific Exaggeration (Scope)",
    agent="argus",
    how_to_use="'Exaggerates' label captures scope overclaiming (mice→humans).",
    transform_fn=transform_sci_exaggeration_scope,
))
