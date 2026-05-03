"""
Unified dataset loader — downloads from HuggingFace and transforms into
BiasScan-compatible test cases.

Each test case has:
  - input_text: text to feed into the pipeline (or individual agent)
  - expected_bias_types: list of bias codes that SHOULD be detected
  - expected_label: ground truth label from the dataset
  - metadata: original dataset fields for debugging
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "datasets" / ".cache"


@dataclass
class TestCase:
    """A single evaluation test case."""
    id: str
    input_text: str
    expected_bias_types: list[str] = field(default_factory=list)
    expected_label: str = ""
    bias_present: bool = True           # Should the pipeline detect bias here?
    metadata: dict = field(default_factory=dict)
    dataset_name: str = ""


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


# ── SciFact Loader ──────────────────────────────────────────────────────────

def load_scifact(sample_size: int = 50, seed: int = 42) -> list[TestCase]:
    """
    Load SciFact for B01 (Confirmation Bias) evaluation.

    Strategy: Take REFUTED claims and pair them with only supporting evidence.
    The pipeline should detect confirmation bias (one-sided evidence).
    Also include SUPPORTS claims as negative controls (no bias expected).
    """
    from datasets import load_dataset

    logger.info("Loading SciFact dataset...")
    # SciFact uses a legacy loading script; fall back to raw parquet or alt config
    try:
        ds = load_dataset("allenai/scifact", "claims")
    except Exception:
        try:
            ds = load_dataset("allenai/scifact_open")
        except Exception:
            logger.warning("SciFact not loadable — using fallback synthetic claims")
            return _scifact_fallback(sample_size, seed)

    # Use validation split (SciFact test has no labels)
    split = ds.get("validation", ds.get("train"))

    cases = []
    rng = random.Random(seed)

    items = list(split)
    rng.shuffle(items)

    for i, row in enumerate(items[:sample_size * 2]):  # oversample then trim
        claim = row.get("claim", "")
        evidence = row.get("evidence", {})
        label = row.get("label", "")  # field name varies

        if not claim:
            continue

        # Construct a mini "synthesis" paragraph from the claim
        # For REFUTES: this is biased text (presents refuted claim as supported)
        # For SUPPORTS: this is clean text (negative control)
        if isinstance(evidence, dict):
            evidence_text = "; ".join(str(v) for v in evidence.values() if v)
        elif isinstance(evidence, list):
            evidence_text = "; ".join(str(e) for e in evidence if e)
        else:
            evidence_text = str(evidence) if evidence else ""

        synth_text = (
            f"Evidence from the literature suggests that {claim.lower()} "
            f"This finding is supported by multiple studies. {evidence_text}"
        )

        # Determine if this is a bias case
        is_refuted = "refut" in str(label).lower() or "contra" in str(label).lower()

        cases.append(TestCase(
            id=f"scifact-{_hash_text(claim)}-{i}",
            input_text=synth_text,
            expected_bias_types=["B01"] if is_refuted else [],
            expected_label=str(label),
            bias_present=is_refuted,
            metadata={"claim": claim, "original_label": str(label)},
            dataset_name="SciFact",
        ))

        if len(cases) >= sample_size:
            break

    logger.info(f"Loaded {len(cases)} SciFact test cases "
                f"({sum(1 for c in cases if c.bias_present)} biased, "
                f"{sum(1 for c in cases if not c.bias_present)} clean)")
    return cases


# ── Scientific Exaggeration Loader ──────────────────────────────────────────

def load_scientific_exaggeration(sample_size: int = 50, seed: int = 42) -> list[TestCase]:
    """
    Load Scientific Exaggeration Detection for B03 (Certainty Inflation) / B04 (Overgeneralization).

    Strategy: "EXAGGERATES" pairs = certainty inflation (press release overclaims).
    "SAME" pairs = negative control (faithful reporting).
    """
    from datasets import load_dataset

    logger.info("Loading Scientific Exaggeration dataset...")
    ds = load_dataset("copenlu/scientific-exaggeration-detection", trust_remote_code=False)

    split = ds.get("test", ds.get("validation", ds.get("train")))
    items = list(split)

    rng = random.Random(seed)
    rng.shuffle(items)

    cases = []
    for i, row in enumerate(items[:sample_size * 2]):
        abstract = row.get("abstract_conclusion", row.get("abstract", ""))
        press = row.get("press_release_conclusion", row.get("press_release", ""))
        label = row.get("exaggeration_label", row.get("label", ""))

        if not press:
            continue

        # Use the press release text as the "synthesis" to scan
        synth_text = (
            f"Research findings: {press} "
            f"The original study abstract stated: {abstract}"
        )

        # Map labels — "exaggerates" is biased, "same" is clean, "downplays" is also biased
        label_str = str(label).lower().strip()
        is_exaggerated = label_str in ("exaggerates", "downplays")

        bias_codes = []
        if is_exaggerated:
            bias_codes = ["B03", "B04"]

        cases.append(TestCase(
            id=f"sciexag-{_hash_text(press)}-{i}",
            input_text=synth_text,
            expected_bias_types=bias_codes,
            expected_label=label_str,
            bias_present=is_exaggerated,
            metadata={"abstract": abstract, "press_release": press, "label": label_str},
            dataset_name="Scientific Exaggeration",
        ))

        if len(cases) >= sample_size:
            break

    logger.info(f"Loaded {len(cases)} Scientific Exaggeration test cases")
    return cases


# ── PubHealth Loader ────────────────────────────────────────────────────────

def load_pubhealth(sample_size: int = 50, seed: int = 42) -> list[TestCase]:
    """
    Load PubHealth for B04 (Overgeneralization) evaluation.

    Strategy: FALSE/MIXTURE claims contain overgeneralizations.
    TRUE claims are negative controls.
    """
    from datasets import load_dataset

    logger.info("Loading PubHealth dataset...")
    try:
        ds = load_dataset("ImperialCollegeLondon/health_fact")
    except Exception:
        logger.warning("PubHealth not loadable — using fallback claims")
        return _pubhealth_fallback(sample_size, seed)

    split = ds.get("test", ds.get("validation", ds.get("train")))
    items = list(split)

    rng = random.Random(seed)
    rng.shuffle(items)

    cases = []
    for i, row in enumerate(items[:sample_size * 3]):
        claim = row.get("claim", "")
        explanation = row.get("explanation", "")
        label = row.get("label", "")

        if not claim or len(claim) < 20:
            continue

        synth_text = (
            f"Health evidence synthesis: {claim} "
            f"Supporting rationale: {explanation}" if explanation else claim
        )

        label_str = str(label).lower()
        is_biased = label_str in ("false", "mixture", "0", "2")

        cases.append(TestCase(
            id=f"pubhealth-{_hash_text(claim)}-{i}",
            input_text=synth_text,
            expected_bias_types=["B04"] if is_biased else [],
            expected_label=label_str,
            bias_present=is_biased,
            metadata={"claim": claim, "label": label_str},
            dataset_name="PubHealth",
        ))

        if len(cases) >= sample_size:
            break

    logger.info(f"Loaded {len(cases)} PubHealth test cases")
    return cases


# ── Corr2Cause Loader ──────────────────────────────────────────────────────

def load_corr2cause(sample_size: int = 50, seed: int = 42) -> list[TestCase]:
    """
    Load Corr2Cause for B08 (Causal Inference Error) evaluation.

    Strategy: Invalid causal inferences from correlations should be flagged.
    Valid inferences are negative controls.
    """
    from datasets import load_dataset

    logger.info("Loading Corr2Cause dataset...")
    ds = load_dataset("causal-nlp/corr2cause", trust_remote_code=False)

    split = ds.get("test", ds.get("validation", ds.get("train")))
    items = list(split)

    rng = random.Random(seed)
    rng.shuffle(items)

    cases = []
    for i, row in enumerate(items[:sample_size * 2]):
        premise = row.get("input", row.get("premise", row.get("text", "")))
        label = row.get("label", "")

        if not premise or len(str(premise)) < 20:
            continue

        # Wrap as synthesis-like text
        synth_text = (
            f"Evidence synthesis: {premise} "
            f"Based on this evidence, the causal relationship appears well-established."
        )

        # label=0 typically means "invalid causal inference" (bias present)
        is_invalid = str(label) in ("0", "invalid", "false")

        cases.append(TestCase(
            id=f"corr2cause-{_hash_text(str(premise))}-{i}",
            input_text=synth_text,
            expected_bias_types=["B08"] if is_invalid else [],
            expected_label=str(label),
            bias_present=is_invalid,
            metadata={"premise": str(premise), "label": str(label)},
            dataset_name="Corr2Cause",
        ))

        if len(cases) >= sample_size:
            break

    logger.info(f"Loaded {len(cases)} Corr2Cause test cases")
    return cases


# ── SciCite Loader ──────────────────────────────────────────────────────────

def load_scicite(sample_size: int = 50, seed: int = 42) -> list[TestCase]:
    """
    Load SciCite for LENS (citation intent) evaluation.

    Strategy: Citation contexts with known intent labels.
    Tests whether LENS correctly classifies citation function.
    """
    from datasets import load_dataset

    logger.info("Loading SciCite dataset...")
    try:
        ds = load_dataset("allenai/scicite")
    except Exception:
        logger.warning("SciCite not loadable — using fallback citation contexts")
        return _scicite_fallback(sample_size, seed)

    split = ds.get("test", ds.get("validation", ds.get("train")))
    items = list(split)

    rng = random.Random(seed)
    rng.shuffle(items)

    cases = []
    for i, row in enumerate(items[:sample_size * 2]):
        text = row.get("string", row.get("text", ""))
        label = row.get("label", "")
        section = row.get("sectionName", "")

        if not text or len(text) < 20:
            continue

        # Map numeric labels to intent names
        label_map = {0: "background", 1: "method", 2: "result"}
        label_name = label_map.get(label, str(label)) if isinstance(label, int) else str(label)

        cases.append(TestCase(
            id=f"scicite-{_hash_text(text)}-{i}",
            input_text=text,
            expected_bias_types=[],  # SciCite tests classification, not bias detection
            expected_label=label_name,
            bias_present=False,  # This is for citation intent classification
            metadata={"section": section, "intent": label_name},
            dataset_name="SciCite",
        ))

        if len(cases) >= sample_size:
            break

    logger.info(f"Loaded {len(cases)} SciCite test cases")
    return cases


# ── Unified Loader ──────────────────────────────────────────────────────────

def _pubhealth_fallback(sample_size: int, seed: int) -> list[TestCase]:
    """Synthetic PubHealth-like test cases when the real dataset can't load."""
    rng = random.Random(seed)
    claims = [
        ("Drinking bleach cures COVID-19", True),
        ("Vaccines cause autism in all children", True),
        ("Regular exercise reduces cardiovascular disease risk", False),
        ("Eating raw garlic eliminates all infections", True),
        ("Adequate sleep improves immune function", False),
        ("5G towers cause cancer", True),
        ("Fluoride in water is safe at recommended levels", False),
        ("Essential oils cure diabetes", True),
        ("Handwashing reduces transmission of respiratory infections", False),
        ("Homeopathy is more effective than antibiotics", True),
    ]
    rng.shuffle(claims)
    cases = []
    for i, (claim, biased) in enumerate(claims[:sample_size]):
        cases.append(TestCase(
            id=f"pubhealth-fallback-{i}",
            input_text=f"Health evidence synthesis: {claim}. This has been confirmed by multiple sources.",
            expected_bias_types=["B04"] if biased else [],
            expected_label="false" if biased else "true",
            bias_present=biased,
            metadata={"claim": claim, "synthetic": True},
            dataset_name="PubHealth",
        ))
    return cases


def _scicite_fallback(sample_size: int, seed: int) -> list[TestCase]:
    """Synthetic SciCite-like test cases when the real dataset can't load."""
    rng = random.Random(seed)
    citations = [
        ("Previous work has established that neural networks can approximate any function [1].", "background"),
        ("We follow the methodology of Smith et al. [2] for data preprocessing.", "method"),
        ("Our results confirm the findings of Johnson [3] showing 15% improvement.", "result"),
        ("The theoretical framework was first proposed by Lee [4].", "background"),
        ("We adapted the algorithm from Chen et al. [5] with modifications.", "method"),
        ("Similar accuracy gains were reported by Wang [6] in a different domain.", "result"),
        ("Deep learning has revolutionized natural language processing [7].", "background"),
        ("Data augmentation was performed using the technique of Brown [8].", "method"),
        ("Our error rate of 2.3% is lower than the 3.1% reported by Davis [9].", "result"),
        ("Attention mechanisms were introduced by Vaswani et al. [10].", "background"),
    ]
    rng.shuffle(citations)
    cases = []
    for i, (text, intent) in enumerate(citations[:sample_size]):
        cases.append(TestCase(
            id=f"scicite-fallback-{i}",
            input_text=text,
            expected_bias_types=[],
            expected_label=intent,
            bias_present=False,
            metadata={"intent": intent, "synthetic": True},
            dataset_name="SciCite",
        ))
    return cases


def _scifact_fallback(sample_size: int, seed: int) -> list[TestCase]:
    """Synthetic SciFact-like test cases when the real dataset can't load."""
    rng = random.Random(seed)
    claims = [
        ("Vitamin D supplementation reduces MS relapse rates", True),
        ("Exercise has no effect on neurological disease progression", True),
        ("High fiber diet protects against autoimmune conditions", False),
        ("Gut microbiome composition is unrelated to immune function", True),
        ("Probiotics consistently improve MS outcomes in all patients", True),
        ("Smoking increases risk of multiple sclerosis", False),
        ("There is no association between diet and MS severity", True),
        ("Interferon therapy eliminates all disease activity", True),
        ("Moderate exercise benefits quality of life in MS patients", False),
        ("Fecal transplant cures autoimmune diseases", True),
    ]
    rng.shuffle(claims)
    cases = []
    for i, (claim, biased) in enumerate(claims[:sample_size]):
        cases.append(TestCase(
            id=f"scifact-fallback-{i}",
            input_text=f"Evidence synthesis: {claim}. Multiple studies support this conclusion.",
            expected_bias_types=["B01"] if biased else [],
            expected_label="REFUTED" if biased else "SUPPORTS",
            bias_present=biased,
            metadata={"claim": claim, "synthetic": True},
            dataset_name="SciFact",
        ))
    return cases


LOADER_REGISTRY = {
    "SciFact": load_scifact,
    "Scientific Exaggeration": load_scientific_exaggeration,
    "PubHealth": load_pubhealth,
    "Corr2Cause": load_corr2cause,
    "SciCite": load_scicite,
}


def load_dataset_by_name(name: str, sample_size: int = 50, seed: int = 42) -> list[TestCase]:
    """Load a dataset by name from the registry."""
    if name not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADER_REGISTRY.keys())}")
    return LOADER_REGISTRY[name](sample_size=sample_size, seed=seed)


def load_all_datasets(sample_size: int = 50, seed: int = 42) -> dict[str, list[TestCase]]:
    """Load all registered datasets."""
    results = {}
    for name in LOADER_REGISTRY:
        try:
            results[name] = load_dataset_by_name(name, sample_size, seed)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            results[name] = []
    return results
