"""
B03 — Certainty Inflation transforms.

Datasets: Scientific Exaggeration Detection, BioScope, CoNLL-2010
Transform logic from EVAL_DATASETS:
  - Sci Exaggeration: Pairs labeled SAME/EXAGGERATES/DOWNPLAYS.
    "Exaggerates" class directly tests certainty inflation — press releases
    that overclaim findings.
  - BioScope: Medical texts annotated for speculation/negation cues.
    Strip hedge cues from uncertain sentences → creates certainty-inflated text.
  - CoNLL-2010: Sentence-level hedge classification.
    Strip hedge cues from uncertain sentences → certainty-inflated text.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)


# ── Scientific Exaggeration Detection ─────────────────────────────────────

def transform_sci_exaggeration(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    Scientific Exaggeration B03 transform.

    HOW TO USE: Pairs labeled SAME/EXAGGERATES/DOWNPLAYS. The "exaggerates"
    class directly tests certainty inflation — press releases that overclaim.

    BIASED: Press release text that EXAGGERATES the abstract finding.
    CONTROL: Press release text that faithfully represents the finding (SAME).
    """
    from datasets import load_dataset

    logger.info("Loading Scientific Exaggeration dataset...")
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
        abstract_strength = row.get("abstract_strength", "")
        press_strength = row.get("press_release_strength", "")

        if not press_conclusion or not abstract_conclusion:
            continue

        if label == "exaggerates" and biased_count < max_samples // 2:
            # BIASED: Press release overclaims the abstract finding
            biased_text = (
                f"Research summary: {press_conclusion} "
                f"This conclusion is firmly established by the study findings. "
                f"The evidence definitively demonstrates this effect."
            )

            samples.append(EvalSample(
                id=f"B03_sciexag_{len(samples)}",
                dataset_id="B03_sci_exaggeration",
                bias_type="B03",
                agent="argus",
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
                    "Press release EXAGGERATES abstract finding. "
                    f"Abstract strength: {abstract_strength}, Press strength: {press_strength}."
                ),
            ))
            biased_count += 1

        elif label == "same" and control_count < max_samples // 2:
            # CONTROL: Press release faithfully represents the finding
            control_text = (
                f"Research summary: {press_conclusion} "
                f"Original study conclusion: {abstract_conclusion}"
            )

            samples.append(EvalSample(
                id=f"B03_sciexag_{len(samples)}",
                dataset_id="B03_sci_exaggeration",
                bias_type="B03",
                agent="argus",
                input_text=control_text,
                label="control",
                bias_present=False,
                original_fields={
                    "abstract_conclusion": abstract_conclusion,
                    "press_conclusion": press_conclusion,
                    "exaggeration_label": label,
                },
                transform_description="Press release faithfully represents abstract (SAME label).",
            ))
            control_count += 1

    logger.info(f"Sci Exaggeration: {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


# ── BioScope ──────────────────────────────────────────────────────────────
# BioScope is not on HuggingFace — requires manual download from
# https://rgai.inf.u-szeged.hu/node/105. Use synthetic fallback.

# Common hedge cues used in biomedical text (Hyland markers)
HEDGE_CUES = [
    "may", "might", "could", "possibly", "perhaps", "suggests",
    "appears to", "seems to", "likely", "probable", "potential",
    "it is possible that", "it is likely that", "we hypothesize",
    "our results suggest", "the data indicate", "presumably",
    "apparently", "to our knowledge", "it remains unclear",
    "further research is needed", "tentatively",
]


def _strip_hedge_cues(text: str) -> str:
    """Strip hedge cues from text to simulate certainty inflation."""
    result = text
    replacements = {
        "may": "will",
        "might": "will",
        "could": "does",
        "possibly": "certainly",
        "perhaps": "clearly",
        "suggests": "demonstrates",
        "appears to": "is shown to",
        "seems to": "is confirmed to",
        "likely": "definitively",
        "probable": "certain",
        "potential": "confirmed",
        "it is possible that": "it is established that",
        "it is likely that": "it is confirmed that",
        "our results suggest": "our results demonstrate",
        "the data indicate": "the data prove",
        "presumably": "certainly",
        "tentatively": "conclusively",
    }
    for hedge, certain in replacements.items():
        result = re.sub(rf'\b{re.escape(hedge)}\b', certain, result, flags=re.IGNORECASE)
    return result


def transform_bioscope(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    BioScope B03 transform.

    HOW TO USE: Medical texts annotated for speculation/negation cues and their scope.
    Sentences marked "certain" vs "speculative" — test if ARGUS catches hedging gaps
    where certainty language is used on speculative claims.

    BIASED: Take speculative sentences, strip hedge cues → simulates certainty inflation.
    CONTROL: Present speculative sentences with hedging intact.

    NOTE: BioScope requires manual download. Uses synthetic biomedical examples.
    """
    rng = random.Random(seed)

    # Synthetic speculative biomedical sentences (modeled on BioScope patterns)
    speculative_sentences = [
        "The treatment may reduce inflammation in patients with rheumatoid arthritis.",
        "Our findings suggest that the novel compound could inhibit tumor growth.",
        "It is possible that dietary intervention might improve metabolic outcomes.",
        "The association between gene X and disease Y appears to be significant.",
        "Preliminary data indicate that the vaccine might provide long-term immunity.",
        "The biomarker could potentially serve as an early diagnostic indicator.",
        "These results suggest a possible link between gut microbiota and depression.",
        "It seems likely that epigenetic modifications play a role in disease progression.",
        "The drug may have neuroprotective effects in Alzheimer's disease models.",
        "Our analysis tentatively supports the hypothesis that exercise reduces relapse rates.",
        "This pathway might be involved in the regulation of immune response.",
        "Preliminary evidence suggests that combination therapy could be more effective.",
        "The correlation appears to indicate a possible causal relationship.",
        "It is probable that genetic variants contribute to treatment response variability.",
        "The observed effects may be attributable to changes in protein expression levels.",
        "Further analysis suggests that the metabolite could serve as a therapeutic target.",
        "The intervention might reduce cardiovascular risk in high-risk populations.",
        "Our data indicate that the compound may exhibit anti-inflammatory properties.",
        "It is likely that environmental factors modulate the genetic predisposition.",
        "The preliminary findings suggest a potential role for the receptor in pathogenesis.",
    ]

    certain_sentences = [
        "Regular physical exercise reduces cardiovascular mortality in adults over 50.",
        "Aspirin inhibits cyclooxygenase and reduces platelet aggregation.",
        "Insulin regulates blood glucose levels through receptor-mediated signaling.",
        "Smoking is a well-established risk factor for lung cancer.",
        "Vaccination against measles provides long-lasting immunity in over 95% of recipients.",
        "The BRCA1 mutation increases breast cancer risk significantly.",
        "Hypertension is a leading cause of stroke and cardiovascular disease.",
        "Antibiotics targeting the cell wall are effective against gram-positive bacteria.",
        "Chronic alcohol consumption leads to liver cirrhosis over time.",
        "Folate supplementation during pregnancy reduces neural tube defect risk.",
    ]

    rng.shuffle(speculative_sentences)
    rng.shuffle(certain_sentences)

    samples = []

    # BIASED: Strip hedge cues from speculative sentences
    for i, sent in enumerate(speculative_sentences[:max_samples // 2]):
        inflated = _strip_hedge_cues(sent)
        samples.append(EvalSample(
            id=f"B03_bioscope_{len(samples)}",
            dataset_id="B03_bioscope",
            bias_type="B03",
            agent="argus",
            input_text=f"Evidence synthesis: {inflated}",
            label="biased",
            bias_present=True,
            original_fields={
                "original_speculative": sent,
                "certainty_inflated": inflated,
                "synthetic": True,
            },
            transform_description=(
                "Speculative sentence with hedge cues stripped to simulate certainty inflation."
            ),
        ))

    # CONTROL: Genuinely certain sentences
    for i, sent in enumerate(certain_sentences[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B03_bioscope_{len(samples)}",
            dataset_id="B03_bioscope",
            bias_type="B03",
            agent="argus",
            input_text=f"Evidence synthesis: {sent}",
            label="control",
            bias_present=False,
            original_fields={
                "certain_sentence": sent,
                "synthetic": True,
            },
            transform_description="Genuinely certain biomedical statement (no hedge stripping needed).",
        ))

    rng.shuffle(samples)
    logger.info(f"BioScope (synthetic): {len(samples)} samples")
    return samples


# ── CoNLL-2010 ────────────────────────────────────────────────────────────

def transform_conll2010(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    CoNLL-2010 B03 transform.

    HOW TO USE: Sentence-level hedge classification (certain/uncertain).
    Strip hedge cues from uncertain sentences → creates "certainty inflated"
    text → test if ARGUS detects it.

    NOTE: CoNLL-2010 requires manual download. Uses synthetic examples modeled
    on the task's biological and Wikipedia domains.
    """
    rng = random.Random(seed)

    # Synthetic uncertain sentences (modeled on CoNLL-2010 bio domain)
    uncertain_sentences = [
        "These observations suggest that the protein may function as a transcription factor.",
        "It appears that the deletion of this gene could affect cell proliferation.",
        "The mechanism by which the drug acts is not entirely understood but might involve receptor binding.",
        "Our results indicate that the pathway is possibly regulated by phosphorylation.",
        "It seems that exposure to the compound may induce apoptosis in certain cell lines.",
        "The evidence tentatively supports a role for this cytokine in inflammatory response.",
        "This interaction could potentially explain the observed phenotype in knockout models.",
        "Preliminary screening suggests that the inhibitor might be effective at low concentrations.",
        "The authors hypothesize that oxidative stress may contribute to neurodegeneration.",
        "Based on limited data, the mutation appears to confer resistance to the drug.",
        "It is conceivable that alternative splicing plays a role in tissue-specific expression.",
        "The findings raise the possibility that the virus could exploit this pathway for entry.",
        "While not conclusive, the data suggest a correlation between expression and survival.",
        "The therapeutic potential of this approach remains to be established.",
        "It is thought that epigenetic modifications might explain the variable penetrance.",
    ]

    certain_control = [
        "DNA replication is semiconservative and occurs during the S phase of the cell cycle.",
        "The enzyme catalyzes the hydrolysis of ATP to ADP and inorganic phosphate.",
        "Mitochondria are the primary site of oxidative phosphorylation in eukaryotic cells.",
        "The sodium-potassium pump actively transports ions against their concentration gradients.",
        "Hemoglobin binds oxygen cooperatively, displaying a sigmoidal saturation curve.",
        "Transfer RNA molecules deliver amino acids to the ribosome during translation.",
        "The lac operon is induced in the presence of lactose and absence of glucose.",
        "Photosystem II splits water molecules and releases oxygen as a byproduct.",
        "The krebs cycle generates electron carriers NADH and FADH2 for oxidative phosphorylation.",
        "Restriction enzymes cleave DNA at specific palindromic recognition sequences.",
    ]

    rng.shuffle(uncertain_sentences)
    rng.shuffle(certain_control)

    samples = []

    for i, sent in enumerate(uncertain_sentences[:max_samples // 2]):
        inflated = _strip_hedge_cues(sent)
        samples.append(EvalSample(
            id=f"B03_conll_{len(samples)}",
            dataset_id="B03_conll2010",
            bias_type="B03",
            agent="argus",
            input_text=f"Evidence synthesis: {inflated}",
            label="biased",
            bias_present=True,
            original_fields={
                "original_uncertain": sent,
                "certainty_inflated": inflated,
                "synthetic": True,
            },
            transform_description="Uncertain sentence with hedge cues stripped (certainty inflation).",
        ))

    for i, sent in enumerate(certain_control[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B03_conll_{len(samples)}",
            dataset_id="B03_conll2010",
            bias_type="B03",
            agent="argus",
            input_text=f"Evidence synthesis: {sent}",
            label="control",
            bias_present=False,
            original_fields={"certain_sentence": sent, "synthetic": True},
            transform_description="Genuinely certain biological statement.",
        ))

    rng.shuffle(samples)
    logger.info(f"CoNLL-2010 (synthetic): {len(samples)} samples")
    return samples


# ── Register ──────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="B03_sci_exaggeration",
    bias_type="B03",
    dataset_name="Scientific Exaggeration Detection",
    agent="argus",
    how_to_use="Pairs labeled SAME/EXAGGERATES/DOWNPLAYS. 'Exaggerates' class tests certainty inflation.",
    transform_fn=transform_sci_exaggeration,
))

register_transform(DatasetSpec(
    dataset_id="B03_bioscope",
    bias_type="B03",
    dataset_name="BioScope Corpus",
    agent="argus",
    how_to_use="Strip hedge cues from speculative sentences to create certainty-inflated text.",
    transform_fn=transform_bioscope,
))

register_transform(DatasetSpec(
    dataset_id="B03_conll2010",
    bias_type="B03",
    dataset_name="CoNLL-2010 Shared Task",
    agent="argus",
    how_to_use="Strip hedge cues from uncertain sentences → certainty-inflated text.",
    transform_fn=transform_conll2010,
))
