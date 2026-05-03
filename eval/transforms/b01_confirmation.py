"""
B01 — Confirmation Bias transforms.

Datasets: SciFact, IBM Claim Stance
Transform logic from EVAL_DATASETS:
  - SciFact: Claims labeled SUPPORTS/REFUTES/NOINFO with rationale sentences.
    Construct biased inputs by selecting SUPPORTS-only evidence and removing REFUTES.
    Test whether model flags one-sided reasoning.
  - IBM Claim Stance: Claims labeled Pro/Con for each topic.
    Construct one-sided passages using only Pro claims — tests if ARGUS catches
    missing counter-evidence.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)


# ── SciFact ─────────────────────────────────────────────────────────────────
# SciFact HF uses legacy loading scripts. Download raw JSON from GitHub.

def transform_scifact(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    SciFact B01 transform.

    HOW TO USE: Claims labeled SUPPORTS/REFUTES/NOINFO with rationale sentences.
    Construct biased inputs by selecting SUPPORTS-only evidence and removing REFUTES.
    Test whether model flags one-sided reasoning.

    BIASED samples: Take a REFUTED claim, present it with only supporting language
    as if it were true — simulates cherry-picking / confirmation bias.

    CONTROL samples: Present claims with balanced evidence (both supporting and
    refuting context).
    """
    import json
    import urllib.request
    import tempfile
    import os

    logger.info("Loading SciFact via GitHub raw data...")

    # Download claims and corpus from SciFact GitHub
    claims_url = "https://raw.githubusercontent.com/allenai/scifact/master/data/claims_dev.jsonl"
    corpus_url = "https://raw.githubusercontent.com/allenai/scifact/master/data/corpus.jsonl"

    tmpdir = tempfile.mkdtemp()
    claims_path = os.path.join(tmpdir, "claims.jsonl")
    corpus_path = os.path.join(tmpdir, "corpus.jsonl")

    try:
        urllib.request.urlretrieve(claims_url, claims_path)
        urllib.request.urlretrieve(corpus_url, corpus_path)
    except Exception as e:
        logger.warning(f"Could not download SciFact from GitHub: {e}. Using synthetic fallback.")
        return _scifact_synthetic_fallback(max_samples, seed)

    # Parse corpus: doc_id -> {title, abstract_sentences}
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            corpus[str(doc["doc_id"])] = {
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", []),
            }

    # Parse claims
    claims = []
    with open(claims_path) as f:
        for line in f:
            claims.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(claims)

    samples = []
    biased_count = 0
    control_count = 0

    for claim_obj in claims:
        if len(samples) >= max_samples:
            break

        claim_text = claim_obj.get("claim", "")
        evidence = claim_obj.get("evidence", {})

        if not claim_text or not evidence:
            continue

        # Determine claim verdict from evidence
        has_support = False
        has_refute = False
        support_sentences = []
        refute_sentences = []

        for doc_id, sent_annotations in evidence.items():
            doc = corpus.get(str(doc_id), {})
            abstract = doc.get("abstract", [])

            for annotation in sent_annotations:
                label = annotation.get("label", "")
                sent_indices = annotation.get("sentences", [])
                sents = [abstract[i] for i in sent_indices if i < len(abstract)]

                if label == "SUPPORT":
                    has_support = True
                    support_sentences.extend(sents)
                elif label == "CONTRADICT":
                    has_refute = True
                    refute_sentences.extend(sents)

        if has_refute and biased_count < max_samples // 2:
            # BIASED: Present refuted claim with ONLY supporting language
            # This simulates cherry-picking (confirmation bias)
            biased_text = (
                f"Evidence synthesis: {claim_text} "
                f"This conclusion is well-supported by the literature. "
            )
            if support_sentences:
                biased_text += "Supporting evidence: " + " ".join(support_sentences[:3]) + " "
            biased_text += (
                "The consistency of these findings across studies strengthens "
                "confidence in this conclusion."
            )
            # Note: we deliberately omit the refuting evidence

            samples.append(EvalSample(
                id=f"B01_scifact_{len(samples)}",
                dataset_id="B01_scifact",
                bias_type="B01",
                agent="argus",
                input_text=biased_text,
                label="biased",
                bias_present=True,
                original_fields={
                    "claim": claim_text,
                    "has_support": has_support,
                    "has_refute": has_refute,
                    "omitted_refuting": refute_sentences[:2],
                    "included_supporting": support_sentences[:3],
                },
                transform_description=(
                    "Claim is REFUTED but presented with supporting evidence only. "
                    "Refuting evidence deliberately omitted to simulate cherry-picking."
                ),
            ))
            biased_count += 1

        elif has_support and not has_refute and control_count < max_samples // 2:
            # CONTROL: Claim genuinely supported — present balanced
            control_text = (
                f"Evidence synthesis: {claim_text} "
                f"Available evidence from the literature: "
            )
            if support_sentences:
                control_text += " ".join(support_sentences[:3])

            samples.append(EvalSample(
                id=f"B01_scifact_{len(samples)}",
                dataset_id="B01_scifact",
                bias_type="B01",
                agent="argus",
                input_text=control_text,
                label="control",
                bias_present=False,
                original_fields={
                    "claim": claim_text,
                    "has_support": has_support,
                    "has_refute": has_refute,
                    "supporting": support_sentences[:3],
                },
                transform_description=(
                    "Claim genuinely supported by evidence. No evidence omitted."
                ),
            ))
            control_count += 1

    logger.info(f"SciFact: {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


def _scifact_synthetic_fallback(max_samples: int, seed: int) -> list[EvalSample]:
    """Fallback if GitHub download fails."""
    rng = random.Random(seed)
    biased_claims = [
        ("Vitamin D supplementation reduces MS relapse rates",
         "One trial reported reduced relapses with vitamin D.",
         "However, three larger RCTs found no significant benefit."),
        ("High-dose statins prevent Alzheimer's disease",
         "An observational study linked statin use to lower dementia risk.",
         "Randomized trials showed no cognitive benefit from statins."),
        ("Probiotics cure irritable bowel syndrome",
         "Several small studies report symptom improvement with probiotics.",
         "Meta-analyses show inconsistent effects with high heterogeneity."),
    ]
    control_claims = [
        ("Regular exercise improves cardiovascular health",
         "Multiple RCTs demonstrate reduced cardiovascular events with exercise.",
         ""),
        ("Smoking increases lung cancer risk",
         "Decades of epidemiological evidence consistently show elevated risk.",
         ""),
        ("Hand hygiene reduces hospital-acquired infections",
         "Intervention studies show 20-40% reduction in HAI rates.",
         ""),
    ]
    samples = []
    for i, (claim, support, refute) in enumerate(biased_claims[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B01_scifact_synth_{i}",
            dataset_id="B01_scifact",
            bias_type="B01", agent="argus",
            input_text=f"Evidence synthesis: {claim}. {support} The evidence strongly supports this conclusion.",
            label="biased", bias_present=True,
            original_fields={"claim": claim, "omitted_refuting": refute, "synthetic": True},
            transform_description="Synthetic: refuting evidence omitted to simulate cherry-picking.",
        ))
    for i, (claim, support, _) in enumerate(control_claims[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B01_scifact_synth_ctrl_{i}",
            dataset_id="B01_scifact",
            bias_type="B01", agent="argus",
            input_text=f"Evidence synthesis: {claim}. {support}",
            label="control", bias_present=False,
            original_fields={"claim": claim, "synthetic": True},
            transform_description="Synthetic: claim genuinely supported, no evidence omitted.",
        ))
    return samples


# ── IBM Claim Stance ────────────────────────────────────────────────────────

def transform_ibm_claim_stance(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    IBM Claim Stance B01 transform.

    HOW TO USE: Claims labeled Pro/Con for each topic. Construct one-sided
    passages using only Pro claims — tests if ARGUS catches missing counter-evidence.

    BIASED: For a topic, take only PRO claims and present as balanced synthesis.
    CONTROL: For a topic, present both PRO and CON claims.
    """
    from datasets import load_dataset

    logger.info("Loading IBM Claim Stance...")
    ds = load_dataset("ibm-research/claim_stance", "claim_stance", split="train")

    # Group claims by topic
    topics: dict[int, dict] = {}
    for row in ds:
        tid = row["topicId"]
        if tid not in topics:
            topics[tid] = {"text": row["topicText"], "pro": [], "con": []}
        stance = row["claims.stance"]
        claim = row["claims.claimCorrectedText"]
        if stance == "PRO":
            topics[tid]["pro"].append(claim)
        elif stance == "CON":
            topics[tid]["con"].append(claim)

    # Filter topics that have both pro and con claims
    balanced_topics = {k: v for k, v in topics.items() if v["pro"] and v["con"]}

    rng = random.Random(seed)
    topic_ids = list(balanced_topics.keys())
    rng.shuffle(topic_ids)

    samples = []
    for tid in topic_ids:
        if len(samples) >= max_samples:
            break
        t = balanced_topics[tid]
        topic_text = t["text"]
        pro_claims = t["pro"][:4]
        con_claims = t["con"][:4]

        # BIASED: Only pro claims, presented as comprehensive review
        biased_text = (
            f"Comprehensive evidence review on: {topic_text}\n\n"
            f"The evidence consistently supports this position. "
        )
        for c in pro_claims:
            biased_text += f"{c} "
        biased_text += (
            "\nOverall, the weight of evidence strongly favors this conclusion."
        )

        samples.append(EvalSample(
            id=f"B01_ibm_{len(samples)}",
            dataset_id="B01_ibm_claim",
            bias_type="B01",
            agent="argus",
            input_text=biased_text,
            label="biased",
            bias_present=True,
            original_fields={
                "topic": topic_text,
                "pro_count": len(t["pro"]),
                "con_count": len(t["con"]),
                "omitted_con_claims": con_claims[:2],
            },
            transform_description=(
                f"One-sided: {len(pro_claims)} PRO claims included, "
                f"{len(con_claims)} CON claims deliberately omitted."
            ),
        ))

        # CONTROL: Both pro and con claims
        control_text = (
            f"Evidence review on: {topic_text}\n\n"
            f"Supporting arguments: "
        )
        for c in pro_claims[:2]:
            control_text += f"{c} "
        control_text += "\nOpposing arguments: "
        for c in con_claims[:2]:
            control_text += f"{c} "
        control_text += "\nThe evidence presents a mixed picture."

        samples.append(EvalSample(
            id=f"B01_ibm_{len(samples)}",
            dataset_id="B01_ibm_claim",
            bias_type="B01",
            agent="argus",
            input_text=control_text,
            label="control",
            bias_present=False,
            original_fields={
                "topic": topic_text,
                "pro_included": pro_claims[:2],
                "con_included": con_claims[:2],
            },
            transform_description="Balanced: both PRO and CON claims included.",
        ))

    logger.info(f"IBM Claim Stance: {len(samples)} samples")
    return samples


# ── Register ────────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="B01_scifact",
    bias_type="B01",
    dataset_name="SciFact",
    agent="argus",
    how_to_use="Construct biased inputs by selecting SUPPORTS-only evidence and removing REFUTES.",
    transform_fn=transform_scifact,
))

register_transform(DatasetSpec(
    dataset_id="B01_ibm_claim",
    bias_type="B01",
    dataset_name="IBM Claim Stance",
    agent="argus",
    how_to_use="Construct one-sided passages using only Pro claims — tests if ARGUS catches missing counter-evidence.",
    transform_fn=transform_ibm_claim_stance,
))
