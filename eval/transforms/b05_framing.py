"""
B05 — Framing Effect transforms.

Datasets: Media Frames Corpus, MBIC, SemEval-2023 Task 3
Transform logic from EVAL_DATASETS:
  - Media Frames Corpus: News articles annotated with 15 framing dimensions.
    Test whether ARGUS detects one-sided framing when text only uses one frame.
  - MBIC: Word-level and sentence-level bias annotations.
    Directly tests framing bias detection in language.
  - SemEval-2023 Task 3: Span-level persuasion technique labels including
    "loaded language", "appeal to authority" — maps to framing bias subtypes.

NOTE: All three datasets require manual download (GitHub/Kaggle/competition site).
Using synthetic examples modeled on their annotation schemas.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)

# MFC's 15 framing dimensions
MFC_FRAMES = [
    "economic", "capacity_and_resources", "morality", "fairness_and_equality",
    "legality_constitutionality", "policy_prescription", "crime_and_punishment",
    "security_and_defense", "health_and_safety", "quality_of_life",
    "cultural_identity", "public_opinion", "political", "external_regulation",
    "other",
]


# ── Media Frames Corpus ──────────────────────────────────────────────────

def transform_media_frames(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    Media Frames Corpus B05 transform.

    HOW TO USE: News articles annotated with 15 framing dimensions.
    Test whether ARGUS detects one-sided framing when text only uses one frame.

    BIASED: Text that presents an issue using ONLY ONE framing dimension,
    ignoring all other relevant perspectives.
    CONTROL: Text that presents multiple framing dimensions for the same issue.

    NOTE: MFC requires cloning from GitHub. Uses synthetic examples.
    """
    rng = random.Random(seed)

    # Topics with natural multi-frame coverage
    topics_with_frames = [
        {
            "topic": "immigration policy",
            "one_sided": {
                "frame": "economic",
                "text": (
                    "Immigration policy analysis: Immigrants contribute $2 trillion annually "
                    "to GDP. Studies show net positive fiscal impact. Labor market data indicates "
                    "immigrants fill critical skill gaps. Economic modeling projects continued "
                    "growth from immigration. Tax revenue data shows immigrants contribute more "
                    "than they consume in services."
                ),
            },
            "balanced": {
                "frames": ["economic", "security_and_defense", "cultural_identity", "fairness_and_equality"],
                "text": (
                    "Immigration policy analysis: Economically, immigrants contribute significantly "
                    "to GDP, though impacts vary by sector. Security considerations include border "
                    "enforcement costs and public safety data showing mixed outcomes. Cultural "
                    "integration presents both enrichment opportunities and social cohesion challenges. "
                    "Fairness questions arise regarding both humanitarian obligations and equitable "
                    "resource distribution."
                ),
            },
        },
        {
            "topic": "healthcare reform",
            "one_sided": {
                "frame": "morality",
                "text": (
                    "Healthcare reform: Every person deserves access to medical care as a basic "
                    "human right. It is morally unacceptable that millions lack coverage. The "
                    "ethical imperative demands universal access. Denying care based on ability "
                    "to pay violates fundamental principles of human dignity and compassion."
                ),
            },
            "balanced": {
                "frames": ["morality", "economic", "policy_prescription", "quality_of_life"],
                "text": (
                    "Healthcare reform: While moral arguments support universal access, economic "
                    "constraints shape feasible policy options. Quality-of-life data shows coverage "
                    "gaps affect health outcomes. Various policy approaches — single-payer, "
                    "public option, market-based — each present trade-offs between cost, access, "
                    "and quality. Implementation complexity varies across proposals."
                ),
            },
        },
        {
            "topic": "climate change policy",
            "one_sided": {
                "frame": "security_and_defense",
                "text": (
                    "Climate change represents the greatest security threat of our generation. "
                    "Pentagon analyses identify climate-driven instability as a major risk. "
                    "Resource conflicts will intensify. National security requires immediate "
                    "climate action. Military installations face direct climate threats."
                ),
            },
            "balanced": {
                "frames": ["security_and_defense", "economic", "health_and_safety", "policy_prescription"],
                "text": (
                    "Climate policy intersects multiple dimensions: security analyses identify "
                    "geopolitical risks from climate disruption. Economic assessments weigh "
                    "transition costs against projected damages. Public health data shows "
                    "differential impacts across populations. Policy options range from carbon "
                    "pricing to regulation, each with distinct implementation trade-offs."
                ),
            },
        },
        {
            "topic": "gun control",
            "one_sided": {
                "frame": "crime_and_punishment",
                "text": (
                    "Gun violence is primarily a law enforcement issue. Crime statistics show "
                    "that areas with stricter gun laws do not necessarily have lower crime rates. "
                    "Criminals obtain weapons regardless of legislation. The solution lies in "
                    "better enforcement of existing laws and harsher penalties for gun crimes."
                ),
            },
            "balanced": {
                "frames": ["crime_and_punishment", "health_and_safety", "legality_constitutionality", "public_opinion"],
                "text": (
                    "Gun policy debates span multiple domains: crime data shows complex "
                    "relationships between regulation and violence rates. Public health research "
                    "treats gun deaths as an epidemic. Constitutional scholars debate Second "
                    "Amendment scope. Public opinion polls show majority support for some "
                    "measures while opposing others."
                ),
            },
        },
        {
            "topic": "drug policy",
            "one_sided": {
                "frame": "health_and_safety",
                "text": (
                    "Drug policy should be viewed through a public health lens. Addiction is "
                    "a medical condition requiring treatment, not punishment. Harm reduction "
                    "approaches save lives. Safe injection sites reduce overdose deaths. "
                    "Medical evidence supports treatment over incarceration."
                ),
            },
            "balanced": {
                "frames": ["health_and_safety", "crime_and_punishment", "economic", "morality"],
                "text": (
                    "Drug policy involves multiple considerations: public health evidence "
                    "supports treatment approaches for addiction. Criminal justice data shows "
                    "varied outcomes from enforcement strategies. Economic analyses compare "
                    "costs of incarceration versus treatment. Moral perspectives differ on "
                    "individual responsibility versus societal obligation to provide support."
                ),
            },
        },
        {
            "topic": "education funding",
            "one_sided": {
                "frame": "fairness_and_equality",
                "text": (
                    "Education funding disparities perpetuate systemic inequality. Low-income "
                    "districts receive less funding per student. This inequity determines "
                    "life outcomes. Every child deserves equal access to quality education "
                    "regardless of zip code. The funding gap is the central civil rights issue."
                ),
            },
            "balanced": {
                "frames": ["fairness_and_equality", "economic", "quality_of_life", "policy_prescription"],
                "text": (
                    "Education funding involves equity concerns about per-pupil spending "
                    "disparities. Economic research shows mixed results on whether increased "
                    "spending directly improves outcomes. Quality-of-life data links education "
                    "access to long-term wellbeing. Policy options include formula changes, "
                    "targeted grants, and structural reforms with varying trade-offs."
                ),
            },
        },
    ]

    rng.shuffle(topics_with_frames)

    samples = []
    biased_count = 0
    control_count = 0

    for topic in topics_with_frames:
        if len(samples) >= max_samples:
            break

        # BIASED: One-sided framing
        if biased_count < max_samples // 2:
            samples.append(EvalSample(
                id=f"B05_mfc_{len(samples)}",
                dataset_id="B05_media_frames",
                bias_type="B05",
                agent="argus",
                input_text=topic["one_sided"]["text"],
                label="biased",
                bias_present=True,
                original_fields={
                    "topic": topic["topic"],
                    "dominant_frame": topic["one_sided"]["frame"],
                    "missing_frames": [f for f in topic["balanced"]["frames"]
                                       if f != topic["one_sided"]["frame"]],
                    "synthetic": True,
                },
                transform_description=(
                    f"One-sided framing: only '{topic['one_sided']['frame']}' frame used. "
                    f"Missing: {', '.join(f for f in topic['balanced']['frames'] if f != topic['one_sided']['frame'])}."
                ),
            ))
            biased_count += 1

        # CONTROL: Multi-frame coverage
        if control_count < max_samples // 2:
            samples.append(EvalSample(
                id=f"B05_mfc_{len(samples)}",
                dataset_id="B05_media_frames",
                bias_type="B05",
                agent="argus",
                input_text=topic["balanced"]["text"],
                label="control",
                bias_present=False,
                original_fields={
                    "topic": topic["topic"],
                    "frames_included": topic["balanced"]["frames"],
                    "synthetic": True,
                },
                transform_description=(
                    f"Balanced framing: {len(topic['balanced']['frames'])} frames included."
                ),
            ))
            control_count += 1

    logger.info(f"Media Frames (synthetic): {len(samples)} samples ({biased_count} biased, {control_count} control)")
    return samples


# ── MBIC ──────────────────────────────────────────────────────────────────

def transform_mbic(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    MBIC B05 transform.

    HOW TO USE: Word-level and sentence-level bias annotations.
    Directly tests framing bias detection in language.

    BIASED: Sentences annotated as biased by majority of annotators.
    CONTROL: Sentences annotated as unbiased.

    NOTE: MBIC requires Kaggle download. Uses synthetic examples modeled on
    the annotation schema (biased/non-biased with word-level bias spans).
    """
    rng = random.Random(seed)

    biased_sentences = [
        ("The radical policy will devastate working families.",
         ["radical", "devastate"]),
        ("Critics slammed the disastrous proposal as a reckless gamble.",
         ["slammed", "disastrous", "reckless gamble"]),
        ("The controversial leader pushed through the extreme agenda.",
         ["controversial", "pushed through", "extreme"]),
        ("Opponents of the measure are clearly motivated by greed.",
         ["clearly", "greed"]),
        ("The failed policy has been a catastrophic waste of taxpayer money.",
         ["failed", "catastrophic", "waste"]),
        ("The so-called experts have once again been proven wrong.",
         ["so-called", "once again"]),
        ("This reckless spending spree threatens our children's future.",
         ["reckless", "spending spree", "threatens"]),
        ("The extremist group continues to undermine democratic institutions.",
         ["extremist", "undermine"]),
        ("Despite overwhelming evidence, deniers cling to their discredited theories.",
         ["deniers", "cling", "discredited"]),
        ("The corrupt establishment refuses to address the will of the people.",
         ["corrupt", "establishment", "refuses"]),
        ("The draconian measures imposed by the regime restrict basic freedoms.",
         ["draconian", "imposed", "regime"]),
        ("Supporters blindly follow the charismatic but dangerous leader.",
         ["blindly", "dangerous"]),
    ]

    unbiased_sentences = [
        "The proposed policy would affect approximately 30 million households.",
        "The study found a 15% difference between the treatment and control groups.",
        "Representatives from both parties expressed concerns about the timeline.",
        "The budget allocates $4.2 billion to infrastructure projects over five years.",
        "According to the report, unemployment rates varied across regions.",
        "The committee heard testimony from stakeholders on both sides of the issue.",
        "The amendment was approved with 58 votes in favor and 42 against.",
        "Research participants completed surveys at three-month intervals.",
        "The organization released its annual report documenting program outcomes.",
        "Three independent analyses reached different conclusions about the policy's impact.",
    ]

    rng.shuffle(biased_sentences)
    rng.shuffle(unbiased_sentences)

    samples = []

    for i, (sent, bias_words) in enumerate(biased_sentences[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B05_mbic_{len(samples)}",
            dataset_id="B05_mbic",
            bias_type="B05",
            agent="argus",
            input_text=f"Analysis: {sent}",
            label="biased",
            bias_present=True,
            original_fields={
                "sentence": sent,
                "bias_words": bias_words,
                "synthetic": True,
            },
            transform_description=f"Sentence with biased language: {', '.join(bias_words)}.",
        ))

    for i, sent in enumerate(unbiased_sentences[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B05_mbic_{len(samples)}",
            dataset_id="B05_mbic",
            bias_type="B05",
            agent="argus",
            input_text=f"Analysis: {sent}",
            label="control",
            bias_present=False,
            original_fields={"sentence": sent, "synthetic": True},
            transform_description="Neutral, factual statement without framing bias.",
        ))

    rng.shuffle(samples)
    logger.info(f"MBIC (synthetic): {len(samples)} samples")
    return samples


# ── SemEval-2023 Task 3 ──────────────────────────────────────────────────

PERSUASION_TECHNIQUES = [
    "loaded_language", "name_calling", "appeal_to_authority",
    "black_and_white_fallacy", "appeal_to_fear", "flag_waving",
    "causal_oversimplification", "exaggeration", "repetition",
    "doubt", "straw_man", "whataboutism",
]


def transform_semeval2023(max_samples: int = 50, seed: int = 42) -> list[EvalSample]:
    """
    SemEval-2023 Task 3 B05 transform.

    HOW TO USE: Span-level persuasion technique labels including "loaded language",
    "appeal to authority", "black-and-white fallacy" — maps to framing bias subtypes.

    BIASED: Text containing identifiable persuasion techniques.
    CONTROL: Informative text without persuasion techniques.

    NOTE: Competition data requires registration. Uses synthetic examples
    modeled on the 23 persuasion technique categories.
    """
    rng = random.Random(seed)

    # Texts with clear persuasion techniques
    persuasive_texts = [
        {
            "text": "Every patriotic citizen must support this measure. Those who oppose it are betraying our nation's values and everything our ancestors fought for.",
            "techniques": ["flag_waving", "appeal_to_fear", "black_and_white_fallacy"],
        },
        {
            "text": "As the world's leading expert confirmed, this is the only viable solution. Anyone suggesting alternatives is simply ignorant of the facts.",
            "techniques": ["appeal_to_authority", "black_and_white_fallacy", "name_calling"],
        },
        {
            "text": "This catastrophic policy will destroy families, devastate communities, and plunge the nation into chaos. We must act now or face total ruin.",
            "techniques": ["loaded_language", "exaggeration", "appeal_to_fear"],
        },
        {
            "text": "Critics point to problems elsewhere, but what about the far worse failures of the opposing side? Their track record speaks for itself.",
            "techniques": ["whataboutism", "doubt"],
        },
        {
            "text": "The corrupt elites want to silence ordinary people. This is about freedom versus tyranny — there is no middle ground.",
            "techniques": ["loaded_language", "name_calling", "black_and_white_fallacy"],
        },
        {
            "text": "Nobel laureate Dr. Smith confirmed it, Professor Jones endorsed it, and the National Academy recommended it. The science is absolutely settled.",
            "techniques": ["appeal_to_authority", "repetition", "exaggeration"],
        },
        {
            "text": "Opponents claim this helps the economy, but they really just want to enrich their corporate donors at the expense of hardworking families.",
            "techniques": ["straw_man", "loaded_language", "appeal_to_fear"],
        },
        {
            "text": "The devastating consequences of inaction will be felt for generations. Only a complete reversal of policy can avert this impending disaster.",
            "techniques": ["appeal_to_fear", "exaggeration", "causal_oversimplification"],
        },
    ]

    neutral_texts = [
        {
            "text": "The proposed legislation would modify existing regulations in three areas: emissions standards, fuel efficiency requirements, and reporting timelines.",
            "techniques": [],
        },
        {
            "text": "According to the Bureau of Statistics, the unemployment rate decreased by 0.3 percentage points to 4.2% in the most recent quarter.",
            "techniques": [],
        },
        {
            "text": "The committee report outlined advantages and disadvantages of each approach, noting that implementation costs would vary by region.",
            "techniques": [],
        },
        {
            "text": "Research published in the journal found a correlation between the variables, though the authors noted several limitations in their methodology.",
            "techniques": [],
        },
        {
            "text": "The audit identified 12 areas where procedures could be improved, with estimated savings ranging from $2M to $5M annually.",
            "techniques": [],
        },
        {
            "text": "Representatives from industry, advocacy groups, and academic institutions provided testimony during the three-day hearing.",
            "techniques": [],
        },
    ]

    rng.shuffle(persuasive_texts)
    rng.shuffle(neutral_texts)

    samples = []

    for i, item in enumerate(persuasive_texts[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B05_semeval_{len(samples)}",
            dataset_id="B05_semeval2023",
            bias_type="B05",
            agent="argus",
            input_text=item["text"],
            label="biased",
            bias_present=True,
            original_fields={
                "techniques": item["techniques"],
                "synthetic": True,
            },
            transform_description=f"Contains persuasion techniques: {', '.join(item['techniques'])}.",
        ))

    for i, item in enumerate(neutral_texts[:max_samples // 2]):
        samples.append(EvalSample(
            id=f"B05_semeval_{len(samples)}",
            dataset_id="B05_semeval2023",
            bias_type="B05",
            agent="argus",
            input_text=item["text"],
            label="control",
            bias_present=False,
            original_fields={"techniques": [], "synthetic": True},
            transform_description="Neutral informative text without persuasion techniques.",
        ))

    rng.shuffle(samples)
    logger.info(f"SemEval-2023 (synthetic): {len(samples)} samples")
    return samples


# ── Register ──────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="B05_media_frames",
    bias_type="B05",
    dataset_name="Media Frames Corpus",
    agent="argus",
    how_to_use="Test one-sided framing when text only uses one of 15 framing dimensions.",
    transform_fn=transform_media_frames,
))

register_transform(DatasetSpec(
    dataset_id="B05_mbic",
    bias_type="B05",
    dataset_name="MBIC (Media Bias Annotation)",
    agent="argus",
    how_to_use="Word-level and sentence-level bias annotations test framing bias detection.",
    transform_fn=transform_mbic,
))

register_transform(DatasetSpec(
    dataset_id="B05_semeval2023",
    bias_type="B05",
    dataset_name="SemEval-2023 Task 3",
    agent="argus",
    how_to_use="Span-level persuasion technique labels map to framing bias subtypes.",
    transform_fn=transform_semeval2023,
))
