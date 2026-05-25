"""
Real HF-backed transforms — supplements the synthetic/script-based datasets
that became inaccessible after `datasets>=3.0` removed loading-script support.

Adds three transforms backed by real, scriptless HuggingFace datasets:

  - B01_mbib_cognitive  →  mediabiasgroup/mbib-base/cognitive_bias   (~7K rows)
  - B05_babe            →  mediabiasgroup/BABE                       (~3K rows)
  - B05_mbib_linguistic →  mediabiasgroup/mbib-base/linguistic_bias  (capped at 5K)

Each dataset has binary labels (1=biased, 0=neutral) and short sentences,
which are exactly what the Evidence RAG cross-check needs to give the
BM25 / TF-IDF index real signal on flagged spans.
"""
from __future__ import annotations

import logging
import random

from eval.transforms.registry import EvalSample, DatasetSpec, register_transform

logger = logging.getLogger(__name__)


# Hard upper bound per dataset — the Evidence RAG in-memory BM25 index
# starts to drag past this. The full corpora are accessible if needed.
_LINGUISTIC_CAP = 5000


def _wrap_synthesis(sentence: str) -> str:
    """Embed a bare sentence in a short synthesis-style stub.

    The downstream agents see review-style text in production, so wrapping
    short news/Wikipedia sentences in this preamble keeps the distribution
    aligned with what the agents are trained to flag. The wrapper text is
    identical for biased and control samples so it cannot leak the label.
    """
    return f"Evidence synthesis: {sentence}"


# ── mbib-base/cognitive_bias → B01 ────────────────────────────────────────

def transform_mbib_cognitive(max_samples: int = 99_999, seed: int = 42) -> list[EvalSample]:
    """
    Real cognitive bias detection samples from mbib-base.

    Source: mediabiasgroup/mbib-base — `cognitive_bias` split (7,092 sentences,
    perfectly balanced 50/50). Sentences labelled 1 = exhibit cognitive bias
    (selective framing, confirmation patterns), 0 = neutral baseline.

    Transform: no rewriting — the labels are real human annotations. We just
    wrap each sentence in a synthesis preamble (identical for both classes)
    so the text shape matches what BiasScan agents see in production.
    """
    from datasets import load_dataset

    logger.info("Loading mediabiasgroup/mbib-base · cognitive_bias ...")
    ds = load_dataset("mediabiasgroup/mbib-base", split="cognitive_bias")

    rng = random.Random(seed)
    rows = list(range(len(ds)))
    rng.shuffle(rows)

    samples: list[EvalSample] = []
    biased_n = control_n = 0
    half = max_samples // 2

    for i in rows:
        if len(samples) >= max_samples:
            break
        row = ds[i]
        text = (row.get("text") or "").strip()
        if len(text) < 25:
            continue
        label = int(row.get("label", -1))

        if label == 1 and biased_n < half:
            samples.append(EvalSample(
                id=f"B01_mbib_cog_{len(samples)}",
                dataset_id="B01_mbib_cognitive",
                bias_type="B01",
                agent="argus",
                input_text=_wrap_synthesis(text),
                label="biased",
                bias_present=True,
                original_fields={"raw_text": text, "mbib_label": label, "row_id": row.get("id")},
                transform_description=(
                    "Real cognitive-bias-labelled sentence from mbib-base. "
                    "No rewriting — annotation is the ground truth."
                ),
            ))
            biased_n += 1
        elif label == 0 and control_n < half:
            samples.append(EvalSample(
                id=f"B01_mbib_cog_{len(samples)}",
                dataset_id="B01_mbib_cognitive",
                bias_type="B01",
                agent="argus",
                input_text=_wrap_synthesis(text),
                label="control",
                bias_present=False,
                original_fields={"raw_text": text, "mbib_label": label, "row_id": row.get("id")},
                transform_description="Real cognitive-bias control sentence (label=0).",
            ))
            control_n += 1

    logger.info(f"mbib cognitive_bias: {len(samples)} samples ({biased_n} biased, {control_n} control)")
    return samples


# ── mediabiasgroup/BABE → B05 ────────────────────────────────────────────

def transform_babe(max_samples: int = 99_999, seed: int = 42) -> list[EvalSample]:
    """
    BABE — Bias Annotations By Experts. ~3K sentences from US news outlets,
    each annotated for media bias at the sentence level, with per-word bias
    spans where applicable.

    Source: mediabiasgroup/BABE — `train` split, columns include
    `text`, `label` (0/1), `biased_words` (list).

    Transform: wrap each sentence as a one-line synthesis. The `biased_words`
    list is retained in `original_fields` for downstream span-level eval.
    """
    from datasets import load_dataset

    logger.info("Loading mediabiasgroup/BABE ...")
    ds = load_dataset("mediabiasgroup/BABE", split="train")

    rng = random.Random(seed)
    rows = list(range(len(ds)))
    rng.shuffle(rows)

    samples: list[EvalSample] = []
    biased_n = control_n = 0
    half = max_samples // 2

    for i in rows:
        if len(samples) >= max_samples:
            break
        row = ds[i]
        text = (row.get("text") or "").strip()
        if len(text) < 25:
            continue
        label = int(row.get("label", -1))
        biased_words = row.get("biased_words") or []

        if label == 1 and biased_n < half:
            samples.append(EvalSample(
                id=f"B05_babe_{len(samples)}",
                dataset_id="B05_babe",
                bias_type="B05",
                agent="argus",
                input_text=_wrap_synthesis(text),
                label="biased",
                bias_present=True,
                original_fields={
                    "raw_text": text,
                    "babe_label": label,
                    "biased_words": list(biased_words) if isinstance(biased_words, (list, tuple)) else biased_words,
                    "outlet": row.get("outlet"),
                    "topic": row.get("topic"),
                },
                transform_description=(
                    "Real expert-annotated biased sentence from BABE. "
                    "Span-level biased words preserved in original_fields."
                ),
            ))
            biased_n += 1
        elif label == 0 and control_n < half:
            samples.append(EvalSample(
                id=f"B05_babe_{len(samples)}",
                dataset_id="B05_babe",
                bias_type="B05",
                agent="argus",
                input_text=_wrap_synthesis(text),
                label="control",
                bias_present=False,
                original_fields={
                    "raw_text": text,
                    "babe_label": label,
                    "outlet": row.get("outlet"),
                    "topic": row.get("topic"),
                },
                transform_description="Real BABE control sentence (label=0, no bias annotated).",
            ))
            control_n += 1

    logger.info(f"BABE: {len(samples)} samples ({biased_n} biased, {control_n} control)")
    return samples


# ── mbib-base/linguistic_bias → B05 ──────────────────────────────────────

def transform_mbib_linguistic(max_samples: int = _LINGUISTIC_CAP, seed: int = 42) -> list[EvalSample]:
    """
    Linguistic bias / framing samples from mbib-base.

    Source: mediabiasgroup/mbib-base — `linguistic_bias` split (~400K rows,
    balanced 50/50). We hard-cap at 5K to keep the Evidence RAG index size
    bounded; the cap is enforced here regardless of caller's max_samples
    because the full set is too large for an in-memory BM25 index.
    """
    from datasets import load_dataset

    cap = min(max_samples, _LINGUISTIC_CAP)
    logger.info(f"Loading mbib-base · linguistic_bias (cap={cap}) ...")
    ds = load_dataset("mediabiasgroup/mbib-base", split="linguistic_bias")

    rng = random.Random(seed)
    rows = list(range(len(ds)))
    rng.shuffle(rows)

    samples: list[EvalSample] = []
    biased_n = control_n = 0
    half = cap // 2

    for i in rows:
        if len(samples) >= cap:
            break
        row = ds[i]
        text = (row.get("text") or "").strip()
        if len(text) < 25:
            continue
        label = int(row.get("label", -1))

        if label == 1 and biased_n < half:
            samples.append(EvalSample(
                id=f"B05_mbib_ling_{len(samples)}",
                dataset_id="B05_mbib_linguistic",
                bias_type="B05",
                agent="argus",
                input_text=_wrap_synthesis(text),
                label="biased",
                bias_present=True,
                original_fields={"raw_text": text, "mbib_label": label, "row_id": row.get("id")},
                transform_description="Real linguistic-bias sentence from mbib-base.",
            ))
            biased_n += 1
        elif label == 0 and control_n < half:
            samples.append(EvalSample(
                id=f"B05_mbib_ling_{len(samples)}",
                dataset_id="B05_mbib_linguistic",
                bias_type="B05",
                agent="argus",
                input_text=_wrap_synthesis(text),
                label="control",
                bias_present=False,
                original_fields={"raw_text": text, "mbib_label": label, "row_id": row.get("id")},
                transform_description="Real linguistic-bias control sentence.",
            ))
            control_n += 1

    logger.info(f"mbib linguistic_bias: {len(samples)} samples ({biased_n} biased, {control_n} control)")
    return samples


# ── Register ─────────────────────────────────────────────────────────────

register_transform(DatasetSpec(
    dataset_id="B01_mbib_cognitive",
    bias_type="B01",
    dataset_name="MBIB Cognitive Bias",
    agent="argus",
    how_to_use="Real cognitive-bias-annotated sentences from MBIB (~7K, balanced).",
    transform_fn=transform_mbib_cognitive,
))

register_transform(DatasetSpec(
    dataset_id="B05_babe",
    bias_type="B05",
    dataset_name="BABE (Bias Annotations By Experts)",
    agent="argus",
    how_to_use="Real expert-annotated media-bias sentences from BABE (~3K, with biased-word spans).",
    transform_fn=transform_babe,
))

register_transform(DatasetSpec(
    dataset_id="B05_mbib_linguistic",
    bias_type="B05",
    dataset_name="MBIB Linguistic Bias",
    agent="argus",
    how_to_use="Real linguistic-bias / framing sentences from MBIB (capped at 5K).",
    transform_fn=transform_mbib_linguistic,
))
