# BiasScan Evaluation Datasets — Research Report

> **Purpose**: Map every BiasScan agent and bias type to concrete, downloadable datasets
> for building automated evaluation benchmarks.

---

## Master Dataset Table

### ARGUS Agent — Bias Detection (B01–B11)

| Bias Type | Dataset | Size | Format | Link | How to Use |
|-----------|---------|------|--------|------|------------|
| **B01 — Confirmation Bias** | **SciFact** (AllenAI) | 1,409 claims + 5,183 abstracts | JSON | [HuggingFace](https://huggingface.co/datasets/allenai/scifact) / [GitHub](https://github.com/allenai/scifact) | Claims labeled SUPPORTS/REFUTES/NOINFO with rationale sentences. Test whether ARGUS flags text that only cites supporting evidence and ignores refuting abstracts. Feed the "SUPPORTS-only" subset as biased text. |
| **B01 — Confirmation Bias** | **FEVER** | 185,445 claims | JSON | [HuggingFace](https://huggingface.co/datasets/fever/fever) / [fever.ai](https://fever.ai/dataset/fever.html) | Claims verified against Wikipedia as SUPPORTED/REFUTED/NOT_ENOUGH_INFO. Create test cases by pairing REFUTED claims with only supporting evidence — simulates cherry-picking. |
| **B01 — Confirmation Bias** | **IBM Claim Stance** | 2,394 claims for 55 topics | JSON | [HuggingFace](https://huggingface.co/datasets/ibm-research/claim_stance) | Claims labeled Pro/Con for each topic. Construct one-sided passages using only Pro claims — tests if ARGUS catches the missing counter-evidence. |
| **B03 — Certainty Inflation** | **BioScope Corpus** | 20,000+ sentences | XML | [Official site](https://rgai.inf.u-szeged.hu/node/105) | Medical texts, bio papers, and abstracts annotated for speculation/negation cues and their scope. Sentences marked "certain" vs "speculative" — test if ARGUS catches hedging gaps where certainty language is used on speculative claims. |
| **B03 — Certainty Inflation** | **CoNLL-2010 Shared Task** | ~11,000 sentences (bio) + ~11,000 (Wikipedia) | CoNLL/IOB | [Shared task data](https://rgai.inf.u-szeged.hu/node/105) / [Paper](https://aclanthology.org/W10-3001.pdf) | Sentence-level hedge classification (certain/uncertain). Strip hedge cues from uncertain sentences → creates "certainty inflated" text → test if ARGUS detects it. |
| **B03 — Certainty Inflation** | **Scientific Exaggeration Detection** (CopenNLU) | 663 abstract/press-release pairs | JSON | [GitHub](https://github.com/copenlu/scientific-exaggeration-detection) / [HuggingFace](https://huggingface.co/datasets/copenlu/scientific-exaggeration-detection) | Pairs labeled SAME/EXAGGERATES/DOWNPLAYS. The "exaggerates" class directly tests certainty inflation — press releases that overclaim findings. |
| **B04 — Overgeneralization** | **Scientific Exaggeration Detection** | 663 pairs | JSON | [GitHub](https://github.com/copenlu/scientific-exaggeration-detection) | "Exaggerates" label often captures scope overclaiming (e.g., "mice study" → "humans"). Extract these and test if ARGUS catches population/scope leaps. |
| **B04 — Overgeneralization** | **PubHealth** | 11,832 claims | JSON | [GitHub](https://github.com/neemakot/Health-Fact-Checking) / [HuggingFace](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact) | Health claims labeled TRUE/FALSE/UNPROVEN/MIXTURE with explanations. "FALSE" claims often contain overgeneralizations (e.g., "X cures all cancer"). Extract the explanation text as ground truth for overgeneralization detection. |
| **B04 — Overgeneralization** | **ClaimBuster** | 23,533 sentences | CSV | [Zenodo](https://zenodo.org/records/3609356) | U.S. presidential debate sentences classified by claim-worthiness (NFS/UFS/CFS). Check-worthy factual statements often contain broad claims — useful for testing scope detection. |
| **B05 — Framing Effect** | **Media Frames Corpus** (MFC) | 20,000+ annotated spans across 6 issues | JSON/CSV | [GitHub](https://github.com/dallascard/media_frames_corpus) | News articles annotated with 15 framing dimensions (economic, morality, fairness, etc.). Test whether ARGUS detects one-sided framing when text only uses one frame. |
| **B05 — Framing Effect** | **MBIC (Media Bias Annotation)** | 1,700 statements, 10 annotators each | CSV | [Kaggle](https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset) | Word-level and sentence-level bias annotations. Directly tests framing bias detection in language. |
| **B05 — Framing Effect** | **SemEval-2023 Task 3** — Persuasion Techniques | 9,000+ annotated texts, 23 techniques | JSON | [Official site](https://propaganda.math.unipd.it/semeval2023task3/) / [Zenodo](https://zenodo.org/records/3952415) | Fine-grained span-level persuasion technique labels including "loaded language", "appeal to authority", "black-and-white fallacy" — maps to framing bias subtypes. |
| **B08 — Causal Inference** | **Corr2Cause** (ICLR 2024) | 400,000+ samples | JSON | [HuggingFace](https://huggingface.co/datasets/causal-nlp/corr2cause) / [GitHub](https://github.com/causalNLP/corr2cause) | Correlational statements → valid/invalid causal inferences. Directly tests whether the system can detect when correlation is misrepresented as causation. The gold standard for B08. |
| **B08 — Causal Inference** | **SciFact** | 1,409 claims | JSON | [HuggingFace](https://huggingface.co/datasets/allenai/scifact) | Many scientific claims involve causal language. Filter for causal claims ("causes", "leads to", "results in") and check if ARGUS correctly flags those not supported by causal evidence. |

---

### LIBRA Agent — Scope & Hedging Analysis

| What It Evaluates | Dataset | Size | Format | Link | How to Use |
|-------------------|---------|------|--------|------|------------|
| **Hedge detection** | **BioScope Corpus** | 20,000+ sentences | XML | [Official site](https://rgai.inf.u-szeged.hu/node/105) | Gold standard for biomedical hedging. Annotations mark hedge cues ("may", "possibly", "suggests") and their scope. Compare LIBRA's Hyland-based hedge counts against BioScope annotations. |
| **Hedge detection** | **CoNLL-2010 Shared Task** | ~22,000 sentences | CoNLL/IOB | [Official site](https://rgai.inf.u-szeged.hu/node/105) | Two domains: biological papers + Wikipedia. Sentence-level uncertain/certain labels + in-sentence hedge scope spans. Benchmark LIBRA's sentence-level classification accuracy. |
| **Hedge/booster detection** | **HedgePeer** | Peer review sentences | Text | [ACM DL](https://dl.acm.org/doi/10.1145/3529372.3533300) | Hedging detection in scientific peer review text. Tests LIBRA on academic writing style closer to systematic reviews. |
| **Scope overclaiming** | **Scientific Exaggeration Detection** | 663 pairs | JSON | [GitHub](https://github.com/copenlu/scientific-exaggeration-detection) | Abstract→press release exaggeration. The "exaggerates" label captures temporal overclaiming, population overclaiming, and scope extension — all LIBRA targets. |
| **Hyland markers** | **BERT Uncertainty Detection** | CoNLL-2010 data + models | Python | [GitHub](https://github.com/PeterZhizhin/BERTUncertaintyDetection) | BERT-based hedge detection trained on CoNLL-2010. Provides pre-trained models + processed data. Compare LIBRA's rule-based Hyland counting against BERT neural approach. |
| **Tone calibration** | **PubHealth** | 11,832 claims | JSON | [HuggingFace](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact) | Claims with veracity labels + explanations. "MIXTURE" claims often have poorly calibrated hedging — good test cases for tone calibration scoring. |

---

### LENS Agent — Discourse Analysis

| What It Evaluates | Dataset | Size | Format | Link | How to Use |
|-------------------|---------|------|--------|------|------------|
| **Citation intent** | **SciCite** (AllenAI) | ~11,000 annotated citations | JSONL | [HuggingFace](https://huggingface.co/datasets/allenai/scicite) / [GitHub](https://github.com/allenai/scicite) | Citations classified as BACKGROUND/METHOD/RESULT. Test LENS's ability to detect citation function and whether support/contradict direction is correctly identified. |
| **Citation polarity** | **ACL-ARC Citation Context** | 8,736 annotated citations | XML | [ACL Anthology](https://aclanthology.org/L08-1005/) / [GitHub](https://github.com/languagerecipes/the-acl-rd-tec) | Citations annotated for polarity (positive/negative/neutral) + implicit/explicit. Directly benchmarks LENS's citation sentiment classification. |
| **Argumentation mining** | **OpenDebateEvidence** (NeurIPS 2024) | 3.5M+ documents | JSON | [GitHub](https://arxiv.org/abs/2406.14657) | Massive debate evidence corpus with argument structure annotations. Test LENS's premise-conclusion detection on structured argumentative text. |
| **Argumentation mining** | **IBM Debater — Evidence Sentences** | Varies by subtask | JSON | [IBM Research](https://research.ibm.com/publications/a-recorded-debating-dataset) | Claims + evidence sentences with quality scores. Tests whether LENS correctly identifies claim-evidence relationships. |
| **Selective reporting / Spin** | **SciFact** | 1,409 claims | JSON | [HuggingFace](https://huggingface.co/datasets/allenai/scifact) | Claims REFUTED by evidence but may be presented as supported. Tests LENS's ability to detect when conclusions don't match evidence direction. |
| **Discourse structure** | **SciCite + SciFact combined** | ~12,000 examples | JSON | Both HuggingFace | Cross-reference citation intent with claim verification — if a citation is used as BACKGROUND but the cited paper actually REFUTES the claim, that's a discourse-level bias LENS should catch. |

---

### QUILL Agent — Revision Quality

| What It Evaluates | Dataset | Size | Format | Link | How to Use |
|-------------------|---------|------|--------|------|------------|
| **Edit quality** | Any of the above | — | — | — | Run ARGUS+LIBRA+LENS on raw text → get findings → run QUILL → re-run ARGUS+LIBRA+LENS on revised text. Score improvement = QUILL effectiveness. No external dataset needed — QUILL is evaluated by the pipeline's own convergence. |
| **Paraphrase quality** | **Scientific Exaggeration** | 663 pairs | JSON | [GitHub](https://github.com/copenlu/scientific-exaggeration-detection) | Use the "same" labeled pairs as ground truth for good paraphrasing that preserves meaning without exaggeration. |

---

### VIGIL Agent — Integrity Gate

| What It Evaluates | Dataset | Size | Format | Link | How to Use |
|-------------------|---------|------|--------|------|------------|
| **Citation preservation** | Any cited text from SciFact | — | — | — | VIGIL is rule-based — test by running QUILL on text with known citations and verifying VIGIL catches any dropped ones. |
| **Fact preservation** | **PubHealth** | 11,832 | JSON | [HuggingFace](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact) | Use claims with numerical data. After QUILL revision, VIGIL should catch if numbers changed. |

---

## Cross-Bias Dataset Coverage Matrix

| Dataset | B01 Confirmation | B03 Certainty | B04 Overgen. | B05 Framing | B08 Causal | LIBRA Hedge | LENS Discourse | Free? |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SciFact** | ✅ primary | — | — | — | ✅ secondary | — | ✅ secondary | ✅ |
| **FEVER** | ✅ primary | — | — | — | — | — | — | ✅ |
| **BioScope** | — | ✅ primary | — | — | — | ✅ primary | — | ✅ |
| **CoNLL-2010** | — | ✅ primary | — | — | — | ✅ primary | — | ✅ |
| **Sci-Exaggeration** | — | ✅ primary | ✅ primary | — | — | ✅ secondary | — | ✅ |
| **PubHealth** | — | — | ✅ primary | — | — | ✅ secondary | — | ✅ |
| **Media Frames Corpus** | — | — | — | ✅ primary | — | — | — | ✅ |
| **MBIC** | — | — | — | ✅ primary | — | — | — | ✅ |
| **SemEval-2023 Task 3** | — | — | — | ✅ primary | — | — | — | ✅ |
| **Corr2Cause** | — | — | — | — | ✅ primary | — | — | ✅ |
| **IBM Claim Stance** | ✅ secondary | — | — | — | — | — | — | ✅ |
| **ClaimBuster** | — | — | ✅ secondary | — | — | — | — | ✅ |
| **SciCite** | — | — | — | — | — | — | ✅ primary | ✅ |
| **ACL-ARC** | — | — | — | — | — | — | ✅ primary | ✅ |
| **OpenDebateEvidence** | — | — | — | — | — | — | ✅ secondary | ✅ |

**All 15 datasets are free and open-source.**

---

## Priority Ranking — Start Here

For a first pass automated evaluation, use these **5 datasets** (one per bias type):

1. **B01 Confirmation Bias** → **SciFact** — best alignment, scientific domain, claim+evidence pairs
2. **B03 Certainty Inflation** → **Scientific Exaggeration Detection** — directly measures overclaiming in health science
3. **B04 Overgeneralization** → **PubHealth** — health claims with FALSE/MIXTURE labels catching scope overreach
4. **B05 Framing Effect** → **Media Frames Corpus** — 15 framing dimensions, well-annotated
5. **B08 Causal Inference** → **Corr2Cause** — 400K samples, specifically designed for correlation→causation errors

For LIBRA → **BioScope** (gold standard hedging in biomedical text)
For LENS → **SciCite** (citation intent in scientific papers)

---

## Download Commands (Quick Start)

```python
# Install
pip install datasets

# SciFact (B01)
from datasets import load_dataset
scifact = load_dataset("allenai/scifact")

# FEVER (B01)
fever = load_dataset("fever/fever", "v1.0")

# Corr2Cause (B08)
corr2cause = load_dataset("causal-nlp/corr2cause")

# PubHealth (B04)
pubhealth = load_dataset("ImperialCollegeLondon/health_fact")

# SciCite (LENS)
scicite = load_dataset("allenai/scicite")

# IBM Claim Stance (B01)
claim_stance = load_dataset("ibm-research/claim_stance")

# Scientific Exaggeration (B03/B04)
exaggeration = load_dataset("copenlu/scientific-exaggeration-detection")
```

For BioScope and CoNLL-2010: download from [rgai.inf.u-szeged.hu](https://rgai.inf.u-szeged.hu/node/105)
For Media Frames Corpus: clone [GitHub repo](https://github.com/dallascard/media_frames_corpus)
For MBIC: download from [Kaggle](https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset)
For SemEval-2023: register at [propaganda.math.unipd.it](https://propaganda.math.unipd.it/semeval2023task3/)
For ClaimBuster: download from [Zenodo](https://zenodo.org/records/3609356)
