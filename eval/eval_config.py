"""
Evaluation framework configuration — models to benchmark, dataset settings, scoring weights.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ── Models to benchmark ─────────────────────────────────────────────────────
# Each entry: (provider, model, label_for_charts)
# Covers free + paid as requested

BENCHMARK_MODELS = [
    # Free tier
    ("groq",      "llama-3.3-70b-versatile",   "Llama-3.3-70B (Groq)"),
    ("groq",      "gemma2-9b-it",              "Gemma2-9B (Groq)"),
    ("google",    "gemini-2.5-flash",           "Gemini-2.5-Flash"),
    ("mock",      "mock",                       "Mock (baseline)"),
    # Paid tier
    ("anthropic", "claude-sonnet-4-6",     "Claude Sonnet 4"),
    ("anthropic", "claude-haiku-4-5-20251001",  "Claude Haiku 3.5"),
    ("openai",    "gpt-4o",                     "GPT-4o"),
    ("openai",    "gpt-4o-mini",                "GPT-4o Mini"),
    ("deepseek",  "deepseek-chat",              "DeepSeek-V3"),
    ("xai",       "grok-3-mini",                "Grok-3 Mini"),
]


# ── Automated eval — dataset config ─────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration for one evaluation dataset."""
    name: str
    hf_path: str                     # HuggingFace dataset path
    hf_subset: str = ""              # optional subset/config name
    split: str = "test"              # which split to use
    bias_types: list[str] = field(default_factory=list)  # which B-codes it tests
    agent: str = "argus"             # primary agent being tested
    sample_size: int = 50            # smoke test sample count
    description: str = ""


# Priority datasets for smoke test (one per bias type)
EVAL_DATASETS = [
    DatasetConfig(
        name="SciFact",
        hf_path="allenai/scifact",
        hf_subset="claims",
        split="test",
        bias_types=["B01"],
        agent="argus",
        sample_size=50,
        description="Scientific claims with SUPPORTS/REFUTES labels — tests confirmation bias detection",
    ),
    DatasetConfig(
        name="Scientific Exaggeration",
        hf_path="copenlu/scientific-exaggeration-detection",
        split="test",
        bias_types=["B03", "B04"],
        agent="argus",
        sample_size=50,
        description="Abstract/press-release pairs labeled SAME/EXAGGERATES — tests certainty inflation & overgeneralization",
    ),
    DatasetConfig(
        name="PubHealth",
        hf_path="ImperialCollegeLondon/health_fact",
        split="test",
        bias_types=["B04"],
        agent="argus",
        sample_size=50,
        description="Health claims with TRUE/FALSE/MIXTURE labels — tests overgeneralization in health contexts",
    ),
    DatasetConfig(
        name="Corr2Cause",
        hf_path="causal-nlp/corr2cause",
        split="test",
        bias_types=["B08"],
        agent="argus",
        sample_size=50,
        description="Correlation→causation inference pairs — tests causal inference bias detection",
    ),
    DatasetConfig(
        name="SciCite",
        hf_path="allenai/scicite",
        split="test",
        bias_types=["L1"],
        agent="lens",
        sample_size=50,
        description="Citation intent classification (BACKGROUND/METHOD/RESULT) — tests discourse analysis",
    ),
]


# ── Manual eval — scoring rubric ────────────────────────────────────────────

MANUAL_SCORING_RUBRIC = {
    "precision": {
        "description": "What fraction of reported findings are genuine biases (not false positives)?",
        "scale": "1-5 (1=mostly false positives, 5=all findings are real biases)",
    },
    "recall": {
        "description": "Did the pipeline catch the biases you can see by reading the text?",
        "scale": "1-5 (1=missed most biases, 5=caught everything)",
    },
    "severity_accuracy": {
        "description": "Are the severity labels (LOW/MODERATE/HIGH) correct?",
        "scale": "1-5 (1=completely wrong, 5=perfectly calibrated)",
    },
    "explanation_quality": {
        "description": "Are the bias explanations clear, specific, and actionable?",
        "scale": "1-5 (1=vague/wrong, 5=clear and educational)",
    },
    "revision_quality": {
        "description": "Did QUILL's edits actually fix the bias without distorting meaning?",
        "scale": "1-5 (1=made it worse, 5=perfect debiasing)",
    },
}


# ── Pipeline run settings for eval ──────────────────────────────────────────

EVAL_PIPELINE_SETTINGS = {
    "max_iterations": 2,        # Keep eval runs short
    "patience": 2,
    "threshold": 0.05,
}
