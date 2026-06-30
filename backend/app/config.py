"""Static config. No API keys here — keys are per-request and never stored."""
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"

DEFAULT_ANALYSIS_MODE = "systematic_review"
PROMPT_VERSION_BY_ANALYSIS_MODE = {
    "systematic_review": "v1",
    "general_research": "v2",
}
AVAILABLE_ANALYSIS_MODES = tuple(PROMPT_VERSION_BY_ANALYSIS_MODE.keys())

# Legacy default for code paths that still need a single string. Runtime prompt
# selection should go through the resolver functions below instead.
PROMPT_VERSION = PROMPT_VERSION_BY_ANALYSIS_MODE[DEFAULT_ANALYSIS_MODE]


def resolve_prompt_version(analysis_mode: str) -> str:
    return PROMPT_VERSION_BY_ANALYSIS_MODE.get(
        analysis_mode,
        PROMPT_VERSION_BY_ANALYSIS_MODE[DEFAULT_ANALYSIS_MODE],
    )


def resolve_prompt_filename(prompt_filename: str, prompt_version: str) -> str:
    if re.search(r"_v\d+\.\d+\.txt$", prompt_filename):
        return re.sub(r"_v\d+\.\d+\.txt$", f"_{prompt_version}.0.txt", prompt_filename)
    return prompt_filename


def resolve_prompt_path(prompt_filename: str, analysis_mode: str) -> Path:
    prompt_version = resolve_prompt_version(analysis_mode)
    resolved_name = resolve_prompt_filename(prompt_filename, prompt_version)
    return PROMPTS_DIR / prompt_version / resolved_name

# Tuning knobs (override via env if you really want).
CONFIDENCE_FLOOR = float(os.getenv("BIASSCAN_CONFIDENCE_FLOOR", "0.5"))
DEFAULT_MAX_TOKENS = int(os.getenv("BIASSCAN_MAX_TOKENS", "4096"))
PROVIDER_TIMEOUT_S = float(os.getenv("BIASSCAN_PROVIDER_TIMEOUT", "180"))
# Max concurrent provider calls in flight across the selected agents. Lower this
# to 1 for low-tier providers (e.g. Mistral's free Experiment plan, ~1 req/s)
# that return 429 when several agents call at once.
MAX_CONCURRENCY = max(1, int(os.getenv("BIASSCAN_MAX_CONCURRENCY", "3")))
