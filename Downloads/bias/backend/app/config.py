"""Static config. No API keys here — keys are per-request and never stored."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"

PROMPT_VERSION = "v1"

# Tuning knobs (override via env if you really want).
CONFIDENCE_FLOOR = float(os.getenv("BIASSCAN_CONFIDENCE_FLOOR", "0.5"))
DEFAULT_MAX_TOKENS = int(os.getenv("BIASSCAN_MAX_TOKENS", "4096"))
PROVIDER_TIMEOUT_S = float(os.getenv("BIASSCAN_PROVIDER_TIMEOUT", "180"))
