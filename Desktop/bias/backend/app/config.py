import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("BIASSCAN_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = int(os.getenv("BIASSCAN_MAX_TOKENS", "4096"))

PROMPT_VERSION = "v1"

CONFIDENCE_FLOOR = float(os.getenv("BIASSCAN_CONFIDENCE_FLOOR", "0.5"))
ADAPTIVE_ESCALATE_BELOW = float(os.getenv("BIASSCAN_ADAPTIVE_ESCALATE_BELOW", "0.6"))
