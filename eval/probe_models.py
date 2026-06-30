"""Quick probe: send each candidate model the real ARGUS system prompt + one
sentence, and print raw response so we can see whether it emits parseable JSON."""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.argus import ArgusAgent  # noqa: E402
from app.providers import build_provider  # noqa: E402
from app.providers.base import ProviderConfig  # noqa: E402
from app.agents.base import _extract_json  # noqa: E402

MODELS = [
    "openai/gpt-4.1-nano",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-4-maverick",
]
SENTENCE = ("Today, many news stations are expected to curate media that supports "
            "the political perspective of the news organization's owners.")


async def probe(model_id: str, api_key: str) -> None:
    print(f"\n{'='*80}\n  {model_id}\n{'='*80}")
    cfg = ProviderConfig(provider="openrouter", model=model_id, api_key=api_key, base_url=None)
    provider = build_provider(cfg)
    agent = ArgusAgent()
    user_msg = agent.build_user_message(SENTENCE, references=None, mode="lite")
    try:
        raw = await provider.complete(
            system_prompt=agent.load_prompt(),
            user_message=user_msg,
            max_tokens=1500,
        )
    except Exception as e:
        print(f"  CALL ERROR: {type(e).__name__}: {e}")
        return
    print(f"  raw_length: {len(raw) if raw else 0}")
    if raw is None:
        print("  RAW IS NONE — provider returned no body")
        return
    print(f"  raw[0:300]: {raw[:300]!r}")
    parsed = _extract_json(raw)
    if parsed is None:
        print("  ✗ parser returned None — would fail")
    elif isinstance(parsed, dict) and "annotations" in parsed:
        anns = parsed.get("annotations", [])
        print(f"  ✓ parses as dict with 'annotations' (count={len(anns)})")
    else:
        print(f"  ◐ parses but no 'annotations' key — top-level keys: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}")


async def main() -> int:
    key = os.environ["BIASSCAN_API_KEY"]
    for m in MODELS:
        await probe(m, key)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
