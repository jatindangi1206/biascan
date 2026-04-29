from __future__ import annotations
import httpx

from ..config import PROVIDER_TIMEOUT_S
from .base import LLMError, ProviderConfig


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise LLMError("Anthropic provider requires an API key.")
        self._api_key = config.api_key
        self._model = config.model
        self._base_url = (config.base_url or "https://api.anthropic.com").rstrip("/")

    async def complete(
        self, *, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        url = f"{self._base_url}/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    # Free perf win when the same agent is called repeatedly.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_message}],
        }
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as e:
            raise LLMError(f"Cannot reach Anthropic: {e}") from e
        if resp.status_code >= 400:
            raise LLMError(f"Anthropic error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        text = "".join(parts).strip()
        if not text:
            raise LLMError(f"Anthropic returned no text: {data}")
        return text
