from __future__ import annotations
import httpx

from ..config import PROVIDER_TIMEOUT_S
from .base import LLMError, ProviderConfig


class OpenAIProvider:
    name = "openai"

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise LLMError("OpenAI provider requires an API key.")
        self._api_key = config.api_key
        self._model = config.model
        self._base_url = (config.base_url or "https://api.openai.com").rstrip("/")

    async def complete(
        self, *, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as e:
            raise LLMError(f"Cannot reach OpenAI: {e}") from e
        if resp.status_code >= 400:
            raise LLMError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected OpenAI response shape: {data}") from e
