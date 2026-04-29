from __future__ import annotations
import httpx

from ..config import PROVIDER_TIMEOUT_S
from .base import LLMError, ProviderConfig

# Lightning AI hosts open-source models via an OpenAI-compatible chat-completions
# endpoint, but the base path is /api/v1 (not /v1).  Content is sent as an array
# of typed blocks, which is what their examples show — plain strings also work but
# the array form is more future-proof for multi-modal payloads.


class LightningProvider:
    name = "lightning"

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise LLMError("Lightning AI provider requires an API key (Bearer token).")
        self._api_key = config.api_key
        self._model = config.model
        # Allow base_url override; default to the public Lightning AI endpoint.
        base = (config.base_url or "https://lightning.ai").rstrip("/")
        self._url = f"{base}/api/v1/chat/completions"

    async def complete(
        self, *, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
                resp = await client.post(self._url, json=payload, headers=headers)
        except httpx.RequestError as e:
            raise LLMError(f"Cannot reach Lightning AI: {e}") from e

        if resp.status_code >= 400:
            raise LLMError(
                f"Lightning AI error {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected Lightning AI response shape: {data}") from e
