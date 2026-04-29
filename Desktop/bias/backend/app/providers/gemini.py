from __future__ import annotations
import httpx

from ..config import PROVIDER_TIMEOUT_S
from .base import LLMError, ProviderConfig


class GeminiProvider:
    name = "gemini"

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise LLMError("Gemini provider requires an API key.")
        self._api_key = config.api_key
        self._model = config.model
        self._base_url = (
            config.base_url or "https://generativelanguage.googleapis.com"
        ).rstrip("/")

    async def complete(
        self, *, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        url = (
            f"{self._base_url}/v1beta/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
                resp = await client.post(url, json=payload)
        except httpx.RequestError as e:
            raise LLMError(f"Cannot reach Gemini: {e}") from e
        if resp.status_code >= 400:
            raise LLMError(f"Gemini error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            cand = data["candidates"][0]
            parts = cand["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from e
