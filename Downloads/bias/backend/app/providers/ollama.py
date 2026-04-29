from __future__ import annotations
import httpx

from ..config import PROVIDER_TIMEOUT_S
from .base import LLMError, ProviderConfig


class OllamaProvider:
    name = "ollama"

    def __init__(self, config: ProviderConfig):
        self._model = config.model
        self._base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    async def complete(
        self, *, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "options": {"num_predict": max_tokens, "temperature": 0.2},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
                resp = await client.post(url, json=payload)
        except httpx.RequestError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self._base_url}. Is `ollama serve` running? ({e})"
            ) from e
        if resp.status_code == 404:
            raise LLMError(
                f"Ollama model '{self._model}' not found. Run `ollama pull {self._model}` first."
            )
        if resp.status_code >= 400:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        msg = (data.get("message") or {}).get("content")
        if not msg:
            raise LLMError(f"Ollama returned no content: {data}")
        return msg
