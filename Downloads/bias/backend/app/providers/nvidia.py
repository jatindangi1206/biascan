from __future__ import annotations
import asyncio
import httpx

from ..config import PROVIDER_TIMEOUT_S
from .base import LLMError, ProviderConfig


class NvidiaProvider:
    """NVIDIA NIM API — OpenAI-compatible, hosts DeepSeek, Llama, Mistral and 150+ open-source
    models. API keys available free at build.nvidia.com."""

    name = "nvidia"

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise LLMError("NVIDIA NIM provider requires an API key (free at build.nvidia.com).")
        self._api_key = config.api_key
        self._model = config.model
        self._base_url = (config.base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")

    async def complete(
        self, *, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        url = f"{self._base_url}/chat/completions"
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
            "stream": False,
        }
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                raise LLMError(f"Cannot reach NVIDIA NIM: {e}") from e
            if resp.status_code == 429:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
                continue
            break
        if resp.status_code >= 400:
            raise LLMError(f"NVIDIA NIM error {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected NVIDIA NIM response shape: {data}") from e
