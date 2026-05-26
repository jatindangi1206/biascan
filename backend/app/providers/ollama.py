from __future__ import annotations
import os

import httpx

from ..config import DEFAULT_MAX_TOKENS
from ._http import get_client
from .base import LLMError, ProviderConfig

# Ollama's default num_ctx is 2048, which silently truncates our system prompt
# (~3K tokens) plus any non-trivial synthesis text. Force a larger window so the
# model actually sees the whole paste.
OLLAMA_NUM_CTX = int(os.getenv("BIASSCAN_OLLAMA_NUM_CTX", "16384"))
_OLLAMA_MAX_TOKENS_ENV = os.getenv("BIASSCAN_OLLAMA_MAX_TOKENS")
OLLAMA_MAX_TOKENS = (
    int(_OLLAMA_MAX_TOKENS_ENV) if _OLLAMA_MAX_TOKENS_ENV else DEFAULT_MAX_TOKENS
)


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
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_MAX_TOKENS or max_tokens,
                "temperature": 0.2,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        try:
            resp = await get_client().post(url, json=payload)
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
        message = data.get("message") or {}
        msg = message.get("content")
        if not msg:
            done_reason = data.get("done_reason")
            if message.get("thinking"):
                hint = (
                    "Ollama returned only reasoning with no JSON content. "
                    "Increase BIASSCAN_OLLAMA_MAX_TOKENS or use a JSON-friendly model "
                    "(qwen2.5, gemma2, mistral)."
                )
                if done_reason == "length":
                    hint = f"{hint} Response hit the token limit."
                raise LLMError(hint)
            raise LLMError("Ollama returned empty content.")
        return msg
