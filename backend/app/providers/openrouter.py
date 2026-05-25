from __future__ import annotations
import asyncio
import random
import httpx

from ..config import PROVIDER_TIMEOUT_S
from .base import LLMError, ProviderConfig


class OpenRouterProvider:
    """OpenRouter — OpenAI-compatible gateway that routes to hundreds of models
    from every major lab (OpenAI, Anthropic, Meta, Qwen, Google, DeepSeek, ...)
    behind one API key. Get a key at openrouter.ai/keys. The model id selects the
    upstream model, e.g. "anthropic/claude-3.5-sonnet" or "qwen/qwen-2.5-72b-instruct"."""

    name = "openrouter"

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise LLMError("OpenRouter provider requires an API key (get one at openrouter.ai/keys).")
        self._api_key = config.api_key
        self._model = config.model
        self._base_url = (config.base_url or "https://openrouter.ai/api/v1").rstrip("/")

    async def complete(
        self, *, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Optional ranking headers OpenRouter recommends; harmless to send.
            "HTTP-Referer": "https://github.com/jatindangi1206/biascan",
            "X-Title": "BiasScan",
        }
        # NOTE: no response_format here. OpenRouter routes to many models that
        # don't support json_object mode — notably every Anthropic Claude model,
        # Amazon Nova, Perplexity Sonar, and most ":free" variants — so forcing
        # it would break those. We rely on the agent system prompt (which already
        # demands JSON) plus BaseAgent's defensive JSON parser instead.
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                raise LLMError(f"Cannot reach OpenRouter: {e}") from e
            if resp.status_code == 429 and attempt < max_attempts - 1:
                # Honor the server's Retry-After when present; else exponential
                # backoff. Jitter avoids agents retrying in lockstep. Cap at 30s.
                retry_after = resp.headers.get("Retry-After", "")
                base = float(retry_after) if retry_after.isdigit() else 2 ** attempt
                await asyncio.sleep(min(base + random.uniform(0, 0.5), 30.0))
                continue
            break
        if resp.status_code == 401:
            raise LLMError("OpenRouter 401: API key is missing or invalid.")
        if resp.status_code == 402:
            raise LLMError("OpenRouter 402: insufficient credits. Top up at openrouter.ai/credits.")
        if resp.status_code >= 400:
            raise LLMError(f"OpenRouter error {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected OpenRouter response shape: {data}") from e
