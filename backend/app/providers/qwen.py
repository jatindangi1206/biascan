from __future__ import annotations
import asyncio
import random
import httpx

from ._http import get_client
from .base import LLMError, ProviderConfig


class QwenProvider:
    """Qwen via Alibaba Cloud Model Studio (DashScope) — OpenAI-compatible
    chat completions. Get an API key at bailian.console.aliyun.com. The default
    base URL is the international (Singapore) compatible-mode endpoint; override
    it from the UI to use the China endpoint
    (https://dashscope.aliyuncs.com/compatible-mode/v1) or any Qwen-compatible
    gateway."""

    name = "qwen"

    def __init__(self, config: ProviderConfig):
        if not config.api_key:
            raise LLMError(
                "Qwen provider requires an API key "
                "(Alibaba Cloud Model Studio at bailian.console.aliyun.com)."
            )
        self._api_key = config.api_key
        self._model = config.model
        self._base_url = (
            config.base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")

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
        }
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                resp = await get_client().post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                raise LLMError(f"Cannot reach Qwen (Model Studio): {e}") from e
            if resp.status_code == 429 and attempt < max_attempts - 1:
                # Honor the server's Retry-After when present; else exponential
                # backoff. Jitter avoids agents retrying in lockstep. Cap at 30s.
                retry_after = resp.headers.get("Retry-After", "")
                base = float(retry_after) if retry_after.isdigit() else 2 ** attempt
                await asyncio.sleep(min(base + random.uniform(0, 0.5), 30.0))
                continue
            break
        if resp.status_code == 401:
            raise LLMError("Qwen 401: API key is missing or invalid.")
        if resp.status_code >= 400:
            raise LLMError(f"Qwen error {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected Qwen response shape: {data}") from e
