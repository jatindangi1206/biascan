from __future__ import annotations
from typing import Literal, Protocol, Optional
from pydantic import BaseModel, Field

ProviderName = Literal["ollama", "groq", "together", "nvidia", "anthropic", "openai", "gemini"]

SUPPORTED_PROVIDERS: list[dict] = [
    {
        "name": "ollama",
        "label": "Ollama (local, free)",
        "needs_key": False,
        "needs_base_url": True,
        "default_base_url": "http://localhost:11434",
        "default_model": "qwen2.5:7b",
        "model_hint": "qwen2.5:7b · llama3.1:8b · mistral:7b · gemma2:9b · phi3.5:3.8b",
    },
    {
        "name": "groq",
        "label": "Groq (open-source, free tier)",
        "needs_key": True,
        "needs_base_url": False,
        "default_base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "model_hint": "llama-3.3-70b-versatile · llama3-8b-8192 · mixtral-8x7b-32768 · gemma2-9b-it",
    },
    {
        "name": "together",
        "label": "Together AI (open-source)",
        "needs_key": True,
        "needs_base_url": False,
        "default_base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "model_hint": "meta-llama/Llama-3.3-70B-Instruct-Turbo · mistralai/Mixtral-8x7B-Instruct-v0.1 · Qwen/Qwen2.5-72B-Instruct-Turbo",
    },
    {
        "name": "nvidia",
        "label": "NVIDIA NIM (open-source, free)",
        "needs_key": True,
        "needs_base_url": False,
        "default_base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "meta/llama-3.3-70b-instruct",
        "model_hint": "meta/llama-3.3-70b-instruct · deepseek-ai/deepseek-r1 · mistralai/mixtral-8x7b-instruct-v0.1 · google/gemma-3-27b-it",
    },
    {
        "name": "anthropic",
        "label": "Anthropic (Claude)",
        "needs_key": True,
        "needs_base_url": False,
        "default_base_url": "https://api.anthropic.com",
        "default_model": "claude-haiku-4-5-20251001",
        "model_hint": "claude-haiku-4-5-20251001 · claude-sonnet-4-6 · claude-opus-4-7",
    },
    {
        "name": "openai",
        "label": "OpenAI (GPT)",
        "needs_key": True,
        "needs_base_url": False,
        "default_base_url": "https://api.openai.com",
        "default_model": "gpt-4o-mini",
        "model_hint": "gpt-4o-mini · gpt-4o · gpt-4.1-mini",
    },
    {
        "name": "gemini",
        "label": "Google (Gemini)",
        "needs_key": True,
        "needs_base_url": False,
        "default_base_url": "https://generativelanguage.googleapis.com",
        "default_model": "gemini-2.0-flash",
        "model_hint": "gemini-2.0-flash · gemini-2.0-flash-lite · gemini-1.5-pro",
    },
]


class ProviderConfig(BaseModel):
    """Per-request provider config. Never stored server-side."""
    provider: ProviderName
    model: str
    api_key: Optional[str] = Field(default=None, repr=False)  # never serialised back
    base_url: Optional[str] = None

    def model_dump_safe(self) -> dict:
        d = self.model_dump()
        d.pop("api_key", None)
        return d


class LLMError(Exception):
    """Raised when a provider call fails. Message is safe to surface to the user."""


class LLMProvider(Protocol):
    """Minimal LLM interface. Implementations live in ./{provider}.py.

    `complete` returns the assistant's text content as a single string. Each
    implementation handles its provider-specific JSON-mode hint where available.
    """

    name: str

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> str: ...


def build_provider(config: ProviderConfig) -> LLMProvider:
    from .ollama import OllamaProvider
    from .groq import GroqProvider
    from .together import TogetherProvider
    from .nvidia import NvidiaProvider
    from .anthropic import AnthropicProvider
    from .openai_compat import OpenAIProvider
    from .gemini import GeminiProvider

    if config.provider == "ollama":
        return OllamaProvider(config)
    if config.provider == "groq":
        return GroqProvider(config)
    if config.provider == "together":
        return TogetherProvider(config)
    if config.provider == "nvidia":
        return NvidiaProvider(config)
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    if config.provider == "openai":
        return OpenAIProvider(config)
    if config.provider == "gemini":
        return GeminiProvider(config)
    raise LLMError(f"Unknown provider: {config.provider}")
