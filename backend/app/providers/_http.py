"""Shared httpx.AsyncClient used by all providers.

One client per process keeps the TCP/TLS connection pool alive across calls
on the same warm function instance, saving ~50-200ms per request vs.
creating a fresh client each time. On Vercel Fluid Compute, instances are
reused across requests, so the pool persists for the lifetime of the
container and benefits subsequent /analyze calls too.
"""
from __future__ import annotations

import httpx

from ..config import PROVIDER_TIMEOUT_S

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S)
    return _client
