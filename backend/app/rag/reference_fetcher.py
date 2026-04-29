"""Stub reference fetcher. Premium mode would resolve references via
DOI / PubMed / Semantic Scholar / CrossRef and return full-text or abstracts.

Not implemented in v0.1.
"""
from __future__ import annotations


def parse_references(blob: str) -> list[str]:
    """Split a reference blob into individual reference strings.

    Best-effort heuristic: split on numbered prefixes and blank lines.
    """
    if not blob:
        return []
    lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
    return lines


async def fetch_full_text(reference: str) -> str | None:
    """Resolve a reference and return its abstract or full text.

    Not implemented; always returns None.
    """
    return None
