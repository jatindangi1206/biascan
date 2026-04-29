"""Stub vector store. Replace with ChromaDB / FAISS integration.

The premium-mode pipeline would:
1. Resolve each reference (DOI / PubMed / Semantic Scholar / CrossRef).
2. Fetch full-text or abstract.
3. Chunk semantically (abstract / methods / results / conclusion).
4. Embed with a sentence-transformer model.
5. Store + retrieve top-k chunks per agent query.

Not implemented in the v0.1 release. The orchestrator therefore falls back
to text-only Lite mode and emits a warning whenever Premium / Adaptive is
requested.
"""
from __future__ import annotations


class VectorStore:
    def __init__(self) -> None:
        pass

    def index_references(self, references: list[str]) -> None:
        raise NotImplementedError("RAG vector store is not yet implemented.")

    def query(self, query: str, k: int = 4) -> list[str]:
        raise NotImplementedError("RAG vector store is not yet implemented.")
