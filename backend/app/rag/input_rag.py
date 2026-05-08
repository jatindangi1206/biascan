"""
Input RAG — chunks a user's document and retrieves the most relevant
passages for each agent.

Usage:
    rag = InputRAG()
    rag.index_document(user_text)
    chunks = rag.retrieve_for_agent("ARGUS", top_k=8)
    assembled_text = rag.assemble(chunks)
"""
from __future__ import annotations

from .chunker import Chunk, chunk_document, AGENT_QUERIES
from .vector_store import SearchResult, VectorStore


class InputRAG:
    """Facade over chunker + vector store for user-document retrieval."""

    def __init__(
        self,
        max_chunk_tokens: int = 500,
        overlap_tokens: int = 50,
    ) -> None:
        self._store = VectorStore()
        self._chunks: list[Chunk] = []
        self._max_chunk_tokens = max_chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._indexed = False

    def index_document(self, text: str) -> list[Chunk]:
        """Chunk the document and build the TF-IDF index.

        Returns all chunks (useful for inspection / debugging).
        """
        self._chunks = chunk_document(
            text,
            max_chunk_tokens=self._max_chunk_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        self._store.index(self._chunks)
        self._indexed = True
        return self._chunks

    def retrieve_for_agent(
        self,
        agent_name: str,
        top_k: int = 8,
    ) -> list[SearchResult]:
        """Retrieve chunks relevant to a specific agent's bias focus.

        Uses the predefined AGENT_QUERIES to run multiple semantic queries
        and merges results.
        """
        if not self._indexed:
            raise RuntimeError("Call index_document() before retrieval.")

        queries = AGENT_QUERIES.get(agent_name.upper(), [])
        if not queries:
            # Fallback: return the first top_k chunks in document order
            return [
                SearchResult(chunk=c, score=1.0)
                for c in self._chunks[:top_k]
            ]

        results = self._store.query_multi(queries, top_k=top_k, deduplicate=True)

        # Guarantee every agent gets at least min_chunks content
        min_chunks = min(3, len(self._chunks))
        if len(results) < min_chunks:
            seen = {r.chunk.index for r in results}
            for c in self._chunks:
                if c.index not in seen:
                    results.append(SearchResult(chunk=c, score=0.01))
                    seen.add(c.index)
                if len(results) >= min_chunks:
                    break

        return results

    def retrieve_by_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Ad-hoc retrieval with a custom query string."""
        if not self._indexed:
            raise RuntimeError("Call index_document() before retrieval.")
        return self._store.query(query, top_k=top_k)

    @staticmethod
    def assemble(results: list[SearchResult], include_metadata: bool = True) -> str:
        """Reassemble retrieved chunks into a single text block.

        Chunks are sorted by their original document position so the LLM
        sees them in reading order, not relevance order.
        """
        if not results:
            return ""

        # Sort by original chunk index (document order)
        ordered = sorted(results, key=lambda r: r.chunk.index)

        parts: list[str] = []
        for r in ordered:
            c = r.chunk
            if include_metadata:
                header = f"[Section: {c.section} | Chunk {c.index} | Chars {c.char_start}-{c.char_end}]"
                parts.append(f"{header}\n{c.text}")
            else:
                parts.append(c.text)

        return "\n\n---\n\n".join(parts)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_indexed(self) -> bool:
        return self._indexed
