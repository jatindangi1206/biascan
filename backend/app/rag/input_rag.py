"""
Input RAG — chunks a user's document and retrieves the most relevant
passages for each agent, with adjacent-chunk expansion for context.

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
    """Facade over chunker + hybrid vector store for user-document retrieval.

    Features (from LLM-RAG-Architecture):
      - Hybrid search: TF-IDF dense + BM25 sparse with RRF fusion
      - Adjacent chunk expansion: ±N neighbouring chunks for context
      - Per-agent query routing: each agent gets chunks matching its focus
    """

    def __init__(
        self,
        max_chunk_tokens: int = 500,
        overlap_tokens: int = 50,
        adjacent_chunks: int = 1,
    ) -> None:
        self._store = VectorStore()
        self._chunks: list[Chunk] = []
        self._max_chunk_tokens = max_chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._adjacent_chunks = adjacent_chunks
        self._indexed = False

        # Section-grouped index for fast adjacent lookup
        self._section_chunks: dict[str, list[Chunk]] = {}

    def index_document(self, text: str) -> list[Chunk]:
        """Chunk the document and build the hybrid index.

        Returns all chunks (useful for inspection / debugging).
        """
        self._chunks = chunk_document(
            text,
            max_chunk_tokens=self._max_chunk_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        self._store.index(self._chunks)
        self._indexed = True

        # Build section-grouped lookup for adjacent expansion
        self._section_chunks.clear()
        for c in self._chunks:
            self._section_chunks.setdefault(c.section, []).append(c)
        # Sort each section's chunks by index
        for sec_chunks in self._section_chunks.values():
            sec_chunks.sort(key=lambda c: c.index)

        return self._chunks

    def retrieve_for_agent(
        self,
        agent_name: str,
        top_k: int = 8,
        expand_adjacent: bool = True,
    ) -> list[SearchResult]:
        """Retrieve chunks relevant to a specific agent's bias focus.

        Uses predefined AGENT_QUERIES for hybrid multi-query retrieval,
        then expands with adjacent chunks for context continuity.
        """
        if not self._indexed:
            raise RuntimeError("Call index_document() before retrieval.")

        queries = AGENT_QUERIES.get(agent_name.upper(), [])
        if not queries:
            return [
                SearchResult(chunk=c, score=1.0)
                for c in self._chunks[:top_k]
            ]

        # Retrieve with hybrid search (TF-IDF + BM25 + RRF)
        results = self._store.query_multi(queries, top_k=top_k, deduplicate=True)

        # Adjacent chunk expansion (from LLM-RAG-Architecture)
        if expand_adjacent and self._adjacent_chunks > 0:
            results = self._expand_adjacent(results)

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

    # ── Adjacent Chunk Expansion ────────────────────────────────────────

    def _expand_adjacent(
        self, results: list[SearchResult],
    ) -> list[SearchResult]:
        """For each retrieved chunk, also include ±N adjacent chunks from
        the same section. This gives agents surrounding context — critical
        for bias patterns that span paragraph boundaries.

        Ported from QdrantHybridEmbeddingStore.ExpandWithAdjacentChunksAsync.
        """
        seen_indices: set[int] = {r.chunk.index for r in results}
        expanded = list(results)

        for r in results:
            section = r.chunk.section
            sec_chunks = self._section_chunks.get(section, [])
            if not sec_chunks:
                continue

            # Find position of this chunk in its section
            pos = None
            for i, c in enumerate(sec_chunks):
                if c.index == r.chunk.index:
                    pos = i
                    break
            if pos is None:
                continue

            # Grab ±adjacent_chunks neighbours
            for offset in range(-self._adjacent_chunks, self._adjacent_chunks + 1):
                if offset == 0:
                    continue
                adj_pos = pos + offset
                if 0 <= adj_pos < len(sec_chunks):
                    adj_chunk = sec_chunks[adj_pos]
                    if adj_chunk.index not in seen_indices:
                        seen_indices.add(adj_chunk.index)
                        # Adjacent chunks get a diminished score
                        expanded.append(SearchResult(
                            chunk=adj_chunk,
                            score=r.score * 0.5,
                        ))

        return expanded

    # ── Assembly ────────────────────────────────────────────────────────

    @staticmethod
    def assemble(
        results: list[SearchResult],
        include_metadata: bool = True,
        agent_name: str | None = None,
    ) -> str:
        """Reassemble retrieved chunks into a structured text block.

        Chunks are sorted by their original document position so the LLM
        sees them in reading order, not relevance order.

        The output format is optimized for LLM comprehension:
        - Section headers give document structure context
        - Relevance scores help the LLM prioritize attention
        - Contiguous chunks from the same section are merged to reduce
          token overhead from repeated headers
        - A preamble tells the agent what it's looking at

        Args:
            results: Retrieved search results with chunks and scores.
            include_metadata: Whether to include section/position headers.
            agent_name: If provided, adds a context preamble for the agent.
        """
        if not results:
            return ""

        ordered = sorted(results, key=lambda r: r.chunk.index)

        parts: list[str] = []

        # Preamble: orient the LLM about what it's reading
        if agent_name and include_metadata:
            n_chunks = len(ordered)
            sections = sorted(set(r.chunk.section for r in ordered))
            top_score = max(r.score for r in ordered) if ordered else 0
            parts.append(
                f"[DOCUMENT EXCERPTS for {agent_name} analysis | "
                f"{n_chunks} passages | "
                f"Sections: {', '.join(sections)} | "
                f"Top relevance: {top_score:.3f}]"
            )

        # Group contiguous same-section chunks to reduce header repetition
        current_section = None
        for r in ordered:
            c = r.chunk
            if include_metadata:
                if c.section != current_section:
                    # New section header
                    current_section = c.section
                    relevance_tag = (
                        "HIGH" if r.score > 0.01 else
                        "MED" if r.score > 0.005 else
                        "LOW"
                    )
                    header = (
                        f"[Section: {c.section} | "
                        f"Chunk {c.index} | "
                        f"Relevance: {relevance_tag}]"
                    )
                    parts.append(f"{header}\n{c.text}")
                else:
                    # Same section, just append text with a light separator
                    parts.append(c.text)
            else:
                parts.append(c.text)

        return "\n\n---\n\n".join(parts)

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_indexed(self) -> bool:
        return self._indexed
