"""
Lightweight vector store using TF-IDF + cosine similarity.

No external embedding model required — uses scikit-learn's TfidfVectorizer
with numpy for cosine similarity. Drop-in replaceable with FAISS or
sentence-transformers when heavier hardware is available.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .chunker import Chunk


@dataclass
class SearchResult:
    """A single retrieval result."""
    chunk: Chunk
    score: float          # cosine similarity, 0–1


class VectorStore:
    """TF-IDF vector store with cosine-similarity retrieval.

    Lifecycle:
        store = VectorStore()
        store.index(chunks)          # builds vocabulary + TF-IDF matrix
        results = store.query("...", top_k=5)
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vocab: dict[str, int] = {}     # token → column index
        self._idf: np.ndarray | None = None  # (V,)
        self._tfidf: np.ndarray | None = None  # (N, V) L2-normalised rows

    # ── indexing ────────────────────────────────────────────────────────

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build TF-IDF matrix from chunks."""
        self._chunks = list(chunks)
        if not self._chunks:
            return

        # Tokenise each chunk
        token_lists: list[list[str]] = [_tokenise(c.text) for c in self._chunks]

        # Build vocabulary from all tokens
        vocab_set: set[str] = set()
        for tl in token_lists:
            vocab_set.update(tl)
        self._vocab = {tok: i for i, tok in enumerate(sorted(vocab_set))}
        V = len(self._vocab)
        N = len(self._chunks)

        if V == 0:
            self._tfidf = np.zeros((N, 1))
            self._idf = np.ones(1)
            return

        # Term frequency matrix (N, V)
        tf = np.zeros((N, V), dtype=np.float32)
        for row, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                col = self._vocab.get(tok)
                if col is not None:
                    tf[row, col] = cnt

        # Sub-linear TF: 1 + log(tf) for tf > 0
        mask = tf > 0
        tf[mask] = 1.0 + np.log(tf[mask])

        # IDF: log(N / df) + 1  (smoothed)
        df = np.count_nonzero(tf, axis=0).astype(np.float32)
        df = np.maximum(df, 1.0)  # avoid division by zero
        self._idf = np.log(N / df) + 1.0

        # TF-IDF
        tfidf = tf * self._idf  # (N, V)

        # L2-normalise each row
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._tfidf = tfidf / norms

    # ── querying ────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5) -> list[SearchResult]:
        """Return top-k chunks most similar to the query."""
        if self._tfidf is None or len(self._chunks) == 0:
            return []

        q_vec = self._vectorise(query_text)
        if q_vec is None:
            return []

        # Cosine similarity: dot product (rows are already L2-normed)
        scores = self._tfidf @ q_vec  # (N,)

        # Top-k
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[SearchResult] = []
        for idx in top_indices:
            s = float(scores[idx])
            if s > 0:
                results.append(SearchResult(chunk=self._chunks[idx], score=s))

        return results

    def query_multi(
        self,
        queries: list[str],
        top_k: int = 5,
        deduplicate: bool = True,
    ) -> list[SearchResult]:
        """Run multiple queries and merge results, keeping top_k overall."""
        seen_indices: set[int] = set()
        all_results: list[SearchResult] = []

        for q in queries:
            for r in self.query(q, top_k=top_k):
                if deduplicate and r.chunk.index in seen_indices:
                    continue
                seen_indices.add(r.chunk.index)
                all_results.append(r)

        # Sort by score descending, keep top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    # ── internal ────────────────────────────────────────────────────────

    def _vectorise(self, text: str) -> np.ndarray | None:
        """Convert text to a TF-IDF vector using the stored vocabulary."""
        tokens = _tokenise(text)
        if not tokens or not self._vocab:
            return None

        V = len(self._vocab)
        vec = np.zeros(V, dtype=np.float32)
        counts = Counter(tokens)
        for tok, cnt in counts.items():
            col = self._vocab.get(tok)
            if col is not None:
                vec[col] = 1.0 + math.log(cnt) if cnt > 0 else 0.0

        vec *= self._idf  # type: ignore[arg-type]

        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return None
        return vec / norm

    @property
    def size(self) -> int:
        return len(self._chunks)


# ── tokenisation ────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)

# Common English stop words (kept minimal)
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "we", "i", "you",
    "not", "no", "as", "if", "so", "than", "very", "just", "also", "about",
    "which", "what", "who", "whom", "when", "where", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "into", "over", "after", "before", "between", "under",
    "again", "further", "then", "once", "here", "there", "any", "up", "out",
})


def _tokenise(text: str) -> list[str]:
    """Tokenise text to lowercase alpha-numeric tokens, removing stop words."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
