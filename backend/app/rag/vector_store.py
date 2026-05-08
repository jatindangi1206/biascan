"""
Hybrid vector store: TF-IDF dense + BM25 sparse with RRF fusion.

Inspired by matt-bentley/LLM-RAG-Architecture — ported from C#/Qdrant to
pure Python/numpy. No external embedding model or vector DB required.

Two retrieval paths run in parallel:
  1. TF-IDF cosine similarity (semantic — good for paraphrased queries)
  2. BM25 keyword scoring (lexical — good for exact terminology like
     "may, might, could" that TF-IDF normalises away)

Results are fused using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import hashlib
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
    score: float          # fused score (higher = more relevant)


# ═══════════════════════════════════════════════════════════════════════
#  Hybrid Vector Store
# ═══════════════════════════════════════════════════════════════════════

class VectorStore:
    """Hybrid TF-IDF + BM25 store with RRF fusion.

    Lifecycle:
        store = VectorStore()
        store.index(chunks)
        results = store.query("...", top_k=5)
    """

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        self._chunks: list[Chunk] = []

        # Dense (TF-IDF cosine)
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._tfidf: np.ndarray | None = None  # (N, V) L2-normed rows

        # Sparse (BM25)
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._doc_token_counts: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_len: float = 0.0
        self._df: Counter = Counter()  # document frequency per token
        self._N: int = 0

        # Fusion weights
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    # ── indexing ────────────────────────────────────────────────────────

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build both TF-IDF and BM25 indices from chunks."""
        self._chunks = list(chunks)
        if not self._chunks:
            return

        token_lists: list[list[str]] = [_tokenise(c.text) for c in self._chunks]
        self._N = len(self._chunks)

        # ── Build vocabulary ──
        vocab_set: set[str] = set()
        for tl in token_lists:
            vocab_set.update(tl)
        self._vocab = {tok: i for i, tok in enumerate(sorted(vocab_set))}
        V = len(self._vocab)

        if V == 0:
            self._tfidf = np.zeros((self._N, 1))
            self._idf = np.ones(1)
            return

        # ── Dense: TF-IDF matrix ──
        tf = np.zeros((self._N, V), dtype=np.float32)
        for row, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            for tok, cnt in counts.items():
                col = self._vocab.get(tok)
                if col is not None:
                    tf[row, col] = cnt

        # Sub-linear TF: 1 + log(tf)
        mask = tf > 0
        tf_log = tf.copy()
        tf_log[mask] = 1.0 + np.log(tf[mask])

        # IDF: log(N / df) + 1
        df_arr = np.count_nonzero(tf, axis=0).astype(np.float32)
        df_arr = np.maximum(df_arr, 1.0)
        self._idf = np.log(self._N / df_arr) + 1.0

        tfidf = tf_log * self._idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._tfidf = tfidf / norms

        # ── Sparse: BM25 statistics ──
        self._doc_token_counts = [Counter(tl) for tl in token_lists]
        self._doc_lengths = [len(tl) for tl in token_lists]
        self._avg_doc_len = sum(self._doc_lengths) / max(self._N, 1)

        self._df = Counter()
        for tl in token_lists:
            for tok in set(tl):
                self._df[tok] += 1

    # ── querying ────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5) -> list[SearchResult]:
        """Hybrid query: TF-IDF + BM25 fused with RRF."""
        if self._tfidf is None or len(self._chunks) == 0:
            return []

        tokens = _tokenise(query_text)
        if not tokens:
            return []

        # 1. Dense retrieval (TF-IDF cosine)
        dense_ranking = self._dense_query(tokens, top_k=top_k * 3)

        # 2. Sparse retrieval (BM25)
        sparse_ranking = self._bm25_query(tokens, top_k=top_k * 3)

        # 3. RRF fusion
        return self._rrf_fuse(dense_ranking, sparse_ranking, top_k)

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

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    # ── Dense (TF-IDF) ──────────────────────────────────────────────────

    def _dense_query(
        self, tokens: list[str], top_k: int,
    ) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_score) ranked by TF-IDF similarity."""
        q_vec = self._vectorise_tokens(tokens)
        if q_vec is None:
            return []

        scores = self._tfidf @ q_vec  # type: ignore[union-attr]
        k = min(top_k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _vectorise_tokens(self, tokens: list[str]) -> np.ndarray | None:
        if not self._vocab:
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

    # ── Sparse (BM25) ───────────────────────────────────────────────────

    def _bm25_query(
        self, tokens: list[str], top_k: int,
    ) -> list[tuple[int, float]]:
        """Return (chunk_index, bm25_score) ranked by BM25."""
        query_tokens = Counter(tokens)
        scores: list[float] = []

        for doc_idx in range(self._N):
            score = 0.0
            doc_counts = self._doc_token_counts[doc_idx]
            doc_len = self._doc_lengths[doc_idx]

            for tok, qtf in query_tokens.items():
                tf = doc_counts.get(tok, 0)
                if tf == 0:
                    continue
                df = self._df.get(tok, 0)
                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
                # TF saturation: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
                tf_sat = (tf * (self._bm25_k1 + 1)) / (
                    tf + self._bm25_k1 * (
                        1 - self._bm25_b + self._bm25_b * doc_len / max(self._avg_doc_len, 1)
                    )
                )
                score += idf * tf_sat

            scores.append(score)

        # Top-k
        k = min(top_k, self._N)
        arr = np.array(scores)
        top_idx = np.argsort(arr)[::-1][:k]
        return [(int(i), float(arr[i])) for i in top_idx if arr[i] > 0]

    # ── RRF Fusion ──────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense_ranking: list[tuple[int, float]],
        sparse_ranking: list[tuple[int, float]],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of two ranked lists.

        RRF score = w_dense * 1/(k + rank_dense) + w_sparse * 1/(k + rank_sparse)
        """
        fused: dict[int, float] = {}

        for rank, (idx, _score) in enumerate(dense_ranking):
            fused[idx] = fused.get(idx, 0.0) + self._dense_weight / (rrf_k + rank + 1)

        for rank, (idx, _score) in enumerate(sparse_ranking):
            fused[idx] = fused.get(idx, 0.0) + self._sparse_weight / (rrf_k + rank + 1)

        # Sort by fused score
        sorted_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in sorted_items[:top_k]:
            results.append(SearchResult(chunk=self._chunks[idx], score=score))

        return results

    @property
    def size(self) -> int:
        return len(self._chunks)


# ═══════════════════════════════════════════════════════════════════════
#  Tokenisation
# ═══════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)

# Stop words — deliberately EXCLUDE hedging/certainty words that agents need
# to find.  "may", "might", "could", "should" are kept because LIBRA searches
# for them. Standard function words are still removed.
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "i", "you", "not", "no", "as", "if", "so", "than", "very",
    "just", "also", "about", "which", "what", "who", "whom", "when",
    "where", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "into", "over",
    "after", "before", "between", "under", "again", "further", "then",
    "once", "here", "there", "any", "up", "out",
})
# NOTE: "may", "might", "could", "should", "can", "shall" are intentionally
# NOT in the stop list so BM25 can match LIBRA's hedging queries.


def _tokenise(text: str) -> list[str]:
    """Tokenise text to lowercase alpha-numeric tokens, removing stop words."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
