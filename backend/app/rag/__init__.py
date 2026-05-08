"""
BiasScan RAG package.

Two-RAG architecture:
  - InputRAG: chunks user documents, retrieves relevant passages per agent
  - EvidenceRAG: indexes bias patterns + dataset exemplars for cross-checking
"""
from .input_rag import InputRAG
from .evidence_rag import EvidenceRAG

__all__ = ["InputRAG", "EvidenceRAG"]
