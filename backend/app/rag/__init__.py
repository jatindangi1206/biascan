"""
BiasScan RAG package.

InputRAG chunks the user's document and reassembles it in reading order so the
full text is passed cleanly to each agent without exceeding context windows.
"""
from .input_rag import InputRAG

__all__ = ["InputRAG"]
