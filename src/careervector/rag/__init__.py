"""Retrieval-augmented generation support for CareerVector."""

from careervector.rag.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
)
from careervector.rag.service import RAGCareerAdvisor, RAGResponse

__all__ = [
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaModelNotFoundError",
    "RAGCareerAdvisor",
    "RAGResponse",
]
