from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from careervector.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
from careervector.embedding_model import EmbeddingCareerVectorModel
from careervector.profile import CareerProfile
from careervector.rag.context import build_rag_context
from careervector.rag.ollama import OllamaClient
from careervector.rag.prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, object]]
    generator_model: str
    retrieval_engine: str = "sentence_embeddings"


class RAGCareerAdvisor:
    """CareerVector's retrieval-augmented generation pipeline.

    Retrieval remains deterministic and independently testable. The local LLM
    receives only the user's structured profile and the evidence returned by the
    existing embedding retriever.
    """

    def __init__(
        self,
        retriever: EmbeddingCareerVectorModel,
        generator: Any,
        *,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.2,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.model_name = model_name
        self.temperature = float(temperature)

    @classmethod
    def load(
        cls,
        *,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 120.0,
    ) -> "RAGCareerAdvisor":
        return cls(
            EmbeddingCareerVectorModel.load(),
            OllamaClient(ollama_base_url, timeout=timeout),
            model_name=model_name,
        )

    def advise(
        self,
        profile: CareerProfile,
        *,
        top_k: int = 5,
        avoid_weight: float = 0.35,
        salary_metric: str = "median_salary",
    ) -> RAGResponse:
        sources = self.retriever.recommend(
            profile,
            top_k=top_k,
            avoid_weight=avoid_weight,
            salary_metric=salary_metric,
        )
        if not sources:
            return RAGResponse(
                answer="",
                sources=[],
                generator_model=self.model_name,
            )

        context = build_rag_context(sources)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(profile, context)},
        ]
        answer = self.generator.chat(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return RAGResponse(
            answer=answer,
            sources=sources,
            generator_model=self.model_name,
        )
