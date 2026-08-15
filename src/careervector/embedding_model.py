from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from careervector.config import (
    ARTIFACT_DIR,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_INFO_PATH,
    EMBEDDING_MATRIX_PATH,
    EMBEDDING_METADATA_PATH,
)
from careervector.profile import CareerProfile


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            'Sentence embeddings are not installed. Run `pip install -e ".[embeddings]"` '
            'or `pip install -e ".[ui]"`.'
        ) from exc
    return SentenceTransformer(model_name)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _number_or_none(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(number) else float(number)


def _pipe_list(value: object, limit: int) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()][:limit]


def _matching_titles(all_titles: object, query_text: str, limit: int = 4) -> list[str]:
    titles = [
        x.strip()
        for x in str(all_titles or "").split("|")
        if x.strip() and x.strip().lower() != "nan"
    ]
    if not titles:
        return []
    query_tokens = {
        token.lower()
        for token in query_text.replace("/", " ").replace("-", " ").split()
        if len(token) > 2
    }
    ranked = sorted(
        titles,
        key=lambda title: sum(token in title.lower() for token in query_tokens),
        reverse=True,
    )
    return ranked[:limit]


def _semantic_document(row: pd.Series) -> str:
    """Build human-readable text for dense retrieval.

    TF-IDF intentionally repeats important phrases. Dense embeddings work better with a natural,
    compact representation, so this builder uses the same O*NET fields without TF repetition.
    """
    sections = [
        f"Career: {row.get('title', '')}.",
        f"Description: {row.get('description', '')}.",
        f"Related job titles: {row.get('job_titles', '')}.",
        f"Interests: {row.get('top_interests', '')}.",
        f"Skills: {row.get('top_skills', '')}.",
        f"Knowledge: {row.get('top_knowledge', '')}.",
        f"Work activities: {row.get('top_activities', '')}.",
        f"Core tasks: {row.get('core_tasks', '')}.",
        f"Software and technologies: {row.get('software_skills', '')}.",
    ]
    return " ".join(str(section) for section in sections if str(section).strip())


def _chunk_words(text: str, *, chunk_words: int = 140, overlap_words: int = 20) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")

    step = chunk_words - overlap_words
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


def _encode_documents(model, texts: list[str], *, batch_size: int) -> np.ndarray:
    # encode_document / encode_query are the current retrieval-specific APIs. Fall back to encode
    # so the project remains compatible with older Sentence Transformers versions.
    encoder = getattr(model, "encode_document", None) or model.encode
    vectors = encoder(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _encode_query(model, text: str) -> np.ndarray:
    encoder = getattr(model, "encode_query", None) or model.encode
    vector = encoder(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vector[0], dtype=np.float32)


class EmbeddingCareerVectorModel:
    """Dense semantic career recommender using a local Sentence Transformer."""

    def __init__(
        self,
        model,
        matrix: np.ndarray,
        metadata: pd.DataFrame,
        *,
        model_name: str,
    ) -> None:
        self.model = model
        self.matrix = _normalize_rows(matrix)
        self.metadata = metadata.reset_index(drop=True)
        self.model_name = model_name
        if self.matrix.shape[0] != len(self.metadata):
            raise ValueError("Embedding row count does not match occupation metadata row count")

    @classmethod
    def build(
        cls,
        occupations: pd.DataFrame,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        chunk_words: int = 140,
        overlap_words: int = 20,
    ) -> "EmbeddingCareerVectorModel":
        model = _load_sentence_transformer(model_name)

        all_chunks: list[str] = []
        owners: list[int] = []
        for row_index, (_, row) in enumerate(occupations.iterrows()):
            text = _semantic_document(row)
            chunks = _chunk_words(
                text,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
            for chunk in chunks:
                all_chunks.append(chunk)
                owners.append(row_index)

        chunk_vectors = _encode_documents(model, all_chunks, batch_size=batch_size)
        matrix = np.zeros((len(occupations), chunk_vectors.shape[1]), dtype=np.float32)
        counts = np.zeros(len(occupations), dtype=np.float32)

        for owner, vector in zip(owners, chunk_vectors, strict=True):
            matrix[owner] += vector
            counts[owner] += 1.0

        counts[counts == 0] = 1.0
        matrix /= counts[:, None]
        matrix = _normalize_rows(matrix)

        metadata = occupations.drop(columns=["document"], errors="ignore").copy()
        return cls(model, matrix, metadata, model_name=model_name)

    def save(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifact_dir / EMBEDDING_MATRIX_PATH.name, self.matrix.astype(np.float32))
        self.metadata.to_csv(artifact_dir / EMBEDDING_METADATA_PATH.name, index=False)
        info = {
            "model": "sentence_transformer_cosine_similarity",
            "sentence_transformer": self.model_name,
            "num_occupations": int(self.matrix.shape[0]),
            "embedding_dimensions": int(self.matrix.shape[1]),
            "normalized_embeddings": True,
            "aggregation": "mean of normalized overlapping document chunks, then renormalized",
        }
        (artifact_dir / EMBEDDING_INFO_PATH.name).write_text(json.dumps(info, indent=2) + "\n")

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> "EmbeddingCareerVectorModel":
        info_path = artifact_dir / EMBEDDING_INFO_PATH.name
        if not info_path.exists():
            raise FileNotFoundError(
                f"{info_path} does not exist. Run `python scripts/build_embeddings.py` first."
            )
        info = json.loads(info_path.read_text())
        model_name = str(info["sentence_transformer"])
        model = _load_sentence_transformer(model_name)
        matrix = np.load(artifact_dir / EMBEDDING_MATRIX_PATH.name)
        metadata = pd.read_csv(artifact_dir / EMBEDDING_METADATA_PATH.name)
        return cls(model, matrix, metadata, model_name=model_name)

    def recommend(
        self,
        profile: CareerProfile,
        *,
        top_k: int = 10,
        avoid_weight: float = 0.35,
        salary_metric: str = "median_salary",
    ) -> list[dict[str, object]]:
        positive_text = profile.semantic_text()
        if not positive_text.strip():
            raise ValueError(
                "Provide at least a major, interest, specialization, preferred-work item, or keyword"
            )

        query_vec = _encode_query(self.model, positive_text)
        positive_scores = self.matrix @ query_vec

        avoid_scores = np.zeros_like(positive_scores)
        if profile.avoid_text().strip():
            avoid_vec = _encode_query(self.model, profile.avoid_text())
            avoid_scores = self.matrix @ avoid_vec

        final_scores = positive_scores - avoid_weight * avoid_scores
        eligible = np.ones(len(self.metadata), dtype=bool)

        if profile.minimum_salary is not None:
            if salary_metric not in self.metadata.columns:
                raise ValueError(f"Unknown salary metric: {salary_metric}")
            salaries = pd.to_numeric(self.metadata[salary_metric], errors="coerce").to_numpy()
            eligible &= np.isfinite(salaries) & (salaries >= float(profile.minimum_salary))

        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices) == 0:
            return []
        ranked_indices = eligible_indices[np.argsort(final_scores[eligible_indices])[::-1]][:top_k]

        results: list[dict[str, object]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            row = self.metadata.iloc[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "engine": "embeddings",
                    "onet_soc_code": row.get("onet_soc_code"),
                    "occupation": row.get("title"),
                    "match_score": round(float(max(0.0, final_scores[idx])) * 100, 2),
                    "similarity": round(float(positive_scores[idx]) * 100, 2),
                    "avoid_penalty": round(float(avoid_scores[idx]) * avoid_weight * 100, 2),
                    "median_salary": _number_or_none(row.get("median_salary")),
                    "mean_salary": _number_or_none(row.get("mean_salary")),
                    "sample_job_titles": _matching_titles(row.get("job_titles"), positive_text),
                    "description": row.get("description"),
                    "top_interests": _pipe_list(row.get("top_interests"), 5),
                    "top_skills": _pipe_list(row.get("top_skills"), 5),
                    "top_knowledge": _pipe_list(row.get("top_knowledge"), 5),
                }
            )
        return results
