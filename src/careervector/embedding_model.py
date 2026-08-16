from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from careervector.config import (
    ARTIFACT_DIR,
    ARTIFACT_SCHEMA_VERSION,
    CAREER_CORPUS_KIND,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_INFO_PATH,
    EMBEDDING_MATRIX_PATH,
    EMBEDDING_METADATA_PATH,
)
from careervector.profile import CareerProfile
from careervector.ranking import academic_matches_for_row, combine_relevance_scores, select_diverse_indices


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
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


def _text_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _pipe_list(value: object, limit: int) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()][:limit]


def _matching_titles(all_titles: object, query_text: str, limit: int = 4) -> list[str]:
    titles = [x.strip() for x in str(all_titles or "").split("|") if x.strip() and x.strip().lower() != "nan"]
    if not titles:
        return []
    query_tokens = {
        token.lower()
        for token in query_text.replace("/", " ").replace("-", " ").split()
        if len(token) > 2
    }
    ranked = sorted(titles, key=lambda title: sum(token in title.lower() for token in query_tokens), reverse=True)
    return ranked[:limit]


def _semantic_document(row: pd.Series) -> str:
    """Natural representation for dense retrieval over granular career roles."""
    role = row.get("title", "")
    parent = row.get("parent_title", "")
    source = row.get("source", "")
    sections = [
        f"Career role: {role}.",
        f"Occupation family: {parent}." if parent and str(parent) != str(role) else "",
        f"Source taxonomy: {source}.",
        f"Description: {row.get('description', '')}.",
        f"Related job titles: {row.get('job_titles', '')}.",
        f"Compatible majors and fields of study: {row.get('compatible_majors', '')}.",
        f"Interests: {row.get('top_interests', '')}.",
        f"Skills: {row.get('top_skills', '')}.",
        f"Knowledge: {row.get('top_knowledge', '')}.",
        f"Work activities: {row.get('top_activities', '')}.",
        f"Core tasks: {row.get('core_tasks', '')}.",
        f"Software and technologies: {row.get('software_skills', '')}.",
        f"Typical education: {row.get('typical_education', '')}.",
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
    """Dense semantic recommender over specific career-role records."""

    def __init__(self, model, matrix: np.ndarray, metadata: pd.DataFrame, *, model_name: str) -> None:
        self.model = model
        self.matrix = _normalize_rows(matrix)
        self.metadata = metadata.reset_index(drop=True)
        self.model_name = model_name
        if self.matrix.shape[0] != len(self.metadata):
            raise ValueError("Embedding row count does not match career-role metadata row count")

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
            chunks = _chunk_words(text, chunk_words=chunk_words, overlap_words=overlap_words)
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
            "model": "sentence_transformer_hybrid_career_role_ranker",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "corpus_kind": CAREER_CORPUS_KIND,
            "sentence_transformer": self.model_name,
            "num_roles": int(self.matrix.shape[0]),
            "embedding_dimensions": int(self.matrix.shape[1]),
            "normalized_embeddings": True,
            "aggregation": "mean of normalized overlapping document chunks, then renormalized",
            "ranking": {
                "retrieval_weight": 0.58,
                "academic_weight": 0.24,
                "title_weight": 0.13,
                "outlook_weight": 0.05,
            },
        }
        (artifact_dir / EMBEDDING_INFO_PATH.name).write_text(json.dumps(info, indent=2) + "\n")

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> "EmbeddingCareerVectorModel":
        info_path = artifact_dir / EMBEDDING_INFO_PATH.name
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} does not exist. Run `python scripts/build_embeddings.py` first.")
        info = json.loads(info_path.read_text())
        if (
            info.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
            or info.get("corpus_kind") != CAREER_CORPUS_KIND
        ):
            raise RuntimeError(
                "Your embedding artifacts were built by an older CareerVector corpus. "
                "v0.4 ranks specific career roles, so rebuild with: "
                "`python scripts/build_dataset.py` then `python scripts/build_embeddings.py`."
            )
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
                "Provide at least a major, concentration, interest, skill, specialization, preferred-work item, or keyword"
            )

        query_vec = _encode_query(self.model, positive_text)
        positive_scores = self.matrix @ query_vec

        avoid_scores = np.zeros_like(positive_scores)
        if profile.avoid_text().strip():
            avoid_vec = _encode_query(self.model, profile.avoid_text())
            avoid_scores = self.matrix @ avoid_vec

        retrieval_scores = positive_scores - avoid_weight * avoid_scores
        combined_scores, academic_scores, title_scores, outlook = combine_relevance_scores(
            retrieval_scores, self.metadata, profile
        )
        eligible = np.ones(len(self.metadata), dtype=bool)

        if profile.minimum_salary is not None:
            if salary_metric not in self.metadata.columns:
                raise ValueError(f"Unknown salary metric: {salary_metric}")
            salaries = pd.to_numeric(self.metadata[salary_metric], errors="coerce").to_numpy()
            eligible &= np.isfinite(salaries) & (salaries >= float(profile.minimum_salary))

        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices) == 0:
            return []
        ranked_indices = select_diverse_indices(combined_scores, eligible_indices, self.metadata, top_k=top_k)

        results: list[dict[str, object]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            row = self.metadata.iloc[int(idx)]
            academic = float(academic_scores[idx]) if np.isfinite(academic_scores[idx]) else None
            title_score = float(title_scores[idx]) if np.isfinite(title_scores[idx]) else None
            outlook_score = float(outlook[idx]) if np.isfinite(outlook[idx]) else None
            results.append(
                {
                    "rank": rank,
                    "engine": "embeddings",
                    "role_id": row.get("role_id"),
                    "source": row.get("source", "O*NET"),
                    "role_kind": row.get("role_kind"),
                    "onet_soc_code": row.get("onet_soc_code"),
                    "occupation": row.get("title"),
                    "parent_occupation": row.get("parent_title"),
                    "match_score": round(float(max(0.0, combined_scores[idx])) * 100, 2),
                    "similarity": round(float(positive_scores[idx]) * 100, 2),
                    "retrieval_score": round(float(max(0.0, retrieval_scores[idx])) * 100, 2),
                    "academic_alignment": round(academic * 100, 2) if academic is not None else None,
                    "title_alignment": round(title_score * 100, 2) if title_score is not None else None,
                    "outlook_score": round(outlook_score * 100, 2) if outlook_score is not None else None,
                    "avoid_penalty": round(float(avoid_scores[idx]) * avoid_weight * 100, 2),
                    "median_salary": _number_or_none(row.get("median_salary")),
                    "mean_salary": _number_or_none(row.get("mean_salary")),
                    "growth_percent": _number_or_none(row.get("growth_percent")),
                    "annual_openings": _number_or_none(row.get("annual_openings")),
                    "typical_education": _text_or_none(row.get("typical_education")),
                    "academic_matches": academic_matches_for_row(row, profile),
                    "sample_job_titles": _matching_titles(row.get("job_titles"), positive_text),
                    "description": row.get("description"),
                    "top_interests": _pipe_list(row.get("top_interests"), 5),
                    "top_skills": _pipe_list(row.get("top_skills"), 5),
                    "top_knowledge": _pipe_list(row.get("top_knowledge"), 5),
                    "top_activities": _pipe_list(row.get("top_activities"), 5),
                    "core_tasks": _pipe_list(row.get("core_tasks"), 6),
                    "software_skills": _pipe_list(row.get("software_skills"), 8),
                    "p10_salary": _number_or_none(row.get("p10_salary")),
                    "p90_salary": _number_or_none(row.get("p90_salary")),
                }
            )
        return results
