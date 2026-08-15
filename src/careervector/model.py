from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from careervector.config import (
    ARTIFACT_DIR,
    MATRIX_PATH,
    METADATA_PATH,
    MODEL_INFO_PATH,
    VECTORIZER_PATH,
)
from careervector.profile import CareerProfile


class CareerVectorModel:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix: sparse.csr_matrix,
        metadata: pd.DataFrame,
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix.tocsr()
        self.metadata = metadata.reset_index(drop=True)
        if self.matrix.shape[0] != len(self.metadata):
            raise ValueError("TF-IDF row count does not match occupation metadata row count")

    @classmethod
    def train(
        cls,
        occupations: pd.DataFrame,
        *,
        max_features: int = 100_000,
    ) -> "CareerVectorModel":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(occupations["document"].fillna("").astype(str)).tocsr()
        metadata = occupations.drop(columns=["document"]).copy()
        return cls(vectorizer, matrix, metadata)

    def save(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, artifact_dir / VECTORIZER_PATH.name)
        sparse.save_npz(artifact_dir / MATRIX_PATH.name, self.matrix)
        self.metadata.to_csv(artifact_dir / METADATA_PATH.name, index=False)
        info = {
            "model": "tfidf_cosine_similarity",
            "num_occupations": int(self.matrix.shape[0]),
            "num_features": int(self.matrix.shape[1]),
            "vectorizer": {
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "norm": "l2",
                "stop_words": "english",
            },
        }
        (artifact_dir / MODEL_INFO_PATH.name).write_text(json.dumps(info, indent=2) + "\n")

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> "CareerVectorModel":
        vectorizer = joblib.load(artifact_dir / VECTORIZER_PATH.name)
        matrix = sparse.load_npz(artifact_dir / MATRIX_PATH.name).tocsr()
        metadata = pd.read_csv(artifact_dir / METADATA_PATH.name)
        return cls(vectorizer, matrix, metadata)

    def _top_shared_terms(self, query_vec: sparse.csr_matrix, row_index: int, limit: int = 8) -> list[str]:
        overlap = self.matrix[row_index].multiply(query_vec)
        if overlap.nnz == 0:
            return []
        order = np.argsort(overlap.data)[::-1][:limit]
        feature_names = self.vectorizer.get_feature_names_out()
        return [str(feature_names[overlap.indices[i]]) for i in order]

    @staticmethod
    def _matching_titles(all_titles: object, query_text: str, limit: int = 4) -> list[str]:
        titles = [x.strip() for x in str(all_titles or "").split("|") if x.strip() and x.strip().lower() != "nan"]
        if not titles:
            return []
        query_tokens = {t.lower() for t in query_text.replace("/", " ").replace("-", " ").split() if len(t) > 2}
        ranked = sorted(
            titles,
            key=lambda title: sum(token in title.lower() for token in query_tokens),
            reverse=True,
        )
        return ranked[:limit]

    def recommend(
        self,
        profile: CareerProfile,
        *,
        top_k: int = 10,
        avoid_weight: float = 0.35,
        salary_metric: str = "median_salary",
    ) -> list[dict[str, object]]:
        positive_text = profile.positive_text()
        if not positive_text.strip():
            raise ValueError("Provide at least a major, interest, specialization, preferred-work item, or keyword")

        query_vec = self.vectorizer.transform([positive_text]).tocsr()
        positive_scores = cosine_similarity(query_vec, self.matrix).ravel()

        avoid_scores = np.zeros_like(positive_scores)
        if profile.avoid_text().strip():
            avoid_vec = self.vectorizer.transform([profile.avoid_text()]).tocsr()
            avoid_scores = cosine_similarity(avoid_vec, self.matrix).ravel()

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
                    "engine": "tfidf",
                    "onet_soc_code": row.get("onet_soc_code"),
                    "occupation": row.get("title"),
                    "match_score": round(float(max(0.0, final_scores[idx])) * 100, 2),
                    "similarity": round(float(positive_scores[idx]) * 100, 2),
                    "avoid_penalty": round(float(avoid_scores[idx]) * avoid_weight * 100, 2),
                    "median_salary": _number_or_none(row.get("median_salary")),
                    "mean_salary": _number_or_none(row.get("mean_salary")),
                    "sample_job_titles": self._matching_titles(row.get("job_titles"), positive_text),
                    "matched_terms": self._top_shared_terms(query_vec, int(idx)),
                    "description": row.get("description"),
                    "top_interests": _pipe_list(row.get("top_interests"), 5),
                    "top_skills": _pipe_list(row.get("top_skills"), 5),
                    "top_knowledge": _pipe_list(row.get("top_knowledge"), 5),
                }
            )
        return results


def _number_or_none(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(number) else float(number)


def _pipe_list(value: object, limit: int) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()][:limit]
