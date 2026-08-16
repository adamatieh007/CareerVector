from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from careervector.config import (
    ARTIFACT_DIR,
    ARTIFACT_SCHEMA_VERSION,
    CAREER_CORPUS_KIND,
    MATRIX_PATH,
    METADATA_PATH,
    MODEL_INFO_PATH,
    VECTORIZER_PATH,
)
from careervector.profile import CareerProfile
from careervector.ranking import academic_matches_for_row, combine_relevance_scores, select_diverse_indices


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
            raise ValueError("TF-IDF row count does not match career-role metadata row count")

    @classmethod
    def train(cls, occupations: pd.DataFrame, *, max_features: int = 125_000) -> "CareerVectorModel":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.985,
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
            "model": "tfidf_hybrid_career_role_ranker",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "corpus_kind": CAREER_CORPUS_KIND,
            "num_roles": int(self.matrix.shape[0]),
            "num_features": int(self.matrix.shape[1]),
            "ranking": {
                "retrieval_weight": 0.58,
                "academic_weight": 0.24,
                "title_weight": 0.13,
                "outlook_weight": 0.05,
                "academic_source": "NCES CIP 2020 to SOC 2018 crosswalk",
                "outlook_source": "BLS Employment Projections 2024-2034",
            },
        }
        (artifact_dir / MODEL_INFO_PATH.name).write_text(json.dumps(info, indent=2) + "\n")

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> "CareerVectorModel":
        info_path = artifact_dir / MODEL_INFO_PATH.name
        if not info_path.exists():
            raise FileNotFoundError(
                f"{info_path} does not exist. Run `python scripts/train_tfidf.py` first."
            )
        info = json.loads(info_path.read_text())
        if (
            info.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
            or info.get("corpus_kind") != CAREER_CORPUS_KIND
        ):
            raise RuntimeError(
                "Your TF-IDF artifacts were built by an older CareerVector corpus. "
                "v0.4 ranks specific career roles, so rebuild with: "
                "`python scripts/build_dataset.py` then `python scripts/train_tfidf.py`."
            )
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
        ranked = sorted(titles, key=lambda title: sum(token in title.lower() for token in query_tokens), reverse=True)
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
            raise ValueError("Provide at least a major, concentration, interest, skill, specialization, preferred-work item, or keyword")

        query_vec = self.vectorizer.transform([positive_text]).tocsr()
        positive_scores = cosine_similarity(query_vec, self.matrix).ravel()

        avoid_scores = np.zeros_like(positive_scores)
        if profile.avoid_text().strip():
            avoid_vec = self.vectorizer.transform([profile.avoid_text()]).tocsr()
            avoid_scores = cosine_similarity(avoid_vec, self.matrix).ravel()

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
                    "engine": "tfidf",
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
                    "sample_job_titles": self._matching_titles(row.get("job_titles"), positive_text),
                    "matched_terms": self._top_shared_terms(query_vec, int(idx)),
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
