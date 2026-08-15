import numpy as np
import pytest

from careervector.embedding_model import _chunk_words, _normalize_rows


def test_chunk_words_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(12))
    chunks = _chunk_words(text, chunk_words=5, overlap_words=2)
    assert chunks[0] == "w0 w1 w2 w3 w4"
    assert chunks[1] == "w3 w4 w5 w6 w7"


def test_chunk_words_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        _chunk_words("hello world", chunk_words=5, overlap_words=5)


def test_normalize_rows() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = _normalize_rows(matrix)
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.0, 0.0])


def test_embedding_recommender_ranks_semantic_vector() -> None:
    import pandas as pd

    from careervector.embedding_model import EmbeddingCareerVectorModel
    from careervector.profile import CareerProfile

    class FakeModel:
        def encode_query(self, texts, **kwargs):
            text = texts[0].lower()
            if "hardware" in text or "computer engineering" in text:
                return np.array([[1.0, 0.0]], dtype=np.float32)
            return np.array([[0.0, 1.0]], dtype=np.float32)

    metadata = pd.DataFrame(
        [
            {
                "onet_soc_code": "17-2061.00",
                "title": "Computer Hardware Engineers",
                "description": "Design computer hardware.",
                "job_titles": "FPGA Engineer | Hardware Engineer",
                "top_interests": "Engineering",
                "top_skills": "Critical Thinking",
                "top_knowledge": "Computers and Electronics",
                "median_salary": 140000,
                "mean_salary": 150000,
            },
            {
                "onet_soc_code": "27-1013.00",
                "title": "Fine Artists",
                "description": "Create visual art.",
                "job_titles": "Artist",
                "top_interests": "Visual Arts",
                "top_skills": "Creativity",
                "top_knowledge": "Fine Arts",
                "median_salary": 80000,
                "mean_salary": 90000,
            },
        ]
    )
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = EmbeddingCareerVectorModel(
        FakeModel(), matrix, metadata, model_name="fake-model"
    )
    profile = CareerProfile(major="Computer Engineering", interests=["hardware"])
    results = model.recommend(profile, top_k=1)
    assert results[0]["onet_soc_code"] == "17-2061.00"
