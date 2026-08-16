import json

import pytest

from careervector.embedding_model import EmbeddingCareerVectorModel
from careervector.model import CareerVectorModel


def test_tfidf_loader_rejects_pre_v04_artifacts(tmp_path) -> None:
    (tmp_path / "model_info.json").write_text(json.dumps({"model": "tfidf"}))
    with pytest.raises(RuntimeError, match="older CareerVector corpus"):
        CareerVectorModel.load(tmp_path)


def test_embedding_loader_rejects_pre_v04_artifacts(tmp_path) -> None:
    (tmp_path / "embedding_model_info.json").write_text(
        json.dumps({"sentence_transformer": "fake"})
    )
    with pytest.raises(RuntimeError, match="older CareerVector corpus"):
        EmbeddingCareerVectorModel.load(tmp_path)
