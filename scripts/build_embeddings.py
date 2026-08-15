from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careervector.config import (  # noqa: E402
    ARTIFACT_DIR,
    DEFAULT_EMBEDDING_MODEL,
    OCCUPATIONS_PATH,
)
from careervector.embedding_model import EmbeddingCareerVectorModel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local sentence embeddings for O*NET careers")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-words", type=int, default=140)
    parser.add_argument("--overlap-words", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not OCCUPATIONS_PATH.exists():
        raise FileNotFoundError(
            f"{OCCUPATIONS_PATH} does not exist. Run `python scripts/build_dataset.py` first."
        )

    occupations = pd.read_csv(OCCUPATIONS_PATH)
    print(f"Building sentence embeddings for {len(occupations):,} occupation documents...")
    print(f"Model: {args.model}")
    model = EmbeddingCareerVectorModel.build(
        occupations,
        model_name=args.model,
        batch_size=args.batch_size,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    model.save(ARTIFACT_DIR)
    print(f"Embedding matrix shape: {model.matrix.shape}")
    print(f"Saved artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
