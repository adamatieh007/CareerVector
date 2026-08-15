from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careervector.config import ARTIFACT_DIR, OCCUPATIONS_PATH  # noqa: E402
from careervector.model import CareerVectorModel  # noqa: E402


def main() -> None:
    if not OCCUPATIONS_PATH.exists():
        raise FileNotFoundError(
            f"{OCCUPATIONS_PATH} does not exist. Run `python scripts/build_dataset.py` first."
        )

    occupations = pd.read_csv(OCCUPATIONS_PATH)
    print(f"Training TF-IDF on {len(occupations):,} occupation documents...")
    model = CareerVectorModel.train(occupations)
    model.save(ARTIFACT_DIR)
    print(f"TF-IDF matrix shape: {model.matrix.shape}")
    print(f"Saved artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
