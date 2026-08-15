from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careervector.config import OCCUPATIONS_PATH, PROCESSED_DIR, RAW_DIR  # noqa: E402
from careervector.dataset import build_occupation_documents  # noqa: E402


def main() -> None:
    required = ["occupation_data.csv", "job_titles.csv", "task_statements.csv"]
    missing = [name for name in required if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data: {missing}. Run `python scripts/download_data.py` first."
        )

    print("Building weighted occupation documents...")
    frame = build_occupation_documents(RAW_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OCCUPATIONS_PATH, index=False)
    print(f"Wrote {len(frame):,} occupation documents to {OCCUPATIONS_PATH}")
    wage_matches = frame["median_salary"].notna().sum()
    print(f"BLS wage matches: {wage_matches:,}/{len(frame):,}")


if __name__ == "__main__":
    main()
