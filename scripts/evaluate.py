from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careervector.embedding_model import EmbeddingCareerVectorModel  # noqa: E402
from careervector.model import CareerVectorModel  # noqa: E402
from careervector.profile import CareerProfile  # noqa: E402
from careervector.text import normalize_soc  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CareerVector Recall@5")
    parser.add_argument("--method", choices=["tfidf", "embeddings"], default="tfidf")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = json.loads((ROOT / "examples" / "evaluation_profiles.json").read_text())
    model = (
        EmbeddingCareerVectorModel.load()
        if args.method == "embeddings"
        else CareerVectorModel.load()
    )

    hits = 0
    for case in examples:
        profile = CareerProfile(
            major=case.get("major", ""),
            concentration=case.get("concentration", ""),
            interests=case.get("interests", []),
            specializations=case.get("specializations", []),
            skills=case.get("skills", []),
            preferred_work=case.get("preferred_work", []),
        )
        results = model.recommend(profile, top_k=5)
        returned_raw = {str(r["onet_soc_code"]) for r in results}
        expected_raw = set(case["expected_onet_codes"])
        returned = {normalize_soc(code) for code in returned_raw}
        expected = {normalize_soc(code) for code in expected_raw}
        hit = bool(returned & expected)
        hits += int(hit)
        print(f"{'PASS' if hit else 'MISS'}: {case['name']}")
        print("  returned:", [r["onet_soc_code"] for r in results])
        print("  expected one of:", sorted(expected_raw))

    total = len(examples)
    print(f"\n{args.method} Recall@5 cases: {hits}/{total} = {hits / total:.1%}")


if __name__ == "__main__":
    main()
