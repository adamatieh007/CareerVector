from __future__ import annotations

import argparse

from careervector.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
from careervector.embedding_model import EmbeddingCareerVectorModel
from careervector.model import CareerVectorModel
from careervector.profile import CareerProfile
from careervector.rag import RAGCareerAdvisor
from careervector.text import split_csv_text


def _money(value: object) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):,.0f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local career recommendation engine")
    parser.add_argument(
        "--method",
        choices=["tfidf", "embeddings", "rag"],
        default="tfidf",
        help="Recommendation engine to use (default: tfidf)",
    )
    parser.add_argument("--major", default="")
    parser.add_argument("--interests", default="", help="Comma-separated interests")
    parser.add_argument("--specializations", default="", help="Comma-separated specializations")
    parser.add_argument("--preferred-work", default="", help="Comma-separated preferred work")
    parser.add_argument("--keywords", default="", help="Comma-separated extra keywords")
    parser.add_argument("--avoid", default="", help="Comma-separated things to avoid")
    parser.add_argument("--min-salary", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Local Ollama generator model for --method rag (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama API base URL (default: {DEFAULT_OLLAMA_BASE_URL})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = CareerProfile(
        major=args.major,
        interests=split_csv_text(args.interests),
        specializations=split_csv_text(args.specializations),
        preferred_work=split_csv_text(args.preferred_work),
        keywords=split_csv_text(args.keywords),
        avoid=split_csv_text(args.avoid),
        minimum_salary=args.min_salary,
    )

    if args.method == "rag":
        advisor = RAGCareerAdvisor.load(
            model_name=args.llm_model,
            ollama_base_url=args.ollama_url,
        )
        response = advisor.advise(profile, top_k=args.top_k)
        if not response.sources:
            print("No occupations met the supplied constraints.")
            return
        print(
            f"\nCareerVector RAG analysis — generator={response.generator_model}, "
            f"retriever={response.retrieval_engine}\n" + "=" * 72
        )
        print(response.answer)
        print("\nRetrieved evidence\n" + "-" * 72)
        for item in response.sources:
            print(f"[CV{item['rank']}] {item['occupation']} ({item['onet_soc_code']})")
            print(
                f"    Semantic relevance: {item['match_score']:.2f} "
                "(cosine x 100; not a probability)"
            )
            print(
                f"    Median wage: {_money(item['median_salary'])} | "
                f"Mean wage: {_money(item['mean_salary'])}"
            )
        return

    if args.method == "embeddings":
        model = EmbeddingCareerVectorModel.load()
        score_label = "Semantic relevance"
    else:
        model = CareerVectorModel.load()
        score_label = "TF-IDF relevance"

    results = model.recommend(profile, top_k=args.top_k)
    if not results:
        print("No occupations met the supplied constraints.")
        return

    print(f"\nCareerVector recommendations — {args.method}\n" + "=" * 72)
    for item in results:
        print(f"#{item['rank']}  {item['occupation']}  ({item['onet_soc_code']})")
        print(f"    {score_label}: {item['match_score']:.2f} (cosine x 100; not a probability)")
        print(
            f"    Median wage: {_money(item['median_salary'])} | "
            f"Mean wage: {_money(item['mean_salary'])}"
        )
        if item["sample_job_titles"]:
            print("    Related titles: " + "; ".join(item["sample_job_titles"]))
        if args.method == "tfidf" and item.get("matched_terms"):
            print("    Matched terms: " + ", ".join(item["matched_terms"]))
        if item["top_interests"]:
            print("    O*NET interests: " + ", ".join(item["top_interests"]))
        print()


if __name__ == "__main__":
    main()
