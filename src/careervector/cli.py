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
    parser.add_argument("--concentration", default="")
    parser.add_argument("--interests", default="", help="Comma-separated interests")
    parser.add_argument("--specializations", default="", help="Comma-separated technical specializations")
    parser.add_argument("--skills", default="", help="Comma-separated skills")
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
        concentration=args.concentration,
        interests=split_csv_text(args.interests),
        specializations=split_csv_text(args.specializations),
        skills=split_csv_text(args.skills),
        preferred_work=split_csv_text(args.preferred_work),
        keywords=split_csv_text(args.keywords),
        avoid=split_csv_text(args.avoid),
        minimum_salary=args.min_salary,
    )

    if args.method == "rag":
        advisor = RAGCareerAdvisor.load(model_name=args.llm_model, ollama_base_url=args.ollama_url)
        response = advisor.advise(profile, top_k=args.top_k)
        if not response.sources:
            print("No career roles met the supplied constraints.")
            return
        print(
            f"\nCareerVector RAG analysis — generator={response.generator_model}, "
            f"retriever={response.retrieval_engine}\n" + "=" * 72
        )
        print(response.answer)
        print("\nRetrieved evidence\n" + "-" * 72)
        for item in response.sources:
            parent = item.get("parent_occupation")
            parent_text = f" · parent: {parent}" if parent and parent != item.get("occupation") else ""
            print(f"[CV{item['rank']}] {item['occupation']} ({item.get('onet_soc_code') or item.get('source')}){parent_text}")
            print(f"    Combined score: {item['match_score']:.2f}")
            print(f"    Retrieval: {item.get('retrieval_score', 0):.2f} | Academic: {item.get('academic_alignment') or 0:.2f}")
            print(f"    Median wage: {_money(item.get('median_salary'))} | Growth: {item.get('growth_percent')}")
        return

    if args.method == "embeddings":
        model = EmbeddingCareerVectorModel.load()
        score_label = "Semantic retrieval"
    else:
        model = CareerVectorModel.load()
        score_label = "TF-IDF retrieval"

    results = model.recommend(profile, top_k=args.top_k)
    if not results:
        print("No career roles met the supplied constraints.")
        return

    print(f"\nCareerVector recommendations — {args.method}\n" + "=" * 72)
    for item in results:
        parent = item.get("parent_occupation")
        parent_text = f" · parent: {parent}" if parent and parent != item.get("occupation") else ""
        print(f"#{item['rank']}  {item['occupation']}  ({item.get('onet_soc_code') or item.get('source')}){parent_text}")
        print(f"    Combined score: {item['match_score']:.2f}")
        print(f"    {score_label}: {item.get('retrieval_score', 0):.2f}")
        if item.get("academic_alignment") is not None:
            print(f"    Academic alignment: {item['academic_alignment']:.2f}")
        if item.get("growth_percent") is not None:
            print(f"    BLS growth 2024-34: {item['growth_percent']:.1f}%")
        print(f"    Median wage: {_money(item.get('median_salary'))} | Mean wage: {_money(item.get('mean_salary'))}")
        if item.get("academic_matches"):
            print("    Related fields of study: " + "; ".join(item["academic_matches"]))
        if item.get("sample_job_titles"):
            print("    Related titles: " + "; ".join(item["sample_job_titles"]))
        if args.method == "tfidf" and item.get("matched_terms"):
            print("    Matched terms: " + ", ".join(item["matched_terms"]))
        print()


if __name__ == "__main__":
    main()
