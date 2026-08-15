from careervector.rag.context import build_rag_context


def test_build_rag_context_has_citations_and_evidence() -> None:
    context = build_rag_context(
        [
            {
                "occupation": "Physicists",
                "onet_soc_code": "19-2012.00",
                "match_score": 71.2,
                "description": "Conduct research into physical phenomena.",
                "sample_job_titles": ["Medical Physicist", "Research Physicist"],
                "median_salary": 120000,
                "mean_salary": 130000,
                "top_interests": ["Physics", "Research"],
                "top_skills": ["Mathematics"],
                "top_knowledge": ["Physics"],
                "top_activities": ["Analyzing Data"],
                "core_tasks": ["Develop scientific models"],
                "software_skills": ["Python"],
            }
        ]
    )
    assert "[CV1]" in context
    assert "[/CV1]" in context
    assert "Medical Physicist" in context
    assert "$120,000" in context
    assert "Develop scientific models" in context


def test_build_rag_context_does_not_emit_nan() -> None:
    context = build_rag_context(
        [{"occupation": "Example", "onet_soc_code": "00-0000.00", "description": float("nan")}]
    )
    assert "nan" not in context.lower()
