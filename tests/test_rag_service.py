from careervector.profile import CareerProfile
from careervector.rag.service import RAGCareerAdvisor


class FakeRetriever:
    def recommend(self, profile, **kwargs):
        return [
            {
                "rank": 1,
                "occupation": "Physicists",
                "onet_soc_code": "19-2012.00",
                "match_score": 72.0,
                "description": "Conduct research into physical phenomena.",
                "sample_job_titles": ["Medical Physicist"],
                "median_salary": 130000,
                "mean_salary": 140000,
                "top_interests": ["Physics"],
                "top_skills": ["Mathematics"],
                "top_knowledge": ["Physics"],
                "top_activities": ["Analyzing Data"],
                "core_tasks": ["Develop scientific models"],
                "software_skills": [],
            }
        ]


class FakeGenerator:
    def __init__(self):
        self.call = None

    def chat(self, **kwargs):
        self.call = kwargs
        return "Physicists is supported by the retrieved evidence [CV1]."


def test_rag_advisor_retrieves_then_generates_grounded_prompt() -> None:
    generator = FakeGenerator()
    advisor = RAGCareerAdvisor(FakeRetriever(), generator, model_name="fake-local-model")
    response = advisor.advise(
        CareerProfile(
            major="Biomedical Physics",
            interests=["Medical Physics", "Dosimetry"],
            preferred_work=["Research"],
        ),
        top_k=3,
    )

    assert response.generator_model == "fake-local-model"
    assert response.sources[0]["occupation"] == "Physicists"
    assert "[CV1]" in response.answer
    messages = generator.call["messages"]
    assert messages[0]["role"] == "system"
    assert "Use only" in messages[0]["content"]
    assert "Medical Physicist" in messages[1]["content"]
    assert "Biomedical Physics" in messages[1]["content"]


def test_rag_advisor_skips_generation_when_no_sources() -> None:
    class EmptyRetriever:
        def recommend(self, profile, **kwargs):
            return []

    generator = FakeGenerator()
    response = RAGCareerAdvisor(EmptyRetriever(), generator).advise(CareerProfile(major="Physics"))
    assert response.sources == []
    assert response.answer == ""
    assert generator.call is None
