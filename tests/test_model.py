import pandas as pd

from careervector.model import CareerVectorModel
from careervector.profile import CareerProfile


def test_tiny_recommender_prefers_hardware() -> None:
    occupations = pd.DataFrame(
        [
            {
                "onet_soc_code": "17-2061.00",
                "title": "Computer Hardware Engineers",
                "description": "Design computer hardware",
                "job_titles": "FPGA Engineer | ASIC Design Engineer",
                "document": "computer hardware fpga digital logic architecture verilog embedded systems",
                "median_salary": 130000,
                "mean_salary": 140000,
            },
            {
                "onet_soc_code": "11-2021.00",
                "title": "Marketing Managers",
                "description": "Plan marketing campaigns",
                "job_titles": "Marketing Manager",
                "document": "marketing sales advertising customers campaigns",
                "median_salary": 120000,
                "mean_salary": 130000,
            },
        ]
    )
    model = CareerVectorModel.train(occupations, max_features=1000)
    result = model.recommend(
        CareerProfile(major="Computer Engineering", interests=["FPGA", "digital hardware"]),
        top_k=1,
    )
    assert result[0]["onet_soc_code"] == "17-2061.00"
