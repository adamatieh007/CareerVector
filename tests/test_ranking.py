import numpy as np
import pandas as pd

from careervector.profile import CareerProfile
from careervector.ranking import combine_relevance_scores, select_diverse_indices, title_match_scores


def test_specific_title_signal_rewards_fpga_role() -> None:
    metadata = pd.DataFrame(
        {
            "title": ["FPGA Engineer", "Engineering Manager"],
            "parent_title": ["Computer Hardware Engineers", "Architectural and Engineering Managers"],
            "base_soc": ["17-2061", "11-9041"],
            "compatible_majors": ["Computer Engineering, General.", "Computer Engineering, General."],
            "growth_percent": [7.3, 3.8],
        }
    )
    profile = CareerProfile(major="Computer Engineering", interests=["FPGA"])
    title_scores = title_match_scores(metadata, profile)
    assert title_scores[0] > title_scores[1]
    combined, _, _, _ = combine_relevance_scores(np.array([0.1, 0.1]), metadata, profile)
    assert combined[0] > combined[1]


def test_diversity_caps_one_parent_family_from_flooding() -> None:
    metadata = pd.DataFrame(
        {
            "base_soc": ["A", "A", "A", "B", "C"],
            "title": ["A1", "A2", "A3", "B1", "C1"],
        }
    )
    scores = np.array([1.0, 0.99, 0.98, 0.8, 0.7])
    selected = select_diverse_indices(scores, np.arange(5), metadata, top_k=4, max_per_parent=2)
    selected_titles = metadata.iloc[selected]["title"].tolist()
    assert selected_titles.count("A1") == 1
    assert len([title for title in selected_titles if title.startswith("A")]) == 2
    assert "B1" in selected_titles and "C1" in selected_titles
