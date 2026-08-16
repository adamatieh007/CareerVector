import numpy as np
import pandas as pd

from careervector.academic import academic_alignment_scores, program_similarity
from careervector.profile import CareerProfile


def test_program_similarity_prefers_matching_major() -> None:
    assert program_similarity("Computer Engineering", "Computer Engineering, General.") > 0.9
    assert program_similarity("Computer Engineering", "Physics, General.") < 0.4


def test_academic_alignment_uses_cip_compatible_programs() -> None:
    metadata = pd.DataFrame(
        {
            "base_soc": ["17-2061", "19-2012"],
            "compatible_majors": [
                "Computer Engineering, General. | Electrical and Computer Engineering.",
                "Physics, General. | Health/Medical Physics.",
            ],
        }
    )
    scores = academic_alignment_scores(metadata, CareerProfile(major="Computer Engineering"))
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]
