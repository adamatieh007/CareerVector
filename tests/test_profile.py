from careervector.profile import CareerProfile


def test_profile_builds_weighted_query() -> None:
    profile = CareerProfile(
        major="Computer Engineering",
        interests=["FPGA"],
        specializations=["Computer Architecture"],
        avoid=["sales"],
    )
    text = profile.positive_text().lower()
    assert text.count("computer engineering") == 2
    assert text.count("fpga") == 3
    assert text.count("computer architecture") == 3
    assert profile.avoid_text() == "sales"


def test_profile_builds_natural_semantic_query() -> None:
    from careervector.profile import CareerProfile

    profile = CareerProfile(
        major="Biomedical Physics",
        interests=["Medical Physics", "Dosimetry"],
        preferred_work=["Research"],
    )
    text = profile.semantic_text()
    assert "Academic background: Biomedical Physics." in text
    assert "Interests: Medical Physics, Dosimetry." in text
    assert text.count("Medical Physics") == 1
