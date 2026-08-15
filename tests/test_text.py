from careervector.text import normalize_soc, split_csv_text


def test_normalize_soc() -> None:
    assert normalize_soc("17-2061.00") == "17-2061"
    assert normalize_soc("15-1252.01") == "15-1252"


def test_split_csv_text() -> None:
    assert split_csv_text("FPGA, embedded systems,  computer architecture ") == [
        "FPGA",
        "embedded systems",
        "computer architecture",
    ]
