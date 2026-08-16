from pathlib import Path

import pandas as pd

from careervector.dataset import _load_bls_projections, _load_cip_soc, load_esco_roles


def test_load_cip_soc_crosswalk(tmp_path: Path) -> None:
    path = tmp_path / "cip.xlsx"
    frame = pd.DataFrame(
        {
            "CIP2020Code": [14.0901, 40.0801],
            "CIP2020Title": ["Computer Engineering, General.", "Physics, General."],
            "SOC2018Code": ["17-2061", "19-2012"],
            "SOC2018Title": ["Computer Hardware Engineers", "Physicists"],
        }
    )
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="CIP-SOC", index=False)
    crosswalk, by_soc = _load_cip_soc(path)
    assert len(crosswalk) == 2
    assert "Computer Engineering" in by_soc["17-2061"]


def test_load_bls_projection_table(tmp_path: Path) -> None:
    path = tmp_path / "projection.xlsx"
    rows = pd.DataFrame(
        [
            [
                "Computer Hardware Engineers", "17-2061", "Line item", 76.8, 82.4,
                0.0, 0.0, 5.6, 7.3, 0.0, 4.7, 155020, "Bachelor's degree", "None", "None", "—"
            ]
        ],
        columns=[
            "2024 National Employment Matrix title",
            "2024 National Employment Matrix code",
            "Occupation type",
            "Employment, 2024",
            "Employment, 2034",
            "Employment distribution, percent, 2024",
            "Employment distribution, percent, 2034",
            "Employment change, numeric, 2024–34",
            "Employment change, percent, 2024–34",
            "Percent self employed, 2024",
            "Occupational openings, 2024–34 annual average",
            "Median annual wage, dollars, 2024[1]",
            "Typical education needed for entry",
            "Work experience in a related occupation",
            "Typical on-the-job training needed to attain competency in the occupation",
            "Related Occupational Outlook Handbook (OOH) content xlsx",
        ],
    )
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["metadata"]]).to_excel(writer, sheet_name="Table 1.2", index=False, header=False)
        rows.to_excel(writer, sheet_name="Table 1.2", index=False, startrow=1)
    result = _load_bls_projections(path)
    assert result.iloc[0]["base_soc"] == "17-2061"
    assert result.iloc[0]["growth_percent"] == 7.3
    assert result.iloc[0]["typical_education"] == "Bachelor's degree"


def test_esco_optional_csv_import(tmp_path: Path) -> None:
    occupations = pd.DataFrame(
        {
            "conceptUri": ["http://example/occupation/1"],
            "preferredLabel": ["robotics engineer"],
            "altLabels": ["robotics specialist"],
            "description": ["Designs and develops robotic systems."],
        }
    )
    occupations.to_csv(tmp_path / "occupations_en.csv", index=False)
    result = load_esco_roles(tmp_path)
    assert len(result) == 1
    assert result.iloc[0]["title"] == "robotics engineer"
    assert result.iloc[0]["source"] == "ESCO"
