from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careervector.config import (  # noqa: E402
    BLS_PROJECTIONS_XLSX,
    CAREER_ROLES_PATH,
    NCES_CIP_SOC_XLSX,
    OCCUPATIONS_PATH,
    PROCESSED_DIR,
    RAW_DIR,
)
from careervector.dataset import build_career_role_documents, build_occupation_documents  # noqa: E402


def main() -> None:
    required = [
        "occupation_data.csv",
        "job_titles.csv",
        "task_statements.csv",
        "essential_skills.csv",
        "knowledge.csv",
        "specific_interest_areas.csv",
        "software_skills.csv",
        "work_activities.csv",
        NCES_CIP_SOC_XLSX,
        BLS_PROJECTIONS_XLSX,
    ]
    missing = [name for name in required if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing v0.4 source/enrichment data: "
            f"{missing}. Run `python scripts/download_data.py` first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building enriched parent occupation documents...")
    occupations = build_occupation_documents(RAW_DIR)
    occupations.to_csv(OCCUPATIONS_PATH, index=False)
    print(f"Wrote {len(occupations):,} parent occupations to {OCCUPATIONS_PATH}")

    print("Building granular career-role corpus from O*NET job titles...")
    roles = build_career_role_documents(RAW_DIR, include_esco=True)
    roles.to_csv(CAREER_ROLES_PATH, index=False)
    print(f"Wrote {len(roles):,} career-role documents to {CAREER_ROLES_PATH}")

    wage_matches = roles["median_salary"].notna().sum()
    academic_matches = roles["compatible_majors"].fillna("").astype(str).str.len().gt(0).sum()
    projection_matches = roles["growth_percent"].notna().sum()
    esco_count = roles["source"].astype(str).eq("ESCO").sum()
    print(f"Salary coverage: {wage_matches:,}/{len(roles):,}")
    print(f"NCES academic-crosswalk coverage: {academic_matches:,}/{len(roles):,}")
    print(f"BLS projections coverage: {projection_matches:,}/{len(roles):,}")
    print(f"Optional ESCO roles imported: {esco_count:,}")


if __name__ == "__main__":
    main()
