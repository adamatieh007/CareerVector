# CareerVector data

CareerVector v0.4 builds a **multi-source career knowledge base**. Large downloaded/generated files are intentionally ignored by Git.

## Automatically downloaded sources

Run:

```bash
python scripts/download_data.py
```

This downloads:

- **O*NET 30.3** occupation records, 57K+ alternate job titles, tasks, skills, knowledge, interests, work activities, and software/technology examples.
- **NCES 2020 CIP → 2018 SOC Crosswalk** for structured major/field-of-study compatibility.
- **BLS OEWS May 2025** national wage data.
- **BLS Employment Projections 2024–2034** for employment growth, annual openings, median wage fallback, and typical entry education.

## Optional ESCO enrichment

ESCO's official download portal uses interactive package selection. To enable it:

1. Download the **ESCO v1.2.1 English CSV classification** from the official ESCO download page.
2. Unzip the CSV package under:

```text
data/raw/esco/
```

3. Run `python scripts/build_dataset.py` again.

CareerVector detects occupation CSVs and, when available, occupation-skill relation files automatically. ESCO-only roles intentionally do not inherit U.S. salary/CIP metadata unless a trustworthy mapping exists.

## Generated files

`python scripts/build_dataset.py` creates:

- `data/processed/occupations.csv` — 1 row per broad O*NET occupation, retained for inspection.
- `data/processed/career_roles.csv` — retrieval corpus with one row per specific role/title. O*NET alternate titles inherit their parent occupation's validated metadata; optional ESCO occupations are appended as extra roles.

Do not commit `data/raw/*`, `data/processed/*`, or generated `artifacts/*`.
