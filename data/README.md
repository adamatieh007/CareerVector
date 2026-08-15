# CareerVector data

The repository intentionally does **not** commit the raw O*NET or BLS datasets.
Run:

```bash
python scripts/download_data.py
```

The downloader fetches the current project-pinned datasets:

## O*NET 30.3 (CSV)

- `occupation_data.csv` — occupation title + description
- `job_titles.csv` — alternate/lay job titles (important for terms such as FPGA Engineer)
- `task_statements.csv` — occupation task text
- `essential_skills.csv` — occupation skill ratings
- `knowledge.csv` — occupation knowledge ratings
- `specific_interest_areas.csv` — occupation-specific interest ratings
- `software_skills.csv` — technology/software examples
- `work_activities.csv` — occupation work-activity ratings

Official database page: https://www.onetcenter.org/database.html

## BLS OEWS May 2025 national estimates

- `national_M2025_dl.xlsx` — national occupation wage estimates, extracted from the
  official `oesm25nat.zip` archive.

Official tables page: https://www.bls.gov/oes/tables.htm

See the upstream sources for their respective licenses/terms. O*NET 30.3 is
published under a Creative Commons license; BLS is a U.S. federal statistical source.
