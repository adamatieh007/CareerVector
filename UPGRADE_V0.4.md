# Upgrading CareerVector from v0.3 to v0.4

CareerVector v0.4 changes the retrieval corpus from broad O*NET occupations to **specific career-role titles** and adds NCES/BLS enrichment. Existing v0.3 TF-IDF and sentence-embedding artifacts are therefore intentionally incompatible and must be rebuilt.

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e ".[ui,dev]"

python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train_tfidf.py
python scripts/build_embeddings.py
python -m pytest -q
streamlit run app.py
```

`download_data.py` keeps existing files unless a download is missing, so your existing O*NET/BLS files do not need to be downloaded again unnecessarily. It will fetch the new NCES CIP-SOC crosswalk and BLS Employment Projections workbook when they are not already present.

## Optional ESCO enrichment

If you want the optional ESCO layer, download the official English CSV classification package and extract it under:

```text
data/raw/esco/
```

Then rerun:

```bash
python scripts/build_dataset.py
python scripts/train_tfidf.py
python scripts/build_embeddings.py
```

## Why rebuilding is required

v0.3 artifacts contain one vector per broad occupation. v0.4 contains one vector per specific role/title, with inherited O*NET evidence plus NCES academic compatibility and BLS labor-market metadata. The application detects stale v0.3 artifacts and tells you to rebuild rather than silently mixing incompatible data.
