# CareerVector

CareerVector is a local-first career recommendation and retrieval-augmented generation project. It compares a user's academic background, concentration, interests, skills, preferred work, and salary constraints against a multi-source career knowledge base.

Version **0.4** moves beyond ranking only ~1,000 broad O*NET occupations. It builds a much more granular corpus from O*NET's alternate job titles and enriches those roles with structured academic and labor-market data.

## What v0.4 adds

CareerVector now combines:

- **O*NET 30.3** — broad occupations, 57K+ alternate job titles, skills, knowledge, tasks, interests, activities, and software/technology signals.
- **NCES CIP → SOC** — official field-of-study to occupation relationships used as an academic compatibility prior.
- **BLS OEWS** — national wage data.
- **BLS Employment Projections 2024–2034** — growth, annual openings, entry education, and wage fallback data.
- **ESCO v1.2.1 (optional)** — additional occupation/skill concepts when its English CSV package is placed locally under `data/raw/esco/`.

The processed corpus therefore contains **specific career roles** such as `FPGA Engineer`, `Radiation Physicist`, `Clinical Research Scientist`, and thousands of other role titles rather than only broad parent occupations.

## Recommendation modes

The Streamlit frontend exposes three modes:

1. **TF-IDF** — lexical retrieval baseline.
2. **Sentence Embeddings** — semantic retrieval with `sentence-transformers/all-MiniLM-L6-v2` by default.
3. **RAG** — embedding retrieval followed by evidence-grounded generation using a local Ollama model.

All three modes share the same structured ranking layer:

```text
retrieval relevance
        +
specific role-title match
        +
NCES academic compatibility
        +
small BLS outlook signal
        ↓
combined ranking
```

Missing structured data is treated as unavailable rather than as a zero. The remaining weights are redistributed for that record.

## Architecture

```text
User profile
   │
   ├── major
   ├── concentration
   ├── interests
   ├── technical specializations
   ├── skills
   ├── preferred work
   └── salary constraint
   │
   ▼
TF-IDF or Sentence-Embedding Retrieval
   │
   ├── role-title specificity
   ├── NCES CIP/SOC academic prior
   └── BLS outlook signal
   │
   ▼
Diversified Top-K Career Roles
   │
   ├── direct result cards
   └── RAG evidence context
            │
            ▼
       Local Ollama LLM
            │
            ▼
   Evidence-cited explanation
```

## Setup

> **Upgrading from v0.3?** The retrieval corpus changed in v0.4, so old TF-IDF/embedding artifacts must be rebuilt. See [`UPGRADE_V0.4.md`](UPGRADE_V0.4.md).

Python 3.10+ is required. Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ui,dev]"
```

## 1. Download the data

```bash
python scripts/download_data.py
```

This downloads the O*NET, NCES, and BLS sources needed by v0.4.

### Optional ESCO

Download the official ESCO English CSV classification package and unzip it to:

```text
data/raw/esco/
```

Then rebuild the dataset. ESCO is optional; the U.S. O*NET/NCES/BLS pipeline works without it.

## 2. Build the knowledge base

```bash
python scripts/build_dataset.py
```

This produces both:

```text
data/processed/occupations.csv
data/processed/career_roles.csv
```

The second file is the retrieval corpus. With O*NET 30.3 it contains tens of thousands of specific role/title records.

## 3. Train TF-IDF

```bash
python scripts/train_tfidf.py
```

## 4. Build sentence embeddings

```bash
python scripts/build_embeddings.py
```

The first run downloads the configured public Sentence Transformers model from Hugging Face. Inference runs locally after the model is cached.

## 5. Run the frontend

```bash
streamlit run app.py
```

CareerVector's project Streamlit config disables the recursive file watcher because Transformers exposes many optional vision modules that are unrelated to this text application.

## 6. RAG with Ollama

Install Ollama separately, then pull a local model, for example:

```bash
ollama pull gemma3
```

Start CareerVector and select:

```text
RAG · Embeddings + Local LLM
```

RAG does **not** let the LLM choose careers from general model memory. The embedding engine first retrieves ranked career-role evidence; the local LLM then receives those source blocks and is prompted to cite them as `[CV1]`, `[CV2]`, etc.

## CLI examples

### TF-IDF

```bash
careervector \
  --method tfidf \
  --major "Computer Engineering" \
  --concentration "Computer Architecture" \
  --interests "FPGA, embedded systems, low-latency systems" \
  --skills "C++, SystemVerilog, CUDA" \
  --top-k 8
```

### Embeddings

```bash
careervector \
  --method embeddings \
  --major "Biomedical Physics" \
  --concentration "Medical Physics" \
  --interests "Radiation oncology, dosimetry, medical imaging" \
  --skills "Physics, research" \
  --top-k 8
```

### Local RAG

```bash
careervector \
  --method rag \
  --major "Biomedical Physics" \
  --concentration "Medical Physics" \
  --interests "Radiation oncology, dosimetry" \
  --preferred-work "Clinical research" \
  --llm-model gemma3 \
  --top-k 5
```

## Scoring design

CareerVector intentionally separates retrieval from structured ranking.

### Retrieval

- TF-IDF: lexical cosine similarity.
- Embeddings: dense cosine similarity.
- Avoid preferences: subtract a configurable negative-similarity penalty.

### Structured signals

- **Title specificity** rewards exact/near-exact user concepts appearing in a specific role title, e.g. `FPGA` → `FPGA Engineer`.
- **Academic alignment** compares the user's major/concentration against NCES CIP programs officially linked to the role's SOC occupation.
- **Outlook** adds a small bounded signal from BLS projected 2024–2034 employment growth.
- **Salary** remains a hard constraint when the user specifies a minimum.

The default blend is approximately:

```text
58% retrieval
24% academic compatibility
13% specific role-title match
 5% BLS outlook
```

If academic or outlook data is unavailable for a record, that missing weight is redistributed rather than treated as a poor score.

A diversity pass also caps repeated aliases from the same parent SOC so one broad occupation does not flood the entire Top-K list.

## Repository layout

```text
CareerVector/
├── app.py
├── artifacts/
├── data/
│   ├── raw/
│   │   └── esco/              # optional
│   └── processed/
├── design_docs/
├── examples/
├── scripts/
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── train_tfidf.py
│   ├── build_embeddings.py
│   └── evaluate.py
├── src/careervector/
│   ├── academic.py
│   ├── ranking.py
│   ├── dataset.py
│   ├── model.py
│   ├── embedding_model.py
│   ├── profile.py
│   └── rag/
└── tests/
```

## Tests

```bash
python -m pytest -q
```

See [`VALIDATION.md`](VALIDATION.md) for the real-data corpus build and the small multi-domain retrieval sanity suite used during v0.4 development.

The suite covers profile construction, TF-IDF, embedding behavior, NCES academic matching, BLS projection parsing, optional ESCO ingestion, title-specific ranking, parent-role diversity, RAG context, Ollama client behavior, CLI flags, and design-document structure.

## Design documentation

Open:

```text
design_docs/index.html
```

The HTML engineering documentation includes Mermaid diagrams for the multi-source knowledge base, role-building pipeline, structured ranking, RAG workflow, UI flow, and testing/evolution strategy.

## Data provenance

CareerVector uses public data from:

- O*NET Resource Center: https://www.onetcenter.org/database.html
- NCES CIP resources: https://nces.ed.gov/ipeds/cipcode/resources.aspx?y=56
- BLS Employment Projections: https://www.bls.gov/emp/data/occupational-data.htm
- BLS OEWS: https://www.bls.gov/oes/
- ESCO: https://esco.ec.europa.eu/en/use-esco/download

Generated/raw data and model artifacts are ignored by Git so the repository stays reproducible instead of committing large government datasets or local model outputs.
