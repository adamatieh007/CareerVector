# CareerVector

CareerVector is a **local career recommendation engine** that ranks O*NET occupations from a user's
major, interests, specializations, preferred work, negative preferences, and salary constraints.

Version 0.2 supports two selectable retrieval engines:

1. **TF-IDF + cosine similarity** — transparent lexical baseline.
2. **Sentence Transformers + cosine similarity** — semantic retrieval that can match related ideas even
   when the same words are not used.

The Streamlit frontend lets users switch between them from the UI, and the CLI exposes the same choice
through `--method tfidf` or `--method embeddings`.

No paid API or hosted inference service is required. The embedding model downloads once and then runs
locally from the local model cache.

## Architecture

```text
                         O*NET + BLS
                              |
                      build_dataset.py
                              |
                 processed occupation records
                    /                   \
                   /                     \
        TF-IDF vectorizer          Sentence Transformer
           sparse matrix             dense embeddings
                   \                     /
                    \                   /
                     selectable ranking engine
                              |
                    cosine similarity
                              |
                  negative preference penalty
                              |
                       salary filter
                              |
                    ranked career results
                              |
                    Streamlit web frontend
```

## Important: embeddings are not RAG

Sentence embeddings are the **retrieval** part: occupation documents and the user's profile are mapped
into a vector space and ranked by semantic similarity. That is not yet a full RAG system. A later RAG
version could retrieve careers with these embeddings and pass the retrieved O*NET evidence to a local
LLM to generate grounded explanations.

## Setup

```bash
git clone <your-repo-url>
cd careervector
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[ui,dev]"
```

If you only want TF-IDF:

```bash
pip install -e .
```

If you only want CLI sentence embeddings:

```bash
pip install -e ".[embeddings]"
```

## Data

CareerVector uses the same O*NET/BLS data for both retrieval engines. You do **not** need a new dataset
for embeddings.

```bash
python scripts/download_data.py
python scripts/build_dataset.py
```

The processed dataset contains 1,016 O*NET occupation records in the current data release. The builder
keeps a natural-language `core_tasks` field in addition to the weighted TF-IDF document so dense
retrieval can consume less repetitive text.

## Build the TF-IDF baseline

```bash
python scripts/train_tfidf.py
```

TF-IDF artifacts:

```text
artifacts/
├── tfidf_vectorizer.joblib
├── occupation_tfidf.npz
├── occupation_metadata.csv
└── model_info.json
```

## Build sentence embeddings

```bash
python scripts/build_embeddings.py
```

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

CareerVector builds a natural semantic document from each occupation's title, description, alternate
job titles, interests, skills, knowledge, work activities, core tasks, and software technologies.
Long documents are split into overlapping chunks. Each chunk is embedded, the chunk vectors are averaged
per occupation, and the final occupation vectors are normalized for cosine ranking.

The embedding query is also built as natural labeled text (major, interests, specializations, and preferred
work once each) instead of reusing TF-IDF's deliberate phrase repetition.

Embedding artifacts:

```text
artifacts/
├── occupation_embeddings.npy
├── embedding_metadata.csv
└── embedding_model_info.json
```

The pretrained Sentence Transformer weights are stored in your local model cache rather than committed
to Git.

## Run the web interface

```bash
streamlit run app.py
```

The UI asks:

- major / academic background
- interests
- specializations
- preferred work
- things to avoid
- extra keywords
- optional minimum salary
- number of recommendations

Use the sidebar to switch between **TF-IDF** and **Sentence Embeddings**.

## CLI: TF-IDF

```bash
careervector \
  --method tfidf \
  --major "Biomedical Physics" \
  --interests "Radiation Oncology, Medical Physics, Dosimetry" \
  --specializations "Radiation Physics, Imaging" \
  --preferred-work "Research, Clinical Research" \
  --min-salary 120000 \
  --top-k 8
```

## CLI: sentence embeddings

Use the exact same profile and change one flag:

```bash
careervector \
  --method embeddings \
  --major "Biomedical Physics" \
  --interests "Radiation Oncology, Medical Physics, Dosimetry" \
  --specializations "Radiation Physics, Imaging" \
  --preferred-work "Research, Clinical Research" \
  --min-salary 120000 \
  --top-k 8
```

That gives the project a clean A/B comparison: the data and profile stay fixed while the retrieval
representation changes.

## Why `all-MiniLM-L6-v2` first?

It is a lightweight Sentence Transformer intended for semantic similarity/search. It produces compact
dense vectors and is small enough to run comfortably on a normal laptop. That makes it a good semantic
baseline before testing larger embedding models.

Later experiments can compare models such as `all-mpnet-base-v2` using the same evaluation set.

## Scoring

### TF-IDF

```text
positive_score = cosine(user_tfidf, occupation_tfidf)
```

### Embeddings

```text
positive_score = cosine(user_embedding, occupation_embedding)
```

Both engines support the same negative preference penalty:

```text
score = positive_score - 0.35 * avoid_similarity
```

A minimum salary is a hard post-ranking filter using BLS median annual wage data.

Displayed relevance is `cosine similarity × 100`. It is a ranking score, **not a probability**.

## Evaluation

Evaluate the lexical baseline:

```bash
python scripts/evaluate.py --method tfidf
```

Evaluate sentence embeddings:

```bash
python scripts/evaluate.py --method embeddings
```

Using the same labeled profiles is important: it lets you quantify whether semantic retrieval actually
beats the TF-IDF baseline instead of assuming it does.

## Repository layout

```text
careervector/
├── app.py
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── examples/
├── scripts/
│   ├── build_dataset.py
│   ├── build_embeddings.py
│   ├── download_data.py
│   ├── evaluate.py
│   └── train_tfidf.py
├── src/careervector/
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── embedding_model.py
│   ├── model.py
│   ├── profile.py
│   └── text.py
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] O*NET + BLS data pipeline
- [x] TF-IDF lexical baseline
- [x] salary filtering
- [x] negative-preference penalty
- [x] sentence-transformer semantic retrieval
- [x] selectable TF-IDF / embedding CLI flag
- [x] local Streamlit frontend
- [ ] larger labeled evaluation set
- [ ] Precision@K / MRR / NDCG comparison dashboard
- [ ] hybrid TF-IDF + embeddings ranker
- [ ] model selector (`MiniLM` vs `MPNet`)
- [ ] local RAG explanations grounded in retrieved O*NET evidence
- [ ] location-specific wage filtering
- [ ] learned reranker from user feedback

## Tests

```bash
python -m pytest -q
```

## Data attribution

This project consumes public occupational data from the U.S. Department of Labor's O*NET program and
wage estimates from the U.S. Bureau of Labor Statistics OEWS program. Review and retain the required
upstream attribution/license notices when redistributing derived data.
