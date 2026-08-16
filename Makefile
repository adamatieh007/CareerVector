.PHONY: install data build-data tfidf embeddings evaluate test app all

install:
	python -m pip install -e ".[ui,dev]"

data:
	python scripts/download_data.py

build-data:
	python scripts/build_dataset.py

tfidf:
	python scripts/train_tfidf.py

embeddings:
	python scripts/build_embeddings.py

evaluate:
	python scripts/evaluate.py --method tfidf
	python scripts/evaluate.py --method embeddings

test:
	python -m pytest -q

app:
	streamlit run app.py

all: build-data tfidf embeddings test
