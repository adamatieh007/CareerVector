.PHONY: install install-ui data build train-tfidf train-embeddings demo-tfidf demo-embeddings demo-rag ui test all clean

install:
	python -m pip install -e .

install-ui:
	python -m pip install -e ".[ui,dev]"

data:
	python scripts/download_data.py

build:
	python scripts/build_dataset.py

train-tfidf:
	python scripts/train_tfidf.py

train-embeddings:
	python scripts/build_embeddings.py

demo-tfidf:
	careervector --method tfidf --major "Computer Engineering" \
		--interests "embedded systems, hardware engineering, computer architecture" \
		--preferred-work "low-level programming, digital hardware" \
		--min-salary 100000 --top-k 8

demo-embeddings:
	careervector --method embeddings --major "Computer Engineering" \
		--interests "embedded systems, hardware engineering, computer architecture" \
		--preferred-work "low-level programming, digital hardware" \
		--min-salary 100000 --top-k 8

demo-rag:
	careervector --method rag --major "Biomedical Physics" \
		--interests "radiation oncology, medical physics, dosimetry" \
		--specializations "radiation physics, imaging" \
		--preferred-work "research, clinical research" \
		--top-k 5 --llm-model gemma3

ui:
	streamlit run app.py

test:
	python -m pytest -q

all: data build train-tfidf train-embeddings

clean:
	rm -rf data/processed/* artifacts/*
	touch data/processed/.gitkeep artifacts/.gitkeep
