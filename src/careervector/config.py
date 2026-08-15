import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

OCCUPATIONS_PATH = PROCESSED_DIR / "occupations.csv"

# TF-IDF artifacts
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = ARTIFACT_DIR / "occupation_tfidf.npz"
METADATA_PATH = ARTIFACT_DIR / "occupation_metadata.csv"
MODEL_INFO_PATH = ARTIFACT_DIR / "model_info.json"

# Sentence-embedding artifacts
EMBEDDING_MATRIX_PATH = ARTIFACT_DIR / "occupation_embeddings.npy"
EMBEDDING_METADATA_PATH = ARTIFACT_DIR / "embedding_metadata.csv"
EMBEDDING_INFO_PATH = ARTIFACT_DIR / "embedding_model_info.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local RAG generator (Ollama)
DEFAULT_OLLAMA_BASE_URL = os.getenv("CAREERVECTOR_OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("CAREERVECTOR_OLLAMA_MODEL", "gemma3")

ONET_RELEASE = "30.3"
ONET_BASE = "https://www.onetcenter.org/dl_files/database/db_30_3_csv"
ONET_FILES = {
    "occupation_data.csv": f"{ONET_BASE}/occupation_data.csv",
    "job_titles.csv": f"{ONET_BASE}/job_titles.csv",
    "task_statements.csv": f"{ONET_BASE}/task_statements.csv",
    "essential_skills.csv": f"{ONET_BASE}/essential_skills.csv",
    "knowledge.csv": f"{ONET_BASE}/knowledge.csv",
    "specific_interest_areas.csv": f"{ONET_BASE}/specific_interest_areas.csv",
    "software_skills.csv": f"{ONET_BASE}/software_skills.csv",
    "work_activities.csv": f"{ONET_BASE}/work_activities.csv",
}

BLS_RELEASE = "May 2025"
BLS_NATIONAL_ZIP_URL = "https://www.bls.gov/oes/special-requests/oesm25nat.zip"
BLS_NATIONAL_XLSX = "national_M2025_dl.xlsx"
