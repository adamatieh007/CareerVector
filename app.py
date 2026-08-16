from __future__ import annotations

from pathlib import Path

import streamlit as st

from careervector.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, EMBEDDING_INFO_PATH
from careervector.embedding_model import EmbeddingCareerVectorModel
from careervector.model import CareerVectorModel
from careervector.profile import CareerProfile
from careervector.rag import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    RAGCareerAdvisor,
)
from careervector.text import split_csv_text

st.set_page_config(page_title="CareerVector", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
.cv-hero { padding: 1.5rem 1.6rem; border: 1px solid rgba(128,128,128,.22); border-radius: 18px; margin-bottom: 1.5rem; background: linear-gradient(135deg, rgba(70,90,255,.10), rgba(90,200,180,.08)); }
.cv-hero h1 { margin: 0 0 .35rem 0; font-size: 2.25rem; font-weight: 750; }
.cv-hero p { margin: 0; opacity: .82; font-size: 1.02rem; line-height: 1.55; }
.cv-rank { display:inline-block; font-weight:700; font-size:.78rem; padding:.2rem .55rem; border-radius:999px; border:1px solid rgba(128,128,128,.3); margin-bottom:.35rem; }
.cv-score { font-size:1.55rem; font-weight:750; }
.cv-muted { opacity:.72; font-size:.92rem; }
.cv-rag { border:1px solid rgba(112,225,195,.32); border-radius:16px; padding:1rem 1.2rem; background:rgba(112,225,195,.055); margin:.5rem 0 1.5rem; }
div[data-testid="stButton"] > button { border-radius:10px; font-weight:650; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_tfidf_model() -> CareerVectorModel:
    return CareerVectorModel.load()


@st.cache_resource
def load_embedding_model() -> EmbeddingCareerVectorModel:
    return EmbeddingCareerVectorModel.load()


def money(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def number(value: object, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.1f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def result_card(item: dict[str, object], method: str) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1.3])
        with left:
            st.markdown(f"<span class='cv-rank'>#{item['rank']}</span>", unsafe_allow_html=True)
            st.subheader(str(item["occupation"]))
            parent = item.get("parent_occupation")
            if parent and str(parent) != str(item.get("occupation")):
                st.caption(f"Parent occupation: {parent} · {item.get('source', 'O*NET')}")
            elif item.get("onet_soc_code"):
                st.caption(f"O*NET-SOC {item['onet_soc_code']} · {item.get('source', 'O*NET')}")
            else:
                st.caption(str(item.get("source", "Career knowledge base")))

        with right:
            st.markdown(
                f"<div class='cv-score'>{float(item['match_score']):.1f}</div><div class='cv-muted'>combined score</div>",
                unsafe_allow_html=True,
            )

        description = item.get("description")
        if description and str(description).lower() != "nan":
            st.write(str(description))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median salary", money(item.get("median_salary")))
        c2.metric("Academic fit", number(item.get("academic_alignment")))
        c3.metric("2024–34 growth", number(item.get("growth_percent"), "%"))
        c4.metric("Retrieval", number(item.get("retrieval_score")))

        matches = item.get("academic_matches") or []
        if matches:
            st.markdown("**Related fields of study**")
            st.write(" · ".join(str(x) for x in matches))

        titles = item.get("sample_job_titles") or []
        if titles:
            st.markdown("**Related job titles**")
            st.write(" · ".join(str(x) for x in titles))

        if method == "tfidf" and item.get("matched_terms"):
            st.markdown("**Strong lexical matches**")
            st.write(" · ".join(str(x) for x in item["matched_terms"]))
        else:
            st.caption(
                "The combined score blends retrieval relevance with NCES academic compatibility and a small BLS outlook signal when those data exist. It is not a probability."
            )

        details: list[tuple[str, object]] = []
        for label, key in [
            ("Typical entry education", "typical_education"),
            ("Average annual openings (thousands)", "annual_openings"),
            ("O*NET interests", "top_interests"),
            ("Top skills", "top_skills"),
            ("Knowledge areas", "top_knowledge"),
            ("Work activities", "top_activities"),
            ("Core tasks", "core_tasks"),
            ("Software / technologies", "software_skills"),
        ]:
            if item.get(key) not in (None, "", []):
                details.append((label, item[key]))

        if details:
            with st.expander("Retrieved evidence"):
                for label, values in details:
                    rendered = ", ".join(str(x) for x in values) if isinstance(values, list) else str(values)
                    st.markdown(f"**{label}:** {rendered}")


st.markdown(
    """
<div class="cv-hero">
<h1>🎯 CareerVector</h1>
<p>
CareerVector now searches a granular career-role knowledge base built from O*NET job titles,
NCES major-to-occupation mappings, BLS wages/projections, and optional ESCO enrichment. Compare
TF-IDF, sentence embeddings, or local RAG explanations with Ollama.
</p>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Choose a recommendation method")
engine_label = st.radio(
    "Recommendation method",
    ["TF-IDF", "Sentence Embeddings", "RAG · Embeddings + Local LLM"],
    horizontal=True,
    label_visibility="collapsed",
)
method = {"TF-IDF": "tfidf", "Sentence Embeddings": "embeddings", "RAG · Embeddings + Local LLM": "rag"}[engine_label]

ollama_base_url = DEFAULT_OLLAMA_BASE_URL
ollama_model = DEFAULT_OLLAMA_MODEL
ollama_models: list[str] = []
ollama_error: str | None = None

if method == "tfidf":
    st.caption("Lexical retrieval + NCES academic prior + BLS outlook signal.")
elif method == "embeddings":
    st.caption("Semantic retrieval + NCES academic prior + BLS outlook signal.")
else:
    st.caption("Semantic retrieval ranks roles first; a local Ollama model then explains the retrieved evidence.")
    with st.expander("Local RAG settings", expanded=True):
        ollama_base_url = st.text_input("Ollama server", value=DEFAULT_OLLAMA_BASE_URL)
        try:
            ollama_models = OllamaClient(ollama_base_url, timeout=2.0).list_models()
        except OllamaError as exc:
            ollama_error = str(exc)
        if ollama_models:
            default_index = next((i for i, name in enumerate(ollama_models) if name == DEFAULT_OLLAMA_MODEL), 0)
            ollama_model = st.selectbox("Local generator model", ollama_models, index=default_index)
        else:
            ollama_model = st.text_input("Local generator model", value=DEFAULT_OLLAMA_MODEL)
            if ollama_error:
                st.warning(ollama_error)
            st.code("ollama pull gemma3", language="bash")

with st.sidebar:
    st.header("Knowledge base")
    st.write("✅ O*NET occupations + job titles")
    st.write("✅ NCES CIP → SOC academic mapping")
    st.write("✅ BLS wages + 2024–34 projections")
    st.write("➕ ESCO if data/raw/esco is present")
    st.divider()
    st.header("Model status")
    st.write("✅ TF-IDF artifacts")
    st.write("✅ Sentence-embedding artifacts" if Path(EMBEDDING_INFO_PATH).exists() else "⚠️ Sentence embeddings not built yet")
    if method == "rag":
        st.write(f"✅ Ollama · {len(ollama_models)} local model(s)" if ollama_models else "⚠️ Ollama unavailable / no models")

st.subheader("Tell CareerVector about yourself")
major = st.text_input("1. Major / academic field", placeholder="e.g. Computer Engineering, Biomedical Physics, Accounting")
concentration = st.text_input("2. Concentration or academic specialization", placeholder="e.g. Computer Architecture, Medical Physics, Finance (optional)")
interests = st.text_area("3. Interests", placeholder="e.g. FPGA design, low-latency systems, GPU architecture", height=95)
specializations = st.text_area("4. Technical areas to emphasize", placeholder="e.g. RTL design, CUDA, dosimetry, medical imaging", height=85)
skills = st.text_area("5. Skills you already have or want to use", placeholder="e.g. C++, SystemVerilog, Python, statistics, microscopy", height=85)
career_preferences = st.text_area("6. What would you like your career to involve?", placeholder="e.g. Hardware design, research, clinical work, quantitative problem solving", height=95)
avoid = st.text_area("7. Anything you want to avoid?", placeholder="e.g. Sales, web development, frequent travel (optional)", height=75)

st.markdown("### Salary preference")
salary_choice = st.selectbox(
    "Minimum median annual salary",
    ["No minimum", "$60,000", "$80,000", "$100,000", "$120,000", "$150,000", "Custom"],
    index=0,
)
if salary_choice == "Custom":
    minimum_salary = float(st.number_input("Enter your minimum annual salary", min_value=0, max_value=1_000_000, value=120_000, step=5_000, format="%d"))
elif salary_choice == "No minimum":
    minimum_salary = None
else:
    minimum_salary = float(salary_choice.replace("$", "").replace(",", ""))

top_k = st.slider("Number of recommendations", min_value=3, max_value=15, value=5 if method == "rag" else 8)
submitted = st.button("Find my careers", type="primary", use_container_width=True)

if submitted:
    profile = CareerProfile(
        major=major,
        concentration=concentration,
        interests=split_csv_text(interests),
        specializations=split_csv_text(specializations),
        skills=split_csv_text(skills),
        preferred_work=split_csv_text(career_preferences),
        avoid=split_csv_text(avoid),
        minimum_salary=minimum_salary,
    )

    if not profile.positive_text().strip():
        st.warning("Add at least an academic field, concentration, interest, skill, specialization, or career preference.")
        st.stop()

    try:
        if method == "tfidf":
            with st.spinner("Ranking specific career roles with TF-IDF + structured signals..."):
                results = load_tfidf_model().recommend(profile, top_k=top_k)
            rag_answer = rag_model = None
        elif method == "embeddings":
            with st.spinner("Computing semantic role matches + academic compatibility..."):
                results = load_embedding_model().recommend(profile, top_k=top_k)
            rag_answer = rag_model = None
        else:
            with st.spinner(f"Retrieving enriched evidence and generating with {ollama_model}..."):
                advisor = RAGCareerAdvisor(load_embedding_model(), OllamaClient(ollama_base_url, timeout=180.0), model_name=ollama_model)
                rag_response = advisor.advise(profile, top_k=top_k)
                results = rag_response.sources
                rag_answer = rag_response.answer
                rag_model = rag_response.generator_model
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("v0.4 changes the retrieval corpus. Run build_dataset, train_tfidf, and build_embeddings again.")
        st.stop()
    except OllamaModelNotFoundError as exc:
        st.error(str(exc)); st.code(f"ollama pull {ollama_model}", language="bash"); st.stop()
    except OllamaConnectionError as exc:
        st.error(str(exc)); st.info("Open/start Ollama, then rerun the request."); st.stop()
    except (OllamaError, RuntimeError, ValueError) as exc:
        st.error(str(exc)); st.stop()

    if not results:
        st.warning("No career roles met the current constraints. Try removing the salary minimum first.")
        st.stop()

    st.divider()
    if method == "rag" and rag_answer:
        st.subheader("RAG career analysis")
        st.markdown(
            f"<div class='cv-rag'><strong>Generator:</strong> {rag_model} · <strong>Retriever:</strong> Sentence Embeddings + structured ranking</div>",
            unsafe_allow_html=True,
        )
        st.markdown(rag_answer)
        st.caption("The local LLM is asked to cite retrieved [CV#] evidence. Expand the cards below to inspect the underlying data.")
        st.subheader(f"Retrieved evidence · {len(results)} career roles")
    else:
        st.subheader(f"Top {len(results)} career roles · {engine_label}")

    st.caption("Combined scores are ranking signals, not probabilities. Academic compatibility comes from the NCES CIP-SOC crosswalk; job outlook comes from BLS projections when available.")
    for result in results:
        result_card(result, method)
