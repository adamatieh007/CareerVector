from __future__ import annotations

from pathlib import Path

import streamlit as st

from careervector.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    EMBEDDING_INFO_PATH,
)
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


st.set_page_config(
    page_title="CareerVector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1080px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.cv-hero {
    padding: 1.5rem 1.6rem;
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 18px;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(70, 90, 255, 0.10), rgba(90, 200, 180, 0.08));
}
.cv-hero h1 { margin: 0 0 0.35rem 0; font-size: 2.25rem; font-weight: 750; }
.cv-hero p { margin: 0; opacity: 0.82; font-size: 1.02rem; line-height: 1.55; }
.cv-rank {
    display: inline-block;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid rgba(128, 128, 128, 0.3);
    margin-bottom: 0.35rem;
}
.cv-score { font-size: 1.55rem; font-weight: 750; }
.cv-muted { opacity: 0.72; font-size: 0.92rem; }
.cv-rag {
    border: 1px solid rgba(112, 225, 195, 0.32);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    background: rgba(112, 225, 195, 0.055);
    margin: 0.5rem 0 1.5rem;
}
div[data-testid="stButton"] > button { border-radius: 10px; font-weight: 650; }
div[data-baseweb="select"] { border-radius: 10px; }
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


def result_card(item: dict[str, object], method: str) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1.3])
        with left:
            st.markdown(f"<span class='cv-rank'>#{item['rank']}</span>", unsafe_allow_html=True)
            st.subheader(str(item["occupation"]))
            if item.get("onet_soc_code"):
                st.caption(f"O*NET-SOC {item['onet_soc_code']}")

        with right:
            st.markdown(
                f"""
<div class="cv-score">{float(item['match_score']):.1f}</div>
<div class="cv-muted">relevance score</div>
                """,
                unsafe_allow_html=True,
            )

        description = item.get("description")
        if description and str(description).lower() != "nan":
            st.write(str(description))

        c1, c2, c3 = st.columns(3)
        c1.metric("Median salary", money(item.get("median_salary")))
        c2.metric("Mean salary", money(item.get("mean_salary")))
        engine_name = {
            "tfidf": "TF-IDF",
            "embeddings": "Embeddings",
            "rag": "RAG retrieval",
        }[method]
        c3.metric("Engine", engine_name)

        titles = item.get("sample_job_titles") or []
        if titles:
            st.markdown("**Related job titles**")
            st.write(" · ".join(str(x) for x in titles))

        if method == "tfidf" and item.get("matched_terms"):
            st.markdown("**Strong lexical matches**")
            st.write(" · ".join(str(x) for x in item["matched_terms"]))
        elif method in {"embeddings", "rag"}:
            st.caption(
                "The retrieval score comes from semantic similarity in embedding space. "
                "It is a ranking signal, not a probability."
            )

        details: list[tuple[str, object]] = []
        for label, key in [
            ("O*NET interests", "top_interests"),
            ("Top skills", "top_skills"),
            ("Knowledge areas", "top_knowledge"),
            ("Work activities", "top_activities"),
            ("Core tasks", "core_tasks"),
            ("Software / technologies", "software_skills"),
        ]:
            if item.get(key):
                details.append((label, item[key]))

        if details:
            with st.expander("Retrieved evidence"):
                for label, values in details:
                    if isinstance(values, list):
                        rendered = ", ".join(str(x) for x in values)
                    else:
                        rendered = str(values)
                    st.markdown(f"**{label}:** {rendered}")


st.markdown(
    """
<div class="cv-hero">
<h1>🎯 CareerVector</h1>
<p>
Describe your background, interests, and career preferences. Compare lexical TF-IDF,
semantic sentence embeddings, or a local RAG workflow that retrieves O*NET/BLS evidence
and asks an Ollama-hosted LLM to explain the matches.
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
    help=(
        "TF-IDF emphasizes shared words and phrases. Sentence Embeddings compare semantic meaning. "
        "RAG uses the embedding retriever, then sends the retrieved evidence to a local Ollama model "
        "for a grounded explanation."
    ),
)
method = {
    "TF-IDF": "tfidf",
    "Sentence Embeddings": "embeddings",
    "RAG · Embeddings + Local LLM": "rag",
}[engine_label]

ollama_base_url = DEFAULT_OLLAMA_BASE_URL
ollama_model = DEFAULT_OLLAMA_MODEL
ollama_models: list[str] = []
ollama_error: str | None = None

if method == "tfidf":
    st.caption("TF-IDF is the lexical baseline and rewards shared words and phrases.")
elif method == "embeddings":
    st.caption("Sentence Embeddings retrieve semantically related careers even when wording differs.")
else:
    st.caption(
        "RAG first retrieves careers with sentence embeddings, then gives only that retrieved "
        "O*NET/BLS evidence to a local Ollama LLM for a cited explanation."
    )
    with st.expander("Local RAG settings", expanded=True):
        ollama_base_url = st.text_input(
            "Ollama server",
            value=DEFAULT_OLLAMA_BASE_URL,
            help="Default local Ollama API address is http://localhost:11434.",
        )
        try:
            ollama_models = OllamaClient(ollama_base_url, timeout=2.0).list_models()
        except OllamaError as exc:
            ollama_error = str(exc)

        if ollama_models:
            default_index = next(
                (i for i, name in enumerate(ollama_models) if name == DEFAULT_OLLAMA_MODEL),
                0,
            )
            ollama_model = st.selectbox(
                "Local generator model",
                ollama_models,
                index=default_index,
                help="These models are reported by your local Ollama /api/tags endpoint.",
            )
        else:
            ollama_model = st.text_input(
                "Local generator model",
                value=DEFAULT_OLLAMA_MODEL,
                help="Example: gemma3. Pull it with `ollama pull gemma3`.",
            )
            if ollama_error:
                st.warning(ollama_error)
            st.code("ollama pull gemma3", language="bash")

with st.sidebar:
    st.header("Model status")
    st.write("✅ TF-IDF artifacts")
    if Path(EMBEDDING_INFO_PATH).exists():
        st.write("✅ Sentence-embedding artifacts")
    else:
        st.write("⚠️ Sentence embeddings not built yet")
    if method == "rag":
        if ollama_models:
            st.write(f"✅ Ollama · {len(ollama_models)} local model(s)")
        else:
            st.write("⚠️ Ollama unavailable / no models")
    st.divider()
    st.caption(
        "Retrieval and generation run locally. Salary filtering uses BLS wage data stored in the project."
    )

st.subheader("Tell CareerVector about yourself")
major = st.text_input(
    "1. What are you studying, or what is your academic background?",
    placeholder="e.g. Computer Engineering",
)
interests = st.text_area(
    "2. What subjects, technologies, or problems are you interested in?",
    placeholder=(
        "e.g. FPGA design, computer architecture, embedded systems\n"
        "or: Radiation oncology, medical physics, dosimetry"
    ),
    height=115,
)
specializations = st.text_area(
    "3. Any specializations or technical areas you want to emphasize?",
    placeholder="e.g. Digital logic, CUDA, radiation dosimetry, medical imaging (optional)",
    height=90,
)
career_preferences = st.text_area(
    "4. What would you like your career to involve?",
    placeholder=(
        "e.g. Hardware design, performance optimization, low-level coding\n"
        "or: Research, clinical work, quantitative problem solving"
    ),
    height=115,
)
avoid = st.text_area(
    "5. Anything you definitely want to avoid?",
    placeholder="e.g. Web development, sales, product management (optional)",
    height=90,
)

st.markdown("### Salary preference")
salary_choice = st.selectbox(
    "Minimum median annual salary",
    ["No minimum", "$60,000", "$80,000", "$100,000", "$120,000", "$150,000", "Custom"],
    index=0,
    help="BLS median wage is applied as a hard filter when selected.",
)
if salary_choice == "Custom":
    minimum_salary = float(
        st.number_input(
            "Enter your minimum annual salary",
            min_value=0,
            max_value=1_000_000,
            value=120_000,
            step=5_000,
            format="%d",
        )
    )
elif salary_choice == "No minimum":
    minimum_salary = None
else:
    minimum_salary = float(salary_choice.replace("$", "").replace(",", ""))

top_k = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=15,
    value=5 if method == "rag" else 8,
    help="For RAG, fewer high-quality sources usually produce a more focused local-LLM context.",
)

submitted = st.button("Find my careers", type="primary", use_container_width=True)

if submitted:
    profile = CareerProfile(
        major=major,
        interests=split_csv_text(interests),
        specializations=split_csv_text(specializations),
        preferred_work=split_csv_text(career_preferences),
        avoid=split_csv_text(avoid),
        minimum_salary=minimum_salary,
    )

    if not profile.positive_text().strip():
        st.warning("Add at least your academic background, an interest, specialization, or career preference.")
        st.stop()

    try:
        if method == "tfidf":
            with st.spinner("Ranking careers with TF-IDF..."):
                results = load_tfidf_model().recommend(profile, top_k=top_k)
            rag_answer = None
            rag_model = None
        elif method == "embeddings":
            with st.spinner("Computing semantic matches..."):
                results = load_embedding_model().recommend(profile, top_k=top_k)
            rag_answer = None
            rag_model = None
        else:
            with st.spinner(f"Retrieving evidence and generating with {ollama_model}..."):
                advisor = RAGCareerAdvisor(
                    load_embedding_model(),
                    OllamaClient(ollama_base_url, timeout=180.0),
                    model_name=ollama_model,
                )
                rag_response = advisor.advise(profile, top_k=top_k)
                results = rag_response.sources
                rag_answer = rag_response.answer
                rag_model = rag_response.generator_model

    except FileNotFoundError as exc:
        st.error(str(exc))
        if method in {"embeddings", "rag"}:
            st.info("Build the embedding artifacts first:")
            st.code("python scripts/build_embeddings.py", language="bash")
        st.stop()
    except OllamaModelNotFoundError as exc:
        st.error(str(exc))
        st.code(f"ollama pull {ollama_model}", language="bash")
        st.stop()
    except OllamaConnectionError as exc:
        st.error(str(exc))
        st.info("Open/start Ollama, then rerun the request.")
        st.stop()
    except (OllamaError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    if not results:
        st.warning(
            "No occupations met the current constraints. If you selected a minimum salary, "
            "try lowering it or choosing 'No minimum' to inspect the underlying recommendations."
        )
        st.stop()

    st.divider()

    if method == "rag" and rag_answer:
        st.subheader("RAG career analysis")
        st.markdown(
            f"<div class='cv-rag'><strong>Generator:</strong> {rag_model} · "
            "<strong>Retriever:</strong> Sentence Embeddings</div>",
            unsafe_allow_html=True,
        )
        st.markdown(rag_answer)
        st.caption(
            "CareerVector asks the local LLM to ground factual career claims in the retrieved "
            "sources labeled [CV1], [CV2], etc. Generated explanations can still be imperfect; "
            "expand the evidence cards below to inspect the source data."
        )
        st.subheader(f"Retrieved evidence · {len(results)} occupations")
    else:
        st.subheader(f"Top {len(results)} recommendations · {engine_label}")

    st.caption(
        "Relevance scores are cosine-similarity ranking signals scaled by 100. "
        "They are not probabilities or percentages of career suitability."
    )
    for result in results:
        result_card(result, method)
