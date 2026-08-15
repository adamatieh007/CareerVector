from __future__ import annotations

from pathlib import Path

import streamlit as st

from careervector.config import EMBEDDING_INFO_PATH
from careervector.embedding_model import EmbeddingCareerVectorModel
from careervector.model import CareerVectorModel
from careervector.profile import CareerProfile
from careervector.text import split_csv_text


# --------------------`-------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="CareerVector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

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
    background: linear-gradient(
        135deg,
        rgba(70, 90, 255, 0.10),
        rgba(90, 200, 180, 0.08)
    );
}

.cv-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 2.25rem;
    font-weight: 750;
}

.cv-hero p {
    margin: 0;
    opacity: 0.82;
    font-size: 1.02rem;
    line-height: 1.55;
}

.cv-rank {
    display: inline-block;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid rgba(128, 128, 128, 0.3);
    margin-bottom: 0.35rem;
}

.cv-score {
    font-size: 1.55rem;
    font-weight: 750;
}

.cv-muted {
    opacity: 0.72;
    font-size: 0.92rem;
}

div[data-testid="stButton"] > button {
    border-radius: 10px;
    font-weight: 650;
}

div[data-baseweb="select"] {
    border-radius: 10px;
}
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource
def load_tfidf_model() -> CareerVectorModel:
    return CareerVectorModel.load()


@st.cache_resource
def load_embedding_model() -> EmbeddingCareerVectorModel:
    return EmbeddingCareerVectorModel.load()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def money(value: object) -> str:
    if value is None:
        return "N/A"

    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def result_card(
    item: dict[str, object],
    method: str,
) -> None:

    with st.container(border=True):

        left, right = st.columns([5, 1.3])

        with left:

            st.markdown(
                f"<span class='cv-rank'>#{item['rank']}</span>",
                unsafe_allow_html=True,
            )

            st.subheader(
                str(item["occupation"])
            )

            code = item.get("onet_soc_code")

            if code:
                st.caption(
                    f"O*NET-SOC {code}"
                )

        with right:

            st.markdown(
                f"""
<div class="cv-score">
{float(item["match_score"]):.1f}
</div>

<div class="cv-muted">
relevance score
</div>
                """,
                unsafe_allow_html=True,
            )

        description = item.get("description")

        if (
            description
            and str(description).lower() != "nan"
        ):
            st.write(
                str(description)
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Median salary",
            money(
                item.get("median_salary")
            ),
        )

        c2.metric(
            "Mean salary",
            money(
                item.get("mean_salary")
            ),
        )

        c3.metric(
            "Engine",
            (
                "TF-IDF"
                if method == "tfidf"
                else "Embeddings"
            ),
        )

        titles = (
            item.get("sample_job_titles")
            or []
        )

        if titles:

            st.markdown(
                "**Related job titles**"
            )

            st.write(
                " · ".join(
                    str(x)
                    for x in titles
                )
            )

        if (
            method == "tfidf"
            and item.get("matched_terms")
        ):

            st.markdown(
                "**Strong lexical matches**"
            )

            st.write(
                " · ".join(
                    str(x)
                    for x in item["matched_terms"]
                )
            )

        elif method == "embeddings":

            st.caption(
                "This score comes from semantic similarity "
                "in embedding space. A strong result does "
                "not require your wording to exactly match "
                "the occupation description."
            )

        details = []

        if item.get("top_interests"):
            details.append(
                (
                    "O*NET interests",
                    item["top_interests"],
                )
            )

        if item.get("top_skills"):
            details.append(
                (
                    "Top skills",
                    item["top_skills"],
                )
            )

        if item.get("top_knowledge"):
            details.append(
                (
                    "Knowledge areas",
                    item["top_knowledge"],
                )
            )

        if details:

            with st.expander(
                "Why this career may fit"
            ):

                for label, values in details:

                    st.markdown(
                        f"**{label}:** "
                        + ", ".join(
                            str(x)
                            for x in values
                        )
                    )


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    """
<div class="cv-hero">
<h1>🎯 CareerVector</h1>
<p>
Describe your background, interests, and career preferences.
Compare a lexical TF-IDF baseline with semantic sentence embeddings
using the same occupational dataset.
</p>
</div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header(
        "Model status"
    )

    st.write(
        "✅ TF-IDF artifacts"
    )

    if Path(
        EMBEDDING_INFO_PATH
    ).exists():

        st.write(
            "✅ Sentence-embedding artifacts"
        )

    else:

        st.write(
            "⚠️ Sentence embeddings not built yet"
        )

    st.divider()

    st.caption(
        "CareerVector ranks occupations locally. "
        "Salary filtering uses the BLS wage data "
        "stored in the project."
    )


# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

st.subheader(
    "Choose a recommendation method"
)

engine_label = st.radio(
    "Recommendation method",
    [
        "TF-IDF",
        "Sentence Embeddings",
    ],
    horizontal=True,
    label_visibility="collapsed",
    help=(
        "TF-IDF emphasizes shared words and phrases. "
        "Sentence Embeddings compare semantic meaning, "
        "so related ideas can match even when wording differs."
    ),
)

method = (
    "tfidf"
    if engine_label == "TF-IDF"
    else "embeddings"
)


if method == "tfidf":

    st.caption(
        "TF-IDF is a lexical baseline. It works best when "
        "your profile and an occupation share similar words "
        "or phrases."
    )

else:

    st.caption(
        "Sentence Embeddings use semantic similarity, allowing "
        "related concepts to match even when the exact wording "
        "is different."
    )


# ---------------------------------------------------------
# PROFILE INPUTS
# ---------------------------------------------------------

st.subheader(
    "Tell CareerVector about yourself"
)


major = st.text_input(
    "1. What are you studying, or what is your academic background?",
    placeholder="e.g. Computer Engineering",
    help=(
        "Enter your degree, major, program, or primary "
        "academic background."
    ),
)


interests = st.text_area(
    "2. What subjects, technologies, or problems are you interested in?",
    placeholder=(
        "e.g. FPGA design, computer architecture, embedded systems\n"
        "or: Radiation oncology, medical physics, dosimetry"
    ),
    height=115,
    help=(
        "Focus on the subjects and technical areas "
        "you want your career to involve."
    ),
)


career_preferences = st.text_area(
    "3. What would you like your career to involve?",
    placeholder=(
        "e.g. Hardware design, performance optimization, low-level coding\n"
        "or: Research, clinical work, quantitative problem solving"
    ),
    height=115,
    help=(
        "Describe the actual type of work you would enjoy doing "
        "day to day."
    ),
)


avoid = st.text_area(
    "4. Anything you definitely want to avoid?",
    placeholder=(
        "e.g. Web development, sales, product management (optional)"
    ),
    height=90,
    help=(
        "Optional. CareerVector will penalize occupations "
        "that strongly resemble these preferences."
    ),
)


# ---------------------------------------------------------
# SALARY
# ---------------------------------------------------------

st.markdown(
    "### Salary preference"
)

salary_choice = st.selectbox(
    "Minimum median annual salary",
    [
        "No minimum",
        "$60,000",
        "$80,000",
        "$100,000",
        "$120,000",
        "$150,000",
        "Custom",
    ],
    index=0,
    help=(
        "CareerVector uses available BLS median wage data "
        "as a hard salary filter."
    ),
)


if salary_choice == "Custom":

    custom_salary = st.number_input(
        "Enter your minimum annual salary",
        min_value=0,
        max_value=1_000_000,
        value=120_000,
        step=5_000,
        format="%d",
        help=(
            "Enter any annual salary threshold you want."
        ),
    )

    minimum_salary = float(
        custom_salary
    )


elif salary_choice == "No minimum":

    minimum_salary = None


else:

    minimum_salary = float(
        salary_choice
        .replace("$", "")
        .replace(",", "")
    )


# ---------------------------------------------------------
# NUMBER OF RESULTS
# ---------------------------------------------------------

top_k = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=15,
    value=8,
)


# ---------------------------------------------------------
# SUBMIT BUTTON
# ---------------------------------------------------------

submitted = st.button(
    "Find my careers",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# RUN MODEL
# ---------------------------------------------------------

if submitted:

    profile = CareerProfile(
        major=major,
        interests=split_csv_text(
            interests
        ),
        preferred_work=split_csv_text(
            career_preferences
        ),
        avoid=split_csv_text(
            avoid
        ),
        minimum_salary=minimum_salary,
    )


    if not profile.positive_text().strip():

        st.warning(
            "Add at least your academic background, "
            "an interest, or a career preference."
        )

        st.stop()


    try:

        if method == "tfidf":

            spinner_message = (
                "Ranking careers with TF-IDF..."
            )

        else:

            spinner_message = (
                "Computing semantic matches..."
            )


        with st.spinner(
            spinner_message
        ):

            if method == "tfidf":

                model = (
                    load_tfidf_model()
                )

            else:

                model = (
                    load_embedding_model()
                )


            results = model.recommend(
                profile,
                top_k=top_k,
            )


    except FileNotFoundError as exc:

        st.error(
            str(exc)
        )

        if method == "embeddings":

            st.info(
                "Build the embedding artifacts first:"
            )

            st.code(
                "python scripts/build_embeddings.py",
                language="bash",
            )

        st.stop()


    except RuntimeError as exc:

        st.error(
            str(exc)
        )

        st.stop()


    # -----------------------------------------------------
    # NO RESULTS
    # -----------------------------------------------------

    if not results:

        st.warning(
            "No occupations met the current constraints. "
            "If you selected a minimum salary, try lowering "
            "it or choosing 'No minimum' to inspect the "
            "underlying recommendations."
        )

        st.stop()


    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        f"Top {len(results)} recommendations · {engine_label}"
    )

    st.caption(
        "Relevance scores are ranking scores derived from "
        "cosine similarity and scaled by 100. They are not "
        "probabilities or percentages of career suitability."
    )


    for result in results:

        result_card(
            result,
            method,
        )