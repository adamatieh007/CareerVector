from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from careervector.config import BLS_NATIONAL_XLSX
from careervector.text import clean_text, normalize_soc, repeat_phrases

CODE = "O*NET-SOC Code"


def _group_strings(
    df: pd.DataFrame,
    value_col: str,
    *,
    sort_col: str | None = None,
    ascending: bool = False,
    max_items: int | None = None,
    separator: str = " | ",
) -> dict[str, str]:
    work = df.copy()
    work[value_col] = work[value_col].map(clean_text)
    work = work[work[value_col] != ""]
    if sort_col and sort_col in work.columns:
        work = work.sort_values([CODE, sort_col], ascending=[True, ascending])

    result: dict[str, str] = {}
    for code, group in work.groupby(CODE, sort=False):
        vals = group[value_col].drop_duplicates().tolist()
        if max_items is not None:
            vals = vals[:max_items]
        result[str(code)] = separator.join(vals)
    return result


def _rating_terms(
    df: pd.DataFrame,
    scale_id: str,
    *,
    top_k: int,
    min_value: float | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    work = df[df["Scale ID"].astype(str) == scale_id].copy()
    work["Data Value"] = pd.to_numeric(work["Data Value"], errors="coerce")
    work = work.dropna(subset=["Data Value"])
    if min_value is not None:
        work = work[work["Data Value"] >= min_value]
    work = work.sort_values([CODE, "Data Value"], ascending=[True, False])

    weighted: dict[str, str] = {}
    display: dict[str, str] = {}
    for code, group in work.groupby(CODE, sort=False):
        g = group.head(top_k)
        pairs = list(zip(g["Element Name"].astype(str), g["Data Value"].astype(float)))
        weighted[str(code)] = repeat_phrases(pairs)
        display[str(code)] = " | ".join(clean_text(x) for x in g["Element Name"].tolist())
    return weighted, display


def _load_bls_wages(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["base_soc", "mean_salary", "median_salary", "p10_salary", "p90_salary"])

    wages = pd.read_excel(path)
    wages.columns = [str(c).strip().upper() for c in wages.columns]
    required = {"OCC_CODE", "A_MEAN", "A_MEDIAN"}
    if not required.issubset(wages.columns):
        raise ValueError(f"Unexpected BLS columns in {path.name}: {wages.columns.tolist()}")

    if "O_GROUP" in wages.columns:
        detailed = wages[wages["O_GROUP"].astype(str).str.lower().eq("detailed")]
        if not detailed.empty:
            wages = detailed

    def numeric(col: str) -> pd.Series:
        if col not in wages.columns:
            return pd.Series(np.nan, index=wages.index)
        return pd.to_numeric(wages[col].replace({"*": np.nan, "#": np.nan, "**": np.nan}), errors="coerce")

    out = pd.DataFrame(
        {
            "base_soc": wages["OCC_CODE"].astype(str).str.strip(),
            "mean_salary": numeric("A_MEAN"),
            "median_salary": numeric("A_MEDIAN"),
            "p10_salary": numeric("A_PCT10"),
            "p90_salary": numeric("A_PCT90"),
        }
    )
    return out.drop_duplicates("base_soc")


def build_occupation_documents(raw_dir: Path) -> pd.DataFrame:
    occupations = pd.read_csv(raw_dir / "occupation_data.csv")
    occupations[CODE] = occupations[CODE].astype(str)

    job_titles = pd.read_csv(raw_dir / "job_titles.csv")
    tasks = pd.read_csv(raw_dir / "task_statements.csv")
    skills = pd.read_csv(raw_dir / "essential_skills.csv")
    knowledge = pd.read_csv(raw_dir / "knowledge.csv")
    interests = pd.read_csv(raw_dir / "specific_interest_areas.csv")
    software = pd.read_csv(raw_dir / "software_skills.csv")
    activities = pd.read_csv(raw_dir / "work_activities.csv")

    alt_titles = _group_strings(job_titles, "Job Title", max_items=60)

    core_tasks = tasks[tasks["Task Type"].fillna("").astype(str).str.lower().eq("core")]
    if core_tasks.empty:
        core_tasks = tasks
    task_text = _group_strings(core_tasks, "Task", max_items=20, separator=" ")

    skill_weighted, skill_display = _rating_terms(skills, "IM", top_k=10, min_value=2.0)
    knowledge_weighted, knowledge_display = _rating_terms(knowledge, "IM", top_k=15, min_value=2.0)
    interest_weighted, interest_display = _rating_terms(interests, "OI", top_k=12)
    activity_weighted, activity_display = _rating_terms(activities, "IM", top_k=15, min_value=2.0)

    # Make hot / in-demand software appear more often in the document.
    software = software.copy()
    software["boost"] = 1 + software["Hot Technology"].eq("Y").astype(int) + software["In Demand"].eq("Y").astype(int)
    software["weighted_software"] = software.apply(
        lambda r: " ".join([clean_text(r["Workplace Example"])] * int(r["boost"])), axis=1
    )
    software_text = _group_strings(software, "weighted_software", max_items=40, separator=" ")
    software_display = _group_strings(software, "Workplace Example", max_items=20)

    records: list[dict[str, object]] = []
    for code_value, title_value, description_value in occupations[[CODE, "Title", "Description"]].itertuples(index=False, name=None):
        code = str(code_value)
        title = clean_text(title_value)
        description = clean_text(description_value)

        sections = [
            f"occupation {title} {title} {title}",
            f"description {description} {description}",
            f"job titles {alt_titles.get(code, '')}",
            f"specific interests {interest_weighted.get(code, '')}",
            f"skills {skill_weighted.get(code, '')}",
            f"knowledge {knowledge_weighted.get(code, '')}",
            f"work activities {activity_weighted.get(code, '')}",
            f"tasks {task_text.get(code, '')}",
            f"software technologies {software_text.get(code, '')}",
        ]
        document = " ".join(clean_text(s) for s in sections if clean_text(s))

        records.append(
            {
                "onet_soc_code": code,
                "base_soc": normalize_soc(code),
                "title": title,
                "description": description,
                "job_titles": alt_titles.get(code, ""),
                "top_interests": interest_display.get(code, ""),
                "top_skills": skill_display.get(code, ""),
                "top_knowledge": knowledge_display.get(code, ""),
                "top_activities": activity_display.get(code, ""),
                "software_skills": software_display.get(code, ""),
                "core_tasks": task_text.get(code, ""),
                "document": document,
            }
        )

    frame = pd.DataFrame(records)
    wages = _load_bls_wages(raw_dir / BLS_NATIONAL_XLSX)
    if not wages.empty:
        frame = frame.merge(wages, on="base_soc", how="left")
    else:
        for col in ["mean_salary", "median_salary", "p10_salary", "p90_salary"]:
            frame[col] = np.nan

    return frame.sort_values("onet_soc_code").reset_index(drop=True)
