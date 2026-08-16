from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from careervector.config import (
    BLS_NATIONAL_XLSX,
    BLS_PROJECTIONS_XLSX,
    ESCO_DIR,
    NCES_CIP_SOC_XLSX,
)
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


def _numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(
        frame[col].replace({"*": np.nan, "#": np.nan, "**": np.nan, "—": np.nan, "-": np.nan}),
        errors="coerce",
    )


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

    out = pd.DataFrame(
        {
            "base_soc": wages["OCC_CODE"].astype(str).str.strip(),
            "mean_salary": _numeric_series(wages, "A_MEAN"),
            "median_salary": _numeric_series(wages, "A_MEDIAN"),
            "p10_salary": _numeric_series(wages, "A_PCT10"),
            "p90_salary": _numeric_series(wages, "A_PCT90"),
        }
    )
    return out.drop_duplicates("base_soc")


def _load_cip_soc(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load NCES 2020 CIP -> 2018 SOC relationships.

    Returns both the raw normalized crosswalk and a SOC -> pipe-delimited program-title map.
    """
    if not path.exists():
        return pd.DataFrame(columns=["cip_code", "cip_title", "base_soc", "soc_title"]), {}

    frame = pd.read_excel(path, sheet_name="CIP-SOC")
    required = {"CIP2020Code", "CIP2020Title", "SOC2018Code", "SOC2018Title"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected NCES CIP-SOC columns: {frame.columns.tolist()}")

    out = pd.DataFrame(
        {
            "cip_code": frame["CIP2020Code"].astype(str).str.strip(),
            "cip_title": frame["CIP2020Title"].map(clean_text),
            "base_soc": frame["SOC2018Code"].astype(str).str.strip().map(normalize_soc),
            "soc_title": frame["SOC2018Title"].map(clean_text),
        }
    ).dropna(subset=["base_soc"])
    out = out[(out["cip_title"] != "") & (out["base_soc"] != "")]

    by_soc: dict[str, str] = {}
    for soc, group in out.groupby("base_soc", sort=False):
        # Keep the corpus compact while preserving a broad set of official program labels.
        titles = group["cip_title"].drop_duplicates().tolist()[:60]
        by_soc[str(soc)] = " | ".join(titles)
    return out.reset_index(drop=True), by_soc


def _load_bls_projections(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "base_soc",
                "employment_2024",
                "employment_2034",
                "growth_percent",
                "annual_openings",
                "projection_median_salary",
                "typical_education",
                "related_work_experience",
                "on_the_job_training",
            ]
        )

    raw = pd.read_excel(path, sheet_name="Table 1.2", header=1)
    # The BLS workbook uses long human-readable headers. Match by stable prefixes.
    columns = {str(c).strip(): c for c in raw.columns}

    def find(prefix: str) -> object | None:
        return next((orig for text, orig in columns.items() if text.startswith(prefix)), None)

    code_col = find("2024 National Employment Matrix code")
    type_col = find("Occupation type")
    if code_col is None:
        raise ValueError(f"Could not find occupation code column in {path.name}")

    frame = raw.copy()
    if type_col is not None:
        detailed = frame[frame[type_col].astype(str).str.lower().eq("line item")]
        if not detailed.empty:
            frame = detailed

    def values(prefix: str) -> pd.Series:
        col = find(prefix)
        if col is None:
            return pd.Series(np.nan, index=frame.index)
        return frame[col]

    out = pd.DataFrame(
        {
            "base_soc": frame[code_col].astype(str).str.strip().map(normalize_soc),
            "employment_2024": pd.to_numeric(values("Employment, 2024"), errors="coerce"),
            "employment_2034": pd.to_numeric(values("Employment, 2034"), errors="coerce"),
            "growth_percent": pd.to_numeric(values("Employment change, percent"), errors="coerce"),
            "annual_openings": pd.to_numeric(values("Occupational openings"), errors="coerce"),
            "projection_median_salary": pd.to_numeric(values("Median annual wage"), errors="coerce"),
            "typical_education": values("Typical education needed for entry").map(clean_text),
            "related_work_experience": values("Work experience in a related occupation").map(clean_text),
            "on_the_job_training": values("Typical on-the-job training").map(clean_text),
        }
    )
    return out[out["base_soc"] != ""].drop_duplicates("base_soc").reset_index(drop=True)


def _merge_external_metadata(frame: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    _, majors_by_soc = _load_cip_soc(raw_dir / NCES_CIP_SOC_XLSX)
    frame = frame.copy()
    frame["compatible_majors"] = frame["base_soc"].map(majors_by_soc).fillna("")

    projections = _load_bls_projections(raw_dir / BLS_PROJECTIONS_XLSX)
    if not projections.empty:
        frame = frame.merge(projections, on="base_soc", how="left")
    else:
        for col in [
            "employment_2024",
            "employment_2034",
            "growth_percent",
            "annual_openings",
            "projection_median_salary",
        ]:
            frame[col] = np.nan
        for col in ["typical_education", "related_work_experience", "on_the_job_training"]:
            frame[col] = ""

    wages = _load_bls_wages(raw_dir / BLS_NATIONAL_XLSX)
    if not wages.empty:
        frame = frame.merge(wages, on="base_soc", how="left")
    else:
        for col in ["mean_salary", "median_salary", "p10_salary", "p90_salary"]:
            frame[col] = np.nan

    # If OEWS is missing for an occupation, the Employment Projections table still carries
    # a median annual wage. Use it as a transparent fallback, never as a mean/percentile wage.
    if "projection_median_salary" in frame.columns:
        frame["median_salary"] = pd.to_numeric(frame["median_salary"], errors="coerce").fillna(
            pd.to_numeric(frame["projection_median_salary"], errors="coerce")
        )
    return frame


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

    alt_titles = _group_strings(job_titles, "Job Title", max_items=100)

    core_tasks = tasks[tasks["Task Type"].fillna("").astype(str).str.lower().eq("core")]
    if core_tasks.empty:
        core_tasks = tasks
    task_text = _group_strings(core_tasks, "Task", max_items=20, separator=" ")

    skill_weighted, skill_display = _rating_terms(skills, "IM", top_k=10, min_value=2.0)
    knowledge_weighted, knowledge_display = _rating_terms(knowledge, "IM", top_k=15, min_value=2.0)
    interest_weighted, interest_display = _rating_terms(interests, "OI", top_k=12)
    activity_weighted, activity_display = _rating_terms(activities, "IM", top_k=15, min_value=2.0)

    software = software.copy()
    software["boost"] = 1 + software["Hot Technology"].eq("Y").astype(int) + software["In Demand"].eq("Y").astype(int)
    software["weighted_software"] = software.apply(
        lambda r: " ".join([clean_text(r["Workplace Example"])] * int(r["boost"])), axis=1
    )
    software_text = _group_strings(software, "weighted_software", max_items=40, separator=" ")
    software_display = _group_strings(software, "Workplace Example", max_items=25)

    records: list[dict[str, object]] = []
    for code_value, title_value, description_value in occupations[[CODE, "Title", "Description"]].itertuples(index=False, name=None):
        code = str(code_value)
        base_soc = normalize_soc(code)
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
                "base_soc": base_soc,
                "title": title,
                "parent_title": title,
                "source": "O*NET",
                "role_kind": "occupation",
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
    frame = _merge_external_metadata(frame, raw_dir)

    # Enrich lexical documents with official academic compatibility and labor-market metadata.
    academic = frame["compatible_majors"].fillna("").map(lambda x: f"compatible majors fields of study {x}" if x else "")
    labor = frame.apply(
        lambda r: clean_text(
            f"education {r.get('typical_education', '')} employment growth {r.get('growth_percent', '')} "
            f"annual openings {r.get('annual_openings', '')}"
        ),
        axis=1,
    )
    frame["document"] = (frame["document"].fillna("") + " " + academic + " " + labor).map(clean_text)
    return frame.sort_values("onet_soc_code").reset_index(drop=True)


def _role_id(source: str, parent_code: str, title: str) -> str:
    digest = hashlib.sha1(f"{source}|{parent_code}|{title.lower()}".encode("utf-8")).hexdigest()[:12]
    return f"{source.lower()}:{digest}"


def build_career_role_documents(raw_dir: Path, *, include_esco: bool = True) -> pd.DataFrame:
    """Build a granular career-role corpus from official occupation/title data.

    O*NET's ~1K occupation records become tens of thousands of retrievable role-title
    documents. Every alternate title inherits its parent occupation's validated skills,
    interests, tasks, academic crosswalk, wages, and projections.
    """
    occupations = build_occupation_documents(raw_dir)
    titles = pd.read_csv(raw_dir / "job_titles.csv")[[CODE, "Job Title"]].copy()
    titles[CODE] = titles[CODE].astype(str)
    titles["Job Title"] = titles["Job Title"].map(clean_text)
    titles = titles[titles["Job Title"] != ""].drop_duplicates([CODE, "Job Title"])

    titles_by_code: dict[str, list[str]] = {}
    for code, group in titles.groupby(CODE, sort=False):
        titles_by_code[str(code)] = group["Job Title"].drop_duplicates().tolist()

    records: list[dict[str, object]] = []
    for _, parent in occupations.iterrows():
        code = str(parent["onet_soc_code"])
        canonical = clean_text(parent["title"])
        role_titles = [canonical] + titles_by_code.get(code, [])
        seen: set[str] = set()
        role_titles = [t for t in role_titles if t and not (t.lower() in seen or seen.add(t.lower()))]

        all_related = titles_by_code.get(code, [])
        for role_title in role_titles:
            related = [t for t in all_related if t.lower() != role_title.lower()][:30]
            kind = "canonical" if role_title.lower() == canonical.lower() else "alternate_title"

            row = parent.to_dict()
            row.update(
                {
                    "role_id": _role_id("ONET", code, role_title),
                    "title": role_title,
                    "parent_title": canonical,
                    "source": "O*NET",
                    "role_kind": kind,
                    "job_titles": " | ".join(related),
                }
            )
            row["document"] = clean_text(
                " ".join(
                    [
                        f"career role {role_title} {role_title} {role_title}",
                        f"parent occupation {canonical} {canonical}",
                        f"description {row.get('description', '')}",
                        f"related titles {row.get('job_titles', '')}",
                        f"compatible majors fields of study {row.get('compatible_majors', '')}",
                        f"specific interests {row.get('top_interests', '')}",
                        f"skills {row.get('top_skills', '')}",
                        f"knowledge {row.get('top_knowledge', '')}",
                        f"work activities {row.get('top_activities', '')}",
                        f"tasks {row.get('core_tasks', '')}",
                        f"software technologies {row.get('software_skills', '')}",
                        f"typical education {row.get('typical_education', '')}",
                    ]
                )
            )
            records.append(row)

    role_frame = pd.DataFrame(records)

    if include_esco:
        esco = load_esco_roles(ESCO_DIR)
        if not esco.empty:
            # Align schemas before concatenation. ESCO roles are intentionally missing U.S.-specific
            # wage/CIP fields; ranking redistributes those unavailable structured weights.
            for col in role_frame.columns:
                if col not in esco.columns:
                    esco[col] = np.nan if role_frame[col].dtype.kind in "fiu" else ""
            for col in esco.columns:
                if col not in role_frame.columns:
                    role_frame[col] = np.nan if esco[col].dtype.kind in "fiu" else ""
            role_frame = pd.concat([role_frame, esco[role_frame.columns]], ignore_index=True)

    return role_frame.drop_duplicates("role_id").reset_index(drop=True)


def _normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]", "", str(col).lower()): str(col) for col in frame.columns}


def _find_col(frame: pd.DataFrame, *names: str) -> str | None:
    normalized = _normalized_columns(frame)
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in normalized:
            return normalized[key]
    return None


def _find_esco_file(root: Path, keywords: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for path in root.rglob("*.csv"):
        name = path.name.lower()
        if all(keyword in name for keyword in keywords):
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def load_esco_roles(esco_dir: Path) -> pd.DataFrame:
    """Best-effort importer for the official ESCO English CSV classification package.

    ESCO's download portal can change package filenames, so detection is based on both
    filenames and standard concept columns instead of one brittle hard-coded filename.
    """
    if not esco_dir.exists():
        return pd.DataFrame()

    occupation_candidates = [
        p for p in esco_dir.rglob("*.csv")
        if "occupation" in p.name.lower()
        and "skill" not in p.name.lower()
        and "relation" not in p.name.lower()
    ]
    occupation_path = sorted(occupation_candidates)[0] if occupation_candidates else None
    if occupation_path is None:
        return pd.DataFrame()

    occupations = pd.read_csv(occupation_path, low_memory=False)
    uri_col = _find_col(occupations, "conceptUri", "uri")
    title_col = _find_col(occupations, "preferredLabel", "preferredTerm", "title")
    alt_col = _find_col(occupations, "altLabels", "alternativeLabels", "altLabel")
    desc_col = _find_col(occupations, "description", "definition")
    if uri_col is None or title_col is None:
        return pd.DataFrame()

    skill_map: dict[str, list[str]] = {}
    skill_candidates = [
        p for p in esco_dir.rglob("*.csv")
        if "skill" in p.name.lower()
        and "occupation" not in p.name.lower()
        and "relation" not in p.name.lower()
    ]
    skills_path = sorted(skill_candidates)[0] if skill_candidates else None
    relation_path = next(
        (
            p
            for p in sorted(esco_dir.rglob("*.csv"))
            if "occupation" in p.name.lower() and "skill" in p.name.lower()
            and ("relation" in p.name.lower() or "association" in p.name.lower())
        ),
        None,
    )
    if skills_path is not None and relation_path is not None and skills_path != relation_path:
        try:
            skills = pd.read_csv(skills_path, low_memory=False)
            relations = pd.read_csv(relation_path, low_memory=False)
            skill_uri = _find_col(skills, "conceptUri", "skillUri", "uri")
            skill_label = _find_col(skills, "preferredLabel", "preferredTerm", "title")
            rel_occ = _find_col(relations, "occupationUri", "occupationConceptUri")
            rel_skill = _find_col(relations, "skillUri", "skillConceptUri")
            if skill_uri and skill_label and rel_occ and rel_skill:
                labels = dict(zip(skills[skill_uri].astype(str), skills[skill_label].map(clean_text), strict=False))
                for occ_uri, group in relations.groupby(rel_occ, sort=False):
                    vals = [labels.get(str(uri), "") for uri in group[rel_skill].tolist()]
                    skill_map[str(occ_uri)] = [value for value in dict.fromkeys(vals) if value][:25]
        except Exception:
            # ESCO is optional enrichment; an unfamiliar package layout should not break the
            # U.S. O*NET/NCES/BLS build.
            skill_map = {}

    records: list[dict[str, object]] = []
    for _, item in occupations.iterrows():
        uri = clean_text(item.get(uri_col, ""))
        title = clean_text(item.get(title_col, ""))
        if not title:
            continue
        alt = clean_text(item.get(alt_col, "")) if alt_col else ""
        description = clean_text(item.get(desc_col, "")) if desc_col else ""
        esco_skills = skill_map.get(uri, [])
        skills_text = " | ".join(esco_skills)
        records.append(
            {
                "role_id": _role_id("ESCO", uri, title),
                "onet_soc_code": "",
                "base_soc": "",
                "title": title,
                "parent_title": title,
                "source": "ESCO",
                "role_kind": "esco_occupation",
                "description": description,
                "job_titles": alt.replace("\n", " | "),
                "top_interests": "",
                "top_skills": skills_text,
                "top_knowledge": "",
                "top_activities": "",
                "software_skills": "",
                "core_tasks": "",
                "compatible_majors": "",
                "mean_salary": np.nan,
                "median_salary": np.nan,
                "p10_salary": np.nan,
                "p90_salary": np.nan,
                "employment_2024": np.nan,
                "employment_2034": np.nan,
                "growth_percent": np.nan,
                "annual_openings": np.nan,
                "projection_median_salary": np.nan,
                "typical_education": "",
                "related_work_experience": "",
                "on_the_job_training": "",
                "document": clean_text(
                    f"career role {title} {title} description {description} related titles {alt} skills {skills_text}"
                ),
            }
        )
    return pd.DataFrame(records)
