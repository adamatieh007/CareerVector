from __future__ import annotations

from collections.abc import Iterable, Mapping


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return " ".join(text.split())


def _items(value: object, *, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[object] = value.split("|")
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    cleaned = [_clean(item) for item in values]
    return [item for item in cleaned if item][:limit]


def _money(value: object) -> str:
    try:
        if value is None:
            return "Unavailable"
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "Unavailable"


def _percent(value: object) -> str:
    try:
        if value is None:
            return "Unavailable"
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def build_rag_context(
    recommendations: list[Mapping[str, object]],
    *,
    max_chars_per_source: int = 5200,
) -> str:
    """Turn retrieved career roles into citation-addressable grounding context."""
    blocks: list[str] = []
    for index, item in enumerate(recommendations, start=1):
        source_id = f"CV{index}"
        parent = _clean(item.get("parent_occupation"))
        title = _clean(item.get("occupation"))
        lines = [
            f"[{source_id}]",
            f"Career role: {title}",
            f"Parent occupation: {parent}" if parent and parent != title else "",
            f"Taxonomy source: {_clean(item.get('source'))}",
            f"O*NET-SOC: {_clean(item.get('onet_soc_code'))}",
            f"Combined recommendation score: {_clean(item.get('match_score'))}",
            f"Retrieval score: {_clean(item.get('retrieval_score'))}",
            f"Academic alignment score: {_clean(item.get('academic_alignment'))}",
            f"Description: {_clean(item.get('description'))}",
            "Related titles: " + ", ".join(_items(item.get("sample_job_titles"), limit=8)),
            "Matched academic programs: " + ", ".join(_items(item.get("academic_matches"), limit=6)),
            f"Median annual wage: {_money(item.get('median_salary'))}",
            f"Mean annual wage: {_money(item.get('mean_salary'))}",
            f"BLS projected employment growth, 2024-2034: {_percent(item.get('growth_percent'))}",
            f"BLS annual openings, 2024-2034 average (thousands): {_clean(item.get('annual_openings'))}",
            f"Typical entry education: {_clean(item.get('typical_education'))}",
            "Interests: " + ", ".join(_items(item.get("top_interests"), limit=8)),
            "Skills: " + ", ".join(_items(item.get("top_skills"), limit=8)),
            "Knowledge: " + ", ".join(_items(item.get("top_knowledge"), limit=8)),
            "Work activities: " + ", ".join(_items(item.get("top_activities"), limit=8)),
            "Core tasks: " + "; ".join(_items(item.get("core_tasks"), limit=6)),
            "Software/technologies: " + ", ".join(_items(item.get("software_skills"), limit=10)),
            f"[/{source_id}]",
        ]
        block = "\n".join(line for line in lines if line and not line.endswith(": "))
        blocks.append(block[:max_chars_per_source])
    return "\n\n".join(blocks)
