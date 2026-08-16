from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from careervector.profile import CareerProfile

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC = {
    "general",
    "other",
    "miscellaneous",
    "studies",
    "program",
    "programs",
    "degree",
}


def normalize_program_name(value: object) -> str:
    tokens = [t for t in _TOKEN_RE.findall(str(value).lower()) if t not in _GENERIC]
    return " ".join(tokens)


def program_similarity(query: str, candidate: str) -> float:
    """Conservative lexical/fuzzy similarity for academic program names.

    This deliberately does not use the sentence embedding model. The academic prior is
    intended to be an independent structured signal from the federal CIP-SOC crosswalk.
    """
    q = normalize_program_name(query)
    c = normalize_program_name(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q in c or c in q:
        shorter = min(len(q.split()), len(c.split()))
        longer = max(len(q.split()), len(c.split()))
        return max(0.82, shorter / max(1, longer))

    q_tokens = set(q.split())
    c_tokens = set(c.split())
    intersection = len(q_tokens & c_tokens)
    union = len(q_tokens | c_tokens)
    jaccard = intersection / union if union else 0.0
    sequence = SequenceMatcher(None, q, c).ratio()

    # Exact domain words such as engineering, physics, nursing, accounting, etc.
    # should matter more than punctuation/order variations.
    containment = intersection / max(1, min(len(q_tokens), len(c_tokens)))
    return float(min(1.0, 0.50 * containment + 0.30 * jaccard + 0.20 * sequence))


def best_program_matches(query: str, compatible_majors: object, *, limit: int = 4) -> list[tuple[str, float]]:
    if compatible_majors is None or pd.isna(compatible_majors):
        return []
    programs = [p.strip() for p in str(compatible_majors).split("|") if p.strip()]
    scored = [(program, program_similarity(query, program)) for program in programs]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item for item in scored[:limit] if item[1] > 0.0]


def academic_alignment_scores(metadata: pd.DataFrame, profile: CareerProfile) -> np.ndarray:
    """Return a 0..1 academic compatibility prior, NaN when unavailable.

    Scores are computed once per SOC and broadcast to all specific role-title records that
    inherit that SOC. This prevents the 57K title corpus from doing duplicated work.
    """
    query = profile.academic_text()
    scores = np.full(len(metadata), np.nan, dtype=np.float32)
    if not query.strip() or "compatible_majors" not in metadata.columns:
        return scores

    if "base_soc" in metadata.columns:
        groups = metadata.groupby(metadata["base_soc"].fillna(""), sort=False).indices
    else:
        groups = {str(i): np.asarray([i]) for i in range(len(metadata))}

    major_query = profile.major.strip()
    concentration_query = profile.concentration.strip()

    for _, indices in groups.items():
        idx = int(indices[0])
        raw = metadata.iloc[idx].get("compatible_majors")
        if raw is None or pd.isna(raw) or not str(raw).strip():
            continue

        programs = [p.strip() for p in str(raw).split("|") if p.strip()]
        if not programs:
            continue

        major_score = max((program_similarity(major_query, p) for p in programs), default=0.0) if major_query else np.nan
        concentration_score = (
            max((program_similarity(concentration_query, p) for p in programs), default=0.0)
            if concentration_query
            else np.nan
        )

        if np.isfinite(major_score) and np.isfinite(concentration_score):
            score = 0.75 * float(major_score) + 0.25 * float(concentration_score)
        elif np.isfinite(major_score):
            score = float(major_score)
        elif np.isfinite(concentration_score):
            score = float(concentration_score)
        else:
            continue
        scores[np.asarray(indices, dtype=int)] = np.float32(score)

    return scores
