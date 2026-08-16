from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from careervector.academic import academic_alignment_scores, best_program_matches
from careervector.profile import CareerProfile

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def outlook_scores(metadata: pd.DataFrame) -> np.ndarray:
    """Convert BLS 10-year growth into a bounded 0..1 auxiliary score."""
    scores = np.full(len(metadata), np.nan, dtype=np.float32)
    if "growth_percent" not in metadata.columns:
        return scores
    growth = pd.to_numeric(metadata["growth_percent"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(growth)
    scores[valid] = np.clip((growth[valid] + 10.0) / 30.0, 0.0, 1.0).astype(np.float32)
    return scores


def _stem_token(token: str) -> str:
    aliases = {
        "engineering": "engineer", "engineers": "engineer",
        "nursing": "nurse", "nurses": "nurse",
        "accounting": "accountant", "accountants": "accountant",
        "physics": "physicist", "physicists": "physicist",
        "chemistry": "chemist", "chemists": "chemist",
        "biology": "biologist", "biologists": "biologist",
        "psychology": "psychologist", "psychologists": "psychologist",
        "pharmacy": "pharmacist", "pharmacists": "pharmacist",
        "architecture": "architect", "architects": "architect",
        "economics": "economist", "economists": "economist",
        "statistics": "statistician", "statisticians": "statistician",
        "mathematics": "mathematician", "mathematicians": "mathematician",
    }
    if token in aliases:
        return aliases[token]
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: object) -> set[str]:
    return {_stem_token(token) for token in _TOKEN_RE.findall(str(value).lower()) if len(token) > 1}


def title_match_scores(metadata: pd.DataFrame, profile: CareerProfile) -> np.ndarray:
    """Give specific role titles an independent exact/near-exact preference signal.

    This is especially useful after expanding O*NET's 1K occupations into ~58K job-title
    records: a query containing 'FPGA' should favor an FPGA Engineer role over a generic
    engineering-management title even when both share the same academic major.
    """
    values = [
        profile.concentration,
        *profile.interests,
        *profile.specializations,
        *profile.skills,
        *profile.keywords,
    ]
    phrases = [" ".join(_tokens(value)) for value in values if _tokens(value)]
    query_tokens = set().union(*(_tokens(value) for value in values)) if values else set()
    academic_tokens = _tokens(profile.major) | _tokens(profile.concentration)
    if not query_tokens and academic_tokens:
        query_tokens = set(academic_tokens)
    scores = np.zeros(len(metadata), dtype=np.float32)
    if not query_tokens:
        return scores

    titles = metadata.get("title", pd.Series("", index=metadata.index)).fillna("").astype(str)
    parents = metadata.get("parent_title", pd.Series("", index=metadata.index)).fillna("").astype(str)

    for i, (title, parent) in enumerate(zip(titles, parents, strict=False)):
        title_norm = " ".join(_tokens(title))
        title_tokens = _tokens(title)
        parent_tokens = _tokens(parent)
        if not title_tokens:
            continue

        exact_phrase = max((1.0 if phrase and phrase in title_norm else 0.0 for phrase in phrases), default=0.0)
        containment = len(query_tokens & title_tokens) / max(1, min(len(query_tokens), len(title_tokens)))
        jaccard = len(query_tokens & title_tokens) / max(1, len(query_tokens | title_tokens))
        parent_overlap = len(query_tokens & parent_tokens) / max(1, len(query_tokens))
        academic_title_overlap = len(academic_tokens & (title_tokens | parent_tokens)) / max(1, len(academic_tokens)) if academic_tokens else 0.0
        semantic_title = 0.65 * containment + 0.25 * jaccard + 0.10 * parent_overlap
        scores[i] = np.float32(min(1.0, max(exact_phrase, semantic_title, 0.55 * academic_title_overlap)))
    return scores


def combine_relevance_scores(
    retrieval_scores: np.ndarray,
    metadata: pd.DataFrame,
    profile: CareerProfile,
    *,
    retrieval_weight: float = 0.58,
    academic_weight: float = 0.24,
    title_weight: float = 0.13,
    outlook_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend retrieval with title specificity, academic compatibility, and outlook.

    Missing academic/outlook data does not count as zero; its weight is redistributed over
    signals that exist for the row. Title matching always exists but may be zero.
    """
    retrieval = np.clip(np.asarray(retrieval_scores, dtype=np.float32), 0.0, 1.0)
    academic = academic_alignment_scores(metadata, profile)
    title = title_match_scores(metadata, profile)
    outlook = outlook_scores(metadata)

    total = np.full(len(metadata), retrieval_weight + title_weight, dtype=np.float32)
    numerator = retrieval_weight * retrieval + title_weight * title

    academic_valid = np.isfinite(academic)
    numerator[academic_valid] += academic_weight * academic[academic_valid]
    total[academic_valid] += academic_weight

    outlook_valid = np.isfinite(outlook)
    numerator[outlook_valid] += outlook_weight * outlook[outlook_valid]
    total[outlook_valid] += outlook_weight

    total[total == 0] = 1.0
    return numerator / total, academic, title, outlook


def select_diverse_indices(
    scores: np.ndarray,
    eligible_indices: np.ndarray,
    metadata: pd.DataFrame,
    *,
    top_k: int,
    max_per_parent: int = 2,
) -> np.ndarray:
    """Prevent one broad parent occupation from flooding the result list with aliases."""
    order = eligible_indices[np.argsort(scores[eligible_indices])[::-1]]
    selected: list[int] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for idx in order:
        row = metadata.iloc[int(idx)]
        parent_key = str(row.get("base_soc") or row.get("parent_title") or row.get("role_id") or idx)
        if counts[parent_key] >= max_per_parent:
            continue
        selected.append(int(idx))
        counts[parent_key] += 1
        if len(selected) >= top_k:
            break
    return np.asarray(selected, dtype=int)


def academic_matches_for_row(row: pd.Series, profile: CareerProfile, *, limit: int = 4) -> list[str]:
    query = profile.academic_text()
    if not query:
        return []
    return [program for program, _ in best_program_matches(query, row.get("compatible_majors"), limit=limit)]
