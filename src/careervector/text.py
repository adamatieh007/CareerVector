import re
from collections.abc import Iterable

SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return SPACE_RE.sub(" ", text)


def split_csv_text(value: str | None) -> list[str]:
    if not value:
        return []
    return [clean_text(x) for x in value.split(",") if clean_text(x)]


def repeat_phrases(values: Iterable[tuple[str, float]], max_repeat: int = 5) -> str:
    """Repeat high-rated phrases so ratings influence TF-IDF term frequency."""
    chunks: list[str] = []
    for phrase, score in values:
        phrase = clean_text(phrase)
        if not phrase:
            continue
        repetitions = max(1, min(max_repeat, int(round(float(score)))))
        chunks.extend([phrase] * repetitions)
    return " ".join(chunks)


def normalize_soc(onet_soc: str) -> str:
    """Convert O*NET 17-2061.00/17-2061.01 to base SOC 17-2061 for BLS joins."""
    return clean_text(onet_soc).split(".", maxsplit=1)[0]
