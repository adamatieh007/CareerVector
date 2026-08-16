from dataclasses import dataclass, field

from careervector.text import clean_text


@dataclass(slots=True)
class CareerProfile:
    major: str = ""
    concentration: str = ""
    interests: list[str] = field(default_factory=list)
    specializations: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    preferred_work: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    minimum_salary: float | None = None

    def positive_text(self) -> str:
        """Build a weighted lexical query for the TF-IDF baseline."""
        parts: list[str] = []

        if clean_text(self.major):
            parts.extend([self.major] * 2)
        if clean_text(self.concentration):
            parts.extend([self.concentration] * 3)

        for value in self.interests:
            parts.extend([value] * 3)
        for value in self.specializations:
            parts.extend([value] * 3)
        for value in self.skills:
            parts.extend([value] * 3)
        for value in self.preferred_work:
            parts.extend([value] * 2)
        for value in self.keywords:
            parts.extend([value] * 2)

        return " ".join(clean_text(x) for x in parts if clean_text(x))

    def semantic_text(self) -> str:
        """Build natural-language profile text for dense semantic retrieval."""
        parts: list[str] = []
        if clean_text(self.major):
            parts.append(f"Academic background: {clean_text(self.major)}.")
        if clean_text(self.concentration):
            parts.append(f"Concentration: {clean_text(self.concentration)}.")
        if self.interests:
            parts.append("Interests: " + ", ".join(clean_text(x) for x in self.interests if clean_text(x)) + ".")
        if self.specializations:
            parts.append(
                "Technical specializations: "
                + ", ".join(clean_text(x) for x in self.specializations if clean_text(x))
                + "."
            )
        if self.skills:
            parts.append("Skills: " + ", ".join(clean_text(x) for x in self.skills if clean_text(x)) + ".")
        if self.preferred_work:
            parts.append(
                "Preferred work: "
                + ", ".join(clean_text(x) for x in self.preferred_work if clean_text(x))
                + "."
            )
        if self.keywords:
            parts.append(
                "Additional keywords: "
                + ", ".join(clean_text(x) for x in self.keywords if clean_text(x))
                + "."
            )
        return " ".join(parts)

    def academic_text(self) -> str:
        values = [clean_text(self.major), clean_text(self.concentration)]
        return " ".join(value for value in values if value)

    def avoid_text(self) -> str:
        return " ".join(clean_text(x) for x in self.avoid if clean_text(x))
