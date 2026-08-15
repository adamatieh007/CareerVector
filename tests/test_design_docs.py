from pathlib import Path
import re


DOCS = Path(__file__).resolve().parents[1] / "design_docs"
MERMAID_RE = re.compile(r'<div class="mermaid">\s*(.*?)\s*</div>', re.DOTALL)
HREF_RE = re.compile(r'href="([^"]+\.html)"')
SUPPORTED_STARTS = (
    "flowchart LR",
    "flowchart TD",
    "flowchart TB",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram-v2",
)


def test_design_docs_have_valid_links_visible_fallbacks_and_diagram_sources() -> None:
    pages = sorted(DOCS.glob("*.html"))
    assert len(pages) == 7

    diagram_count = 0
    for page in pages:
        text = page.read_text(encoding="utf-8")

        for href in HREF_RE.findall(text):
            assert (DOCS / href).exists(), f"Broken design-doc link in {page.name}: {href}"

        diagrams = MERMAID_RE.findall(text)
        assert diagrams, f"No Mermaid diagrams found in {page.name}"
        diagram_count += len(diagrams)

        for source in diagrams:
            source = source.strip()
            assert source.startswith(SUPPORTED_STARTS), (
                f"Unexpected Mermaid diagram type in {page.name}: {source.splitlines()[0]}"
            )
            for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
                assert source.count(opening) == source.count(closing)

        assert "mermaid.parse(source)" in text
        assert "if (!window.mermaid)" in text
        assert ".mermaid-fallback" in text
        assert "color:#111827" in text or "color: #111827" in text

    assert diagram_count >= 20
