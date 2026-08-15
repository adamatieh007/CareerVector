CareerVector Design Documentation
=================================

Recommended repo location:
  docs/design/

Start page:
  docs/design/index.html

The HTML files use Mermaid 11.16.1 through a classic browser script (not an ES-module import),
which avoids the local-file module-loading failure that caused raw Mermaid source to appear.

Diagram behavior:
- Normal case: Mermaid renders each diagram as SVG.
- If Mermaid cannot load: the Mermaid source remains visible with high-contrast dark text instead
  of becoming unreadable white-on-white text.
- Each diagram is parsed/rendered independently, so a failure in one diagram does not stop the
  other diagrams on the page.
