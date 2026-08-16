# CareerVector v0.4 Validation Notes

These checks are **engineering sanity checks**, not a claim of production recommendation accuracy.

## Real-data corpus build

Using the official O*NET 30.3, NCES CIP-SOC, and BLS Employment Projections files, the v0.4 build completed with:

```text
Parent O*NET occupations:        1,016
Career-role documents:          58,556
Salary coverage:                55,148 / 58,556
NCES academic mapping coverage: 58,556 / 58,556
BLS projections coverage:       56,093 / 58,556
Optional ESCO roles:            0 in this validation run
```

The TF-IDF baseline trained successfully on the role corpus:

```text
TF-IDF matrix shape: (58,556, 125,000)
```

## Retrieval sanity suite

`python scripts/evaluate.py --method tfidf` passed all eight included Recall@5 cases:

```text
computer hardware and FPGA        PASS
biomedical and medical physics    PASS
physics research                  PASS
data science and machine learning PASS
civil infrastructure              PASS
cybersecurity                     PASS
accounting and audit              PASS
registered nursing                PASS

Recall@5 sanity cases: 8 / 8
```

The evaluator normalizes detailed O*NET variants to their base SOC family when checking expected occupation families.

## Automated tests

```text
27 passed
```

Coverage includes profile construction, academic matching, NCES/BLS/ESCO adapters, title ranking, diversity, TF-IDF behavior, embedding helpers, RAG context, Ollama client behavior, CLI options, artifact-version compatibility, and design-document structure.

## Example retrieval behavior

A Computer Engineering / Computer Architecture profile emphasizing FPGA, embedded systems, GPU architecture, low-latency systems, C++, SystemVerilog, Verilog, and CUDA returned specific role titles including:

- Embedded Systems Software Developer
- Hardware Systems Engineer
- FPGA Design Engineer
- Computer Systems Software Architect

A Biomedical Physics / Medical Physics profile emphasizing radiation oncology, dosimetry, medical imaging, radiation physics, and clinical research returned titles including:

- Medical Physics Professor
- Medical Physics Teacher
- Radiation Physicist
- Radiation Control Health Physicist
- Clinical Research Scientist
- Oncology Clinical Research Coordinator

These examples illustrate the v0.4 goal: retrieve specific career-role labels and then preserve the broader parent occupation as supporting taxonomy metadata.
