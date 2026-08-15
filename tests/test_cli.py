from careervector.cli import build_parser


def test_cli_accepts_rag_options() -> None:
    args = build_parser().parse_args(
        [
            "--method",
            "rag",
            "--major",
            "Biomedical Physics",
            "--llm-model",
            "gemma3:4b",
            "--ollama-url",
            "http://localhost:11434",
        ]
    )
    assert args.method == "rag"
    assert args.llm_model == "gemma3:4b"
