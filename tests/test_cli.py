import json
from pathlib import Path

from typer.testing import CliRunner

from paper_rag.cli import app

runner = CliRunner()


def test_evaluate_command_passes_baseline() -> None:
    result = runner.invoke(
        app,
        [
            "evaluate",
            "evaluation/retrieval_baseline.json",
            "--top-k",
            "3",
            "--min-recall",
            "0.8",
            "--min-mrr",
            "0.7",
        ],
    )

    assert result.exit_code == 0
    assert "Recall@3:" in result.stdout
    assert "MRR:" in result.stdout


def test_evaluate_command_compares_vector_and_hybrid_search() -> None:
    result = runner.invoke(
        app,
        [
            "evaluate",
            "evaluation/retrieval_baseline.json",
            "--embedding",
            "hashing",
            "--retrieval-mode",
            "both",
        ],
    )

    assert result.exit_code == 0
    assert "[hashing/vector]" in result.stdout
    assert "[hashing/hybrid]" in result.stdout
    assert "Delta hybrid - vector (hashing)" in result.stdout


def test_evaluate_command_fails_quality_gate(tmp_path: Path) -> None:
    dataset = {
        "documents": [
            {
                "title": "Known paper",
                "source": "known-paper",
                "text": "retrieval embeddings context",
            }
        ],
        "cases": [
            {
                "question": "retrieval embeddings",
                "relevant_sources": ["missing-paper"],
            }
        ],
    }
    path = tmp_path / "failing-dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    result = runner.invoke(app, ["evaluate", str(path), "--min-recall", "1.0"])

    assert result.exit_code == 1
    assert "Gate failed" in result.stderr
