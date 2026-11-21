import json
from pathlib import Path

from pytest import MonkeyPatch
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


def test_ingest_and_ask_commands(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHROMA_DIR", str(tmp_path / "chroma"))
    paper = tmp_path / "paper.md"
    paper.write_text("Chunking keeps context across boundaries.", encoding="utf-8")

    ingested = runner.invoke(app, ["ingest", str(paper), "--title", "Chunking"])
    answered = runner.invoke(app, ["ask", "What keeps context?", "--show-steps"])

    assert ingested.exit_code == 0
    assert "Ingested 1 chunks" in ingested.stdout
    assert answered.exit_code == 0
    assert "retrieve:" in answered.stdout
    assert "[1]  Chunking" in answered.stdout
