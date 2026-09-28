import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import paper_rag.cli as cli
from paper_rag.adapters.generation import ABSTENTION_ANSWER
from paper_rag.bootstrap import Container
from paper_rag.cli import app
from paper_rag.core.contracts import RetrievedChunk
from paper_rag.settings import Settings

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


def test_evaluate_generation_command_reports_metrics_without_real_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = {
        "documents": [
            {"title": "Pilot", "source": "pilot", "text": "The result was 37%."}
        ],
        "cases": [
            {
                "question": "What was the result?",
                "expected_facts": ["37%"],
                "expected_abstention": False,
            },
            {
                "question": "Who funded the pilot?",
                "expected_facts": [],
                "expected_abstention": True,
            },
        ],
    }
    path = tmp_path / "generation.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    monkeypatch.setenv("RAG_GENERATION_BACKEND", "ollama")
    monkeypatch.setenv("RAG_OLLAMA_MODEL", "test-model")

    class FakeGenerator:
        def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
            del contexts
            if question == "Who funded the pilot?":
                return ABSTENTION_ANSWER
            return "The result was 37% [1]."

    original_build_container = cli.build_container

    def build_fake_container(settings: Settings) -> Container:
        container = original_build_container(replace(settings, generation_backend="extractive"))
        container.ask_question.generator = FakeGenerator()
        return container

    monkeypatch.setattr(cli, "build_container", build_fake_container)

    result = runner.invoke(app, ["evaluate-generation", str(path)])

    assert result.exit_code == 0
    assert "Backend/model: ollama/test-model" in result.stdout
    assert "Expected facts in answer: 1/1 (1.000)" in result.stdout
    assert "Valid citations: 1/1 (1.000)" in result.stdout
    assert "Cited evidence supports expected facts: 1/1 (1.000)" in result.stdout
    assert "Abstention decisions: 2/2 (1.000)" in result.stdout
