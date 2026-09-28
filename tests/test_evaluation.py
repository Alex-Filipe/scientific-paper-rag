import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from paper_rag.adapters.generation import ABSTENTION_ANSWER
from paper_rag.bootstrap import build_container
from paper_rag.core.contracts import RetrievedChunk
from paper_rag.evaluation.dataset import load_dataset, load_generation_dataset
from paper_rag.evaluation.metrics import recall_at_k, reciprocal_rank
from paper_rag.evaluation.models import EvaluationDocument, GenerationCase, GenerationDataset
from paper_rag.evaluation.runner import GenerationEvaluator, RetrievalEvaluator
from paper_rag.settings import Settings

BASELINE_DATASET = Path("evaluation/retrieval_baseline.json")


def test_retrieval_metrics() -> None:
    retrieved = ["unrelated", "paper-a", "paper-b"]
    relevant = ["paper-a", "paper-c"]

    assert recall_at_k(retrieved, relevant) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_metrics_require_relevant_sources() -> None:
    with pytest.raises(ValueError, match="relevant source"):
        recall_at_k(["paper-a"], [])
    with pytest.raises(ValueError, match="relevant source"):
        reciprocal_rank(["paper-a"], [])


def test_baseline_retrieval_quality(tmp_path: Path) -> None:
    dataset = load_dataset(BASELINE_DATASET)
    container = build_container(Settings(database_path=tmp_path / "evaluation.db"))
    evaluator = RetrievalEvaluator(container.ingest_document, container.retrieve_chunks)

    report = evaluator.run(dataset, top_k=3)

    assert report.recall_at_k >= 0.8
    assert report.mean_reciprocal_rank >= 0.7
    assert report.meets(minimum_recall=0.8, minimum_mrr=0.7)


def test_generation_baseline_dataset_is_valid() -> None:
    dataset = load_generation_dataset(Path("evaluation/generation_baseline.json"))

    assert len(dataset.documents) == 3
    assert len(dataset.cases) == 6
    assert sum(case.expected_abstention for case in dataset.cases) == 2


def test_generation_dataset_requires_expected_facts_in_evidence(tmp_path: Path) -> None:
    path = tmp_path / "invalid-generation.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {"title": "Pilot", "source": "pilot", "text": "The result was 37%."}
                ],
                "cases": [
                    {
                        "question": "What was the result?",
                        "expected_facts": ["42%"],
                        "expected_abstention": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must appear in the evaluation documents"):
        load_generation_dataset(path)


def test_generation_evaluator_scores_answers_citations_and_abstention(
    tmp_path: Path,
) -> None:
    class FakeGenerator:
        def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
            del contexts
            if question == "Who funded the pilot?":
                return ABSTENTION_ANSWER
            if question == "What was the result?":
                return "The result was 37% [1]."
            return "The result was 37% [2]."

    dataset = GenerationDataset(
        documents=(EvaluationDocument("Pilot", "The result was 37%.", "pilot"),),
        cases=(
            GenerationCase("What was the result?", ("37%",), False),
            GenerationCase("What is the result with a bad citation?", ("37%",), False),
            GenerationCase("Who funded the pilot?", (), True),
        ),
    )
    container = build_container(Settings(database_path=tmp_path / "generation.db", top_k=1))
    container.ask_question.generator = FakeGenerator()
    evaluator = GenerationEvaluator(
        ingest_document=container.ingest_document,
        ask_question=container.ask_question,
        backend="test",
        model="fake",
    )

    report = evaluator.run(dataset)

    assert report.answer_fact_match_rate == 1.0
    assert report.citation_valid_rate == 0.5
    assert report.citation_support_rate == 0.5
    assert report.abstention_accuracy == 1.0
    assert report.cases[1].citation_valid is False
    assert report.cases[2].abstention_correct
