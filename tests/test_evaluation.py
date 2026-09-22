from pathlib import Path

import pytest

from paper_rag.bootstrap import build_container
from paper_rag.evaluation.dataset import load_dataset
from paper_rag.evaluation.metrics import recall_at_k, reciprocal_rank
from paper_rag.evaluation.runner import RetrievalEvaluator
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
