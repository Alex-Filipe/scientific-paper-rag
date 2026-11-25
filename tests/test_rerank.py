from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import KeywordCrossEncoder, ServiceFactory, fake_cross_encoder
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from paper_rag.api import create_app
from paper_rag.models import AnswerStatus, Step
from paper_rag.rerank import CUSTOM_CROSS_ENCODERS, CrossEncoderReranker, _register_custom_model
from paper_rag.service import RagService
from paper_rag.settings import Settings


def make_reranker(tmp_path: Path) -> tuple[CrossEncoderReranker, KeywordCrossEncoder]:
    reranker = CrossEncoderReranker("fake/cross-encoder", tmp_path)
    fake = KeywordCrossEncoder()
    reranker._model = fake
    return reranker, fake


def chunks(texts: Sequence[str]) -> list[Document]:
    return [
        Document(page_content=text, metadata={"rank": rank, "chunk_id": f"c{rank}"}, id=f"c{rank}")
        for rank, text in enumerate(texts, start=1)
    ]


def test_rerank_orders_by_score_and_cuts_top_k(tmp_path: Path) -> None:
    reranker, fake = make_reranker(tmp_path)
    candidates = chunks(["training details", "residual error rate", "top-5 error rate of resnet"])

    reranked = reranker.rerank("top-5 error rate", candidates, top_k=2)

    assert [document.id for document in reranked] == ["c3", "c2"]
    assert [document.metadata["rank"] for document in reranked] == [1, 2]
    assert [document.metadata["search_rank"] for document in reranked] == [3, 2]
    assert reranked[0].metadata["rerank_score"] == 4.0  # top, 5, error, rate
    assert fake.calls == [("top-5 error rate", [c.page_content for c in candidates])]


def test_ties_keep_the_search_order(tmp_path: Path) -> None:
    reranker, _ = make_reranker(tmp_path)

    reranked = reranker.rerank("unrelated", chunks(["a", "b", "c"]), top_k=3)

    assert [document.id for document in reranked] == ["c1", "c2", "c3"]


def test_missing_search_rank_uses_the_input_position(tmp_path: Path) -> None:
    reranker, _ = make_reranker(tmp_path)
    candidates = [Document(page_content="error"), Document(page_content="error rate")]

    reranked = reranker.rerank("error rate", candidates, top_k=2)

    assert [document.metadata["search_rank"] for document in reranked] == [2, 1]


def test_empty_candidates_skip_the_model(tmp_path: Path) -> None:
    reranker, fake = make_reranker(tmp_path)

    assert reranker.rerank("question", [], top_k=4) == []
    assert fake.calls == []


def test_top_k_must_be_positive(tmp_path: Path) -> None:
    reranker, _ = make_reranker(tmp_path)

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("question", chunks(["a"]), top_k=0)


class FakeEncoderClass:
    def __init__(self, registered: Sequence[str]) -> None:
        self.registered = list(registered)
        self.added: list[dict[str, Any]] = []

    def list_supported_models(self) -> list[dict[str, Any]]:
        return [{"model": name} for name in self.registered]

    def add_custom_model(self, **kwargs: Any) -> None:
        self.added.append(kwargs)
        self.registered.append(kwargs["model"])


def test_custom_cross_encoder_is_registered_once() -> None:
    name = next(iter(CUSTOM_CROSS_ENCODERS))
    encoder_class = FakeEncoderClass(["Xenova/ms-marco-MiniLM-L-6-v2"])

    _register_custom_model(encoder_class, name)
    _register_custom_model(encoder_class, name)

    assert len(encoder_class.added) == 1
    added = encoder_class.added[0]
    assert added["model"] == name
    assert added["sources"].hf == name
    assert added["model_file"] == "onnx/model.onnx"


def test_builtin_cross_encoder_is_not_registered() -> None:
    encoder_class = FakeEncoderClass(["Xenova/ms-marco-MiniLM-L-6-v2"])

    _register_custom_model(encoder_class, "Xenova/ms-marco-MiniLM-L-6-v2")

    assert encoder_class.added == []


CORPUS = {
    "Optimization": "We use SGD. The learning rate starts from 0.1 and is divided by 10 when the "
    "error plateaus.",
    "Architecture": "Residual networks add identity shortcuts between stacked layers.",
    "Data": "The ImageNet dataset has 1.28 million training images in 1000 classes.",
    "Results": "ResNet-152 reaches a top-5 error of 4.49 percent on the validation set.",
    "Detection": "Faster R-CNN with ResNet-101 improves object detection on COCO.",
    "CIFAR": "On CIFAR-10 the 110-layer network reaches 6.43 percent error.",
}
QUESTION = "How does the learning rate change when the error plateaus?"


@pytest.fixture
def reranked(make_service: ServiceFactory) -> tuple[RagService, KeywordCrossEncoder]:
    service = make_service(rerank=True, top_k=2)
    fake = fake_cross_encoder(service)
    for title, text in CORPUS.items():
        service.ingest_text(title, text, f"{title}.md")
    return service, fake


def test_settings_turn_the_rerank_node_on(make_service: ServiceFactory) -> None:
    plain = make_service()
    reranking = make_service(rerank=True, rerank_model="org/cross-encoder")

    assert plain.reranker is None
    assert "rerank" not in plain.graph.nodes
    assert plain.config().rerank_model is None
    assert reranking.reranker is not None
    assert reranking.reranker.model_name == "org/cross-encoder"
    assert "rerank" in reranking.graph.nodes
    assert reranking.config().rerank_model == "org/cross-encoder"


def test_rerank_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().rerank is False
    monkeypatch.setenv("RAG_RERANK", "true")
    monkeypatch.setenv("RAG_RERANK_MODEL", "org/cross-encoder")

    settings = Settings.from_environment()

    assert (settings.rerank, settings.rerank_model) == (True, "org/cross-encoder")


def test_rerank_node_keeps_the_best_candidates(
    reranked: tuple[RagService, KeywordCrossEncoder],
) -> None:
    service, fake = reranked

    answer = service.ask(QUESTION)

    assert [step.node for step in answer.steps][:2] == ["retrieve", "rerank"]
    assert answer.steps[0].detail.endswith("6 candidato(s) para reordenar.")
    assert answer.steps[1].detail.startswith("6 candidato(s) reordenados; ficaram 2,")
    assert fake.calls[0][0] == QUESTION
    assert len(fake.calls[0][1]) == len(CORPUS)  # every chunk fits in candidate_k=20
    assert [citation.rank for citation in answer.citations] == [1, 2]
    best, second = answer.citations
    assert best.title == "Optimization"
    assert best.rerank_score is not None and second.rerank_score is not None
    assert best.rerank_score > second.rerank_score
    assert best.search_rank is not None


def test_service_retrieve_matches_the_graph(
    reranked: tuple[RagService, KeywordCrossEncoder],
) -> None:
    service, _ = reranked

    retrieved = service.retrieve(QUESTION, 2)

    assert [document.metadata["rank"] for document in retrieved] == [1, 2]
    assert retrieved[0].metadata["document_title"] == "Optimization"
    assert [c.chunk_id for c in service.ask(QUESTION).citations] == [
        document.metadata["chunk_id"] for document in retrieved
    ]


def test_rerank_without_candidates_abstains(make_service: ServiceFactory) -> None:
    service = make_service(rerank=True)
    fake = fake_cross_encoder(service)

    answer = service.ask("Anything indexed?")

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.steps[1] == Step("rerank", "Nenhum candidato para reordenar.")
    assert fake.calls == []


def test_api_exposes_the_reranker(
    reranked: tuple[RagService, KeywordCrossEncoder], make_service: ServiceFactory
) -> None:
    service, _ = reranked
    client = TestClient(create_app(service))
    plain = make_service()
    plain.ingest_text("Optimization", CORPUS["Optimization"], "Optimization.md")

    config = client.get("/config").json()
    citation = client.post("/questions", json={"question": QUESTION}).json()["citations"][0]
    without = TestClient(create_app(plain)).post("/questions", json={"question": QUESTION})

    assert (config["rerank_model"], config["candidate_k"]) == (service.settings.rerank_model, 20)
    assert citation["search_rank"] >= 1
    assert citation["rerank_score"] > 0
    assert without.json()["citations"][0]["search_rank"] is None
    assert without.json()["citations"][0]["rerank_score"] is None
